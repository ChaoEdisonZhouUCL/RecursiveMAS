# How `run.py` Works: `sequential_light` on MATH-500

**Command:**
```bash
python run.py --style sequential_light --batch_size 32 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

---

## 1. Argument Parsing & Style Resolution (`run.py`)

`run.py` parses all CLI flags and resolves `--style sequential_light` into three HuggingFace model identifiers via `STYLE_SPECS` in [load_from_repo.py](load_from_repo.py):

| Role | Model |
|---|---|
| Planner | `RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B` |
| Refiner (Critic) | `RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B` |
| Solver | `RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B` |
| Outer Adapters | `RecursiveMAS/Sequential-Light-Outerlinks` |

Additional defaults are inferred from the dataset:
- `--max_new_tokens 1000` (inferred for math500)
- `--latent_steps 48` (inferred as `latent_length=48` for math500)
- `--mas_shape chain`

`run.py` then calls `inference_mas.main()` with these resolved arguments.

---

## 2. Dataset Loading (`inference_mas.py`)

- Resolves `math500` → `HuggingFaceH4/MATH-500` on HuggingFace
- Loads the `test` split
- Shuffles with `seed=42` for reproducibility
- Processes all samples (`num_samples=-1` means no cap)

---

## 3. The Recursive Latent Inference Pipeline

This is the core of the system. The method is called `ours_recursive` and runs **3 rounds** by default. Instead of passing text between agents, each agent passes **latent vectors** (raw hidden states from its transformer) to the next.

### What Are Latent Representations?

Each model produces a sequence of hidden-state tensors of shape `(seq_len, hidden_size)`. These are fed directly into the next model's embedding layer, bypassing tokenization entirely. Adapters bridge the dimension mismatch between models.

### Adapter Types (from [modeling.py](modeling.py))

- **Inner Adapter** (`ln_res_adapter`): Projects hidden states back into embedding space within the same model — `LayerNorm → Linear → GELU → Linear + residual`
- **Outer Adapter** (`outer_ln_res_adapter`): Projects embeddings across models — maps hidden dimensions from one model size to another

### Latent Rollout (per batch, per stage)

For each of the 48 latent steps:
1. Feed current embeddings through the model
2. Extract the last hidden state
3. Apply the inner adapter (maps hidden state → embedding)
4. Append result to the embedding sequence and extend the attention mask
5. Repeat

After all steps, apply the outer adapter to produce latents for the next agent.

---

### Round 1 — Cold Start

```
Question
   │
   ▼
[Planner latent rollout, 48 steps]
   │ planner_to_refiner latents (via outer_12 adapter)
   ▼
[Refiner latent rollout, 48 steps]
   │ refiner_to_solver latents (via outer_23 adapter)
   ▼
[Solver feedback latent rollout, 48 steps]  ← only if more rounds remain
   │ feedback_to_planner latents (via outer_31 adapter)
   ▼
  (passed to Round 2)
```

### Rounds 2 & 3 — Feedback-Conditioned

The planner now receives the solver's feedback latents from the previous round injected at a `FEEDBACK_SLOT` in its prompt:

```
[feedback latents from prev round]
   │
   ▼
[Planner (feedback-conditioned) latent rollout, 48 steps]
   │
   ▼
[Refiner latent rollout, 48 steps]
   │
   ▼
[Solver feedback latent rollout, 48 steps]   ← omitted on final round
```

### Final Stage — Text Generation

After the last recursive round, the solver uses the final refiner latents to **generate text**:

```python
model.generate(
    inputs_embeds=...,   # prefix + refiner_latent + suffix
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
```

This produces the actual answer string (e.g., containing `\boxed{...}`).

---

## 4. Answer Extraction & Evaluation

- If a sample is missing a `\boxed{...}` answer, a retry appends `"Final Answer: \boxed{"` and generates 16 more tokens.
- `compare_answers()` (from [inference_utils/answer_utils.py](inference_utils/answer_utils.py)) normalizes and checks each prediction.
- Final output:

```
accuracy=<value>% (<correct>/<total>)
```

`run.py` captures this with a regex and prints it as the result.

---

## 5. Full Execution Flow (Summary)

```
run.py
  ├─ Parse args & resolve sequential_light → 3 model paths + adapter paths
  ├─ Infer latent_steps=48, max_new_tokens=1000
  └─ Call inference_mas.main()
        ├─ Load MATH-500 (HuggingFaceH4/MATH-500, test split, seed=42)
        ├─ For each recursive round (×3):
        │     ├─ Planner latent rollout (48 steps, Qwen3-1.7B)
        │     │     └─ outer_12 adapter → refiner latents
        │     ├─ Refiner latent rollout (48 steps, Llama3.2-1B)
        │     │     └─ outer_23 adapter → solver latents
        │     └─ Solver feedback latent rollout (48 steps, Qwen2.5-Math-1.5B)
        │           └─ outer_31 adapter → feedback latents for next round
        ├─ Final solver text generation (temp=0.6, top_p=0.95, max_new_tokens=1000)
        ├─ Answer retry for missing \boxed{} answers
        └─ Compute & print accuracy=X% (correct/total)
```

---

## 6. Key Files

| File | Role |
|---|---|
| [run.py](run.py) | Entry point, CLI wrapper, accuracy capture |
| [load_from_repo.py](load_from_repo.py) | Style specs, dataset defaults |
| [inference_utils/inference_mas.py](inference_utils/inference_mas.py) | Core 3-agent recursive inference |
| [modeling.py](modeling.py) | Inner/outer adapter + `SharedRecursiveLink` implementations |
| [prompts.py](prompts.py) | Prompt templates with `PLANNER_SLOT`, `REFINED_SLOT`, `FEEDBACK_SLOT` |
| [inference_utils/answer_utils.py](inference_utils/answer_utils.py) | Answer extraction and comparison |

---

## 7. Evaluating Locally Trained Checkpoints

`train_outerlinks_math500.py` saves checkpoints at the end of training into
`outputs/checkpoints/<out_prefix>_<mode>_r<N>/`.  Two new `--style` values
let `run.py` evaluate these trained adapters against the released backbone
models.

### 7a. Original mode (three independent CrossModelAdapters)

After running training with `--mode original` (or `compare`), a checkpoint directory like
`outputs/checkpoints/outerlink_grad_original_r5/` is created.  Pass it via `--ckpt_dir`:

```bash
python run.py \
  --style sequential_light_trained \
  --ckpt_dir outputs/checkpoints/outerlink_grad_original_r5 \
  --batch_size 32 --temperature 0.6 --top_p 0.95 \
  --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

What changes vs the baseline `--style sequential_light` run:
- Backbone models (Planner, Critic, Solver) and their inner adapters are still
  loaded from the shared HF cache — unchanged.
- The three outer-link `.pt` files are loaded from `--ckpt_dir` instead of
  `RecursiveMAS/Sequential-Light-Outerlinks` on HuggingFace.

### 7b. SharedRecursiveLink mode (shared MoE + RoAE)

After running training with `--mode shared_roae` (or `compare`), a checkpoint directory like
`outputs/checkpoints/outerlink_grad_shared_roae_r5/` is created:

```bash
python run.py \
  --style sequential_light_shared_roae \
  --ckpt_dir outputs/checkpoints/outerlink_grad_shared_roae_r5 \
  --batch_size 32 --temperature 0.6 --top_p 0.95 \
  --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

What changes vs `sequential_light_trained`:
- Instead of three separate `CrossModelAdapter` files, a single
  `SharedRecursiveLink` (soft-MoE + Rotary Agent Encoding) is loaded from
  `shared_recursive_link.pt` + `shared_roae_config.json` in `--ckpt_dir`.
- `inference_mas.py` receives `--shared_link_path` and routes all cross-agent
  latent transfers through the shared module (calls `forward(h, src=N, dst=M)`
  with 1-indexed agent indices 1=Planner, 2=Critic, 3=Solver).
- The `--outer_12/23/31_path` flags are set to sentinel values and ignored
  whenever `--shared_link_path` is active.

### 7c. Side-by-side comparison

Run both commands sequentially (or in two separate SLURM jobs) to compare
accuracy on the same dataset.  The `compare` training mode in
`train_outerlinks_math500.py` produces **both** checkpoint directories in a
single training run, so you can evaluate both without retraining:

```bash
CKPT_PREFIX=outputs/checkpoints/outerlink_grad

# Baseline (released weights)
python run.py --style sequential_light \
  --batch_size 32 --temperature 0.6 --top_p 0.95 \
  --dataset math500 --seed 42 --trust_remote_code 1 --device cuda

# Trained original adapters
python run.py --style sequential_light_trained \
  --ckpt_dir ${CKPT_PREFIX}_original_r5 \
  --batch_size 32 --temperature 0.6 --top_p 0.95 \
  --dataset math500 --seed 42 --trust_remote_code 1 --device cuda

# Trained shared_roae adapters
python run.py --style sequential_light_shared_roae \
  --ckpt_dir ${CKPT_PREFIX}_shared_roae_r5 \
  --batch_size 32 --temperature 0.6 --top_p 0.95 \
  --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```
