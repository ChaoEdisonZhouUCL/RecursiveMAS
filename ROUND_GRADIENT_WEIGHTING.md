# Can round-wise learning-rate rescaling fix the per-round gradient imbalance?

`GRADIENT_ROUND_PROFILE.md` establishes *that* the per-round gradient spans five to
six orders of magnitude, and *why* (geometric decay for `shared_roae`, an identity
highway plus asymmetric init for `shared_state`). The natural follow-up is whether
re-weighting the rounds — "a learning rate per round" — would flatten the profile and
improve downstream accuracy.

This document answers that. Short version: not as stated, for a reason that sits
upstream of learning rates; there is a real defect underneath the question that has
now been fixed; and the current evaluation protocol could not have detected the
difference either way.

---

## TL;DR

1. **"A learning rate per round" is not a well-defined lever here.** The link
   parameters are *shared* across rounds, so the optimizer sees one gradient
   `g = Σ_r g_r`. There is no per-round parameter group. And AdamW normalises by the
   gradient's own running RMS, so scaling the *summed* gradient leaves the update
   essentially unchanged. The only intervention with any effect is re-weighting
   **before** the sum, `g = Σ_r c_r·g_r` — gradient surgery, not a learning rate.

2. **bfloat16 was discarding the early rounds before any weighting could act — now
   fixed.** The link was held in bf16 (eps 7.8e-3), and rounds accumulate into one
   `.grad` buffer starting from the largest. A contribution 1e-3 below the running
   total keeps ~70% of its norm but reaches only ~16% of coordinates; one 1e-6 below
   keeps ~2% and reaches 0.01%. The trained link now defaults to float32
   (`--link_dtype`, §3.3) while the frozen backbones stay bf16. The case for this rests
   on the numerics; a 50-step A/B confirms the path is correct but does **not** show it
   trains better (§3.4).

3. **Even so, expect little downstream movement, and note you cannot currently
   measure it.** Across all 46 configurations in `eval_results.md` — 4 link modes ×
   3 learning rates × 5 checkpoint steps — the between-config spread is at or *below*
   single-run binomial sampling noise on every one of the seven benchmarks (§5).
   Nothing tried so far has produced a detectable downstream effect, and a round
   re-weighting would land in the same noise floor.

---

## 1. Why "per-round learning rate" is not a lever

In `shared_roae` / `shared_state` a single `SharedRecursiveLink` (19.4M trainable
parameters) is applied at every round. The staged backward
(`train_outerlinks_math500.py`, `for r in range(cfg.n_rounds - 1, -1, -1)`) walks
rounds from last to first, and each stage accumulates into the *same* `.grad` buffer:

```
p.grad  =  g_R + g_{R-1} + … + g_1
```

Two consequences:

- **No per-round parameter group exists.** An optimizer learning rate multiplies the
  update to a parameter, and the parameter is one object used by all rounds. There is
  nothing for a per-round LR to attach to.
- **Adam absorbs a uniform rescale anyway.** The update is `m/(√v + eps)`, invariant
  to multiplying the gradient by a constant (for gradients well above `eps`). Scaling
  `p.grad` after accumulation changes almost nothing.

So the intervention that the question is really reaching for is

```
p.grad  =  Σ_r  c_r · g_r          # weights applied per stage, before the sum
```

which is implementable — the backward is already staged per round, so each `g_r` is
separable — but it is a different and stronger thing than a learning-rate schedule,
and it should be evaluated as such.

---

## 2. The measured profile

Geometric mean of the per-round gradient norm over all 3000 steps of the two
production runs (jobs `15538492`, `15538493`; `GradMonitor` hooks the link *output*
tensors, so these are activation gradients — the ratios are what matter below):

| mode | link | round 1 | round 2 | round 3 | r3 / r1 |
|---|---|---|---|---|---|
| `shared_roae`  | 1→2 | 1.83e-11 | 1.96e-08 | 2.80e-05 | 1.5e6 |
| `shared_roae`  | 2→3 | 3.45e-09 | 4.74e-06 | 7.96e-03 | 2.3e6 |
| `shared_roae`  | 3→1 | 4.76e-09 | 6.63e-06 | — | — |
| `shared_state` | 1→2 | 5.64e-08 | 1.17e-10 | 2.65e-05 | V-shape |
| `shared_state` | 2→3 | 1.36e-05 | 3.96e-08 | 8.67e-03 | V-shape |
| `shared_state` | 3→1 | 9.75e-11 | 2.38e-05 | — | V-shape |
| `shared_state` | `state` | 1.38e-05 | 1.38e-05 | — | **1.00** |

Exactly the shapes `GRADIENT_ROUND_PROFILE.md` predicts: `ρ ≈ 1e-3` geometric decay
for `shared_roae`, the V for `shared_state`, and a perfectly flat `state` highway.
First-step and last-step values agree to within ~2×, so this is a structural property
of the link design, not a transient of early training.

---

## 3. bfloat16 was discarding the early rounds — **fixed**

### 3.1 Gradient accumulation

Training runs `--dtype bfloat16`, and the link was built at that dtype, so `.grad`
was bf16 with **eps = 7.8e-3**. Backward runs round 3 first, so the buffer already
holds `g_3` when `g_2` and `g_1` arrive. Simulating that accumulation elementwise
over 2^20 coordinates with the measured norms above:

| | round 3 | round 2 | round 1 |
|---|---|---|---|
| **bf16**, `shared_roae` 1→2 | 100% norm, 100% coords | 71%, **16% coords** | **2%**, **0.01% coords** |
| **bf16**, `shared_state` 2→3 | 100%, 100% | **6%**, **0.11% coords** | 94%, 33% coords |
| **fp32**, both | 100%, 100% | 100%, ~100% | 100%, ~96–100% |

The damage is not uniform attenuation. Round-to-nearest keeps a contribution only on
the coordinates where it happens to exceed ~eps × |running total|, so what survives is
a *sparse, biased* sample of the true gradient. Severity tracks the magnitude ratio:
contributions within ~1e-3 of the running total mostly survive, ones below ~1e-5
essentially do not.

The all-reduce compounded it — three pipeline ranks hold partial derivatives of the
same loss and their sum was also formed in bf16.

### 3.2 The recursive state update

`SharedRecursiveStateLink.round_feedback` carries the same problem in the forward
pass:

```python
state = f if state is None else state + self.gamma * f
```

with `gamma` ReZero-initialised at 1e-3 — below bf16 eps. At the production state
shape (1 × 48 × 1024 = 49,152 coordinates):

| `|f|/|z|` | bf16 norm change kept | bf16 coords moved | fp32 |
|---|---|---|---|
| 0.3 | 47.9% | 7.1% | 100% / 100% |
| 1.0 | 81.4% | 22.3% | 100% / 100% |
| 3.0 | 104.5% | 52.7% | 100% / 100% |

So `z` does still move in bf16 — the earlier worry that it was frozen outright is
wrong at this scale, though at small tensor sizes it *is* a complete no-op (see the
smoke test in §3.4). What it receives is a sparsified, biased version of the update.

**This does not change the §2 reading of the `state` link.** Its round-1 and round-2
gradients are identical because the `I + γJ ≈ I` highway is genuinely flat, as
`GRADIENT_ROUND_PROFILE.md` argues — not because `z` failed to update.

### 3.3 The fix

The trained link now has its own dtype, independent of the pipeline's:

```bash
--link_dtype float32   # default
--link_dtype same      # old behaviour: follow --dtype
```

The frozen backbones stay at `--dtype bfloat16`; only the ~19.4M trainable link
parameters move to fp32, so the memory and throughput cost is negligible against
~4.2B frozen parameters.

To make that work without touching every call site, `modeling.py` gained
`run_at_param_dtype()`. Each link's `forward` now evaluates in its own parameter
dtype and returns in the *input's* dtype, so tensors crossing the module boundary
keep the ambient dtype and the pipeline's P2P sends are unchanged.
`round_feedback` additionally keeps the recursive state `z` in the parameter dtype
for the whole chain — `z` never leaves the solver rank, so this costs nothing.
`CrossModelAdapter` (the `original` mode links) got the same treatment.

Downstream effects, all benign:

- `.grad` is fp32, so the all-reduce buffer is fp32 too (its dtype is derived from
  `p.grad`, not hardcoded).
- Adam moments are fp32.
- Checkpoints save fp32 weights (~78MB vs ~39MB). Eval is unaffected: the loaders
  cast to the eval dtype, so old bf16 checkpoints and new fp32 ones both work.
- `GradMonitor` still hooks the post-cast bf16 output tensors, so the `12`/`23`/`31`
  traces stay comparable to earlier runs. The `state` trace is now measured in fp32.

### 3.4 Verification

Unit-level, `modeling.py` with fp32 params and bf16 inputs, for all three shared link
classes plus `CrossModelAdapter`: output dtype is bf16, gradients are fp32, shapes
unchanged. The `shared_state` case makes the forward-pass effect visible directly —
on a small tensor, `z` is bit-identical across rounds in bf16 and changes in fp32:

```
SharedRecursiveStateLink  params=float32   state dtype=float32  z changed=True
SharedRecursiveStateLink  params=bfloat16  state dtype=bfloat16  z changed=False
```

End-to-end, 50 steps of `shared_state` on math500, same seed, differing only in
`--link_dtype` (jobs `15542750` fp32 / `15542751` `same`). Both COMPLETED in ~16 min.

**The fp32 path is correct. It does not demonstrate an improvement, and the gradient
norms move opposite to the prediction.**

| link | round | fp32 | bf16 (`same`) | fp32 / bf16 |
|---|---|---|---|---|
| 1→2 | 1 | 3.43e-07 | 3.77e-07 | 0.91 |
| 1→2 | 2 | 2.31e-10 | 2.64e-10 | 0.87 |
| 1→2 | 3 | 8.68e-04 | 1.99e-04 | 4.37 |
| 2→3 | 1 | 3.50e-05 | 3.24e-04 | 0.11 |
| 2→3 | 2 | 4.37e-08 | 3.28e-07 | 0.13 |
| 3→1 | 1 | 1.68e-11 | 9.41e-10 | **0.02** |
| 3→1 | 2 | 5.79e-05 | 6.03e-04 | 0.10 |
| `state` | 1, 2 | 3.32e-05 | 3.47e-04 | 0.10 |

Final gates: fp32 `alpha=2.85e-03 gamma=1.26e-03`; bf16 `alpha=2.88e-03
gamma=1.08e-03`. Final loss: 3.28 vs 2.77.

What this does and does not establish:

- **Does:** the fp32 link trains inside a bf16 pipeline without incident — gates move,
  all four links log, the checkpoint writes, the all-reduce is unaffected.
- **Does not:** show fp32 trains better. The loss trace at `batch_size 1` over 50 steps
  is wildly non-monotonic (11.79, 6.20, 6.15, 7.38, 5.63, 6.12, 4.75, 3.86, 2.93,
  2.73, 3.28); a 0.5 gap between the two runs is far inside that. By the standard §5
  applies to everything else, this comparison has no power.
- **Does not** isolate the accumulation effect. The dtype changes the *forward* pass
  too — the runs already differ at step 0 (loss 11.79 vs 11.61), before any parameter
  update — so there is no clean "same trajectory, different accumulation" contrast to
  read. Everything downstream of step 0 compares two different points in parameter
  space.

The direction is worth noting honestly: fp32's early-round gradients are *smaller*, by
up to 55× on the 3→1 round-1 link. Candidate explanations — trajectory divergence,
bf16 rounding acting as noise injection that inflates measured gradient norms, or a
genuine reduction in early-round sensitivity once those rounds actually receive
updates — are not separable from this data. **The §3.1–3.2 case for float32 rests on
the numerics, which are directly verifiable; it does not rest on this A/B.**

> **Comparability.** `--link_dtype` defaults to `float32`, so runs from here on are not
> directly comparable to the 46 configurations in `eval_results.md`, all of which were
> trained with what is now `--link_dtype same`. Use `same` to reproduce those.

---

## 4. Would round weighting improve downstream accuracy?

With the dtype fixed, `c_r` weighting becomes a coherent thing to try. It is still
unlikely to move the benchmarks much.

**Against:**

- *Gradient magnitude is not information.* `g_1` is small because the loss is
  genuinely insensitive to round-1 link behaviour. Multiplying by 1e6 gives a louder
  estimate of a weak effect, not a better one — the signal-to-noise ratio is
  unchanged. This is the standard objection to naive gradient normalisation.
- *The model already has this knob and declined to use it.* The ReZero gates are a
  learned scale. Over 3000 steps `gamma` moved 9.99e-04 → 2.41e-03 and stopped;
  `alpha` reached 1.57e-02. It had the freedom to raise the early rounds' influence
  and largely did not.
- *Leverage is bounded.* 19.4M trainable parameters against ~4.2B frozen backbone is
  0.46% of the model.
- *Some benchmarks are capped elsewhere entirely.* LiveCodeBench is limited by the
  solver being `Qwen2.5-Math-1.5B`: on a 200-problem diagnostic (job `15542639`),
  **100%** of outputs parsed as valid Python, yet 97.5% passed zero tests —
  47.5% `EVAL_BAD_OUTPUT`, 43.5% failing to run at all (`CE/RE`, `RE`, `NO_FUNC`),
  6.5% wrong answer, 2% timeout. No optimizer change touches that.

**For:** if rounds 1..R−1 really are contributing nothing to the update, the model is
effectively R=1 with the forward cost of R=3, and each round's link is tuned only for
its last-round role. That is a genuine defect worth fixing on its own terms, whatever
the benchmarks say.

---

## 5. The evaluation protocol cannot currently measure it

Parsing all 46 configuration rows in `eval_results.md` (4 link modes × 3 learning
rates × 5 checkpoint steps) and comparing the observed between-config spread to
single-run binomial sampling noise:

| dataset | n | mean | sd observed | sd binomial | ratio |
|---|---|---|---|---|---|
| math500 | 500 | 76.79 | 2.80 | 1.89 | 1.48 |
| medqa | 300 | 29.34 | 2.23 | 2.63 | **0.85** |
| aime2025 | 30 | 27.10 | 4.19 | 8.12 | **0.52** |
| aime2026 | 30 | 18.12 | 3.76 | 7.03 | **0.53** |
| gpqa | 198 | 27.95 | 3.17 | 3.19 | **0.99** |
| mbppplus | 378 | 32.87 | 1.29 | 2.42 | **0.53** |
| livecodebench | 1055 | 2.01 | 0.45 | 0.43 | **1.05** |

A ratio near 1 means the entire spread across every architecture and learning rate
tried is explained by sampling noise alone. Ratios *below* 1 mean the configurations
differ **less** than chance would produce. Only math500 exceeds 1, and that is driven
by two undertrained `outer_link` checkpoints at steps 600 and 1200 (65.0, 66.0).

Minimum detectable difference between two configurations, 80% power, α = 0.05:

| dataset | 1 seed | 3 seeds | 5 seeds | seeds needed for 2 pt |
|---|---|---|---|---|
| math500 | 7.5 pt | 4.3 pt | 3.3 pt | 14 |
| medqa | 10.4 pt | 6.0 pt | 4.7 pt | 28 |
| gpqa | 12.6 pt | 7.3 pt | 5.6 pt | 40 |
| mbppplus | 9.6 pt | 5.5 pt | 4.3 pt | 23 |
| aime2025 | 32.1 pt | 18.6 pt | 14.4 pt | 259 |
| aime2026 | 27.8 pt | 16.1 pt | 12.4 pt | 194 |
| livecodebench | 1.7 pt | 1.0 pt | 0.8 pt | 1 |

AIME at n=30 quantises to 3.33-point steps: "26.67% vs 30%" is one problem. The paper
reports means over five runs with ±0.41 std; single-seed numbers here are 5–7× noisier
than that.

---

## 6. Recommended experiment order

1. **Land the dtype fix** (done — §3.3) and re-log the per-round profile. If rounds 1
   and 2 move relative to `--link_dtype same`, the effect was real. This is a
   training-side readout: near noise-free compared with any benchmark.
2. **Test the premise directly, cheaply.** Evaluate one existing checkpoint at
   `n_rounds = 1, 2, 3, 5` on math500 alone. If accuracy is flat in R, the extra
   rounds are contributing nothing — which is exactly what round weighting is meant
   to fix, measured with far more power than a seven-dataset sweep.
3. **Only then try `c_r` weighting**, applied per stage before accumulation, with the
   per-round profile as the primary readout.
4. **If downstream evals are the readout**, pool math500 + livecodebench + mbppplus +
   medqa (n = 2233), budget ≥3 seeds, and drop AIME as a discriminator. Also use the
   protocol's `latent_length = 80`; `scripts/config.sh` currently pins
   `EVAL_LATENT_STEPS=48`.

---

## Reproducing

```bash
# Unit check: fp32 link inside a bf16 pipeline
python - <<'EOF'
import torch
from modeling import SharedRecursiveStateLink
m = SharedRecursiveStateLink(hidden_dims=[64,48,80], shared_dim=32,
                             n_experts=3, expert_dim_divisor=3).to(torch.float32)
h = torch.randn(2, 5, 64, dtype=torch.bfloat16)
out = m(h, src=1, dst=2)
print(out.dtype)                                    # bfloat16 — boundary preserved
print({p.grad.dtype for p in m.parameters() if p.grad is not None} or "no grads yet")
EOF

# A/B the dtype end to end (50 steps each)
sbatch outputs/slurm_jobs/abtest_linkdtype_float32.sh
sbatch outputs/slurm_jobs/abtest_linkdtype_same.sh

# Noise analysis of eval_results.md (§5)
python - <<'EOF'
import re, math, statistics as st
N = {"math500":500,"medqa":300,"aime2025":30,"aime2026":30,
     "gpqa":198,"mbppplus":378,"livecodebench":1055}
rows = []
for line in open("eval_results.md"):
    if not line.strip().startswith("|"): continue
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) != 8 or "paper" in c[0].lower(): continue
    m = [re.match(r"^([0-9.]+)%?$", x) for x in c[1:]]
    if all(m): rows.append([float(x.group(1)) for x in m])
for j, ds in enumerate(N):
    v = [r[j] for r in rows]; p = st.mean(v)/100
    print(f"{ds:>14} sd_obs={st.stdev(v):5.2f} "
          f"sd_binom={100*math.sqrt(p*(1-p)/N[ds]):5.2f}")
EOF
```

Related: `GRADIENT_ROUND_PROFILE.md` (why the profile has this shape),
`MAS_WIDTH_DEPTH_THEORY.md` §4 (effective trainable depth `D_eff`),
`EVAL_PROTOCOL.md` (the per-dataset settings §5 measures against).
