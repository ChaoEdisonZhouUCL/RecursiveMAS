# Evaluation Protocol (RecursiveMAS paper)

Per-dataset evaluation settings as described in the paper
([arXiv:2604.25917](https://arxiv.org/abs/2604.25917)), and how they are
applied in this repo. The machine-readable version is
[eval_protocol.yaml](eval_protocol.yaml), which `run.py` loads automatically
when the corresponding `--dataset` is evaluated.

## Per-dataset settings

| Dataset       | Metric                    | Rollouts | Temp | Top-p | Max new tokens | Latent steps | Problems |
|---------------|---------------------------|----------|------|-------|----------------|--------------|----------|
| math500       | accuracy                  | 1        | 0.6  | 0.95  | 2,000          | 80           | 500      |
| medqa         | accuracy (4-choice)       | 1        | 0.6  | 0.95  | 4,000          | 80           | 300      |
| gpqa          | accuracy (4-choice)       | 1        | 0.6  | 0.95  | 4,000          | 80           | 198      |
| aime2025      | **pass@10**               | **10**   | 0.6  | 0.95  | **16,000**     | 80           | 30       |
| aime2026      | **pass@10**               | **10**   | 0.6  | 0.95  | **16,000**     | 80           | 30       |
| mbppplus      | execution pass rate       | 1        | 0.2  | 0.95  | 4,000          | 80           | 378      |
| livecodebench | execution pass rate       | 1        | 0.2  | 0.95  | 4,000          | 80           | 1,055    |

Common to all benchmarks (verbatim from the paper):

- "During inference, we set top-p to 0.95 and use a temperature of 0.6 for
  most reasoning tasks and 0.2 for code generation."
- "the maximum generation length is set for 2000 tokens for MATH500, 4000
  tokens for MedQA, GPQA-Diamond, LiveCodeBench, and MBPP Plus, and 16000
  tokens for AIME2025/2026."
- "For AIME2025/2026, we report Pass@10 accuracy for testing robustness."
- Latent thought length: performance stabilizes "around m=80" (Figure 8);
  headline numbers use r=3 recursion rounds.
- "We ... report the mean performance over five independent runs."
  (average std across 5 runs: ±0.0041)

## How the settings are applied

`run.py` reads `eval_protocol.yaml` and uses the entry matching `--dataset`.
Precedence, highest first:

1. **Explicit CLI flags** to `run.py` — anything actually present on the
   command line wins over the protocol file.
2. **eval_protocol.yaml** — the dataset entry, then its `defaults` section.
3. **run.py built-ins** (legacy `infer_max_new_tokens` / release-recommended
   settings).

Note that `scripts/submit.sh` passes `--batch_size`, `--latent_length`
(from `EVAL_LATENT_STEPS`), and `--greedy` (when `EVAL_GREEDY=true`)
explicitly, so those stay under config.sh control; the protocol file
supplies what submit.sh does not pin: `max_new_tokens`, `num_rollouts`,
and `temperature`/`top_p`. Setting `EVAL_LATENT_STEPS="protocol"` makes
submit.sh omit `--latent_length`, so the yaml's per-dataset
`latent_length` applies as well.

## Reproducing the paper numbers

To match the paper protocol for a dataset, in `scripts/config.sh` set:

```bash
EVAL_GREEDY=false        # paper samples; pass@10 requires sampling
EVAL_LATENT_STEPS=80     # paper's m; release demo configs use 16-48 (faster)
EVAL_N_ROUNDS=3          # headline numbers are r=3
```

and run each eval 5 times with different `--eval_seed`, reporting the mean.
Expect AIME to be ~10x slower than a single-rollout eval (10 rollouts x
16k-token budget). Differences that remain after matching the protocol:
released checkpoints + this pipeline reproduce the paper's setup, but
exact numbers still vary run to run (sampling, 30-problem AIME sets).
