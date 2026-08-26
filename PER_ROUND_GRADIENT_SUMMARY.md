# Per-round gradient profile: why `shared_state` lifts round 1 but not round 2

Measured on three 3000-step runs, identical seed/data/hyperparameters, differing
only in link architecture. Values are `||dL/dW||` over the whole link module,
averaged over the last 600 steps.

| run | mode | wandb |
|---|---|---|
| 15556191 | `original` | [7uz4s72c](https://wandb.ai/chaozhou1/recursivemas-outerlinks/runs/7uz4s72c) |
| 15556710 | `shared_roae` | [lonwktr2](https://wandb.ai/chaozhou1/recursivemas-outerlinks/runs/lonwktr2) |
| 15556711 | `shared_state` | [ckei6ldn](https://wandb.ai/chaozhou1/recursivemas-outerlinks/runs/ckei6ldn) |

Final loss is effectively tied across all three: 0.884 / 0.864 / 0.858.

---

## 1. First, which curve — the V is a per-stage effect, not a whole-link one

In the shared modes there is **one** link module. The `1→2` / `2→3` / `3→1`
panels are the share of that module's gradient contributed by each pipeline
stage; the `shared` panel is the whole module, summed across all three stages.

**Whole link** (`shared` series; `original` has separate modules, so `1→2` shown):

| mode | round 1 | round 2 | round 3 | round1 / round3 |
|---|---|---|---|---|
| `original` | 2.09e-08 | 2.09e-05 | 2.83e-02 | 7.4e-07 |
| `shared_roae` | 2.63e-05 | 2.34e-03 | 1.46e-01 | 1.8e-04 |
| `shared_state` | 3.22e-04 | 2.07e-03 | 1.55e-01 | 2.1e-03 |

**On the whole link, round 2 is not smaller than round 1 — it is 6.5x larger**,
and the profile is monotone increasing in round. The V-shape appears only on the
per-stage panels of `shared_state`:

| `shared_state` stage | round 1 | round 2 | round 3 |
|---|---|---|---|
| `1→2` | 9.19e-05 | 9.91e-08 | 2.20e-02 |
| `2→3` | 3.88e-04 | 6.74e-07 | 2.69e-01 |
| `3→1` | 2.86e-08 | 2.75e-03 | — |

Note `3→1` is the exact mirror: its round 2 is **96,000x larger** than its round 1.
So round 1's gradient arrives almost entirely through the planner and critic
stages, and round 2's almost entirely through the solver→planner stage. Summing
them fills in the V.

Two corrections to the visual reading:

- Round 1 is not "comparable to" round 3. On the whole link it is still **482x
  smaller**. What changed is its *share*: `original` puts round 1 at 7.4e-07 of
  round 3, `shared_state` at 2.1e-03 — a ~2800x improvement, and ~11x better
  than `shared_roae`.
- The V is real but it lives on two of the four curves, not on the total.

---

## 2. Why round 2 is low per-stage: round 1 is privileged, round 2 is not suppressed

The recursive state updates as

```
z^(0) = f^(0)                    # weight 1     <-- round 1 seeds the state
z^(r) = z^(r-1) + gamma * f^(r)  # weight gamma <-- every later round
```

Backward:

| derivative | value | consequence |
|---|---|---|
| `∂z^(r)/∂z^(r-1)` | `I + gamma*J ≈ I` | identity highway, no decay between rounds |
| `∂z^(0)/∂f^(0)` | `I` | **round 1 pays nothing** |
| `∂z^(r)/∂f^(r)`, r≥1 | `gamma*I` | every later round pays `gamma` **once** |

So the per-stage V is not round 2 being damped — it is round 1 being the only
round that bypasses the `gamma` gate, because it happens to be the round that
*seeds* the state. Two levels, not a slope. This is an artifact of the seeding,
not a designed property.

**The prediction is quantitative and it holds.** If that is the whole story,
round1/round2 should equal `1/gamma`. With the run's final `gamma = 1.74e-03`:

| stage | measured round1/round2 | predicted `1/gamma` | error |
|---|---|---|---|
| `2→3` | 575.7 | 574.7 | **0.2%** |
| `1→2` | 927.1 | 574.7 | 61% (same order; carries an extra intra-round Jacobian) |

Round 3 is largest everywhere for a separate reason: it is adjacent to the loss
and never traverses the recursion at all.

For contrast, `shared_roae` has no V — it decays geometrically, ~75-100x per
round, which is the expected behaviour with no identity highway.

---

## 3. Can round 2 be raised?

Four levers, ranked by directness.

**(a) Per-round gradient rescaling `c_r` — the direct lever.** The link's weights
are shared across rounds, so the optimizer only ever sees `Σ_r g_r`; the sole
well-defined intervention is `Σ_r c_r * g_r`, scaling before the sum. That is
feasible here because the pipeline backward is already staged per round, and now
instrumented — the per-round profile above is exactly what calibrates `c_r`.
Pure optimization change, forward pass untouched. Note AdamW's `m/√v` absorbs any
*uniform* rescale, so only the relative `c_r` matter — which is the point.

**(b) Raise `gamma`.** Every round `r≥1` is scaled by `gamma` exactly once, so
round 2 scales linearly with it. `gamma` is ReZero-initialised at 1e-3 and barely
moves — 1e-3 → 1.74e-3 over 3000 steps. Initialising it at ~0.1 would lift every
middle round ~57x. But this changes the forward pass, and trades away the
near-identity highway that makes the cross-round path lossless in the first place.

**(c) Seed the state through the same gate** (`z^(0) = gamma * f^(0)`). This
removes the asymmetry and makes the profile clean — but it equalizes by *lowering
round 1*, not by raising round 2. Diagnostically clarifying, practically a loss.

**(d) Per-round `gamma_r`.** Lets each round learn its own gate. Likely inert:
the gradient reaching `gamma_2` is itself scaled small, and the evidence is that
`gamma` does not move even when it is free to.

**Recommendation:** (a), calibrated from the measured profile, with (b) as a cheap
one-line A/B alongside it.

---

## 4. Caveats

- **No evidence yet that raising round 2 helps anything downstream.** All three
  runs reach the same loss (0.884 / 0.864 / 0.858) despite differing by ~3.5
  orders of magnitude in round-1 gradient. A flatter profile is not
  self-evidently a better model.
- **The eval protocol cannot currently measure the answer.** Minimum detectable
  effect at 1 seed is ~7.5 points on math500, ~12.6 on gpqa; 14-259 seeds would
  be needed to resolve 2 points.
- **Test the premise first.** Evaluate one existing checkpoint at
  `n_rounds = 1, 2, 3, 5` on math500 alone. If accuracy is flat in R, the extra
  rounds contribute nothing — which is what round weighting is meant to fix, and
  measured with far more power than a seven-dataset sweep.

## Measurement notes

- `||dL/dW||` is the whole-module parameter gradient, the quantity the optimizer
  consumes — distinct from the activation gradient `||dL/dy||` also logged.
  They diverge: `dL/dW = dL/dy · x^T`, so a link fed a small input has a small
  weight gradient even when its activation gradient is large.
- Per-round terms are recoverable only because the backward is staged per round:
  snapshot `.grad` before each round's stage, subtract after. Norm **of the
  delta**, not delta of the norm.
- Measured with `--link_dtype float32`. In bf16 the `.grad` buffer's eps is
  7.8e-3, so an early round's contribution — 5-6 orders below the last round's —
  is destroyed by the accumulation itself and reads as exactly zero.
- The shared-link total must be summed across ranks **as a vector** before taking
  the norm; norms are not additive.
