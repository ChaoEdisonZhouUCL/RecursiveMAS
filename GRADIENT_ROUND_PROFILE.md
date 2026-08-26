# Why `shared_roae` decays monotonically but `shared_state` shows a V

Analysis of the per-round gradient profile of the outer links, and a sweep over
link type × recursion depth to verify it.

Reference plots this document explains:

- `outerlink_grad_shared_roae_r3.png` — monotone vanishing, round 3 → 1
- `outerlink_grad_shared_state_r3.png` — vanishing then *rising* again at round 1

> **⚠ Those two PNGs were overwritten by the §4.4 sweep.** `plot_single` writes to a
> fixed path per (mode, n_rounds), so the r3 jobs replaced the original 3000-step
> plots (shared_roae @ lr 1e-3, shared_state @ lr 1e-4, both dated Aug 12) with the
> new 200-step lr-1e-4 versions. The originals are not recoverable: the shared_state
> one was untracked, and the tracked shared_roae one had uncommitted changes. The
> *numbers* from both survive — see §4.1 for their step-0 values and the table below
> for their end-of-training values, read from the job logs
> (`outputs/slurm_logs/job-15518359`, `job-15518361`):
>
> | original plot | mode | lr | `12` r1 | r2 | r3 |
> |---|---|---|---|---|---|
> | `..._shared_roae_r3.png` | shared_roae | 1e-3 | 6.148e-14 | 9.636e-11 | 6.221e-07 |
> | `..._shared_state_r3.png` | shared_state | 1e-4 | 4.695e-07 | 9.879e-10 | 3.036e-05 |
>
> **Fixed.** Figures are no longer written to the project root: each run's plots now
> go into its own `outputs/checkpoints/<prefix>_<mode>_r<N><suffix>_<timestamp>/`
> directory, beside the checkpoints they describe. See "Figures" below.

---

## TL;DR

**The V is not a training effect, not an lr effect, and not noise. It is a direct
consequence of one line in `SharedRecursiveStateLink.round_feedback`:**

```python
state = f if state is None else state + self.gamma * f
```

Round 1's feedback enters the recursive state with weight **1** (it *is* the state
initialiser). Every later round enters with weight **`gamma` ≈ 1e-3**. Since the
state chain `z^(r) = z^(r-1) + gamma·f^(r)` has Jacobian `I + gamma·J ≈ I`, it is an
*identity highway*: it carries gradient backwards essentially undamped. So

| | round 1 | rounds 2 … R−1 | round R |
|---|---|---|---|
| gradient reaching that round's `12` link | `A·G` | `gamma·A·G` | direct from loss |

Round 1 sits a factor **`1/gamma` = 1000×** above the others, and the middle rounds
are **flat** — not decaying. That is the V.

`shared_roae` has no such highway: every round boundary is a plain
`out_proj(latent_core(·))`, so gradient must traverse the full round each time and
decays geometrically, `ρ^(R−r)` with ρ ≈ 1e-3.

**The important consequence:** `shared_state` does *not* merely shift the curve — it
changes the scaling law. `shared_roae`'s earliest-round gradient dies like `ρ^R`
(it underflows float32 to literally `0.0` by R=7). `shared_state`'s plateau is
**depth-independent**: adding rounds does not push the middle rounds any lower.
That is a real win, partly hidden by the cosmetic round-1 spike.

---

## 1. What actually carries gradient between rounds

`latent_rollout()` (`train_outerlinks_math500.py:714`) keeps **only step 0** inside the
autograd graph; steps 1..47 run under `no_grad` and are re-attached as fresh leaves:

```python
grad_ctx = torch.enable_grad() if step == 0 else torch.no_grad()
...
last_h = _last_h[0].detach().requires_grad_(True)   # steps >= 1: fresh leaf
```

So each agent contributes exactly **one** in-graph transformer forward to the
cross-round path. Write `A` for the composite intra-round Jacobian from
`critic_prefix^(r)` (the tensor both plots report, `GradMonitor` link `"12"`) down to
the round boundary. `A` is the same map for every round, so all cross-round
differences come from the round boundary alone.

### 1.1 Two gradients per link, and they are not the same number

`GradMonitor` records two families, both per link and per round:

| wandb key | quantity | what it answers |
|---|---|---|
| `grad_norm/<link>/round_<r>` | `\|\|dL/dy\|\|` at the link's **output activation** | how much loss signal survives back to round r |
| `grad_norm_param/<link>/round_<r>` | `\|\|dL/dW\|\|` over the link's **whole module** | how much round r actually trains the link |
| `grad_norm_param/shared/round_<r>` | the same, summed over all three pipeline stages | shared modes only — see below |

They can diverge, because `dL/dW = dL/dy · x^T`: a link fed a small input has a
small weight gradient even when its activation gradient is large. The second is
the one the optimizer consumes, so it is the one to read when asking whether a
round is contributing to learning at all.

The weights are shared across rounds, so by the end of backward `.grad` holds
`Σ_r g_r` and the individual terms are gone. They are recoverable only because
the pipeline backward is staged one round at a time: `LinkParamGradTracker`
snapshots `.grad` before each round's stage and subtracts after, giving the norm
**of the delta** (not the delta of the norm — those differ whenever two rounds
disagree in direction). A `(link, round)` pair whose stage never ran is recorded as
NaN rather than 0.0, since "no such term" is not "a term that happens to be
zero". It is detected as an exactly-zero delta: `.grad` buffers are zeroed but
kept allocated each step, so an untouched module's buffer stays bit-identical.
This is the solver stage on the final round, in **every** mode — that round ends
in a teacher-forced pass that neither produces feedback nor runs a latent
rollout, so it touches no link at all.

In the shared modes there is **one** link module serving all three transitions,
so `12`/`23`/`31` there are not separate links — they are the share of that one
module's gradient contributed by each pipeline stage's backward (an old debug
line reads `13/29 have grad`: a stage touches only the slice it uses). The
module also doubles as every agent's rollout self-loop (`src=dst=agent_idx`), so
a stage's share covers more than its transition.

The module's actual per-round gradient is the **sum over the three stages**, and
norms are not additive — it has to be summed as a vector, across ranks, before
taking the norm. That is the `shared` key: each rank accumulates its own stage's
per-round delta, and one all-reduce per step recovers the whole-module total.
In `original` mode the three links are genuinely separate modules on separate
ranks, so per-stage and whole-link coincide and `shared` is absent.

> Measure this with `--link_dtype float32` (the default). In bfloat16 the link's
> `.grad` buffer has eps 7.8e-3, so an early round's contribution — five to six
> orders of magnitude below the last round's — is discarded by the accumulation
> itself and the delta reads as exactly 0. The measurement and the training both
> need the wider dtype, for the same reason. See
> [ROUND_GRADIENT_WEIGHTING.md](ROUND_GRADIENT_WEIGHTING.md) §3.

## 2. `shared_roae` — geometric decay

```
f^(r)              = latent_core(h_solver^(r), 3→1)
planner_prefix^(r+1) = out_proj[1]( f^(r) )
```

Gradient from round `r+1` to round `r` must pass through the entire round:

```
‖g_12^(r)‖  =  ρ^(R−1−r) · C ,      ρ = one full round's Jacobian gain ≈ 1e-3
```

Monotone, and each extra round of depth costs another factor ρ. **This is ordinary
deep-recursion gradient vanishing.**

## 3. `shared_state` — identity highway + asymmetric initialisation

```
f^(r) = latent_core(h_solver^(r), 3→1)
z^(0) = f^(0)                       # <-- weight 1
z^(r) = z^(r-1) + gamma · f^(r)     # <-- weight gamma, for r >= 1
planner_prefix^(r+1) = out_proj[1]( z^(r) )
```

Backward:

| derivative | value | consequence |
|---|---|---|
| `∂z^(r)/∂z^(r-1)` | `I + gamma·J ≈ I` | **identity highway** — no decay between rounds |
| `∂z^(r)/∂f^(r)`, r ≥ 1 | `gamma·I` | every later round pays `gamma` **once** |
| `∂z^(0)/∂f^(0)` | `I` | **round 1 pays nothing** |

With `G = ‖∂L/∂z^(R-2)‖`:

```
‖g_12^(0)‖  ≈ A · G            <-- round 1, the state initialiser
‖g_12^(r)‖  ≈ A · gamma · G    <-- rounds 2 … R-1, all identical
```

Two levels, not a slope. Round 1 is high **because it is the only round whose
contribution is not scaled by `gamma`** — an artifact of how the state is seeded, not
a designed property.

---

## 4. Evidence

### 4.1 It is present at step 0, at both learning rates

The two reference plots were run at **different lr** (`shared_roae` 1e-3,
`shared_state` 1e-4), which is a confound. It is not the cause — the profile is
already fully formed at **step 0, before any optimiser update**, and both lrs agree:

| job | mode | lr | `12` r1 | r2 | r3 | shape |
|---|---|---|---|---|---|---|
| 15518361 | shared_roae | 1e-3 | 6.100e-11 | 3.340e-08 | 2.649e-05 | monotone |
| 15518362 | shared_roae | 1e-4 | 3.141e-10 | 9.882e-08 | 5.486e-05 | monotone |
| 15518359 | shared_state | 1e-4 | 1.505e-08 | 4.175e-11 | 1.040e-05 | **V**, r1/r2 = 360 |
| 15518360 | shared_state | 1e-3 | 4.124e-07 | 1.046e-10 | 4.935e-05 | **V**, r1/r2 = 3942 |

End-of-training (last 600 of 3000 steps) agrees, over four independent
`shared_state` runs — `r1/r2` = 475, 1130, 251, 460, straddling the predicted
`1/gamma = 1000`.

### 4.2 `gamma` never moves

Every `shared_state` checkpoint, at every step, at both learning rates:

```
gamma = 0.000999451   ==  bfloat16(1e-3)   (init value, unchanged)
```

while `alpha` on the same module moves freely (1e-3 → 5e-3, and to −1e-2). So
`gamma` is pinned at its ReZero init and the plateau stays exactly 1000× down.
See §6 — this is caused by a separate bug.

### 4.3 Isolated replication with the real modules

`exp_round_grad_profile_toy.py` runs the real `SharedRecursiveLink` /
`SharedRecursiveStateLink` on CPU, replacing only each agent's latent rollout with a
fixed random linear map. No LLM, no distributed, no training:

```
  shared_roae  R=5   r1:3.503e-18  r2:5.248e-15  r3:5.911e-12  r4:7.496e-09  r5:1.063e-05
                     r1/r2=6.7e-4  r2/r3=8.9e-4  r3/r4=7.9e-4  r4/r5=7.1e-4     <- geometric

 shared_state  R=5   r1:7.496e-09  r2:7.496e-12  r3:7.496e-12  r4:7.496e-12  r5:1.063e-05
                     r1/r2=1000    r2/r3=1       r3/r4=1                        <- step + plateau
```

The middle rounds are flat to **all printed digits**, and the round-1 step is exactly
`1/gamma`:

| gamma | measured r1/r2 | 1/gamma |
|---|---|---|
| 1e-1 | 9.999 | 10 |
| 1e-2 | 100.0 | 100 |
| 1e-3 | 1000. | 1000 |
| 1e-4 | 1.000e4 | 10000 |

At R=7, `shared_roae`'s round-1 gradient underflows float32 to **exactly 0.0**, while
`shared_state`'s plateau is unchanged at 7.496e-12.

### 4.4 GPU sweep: link type × recursion depth

6 jobs on JURECA `dc-hwai`, 3 GPUs each, **lr fixed at 1e-4** for all of them
(removing the confound), 200 steps, otherwise identical to the reference runs
(`latent_steps=48`, `batch_size=1`, `grad_accum=4`, `n_experts=3`,
`expert_dim_divisor=3`, `--no_round_skip`, pooled `s1k+m1k+opencodereasoning+arpo_sft`,
`max_seq_len=4096`).

Gradient norm at the `12` link (`critic_prefix`), averaged over the last 40 of 200
steps. Jobs 15535904–15535909.

**`shared_roae` — monotone geometric decay at every depth:**

| R | round 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 3 | 1.902e-11 | 1.271e-08 | 1.067e-05 | | |
| 4 | 9.347e-14 | 5.404e-11 | 2.647e-08 | 1.621e-05 | |
| 5 | 1.618e-13 | 9.423e-12 | 1.225e-09 | 2.077e-07 | 2.966e-05 |

Per-round ratios at R=5: 0.017, 0.0077, 0.0059, 0.0070 — constant to within a factor
of ~3, i.e. clean geometric decay spanning **8 orders of magnitude**.

**`shared_state` — step, then plateau:**

| R | round 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 3 | 1.554e-07 | 1.230e-10 | 2.744e-05 | | |
| 4 | 7.925e-09 | **9.873e-12** | **1.145e-11** | 1.571e-05 | |
| 5 | 1.372e-08 | **5.876e-12** | **5.393e-12** | **6.055e-12** | 6.765e-06 |

The bold intermediate rounds are the predicted plateau. At R=5 they span
5.393e-12 … 6.055e-12 — **flat within ±7% of their mean**, across rounds whose `shared_roae`
counterparts differ by four orders of magnitude. The round-1 step is
`r1/r2` = 803 (R=4) and 2335 (R=5), bracketing the predicted `1/gamma` = 1000.

The same plateau appears independently on the `23` link (R=5: 3.786e-09, 3.664e-09,
4.134e-09) and the `31` link (R=5: 1.609e-11, 1.275e-11, 1.848e-11), so it is a
property of the round boundary, not of one probe point.

**The `state` link is the identity highway, measured directly:**

| R | round 1 | 2 | 3 | 4 | ratio |
|---|---|---|---|---|---|
| 3 | 1.872e-05 | 1.872e-05 | | | 1.0000 |
| 4 | 4.217e-06 | 4.217e-06 | 4.217e-06 | | 1.0000 |
| 5 | 9.057e-06 | 9.057e-06 | 9.057e-06 | 9.057e-06 | 1.0000 |

Identical to **all four significant figures across four rounds** — exactly
`∂z^(r)/∂z^(r-1) = I + gamma·J ≈ I`. (These runs include the §6.1 fix; the same rows
read `0.5000` in every pre-fix run, which is what exposed that bug.)

**Depth scaling.** Normalising each run's plateau by its own final-round gradient to
remove run-to-run scale differences:

| R | `shared_roae` earliest round | `shared_state` plateau |
|---|---|---|
| 3 | 1.8e-6 | 4.5e-6 |
| 4 | 5.8e-9 | 6.8e-7 |
| 5 | 5.5e-9 | 8.6e-7 |

`shared_state`'s plateau is flat in depth from R=4 onward; `shared_roae` loses ~2.5
decades going from R=3 to R=4. (`shared_roae` at R=5 does not fall further below R=4;
at ~1e-13 it is at the noise floor of a bf16 backward, and the toy in §4.3 — free of
that floor — shows the decay continuing to underflow.)

---

## 5. Recommendations

1. **Seed the state symmetrically.** Initialise `z = 0` and use the same update for
   every round:

   ```python
   state = self.gamma * f if state is None else state + self.gamma * f
   ```

   Round 1 then drops onto the plateau and the profile is flat across all
   intermediate rounds — the V disappears. Note this makes round 1's gradient
   *smaller*, not the others larger; it removes an artifact rather than adding
   signal.

2. **The plateau height is `gamma`.** Flatness is free, but the whole plateau sits at
   `gamma·G`. Raising the plateau means raising `gamma` — which requires fixing §6
   first, since `gamma` currently cannot train at all.

3. **Precision matters more than the profile suggests.** The link was held in
   bfloat16, whose eps (7.8e-3) is larger than the ratio between consecutive rounds'
   gradients — so the early rounds were partly discarded during accumulation, before
   any optimizer saw them. The trained link now defaults to float32 via
   `--link_dtype`. See [ROUND_GRADIENT_WEIGHTING.md](ROUND_GRADIENT_WEIGHTING.md) §3.

4. **Do not read the round-1 spike as "shared_state fixed vanishing".** The honest
   claim is the better one, and it is what §4.4 measures: `shared_state` makes the
   intermediate-round gradient **flat across rounds and independent of recursion
   depth**, where `shared_roae` decays as `ρ^R`. Both still vanish relative to the
   final round — `shared_state` replaces a *slope* with a *floor*, it does not remove
   the drop.

---

## 6. Two instrumentation/training bugs found while diagnosing this

### 6.1 `GradMonitor` under-reported the `state` link by exactly 2× — **fixed**

A tensor hook fires once *per `backward()` call* that reaches the tensor, carrying
only that call's incremental gradient. The pipeline backward is staged per round, so
`z^(r)` — read both by round `r`'s `out_proj` and by round `r+1`'s state update — is
reached by two separate `backward()` calls and fires twice. `commit_step` averaged
over firings, halving it. This is why `link state` reported `ratio r1/r2 = 0.5000` in
*every* `shared_state` run — suspiciously exact, and the giveaway.

Fixed in `train_outerlinks_math500.py` (`GradMonitor.commit_step`): sum within the
step, divide by the micro-batch count. Links that fire once per micro-batch are
unaffected (`sum/n == mean`), so the `12`/`23`/`31` rows and both reference plots are
unchanged.

### 6.2 The gradient all-reduce mixed unrelated parameters — **fixed**

`train_outerlinks_math500.py:1856`, before the fix:

```python
grads = [p.grad for p in local_params if p.grad is not None]
flat  = torch.cat([g.reshape(-1) for g in grads])
dist.all_reduce(flat, op=dist.ReduceOp.SUM)
```

`optimizer.zero_grad()` defaults to `set_to_none=True` (torch 2.9), and each pipeline
rank's backward touches a **different subset** of the shared link's parameters. So the
filter yields different parameters, in different order, with different total size on
each rank. Verified on CPU with the real module and real dims
(`hidden_dims=[2048,2048,1536]`, `shared_dim=1024`):

| rank | flat numel | slot 5 holds |
|---|---|---|
| 0 planner | 8,404,992 | `in_proj.1.weight` |
| 1 critic | 7,880,192 | `in_proj.2.weight` |
| 2 solver | 7,356,417 | `gamma` |

Only the first 5 slots (`expert_W1/b1/W2/b2`, `alpha`) line up. Everything after is
summed against an unrelated tensor, and the buffers are not even the same length, so
the NCCL all-reduce itself is undefined behaviour.

This explains §4.2: `gamma`'s slot on the solver receives element 0 of
`in_proj.1.weight`'s gradient from the planner, not its own gradient — so it never
moves. `alpha` sits in the aligned prefix, which is why it trains normally.

There was a second consequence beyond the mixing: each rank's `optimizer.step()` only
updated the parameters it happened to have a non-`None` grad for, so the three
replicas of the shared link **drifted apart** during training — despite the module
docstring's claim that they are "kept in sync via allreduce after each backward step".
Checkpoints save one rank's copy, so the saved link had stale weights for the two
transitions that rank did not own.

**Scope note:** this does *not* affect anything in §1–§5. `GradMonitor` reads
gradients during backward, before any all-reduce; the profile is fully present at
step 0 before the first `optimizer.step()`; and §4.3 reproduces it with no
distributed code at all.

#### The fix

Three changes, all in `run_training`:

1. A fixed, rank-independent sync list built next to `local_params`, with gradient
   buffers materialised up front so the layout is stable from step 0:

   ```python
   sync_params = ([p for p in local_params if p.requires_grad]
                  if shared_link is not None else [])
   for p in sync_params:
       if p.grad is None:
           p.grad = torch.zeros_like(p)
   ```

   `p.requires_grad` is a property of how the module was built, so it is identical on
   every rank; `p.grad is not None` is a property of which backward ran, so it is not.
   Frozen `beta` (under `--no_round_skip`) is excluded, which keeps AdamW from
   applying decoupled weight decay to it.

2. `optimizer.zero_grad(set_to_none=False)`, so those buffers survive each step
   instead of being dropped back to `None`.

3. The all-reduce iterates `sync_params` unconditionally, with no `is not None`
   filter on either the gather or the scatter-back.

The `/world` division is kept: pipeline stages hold partial derivatives of the *same*
loss, so the SUM is already the true gradient and dividing by 3 is technically wrong —
but it is a uniform rescaling of the whole vector, which Adam is invariant to and
which is far below the `clip_grad_norm_` threshold, so removing it would change
nothing except to invalidate comparison with earlier runs.

#### Verification

Re-running the same CPU check against the fixed code path:

| | before | after |
|---|---|---|
| flat numel (planner / critic / solver) | 8,404,992 / 7,880,192 / 7,356,417 | 19,420,673 / 19,420,673 / 19,420,673 |
| slot → parameter identical across ranks | no (diverges at slot 5) | **yes** |
| buffer covers the whole link | no | yes (matches the 19,420,673 trainable count) |

And after a SUM all-reduce, every single-owner parameter now carries exactly its
owner's gradient — `gamma` from the solver, `out_proj.2` from the planner,
`out_proj.3` from the critic, and so on. Previously `gamma`'s slot held element 0 of
the planner's `in_proj.1.weight` gradient.

End-to-end confirmation that `gamma` is no longer pinned is in §6.3.

### 6.3 End-to-end confirmation of the §6.2 fix

Two 50-step runs on 3 GPUs (jobs 15536842 `shared_state`, 15536843 `shared_roae`,
`--dataset math500`, lr 1e-4). Both `COMPLETED`; no hang, no NCCL error beyond the
pre-existing benign device-id/P2P warnings. (The old mismatched-size version did not
hang either, so this only rules out a *new* failure — the correctness argument is the
structural check in §6.2.)

**`gamma` now trains.** It had been pinned at exactly `9.99e-04` for 3000 steps at
both learning rates in every previous run:

| step | 0 | 5 | 10 | 15 | 25 | 35 | 49 |
|---|---|---|---|---|---|---|---|
| `gamma` | 9.92e-04 | 1.02e-03 | 1.17e-03 | 1.34e-03 | 1.34e-03 | 1.25e-03 | 1.24e-03 |

It moves on the very first step (logging happens after `optimizer.step()`), and the
saved checkpoint reads `gamma = 0.0012359619140625` — the first shared_state
checkpoint in the repo whose `gamma` is not its init value.

**Frozen `beta` stayed frozen** at `0.00e+00` throughout, confirming that excluding
non-`requires_grad` parameters from `sync_params` keeps AdamW from applying decoupled
weight decay to it.

**The §1–§5 conclusions are unchanged**, as expected — the round profile is
architectural and was never a consequence of this bug:

```
shared_state   12:  r1:6.066e-05  r2:6.380e-08  r3:2.023e-03    <- V, r1/r2 = 951
shared_roae    12:  r1:1.040e-10  r2:9.405e-08  r3:1.348e-04    <- monotone
shared_state state: r1:6.023e-03  r2:6.023e-03   ratio 1.0000   <- identity highway
```

With `gamma` now free to move, `r1/r2` should track `1/gamma` as it changes rather
than sitting at a fixed 1000 — worth watching on the next full training run.

**Implication for existing checkpoints.** Because each rank previously stepped only
the parameters it had gradients for, the three replicas drifted apart, and every
shared-mode checkpoint in `outputs/checkpoints/` was saved from a single rank's
partially-stale replica. Those checkpoints are affected by more than the frozen
`gamma`; treat shared-mode runs from before this fix as suspect.

---

## Figures

All in `grad_round_sweep_figures/`:

| file | what |
|---|---|
| `sweep_{shared_roae,shared_state}_r{3,4,5}_lr1e-4_s200.png` | the six §4.4 sweep runs |
| `reference_shared_roae_r3_lr1e-3_s3000.png` | reconstruction of the original `shared_roae` reference plot |
| `reference_shared_state_r3_lr1e-4_s3000.png` | reconstruction of the original `shared_state` reference plot |
| `reconstruct_reference_plots.py` | script that builds the two reconstructions |

**Note on the two reference PNGs.** `plot_single` writes to a fixed path per
(mode, n_rounds), so the r=3 sweep runs overwrote
`outerlink_grad_shared_roae_r3.png` and `outerlink_grad_shared_state_r3.png` in the
repo root. The originals were not recoverable from git (the `shared_state` one was
never tracked; the tracked `shared_roae` blob is a different, older 1000-step run).
The reconstructions above are rebuilt from data that *was* preserved — the exact loss
curve from `resume_global.pt`, and the exact final averages plus 300-step gradient
samples from the SLURM logs — and reproduce the originals' loss panel and bar chart
exactly; only the middle panel is coarser (300-step samples rather than per-step).

**This is now fixed.** `plot_single` / `plot_compare` / `plot_sweep` no longer write to
the project root: each figure goes into the run's own output directory,
`outputs/checkpoints/<out_prefix>_<mode>_r<N><suffix>_<timestamp>/`, alongside the
checkpoints it describes. A figure covering several runs (`compare`, `roundskip`,
`sweep`) is written into each participating run's directory. Because the directory
name carries the run timestamp, two runs with the same (mode, n_rounds) can no longer
overwrite each other's plots.

## Reproducing

```bash
# CPU, seconds, no GPU needed — §4.3
python exp_round_grad_profile_toy.py

# GPU sweep — §4.4
cd scripts
# NOTE: config.sh has EVAL=true; set EVAL=false to reach training mode
for mode in shared_roae shared_state; do
  for r in 3 4 5; do
    ./submit.sh --mode $mode --n_rounds $r --steps 200 --lr 1e-4 --slurm_time 04:00:00
  done
done
```
