# Gradient Stability Experiment — RecursiveMAS Outer-Loop Training

## Background

RecursiveMAS (Yang et al., 2026) casts a multi-agent system as a unified recursive
computation. N heterogeneous agents are chained in a loop and the entire system is
unrolled for n forward rounds. Agents communicate through lightweight **RecursiveLink**
adapters rather than text. The outer-loop training objective is a single cross-entropy
loss applied only at the final round's last agent output:

```
L_out = CE( S^(n)( S^(n-1)( ... S^(1)(x) ... ) ), y )      [Eq. 6 in paper]
```

Gradients must flow back through all n rounds and all N agents to update the adapters.
This raises the natural question: **is rollout training stable, or do gradients vanish
/ explode in early rounds?**

---

## The Core Question

When backpropagating a single CE loss from round N back to round 1, does the gradient
norm at each round remain informative, or does it collapse?

**Diagnostic:** compute the gradient norm at the last-agent hidden state in each round
via `tensor.register_hook`. Then measure the ratio:

```
ratio = grad_norm(round 1) / grad_norm(round N)
  < 0.1   →  vanishing  (early rounds receive no useful signal)
  > 10    →  exploding  (training diverges)
  ≈ 1     →  stable
```

The paper's **Theorem 4.1** proves that the latent RecursiveLink maintains near-constant
gradient norms (`‖∂R/∂h‖ ≈ 1`) while text-based interaction vanishes (`‖∂R_text/∂h‖ ≤
O(ε) ≪ 1` when token confidence is high). The experiment verifies this empirically on a
miniaturised proxy system.

---

## Three Models Compared

### 1. RecursiveMAS — paper's design (baseline)

Each agent-to-agent transition uses an **independent** `OuterAdapter` (Eq. 4):

```
R_out(h) = ln_target( proj2(GELU(proj1(ln_source(h)))) + W3·h )
```

- N agents in a closed loop → N outer links total (N-1 within-round + 1 cyclic A_N→A_1)
- Each agent also has an **inner link** `R_in(h) = h + W2·σ(W1·h)` (Eq. 3) for
  self-refinement after latent generation
- LLM backbone weights are **frozen**; only inner/outer adapters are trained
- Residual connection `W3·h` preserves latent semantics and is the key mechanism that
  keeps Jacobian norm ≈ 1

### 2. SharedLink MAS with Rotary Agent Encoding (RoAE) — new idea

A **single** `SharedRecursiveLink` module handles **all** transitions (inner and outer),
conditioned on `(src, dst)` agent indices via a **Rotary Agent Encoding** — a RoPE-style
rotation applied directly to the hidden vector, with zero extra learnable parameters for
the positional part.

#### Why RoPE instead of a lookup table

A naive lookup table (`emb(src) + emb(dst)`) treats agent indices as arbitrary tokens
with no relational structure. RoPE-style encoding is better because:

1. **Relative offset is implicit.** The hidden vector `h` is rotated by `src`-frequencies
   through the MLP path, and by `dst`-frequencies through the residual path. Their
   interaction is automatically sensitive to `dst − src` (the transition distance) without
   any learned table.
2. **Zero extra parameters.** The cos/sin tables are computed analytically from index
   values — no `Embedding` weight matrix is needed.
3. **Structural inductive bias.** An inner call `(i, i)` has `dst − src = 0` and both
   paths see the same rotation. An outer call `(i, i+1)` has offset 1. The cyclic call
   `(N, 1)` has offset `1 − N`. The MLP learns to exploit these geometric relationships.

#### Change 1: heterogeneous hidden dims

In the real system agents have different model sizes (e.g. Qwen-1.7B `d=2048`,
LLaMA-3.2-1B `d=2048`, Qwen2.5-Math-1.5B `d=1536`).  The experiment now supports this
via `--hidden_mults`, e.g. `--hidden 64 --hidden_mults 1 2 1` gives dims `[64, 128, 64]`.

Each chain (RecursiveMAS, SharedLink, Text-MAS) uses `OuterAdapter(in_dim, out_dim)`
with the correct per-link dimensions.  The `SharedRecursiveLink` adds **per-agent
input/output projection layers** (one `Linear` per agent) that map each agent's hidden
dim into a common `shared_dim` before the MoE core and back out:

```
x = in_proj[src](h)      # (B, T, hidden_src) -> (B, T, shared_dim)
...
return out_proj[dst](out) # (B, T, shared_dim) -> (B, T, hidden_dst)
```

This is the minimal change that handles heterogeneity without touching the shared
computation: the per-agent projections are the only place where agent-specific capacity
is spent on dimension alignment.

#### Change 2: Mixture-of-Experts (MoE) core

The single FFN is replaced by K expert FFNs with a **soft router** conditioned on the
RoAE features of the transition:

```
# Step 1: project to shared dim d
x = in_proj[src](h)

# Step 2: RoAE rotations
x_src = RoPE(x, src)        # "who is sending"
x_dst = RoPE(x, dst)        # "who is receiving"

# Step 3: router — uses (src, dst, dst-src) geometry
w = softmax( router( cat(x_src, x_dst, x_dst - x_src) ) )   # (B, T, K)

# Step 4: soft MoE dispatch
normed  = ln_in(x_src)
mixture = sum_k  w[..., k] * expert_k(normed)               # each expert: Linear -> GELU -> Linear

# Step 5: residual + project out
return out_proj[dst]( ln_out(x_src + mixture) )
```

**Why MoE here?**  Each transition type needs qualitatively different processing:

| Transition | RoAE offset | Expected specialisation |
|---|---|---|
| Inner `(i, i)` | 0 | Self-refinement: smooth latent thoughts |
| Forward outer `(i, i+1)` | +1 | Cross-agent alignment: re-encode semantics |
| Cyclic `(N, 1)` | `1-N` | Long-range feedback: distil round summary |

A single FFN must average over all three, risking destructive interference.  The MoE
router sees the RoAE-encoded transition geometry and can dispatch each type to its
specialist expert(s).

#### Full architecture of SharedRecursiveLink

```
SharedRecursiveLink:
  in_proj[1..N]   : Linear(hidden_i, shared_dim)          per-agent, trainable
  out_proj[1..N]  : Linear(shared_dim, hidden_i)          per-agent, trainable
  router          : Linear(3*shared_dim, K)               shared, trainable
  expert_w1[k]    : Linear(shared_dim, shared_dim*2)      K experts, trainable
  expert_w2[k]    : Linear(shared_dim*2, shared_dim)      K experts, trainable
  ln_in, ln_out   : LayerNorm(shared_dim)                 shared, trainable
  RoPE tables     : cos/sin of agent indices              no parameters
```

**Key differences vs. the paper:**

| Property | RecursiveMAS | SharedLink (RoAE + MoE) |
|---|---|---|
| Heterogeneous dims | `OuterAdapter(in, out)` per link | Per-agent `in_proj`/`out_proj` + shared core |
| Number of link modules | 2N independent adapters | **1 shared module** |
| Transition specialisation | separate weights per (src,dst) pair | K experts soft-dispatched by RoAE-conditioned router |
| Positional conditioning | none | Rotary encoding of src, dst (zero extra params) |
| Gradient accumulation | each adapter updated by one path | all N×n_rounds invocations update same params |
| Interference risk | none | router must correctly separate inner/outer/cyclic roles |

**Hypothesis:** the MoE router can use the RoAE geometry (`dst−src = 0, 1, 1−N`) to
separate the three transition types into different experts, avoiding the interference
that a single shared FFN suffers — while still keeping total parameter count lower than
2N independent adapters at large N.

### 3. Text-based MAS — vanishing baseline (Theorem 4.1 contrast)

Replaces outer links with the softmax bottleneck from Assumption A.1:

```
R_text(h) = W_in · softmax(W_out · h)
```

As the model becomes confident (entropy → 0), the softmax saturates and the Jacobian
norm collapses to `O(ε) ≪ 1`. This is the baseline the paper claims RecursiveMAS
improves upon.

---

## Implementation

File: `exp_gradient_stability.py`

### Architecture proxy

| Real RecursiveMAS | Experiment proxy |
|---|---|
| LLM agent (billions of params) | 2-layer MiniTransformerBlock + LM head |
| Inner RecursiveLink `R_in` | `InnerAdapter`: 2-layer MLP residual |
| Outer RecursiveLink `R_out` | `OuterAdapter`: 2-layer MLP + residual proj + LayerNorm |
| SharedLink (new) | `SharedRecursiveLink`: per-agent in/out projections + MoE FFN (K experts) + RoAE router |
| Text-based link | `TextBasedOuterAdapter`: linear → softmax → linear |
| Frozen LLM backbone | explicit `requires_grad_(False)` on embed/blocks/lm_head |
| Cyclic loop A_N→A_1 | `outer_links[N-1]` in `RecursiveMASChain` |

### Gradient measurement

Gradient hooks (`tensor.register_hook`) are attached to the last-agent hidden state
`H[-1]` at each round during the forward pass. After `loss.backward()` the hooks fire
and record `‖grad‖`. This directly measures how much signal survives the backward path
from the loss to each recursion depth.

### What is measured

1. **Loss curve** — convergence speed and stability across training steps
2. **Gradient norm per round** — plotted over training steps (log scale) for each model
3. **Gradient norm ratio** (round-1 / round-N) — primary stability diagnostic
4. **Trainable parameter count** — printed to quantify the sharing compression ratio

### Output

A 5-panel PNG (`gradient_stability.png`):
- Panel 0: training loss curves for all three models
- Panels 1–3: grad-norm-per-round trajectories for RecursiveMAS, SharedLink, Text-MAS
- Panel 4: side-by-side bar chart of average grad norm per round (last 20% of training)

---

## How to Run

```bash
# Default: 3 agents, 4 rounds, 200 steps
python exp_gradient_stability.py

# Stress test with deeper unrolling
python exp_gradient_stability.py --n_rounds 8 --steps 400 --hidden 128

# Sweep over recursion depths to see stability vs. depth
python exp_gradient_stability.py --sweep --steps 200

# Larger hidden size to better approximate real model scale
python exp_gradient_stability.py --n_rounds 4 --hidden 256 --n_agents 3 --steps 300
```

CLI arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--n_agents` | 3 | Number of agents (mirrors planner/critic/solver) |
| `--n_rounds` | 4 | Outer-loop unrolling depth |
| `--hidden` | 64 | Base hidden dimension (multiplied per agent) |
| `--hidden_mults` | `1 2 1` | Per-agent multipliers; `--hidden 64 --hidden_mults 1 2 1` → dims `[64,128,64]` |
| `--n_experts` | 4 | Number of MoE experts K in SharedRecursiveLink |
| `--vocab` | 256 | Vocabulary size |
| `--seq_len` | 32 | Sequence length |
| `--batch_size` | 16 | Batch size |
| `--steps` | 200 | Training steps |
| `--lr` | 1e-3 | AdamW learning rate |
| `--sweep` | off | Run across multiple `n_rounds` values |
| `--sweep_rounds` | 1,2,3,4,6,8 | Depths to sweep |

---

## What to Look For

**Expected result (consistent with paper's Theorem 4.1):**

- **RecursiveMAS:** ratio ≈ 1 across rounds — flat bars in the bar chart — confirming
  that the residual-connection Jacobian `J = I + W2·Σ'·W1` keeps gradient norm near 1.

- **Text-MAS:** ratio → 0 — bars shrink steeply from right to left — confirming
  softmax saturation kills gradient flow.

- **SharedLink:** the interesting middle case:
  - If role conditioning works well → ratio ≈ 1, matching RecursiveMAS stability,
    but with far fewer parameters
  - If inner/outer roles interfere → loss converges more slowly or ratio drifts,
    suggesting the single module cannot simultaneously serve all transition types
  - If gradient accumulation from shared parameters amplifies signal → ratio > 1
    (potential instability at deeper unrolling)

The sweep (`--sweep`) tests whether any of these behaviours change as recursion depth
increases, which is the most practically relevant regime.

---

## Bugs Fixed vs. Original Experiment (paper cross-check)

| # | Bug | Fix |
|---|---|---|
| 1 | Missing inner RecursiveLink `R_in` | Added `InnerAdapter` inside each `AgentModel`; hidden state passes through it before being returned |
| 2 | Wrong number of outer links (N-1 instead of N) | `RecursiveMASChain` now has N outer links; index N-1 is the cyclic A_N→A_1 link |
| 3 | Backbone not frozen | `requires_grad_(False)` applied to embed/pos_embed/blocks/ln_out/lm_head in `AgentModel` |
| 4 | No text-based baseline | Added `TextBasedOuterAdapter` and `TextMASChain` to directly verify Theorem 4.1 |
