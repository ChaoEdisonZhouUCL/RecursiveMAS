# Width vs. Depth in a Recursive Multi-Agent System

*Brainstorm note — theory that could explain the trade-off between **number of agents `N`**
(width) and **number of recursion rounds `R`** (depth) in RecursiveMAS.*

Status: speculative synthesis. Some of this is theorem, some is analogy. I mark which is
which: **[T]** = an actual theorem in the ML/CS literature that transfers directly,
**[A]** = analogy that is suggestive but not proved for MAS, **[E]** = something we can
measure in this repo.

---

## 0. The object we are talking about

In this codebase one *round* is a closed loop over `N = 3` agents (planner → critic →
solver → planner), and the system is unrolled for `R` rounds with a single CE loss at
the end. Agents talk through latent `RecursiveLink` adapters of width `d = shared_dim`,
with frozen LLM backbones.

So the computation graph is:

```
x ─► A₁ ─► A₂ ─► … ─► A_N ─┐          round 1
     ▲                     │
     └───── link ──────────┘
     A₁ ─► A₂ ─► … ─► A_N ─┐          round 2
     ▲                     │
     └───── link ──────────┘
                 ⋮                     R rounds
                                 ─► loss
```

The naïve "MAS = deep net" mapping (`N` = width, `R` = depth) is **wrong in one important
way**, and getting that right is most of the insight:

> Agents within a round are **not** in parallel. They are chained. So `N` is *also* depth.
> The system's true depth is `N·R`. What `N` actually buys is **heterogeneity of the
> per-layer function** (`N` distinct maps) and **parameter count**, not parallelism.

This gives the correct three-axis picture, and I think it is the right frame for the paper:

| axis | symbol | NN analogue | what it buys |
|---|---|---|---|
| sequential steps | `L = N·R` | **depth** | function composition, iterative refinement |
| distinct functions | `N` | **number of untied layers** | expressive variety per step; parameters |
| latent channel | `d = shared_dim` | **width** | information carried between steps |
| repetition | `R` | **weight-tied loop depth** | depth *without* parameters |

Everything below is about how these three trade off.

---

## 1. The classical result: depth is exponentially more valuable than width

**[T] Depth separation.** For ReLU networks:

- Telgarsky (2016): there are functions computable by a network of depth `Θ(k³)` and
  width `O(1)` that **any** network of depth `O(k)` needs width `2^{Ω(k)}` to approximate.
- Eldan & Shamir (2016): a specific radial function needs only 3 layers with poly width,
  but any 2-layer approximation needs width exponential in the input dimension.
- Montúfar et al. (2014): a depth-`L`, width-`w` ReLU net carves at most
  `~ (w/n₀)^{n₀(L−1)} · w^{n₀}` linear regions — **exponential in depth, polynomial in
  width**.

**Transfer to MAS [A]:** if each agent step is (roughly) a nonlinear map of bounded
complexity, then a system with `L = N·R` sequential steps has expressivity that grows
*exponentially in `L`* and only *polynomially in the per-step capacity*. Under a fixed
budget, **buying steps beats buying per-step capacity**.

This is the formal version of your intuition: *"when the number of agents is limited, we
can improve the expressivity of MAS by increasing the number of inference steps."*
And note that with tied links (`shared_roae`), extra rounds cost **zero parameters** —
the only cost is FLOPs and gradient path length. That is an unusually good deal, and it
is exactly why looped/recurrent architectures are interesting.

**Caveat [T]:** the depth-separation theorems assume *untied* layers. A weight-tied loop
is strictly weaker than an untied net of the same depth — see §3.

---

## 2. The complexity-theoretic version: rounds buy *computation*, agents buy *capacity*

This is the framing I would actually put in a paper, because it is crisp and it is about
LLMs specifically.

**[T]** A fixed-depth transformer with log precision is in (uniform) **TC⁰** — a
constant-depth, poly-size threshold circuit class. It provably *cannot* do things like
composed multi-step reasoning, graph connectivity, or iterated state tracking, no matter
how wide you make it (assuming TC⁰ ≠ NC¹).

**[T]** Merrill & Sabharwal (2024), "The Expressive Power of Transformers with Chain of
Thought": adding `t(n)` sequential intermediate steps lifts the class roughly to
`TIME(t(n))`. Logarithmically many steps buys you ~`L`; polynomially many steps buys you
**all of P**. Sequential steps are the *only* thing that lifts the class.

**Transfer [A]:** RecursiveMAS's rounds are chain-of-thought moved into latent space.
So:

> **Width (`N`, `d`) changes what the system can *represent* in one step.
> Depth (`R`) changes what the system can *compute* at all.**
> No amount of width lifts you out of the constant-depth class; `R` does.

This is a hard asymmetry, not a smooth trade-off, and it is the strongest theoretical
statement available: for tasks whose solution has an inherent sequential dependency
chain of length `D*` (multi-step algebra, planning, state tracking, iterative proof),
there is a **hard lower bound `N·R ≥ D*`**. Extra agents cannot substitute. Empirically
this shows up as a sharp accuracy cliff below `R*`, not a gentle decline — a very
testable prediction **[E]**.

**Work–span framing.** In parallel-computing terms, `N` is *work* and `R` is *span*.
Brent's theorem says extra work never reduces the critical path. A problem with span
`D*` needs `D*` sequential steps whatever your work budget. Different tasks sit at
different points on the work/span plane, which predicts task-dependent optimal `(N, R)`.

---

## 3. What weight-tying costs: the "iterated map" ceiling

`shared_roae` ties one link across all rounds, so parameters are `O(N·d + d²)`,
independent of `R`. What is the price?

**[T]** Universal Transformers (Dehghani et al., 2019), ALBERT (Lan et al., 2020),
Looped Transformers (Giannou et al., 2023; Yang et al., 2024), Deep Equilibrium Models
(Bai et al., 2019), and latent recurrent-depth LMs (Geiping et al., 2025) all establish
the same shape of result: a tied loop run `T` times *can* simulate a `T`-layer untied
network for many tasks of interest — programmable-computer constructions show looped
transformers are Turing-complete with enough iterations — **provided the state carried
between iterations is wide enough** and, critically, **provided the input is re-injected
each iteration**.

Two consequences for us:

**(a) Minimum-width theorems become minimum-`shared_dim` theorems. [T→A]**
For ReLU nets of *unbounded* depth, universal approximation requires width
`w ≥ d_in + 1` (Lu et al., 2017; Hanin & Sellke, 2017) — below that, **no amount of
depth recovers expressivity**. The MAS analogue is sharp and testable:

> There is a critical `shared_dim` below which increasing `R` does not help at all.
> The inter-agent latent channel, not the agent count, is the true "width" governing
> universality.

I would bet this is the single most publishable prediction in this note. **[E]** Sweep
`shared_dim ∈ {64, 128, 256, 512, 1024}` × `R ∈ {1..6}` and look for a *knee*: below
critical `d`, the `R`-curve is flat; above it, `R` helps. That is a phase transition,
and phase transitions make good figures.

**(b) Fixed-point saturation.** A tied loop `h ← F(h)` with `F` contractive converges:
`‖h^{(R)} − h*‖ ≤ ρ^R ‖h^{(0)} − h*‖`. Past the mixing time, extra rounds add nothing —
the system has reached its fixed point and is burning FLOPs. So the expressivity gain
from `R` is **not** unbounded; it saturates at `R* ≈ mixing time`, and `R*` is
task-dependent (harder task → slower mixing → larger `R*`). This predicts the
diminishing-returns curve everyone observes with more "rounds of debate."

Note `shared_state`'s `z^{(r+1)} = z^{(r)} + γ·f^{(r+1)}` **escapes** the contraction
ceiling: it is an accumulator, not a contraction. It changes the class from *iterated
map* to *state machine with unbounded memory* — the same jump as vanilla-RNN → LSTM,
or ResNet's identity path. That is a genuine expressivity difference, not just a
gradient trick.

---

## 4. The trainability frontier — this is where our gradient data lives

Expressivity is only half the story. The other half is whether you can *fit* the thing,
and this repo already has the data.

Let `ρ` be the per-round Jacobian gain of the cross-round boundary. Gradient reaching
round `r` from a loss at round `R` scales like `ρ^{R−r}`. From
`GRADIENT_ROUND_PROFILE.md`:

- `shared_roae`: every round boundary is a full `out_proj(latent_core(·))`, `ρ ≈ 1e−3`.
  Round-1 gradient dies like `ρ^R` — it **underflows float32 to exactly 0.0 by R = 7**.
  Monotone decay, round `R` → round 1. This is your first plot.
- `shared_state`: the state chain has Jacobian `I + γJ ≈ I`, an identity highway.
  Middle rounds are **flat, not decaying** — depth-independent — and round 1 sits
  `1/γ ≈ 1000×` above them because it *initialises* the state (enters with weight 1,
  not `γ`). Hence the V. This is your second plot.

So define **effective trainable depth**:

```
D_eff  ≈  log(δ_machine) / log(ρ)
```

- `ρ ≈ 1e−3` (shared_roae, fp32, δ ≈ 1e−38):  `D_eff ≈ 6` rounds. Beyond that, early
  rounds receive *literally zero* gradient — the deep part of the network exists in the
  forward pass but is untrained.
- `ρ → 1` (shared_state, identity highway): `D_eff → ∞`.

**This is the central trade-off of the paper, and it is not the one people usually state.**
The binding constraint on `R` is **not** expressivity and **not** FLOPs — it is
`N·R ≤ D_eff`. And `D_eff` is a property of the *link design*, which is exactly what we
control:

> **Link architecture sets the depth budget. Expressivity theory then tells you how to
> spend it.**

A clean way to say it: expressivity wants `N·R` large; trainability caps `N·R` at
`D_eff(link)`; the contribution of `shared_state` / ReZero / skip-in-latent-space is to
raise the cap rather than to improve any fixed-depth model. That reframes gradient
engineering from "a training trick" to "the thing that unlocks the depth axis" —
precisely the ResNet story (depth was always better; BN + identity paths made it
*reachable*).

---

## 5. The statistical trade-off: `N` reduces variance, `R` reduces bias

A complementary, non-circuit-theoretic account, closer to how the MAS literature talks.

- **More agents = ensembling.** With `N` diverse agents, error variance falls roughly
  `~1/N` under independence, but the shared bias does not move. Parallel sampling gives
  coverage `1 − (1−p)^N` (Brown et al., 2024, "Large Language Monkeys" — pass@k is a
  power law in `k`). Great for *search and verification*; useless if every agent shares
  the same blind spot — and in RecursiveMAS the backbones are frozen and often identical,
  so the independence assumption is weak and `N`'s returns fade fast.
- **More rounds = iterative refinement.** Each round is a Newton-ish/proximal step
  toward a better answer, cutting *bias*, subject to the contraction ceiling in §3(b)
  and to error accumulation if `ρ > 1`.

**[T-adjacent]** Snell et al. (2024) find empirically that optimal test-time compute
allocation flips with difficulty: *easy* problems favour **sequential** revision, *hard*
problems favour **parallel** search (because the sequential process gets stuck in a bad
basin). Bansal/Schwarzschild et al.'s "deep thinking" nets show recurrent depth
extrapolates to harder instances at test time.

Combined prediction **[E]**: **the optimal `(N, R)` split depends on problem difficulty
and on whether errors are *recoverable***. Recoverable errors (arithmetic slips a critic
can catch) → spend on `R`. Unrecoverable errors (wrong strategy chosen at step 1) →
spend on `N`. Since RecursiveMAS's critic is a *refinement* mechanism, I predict our
architecture is `R`-favouring on MATH500 — and that a breadth-favouring baseline
(self-consistency over `N` samples) wins on the subset where the first-round plan is
wrong.

---

## 6. A candidate scaling law to actually fit

If we want one equation for the paper, the Chinchilla-style separable form is the
natural first hypothesis:

```
Error(N, R, d)  =  E∞  +  A / N^α  +  B / R^β  +  C / d^κ
        subject to   FLOPs ≈ c · N · R          (iso-compute)
        subject to   N · R ≤ D_eff(link)         (trainability)
```

Minimising the first two terms under `N·R = C/c` gives the optimal split

```
R* / N*  =  (Bβ / Aα)^{1/(α+β)}   →   a constant ratio along the iso-FLOP frontier,
```

i.e. **`N` and `R` should scale together as a fixed ratio as compute grows** — the same
structural conclusion as compute-optimal model/data scaling. Then the *interesting*
result is what breaks it:

1. `β` is **not** constant — it collapses past the mixing time `R*` (§3b). Expect a
   knee, so fit `B/R^β` only for `R ≤ R*`, or use `B·ρ^R` for a contraction-flavoured fit.
2. The `N·R ≤ D_eff` constraint **binds before** the iso-FLOP optimum for `shared_roae`
   (`D_eff ≈ 6`) but **not** for `shared_state`. So the two link types have *different
   compute-optimal frontiers* — and that is a headline result: link design changes the
   scaling law, not just the constant.
3. `α` should be small (frozen, homogeneous backbones ⇒ weak ensembling), `β` larger.
   If we measure `α ≪ β`, that is direct empirical support for "depth beats width in MAS."

**[E] The experiment:** a grid over `N ∈ {2,3,4,5}` × `R ∈ {1..8}` × link ∈
{`shared_roae`, `shared_state`}, with iso-FLOP diagonals highlighted. Report accuracy
*and* the round-1 gradient norm on the same axes. The story writes itself if the
accuracy frontier turns over exactly where the gradient underflows.

---

## 7. Where the deep-net analogy breaks (worth stating honestly)

1. **Agents are not layers — they are frozen LLMs.** Per-step capacity is enormous and
   fixed; only the thin links train. So we are training a `N·R`-deep network whose
   *layers* are frozen and whose *connections* are learned. Closer to Deep Equilibrium /
   hypernetwork territory than to standard depth-separation setups. Depth-separation
   theorems assume you control the layer functions; we do not.
2. **Non-uniform layers.** Planner/critic/solver are functionally distinct, so the
   round is not `F^N` but `F_N ∘ … ∘ F_1`. Width `N` therefore also buys *heterogeneity*,
   which has no clean analogue in width-vs-depth theory. Might be closer to
   "mixture-of-depths" or modular-network theory.
3. **Truncated BPTT.** `latent_rollout()` keeps only step 0 in the graph; steps 1..47 are
   detached. The *forward* depth and the *backward* depth differ. Any theory statement
   about gradients applies to the truncated graph, not the full unrolled computation —
   we should say this explicitly or a reviewer will.
4. **Discrete vs. latent communication.** The paper's Theorem 4.1 (latent links keep
   `‖∂R/∂h‖ ≈ 1`, text links are `O(ε)`) is itself a width/depth statement: text
   communication makes `ρ ≈ 0`, so `D_eff ≈ 1` and text-based MAS is *structurally
   shallow* regardless of how many rounds you run. That is a nice way to state the
   contribution: **latent links do not make each round better, they make rounds
   composable at all.**

---

## 8. One-paragraph summary

A recursive MAS is a weight-tied deep network of effective depth `L = N·R`, per-step
heterogeneity `N`, and channel width `d = shared_dim`. Classical depth-separation and
circuit-complexity results say depth is exponentially more valuable than width and that
*only* sequential steps lift the computational class — so with a limited number of
agents, extra rounds genuinely do buy expressivity, and with tied links they buy it at
zero parameter cost. Three things bound that: (i) a **minimum channel width** `d`, below
which no amount of recursion recovers universality; (ii) **fixed-point saturation**,
which caps useful rounds at the task's mixing time unless the link carries accumulating
state; and (iii) **trainability**, `N·R ≤ D_eff = log δ / log ρ`, which for
`shared_roae` (`ρ ≈ 1e−3`) is about 6 rounds and for `shared_state` (identity highway,
`ρ ≈ 1`) is unbounded. The practical claim: *link design sets the depth budget, and
expressivity theory tells you how to spend it.*

---

## 9. Open questions for you

1. Do you want the paper's framing to be **complexity-theoretic** (§2, strongest claim,
   hardest to prove for our setting) or **empirical scaling law** (§6, safest, fits what
   we already have infrastructure for)? I lean §6 as the backbone with §2 as motivation.
2. Is `N` variable in our codebase? `SharedRecursiveLink.N_AGENTS = 3` is hard-coded and
   the prompts assume planner/critic/solver. A width sweep needs that generalised —
   worth knowing before promising an `N × R` grid.
3. Is the `shared_dim` sweep (§3a) worth prioritising over the `N × R` sweep? I think
   the minimum-width phase transition is the more novel result, and it is cheaper.
4. Should I set up the `(N, R)` grid via `submit.sh` now, or do you want to settle the
   framing first?

---

## References (pointers, verify before citing)

- Telgarsky, *Benefits of depth in neural networks*, COLT 2016.
- Eldan & Shamir, *The power of depth for feedforward neural networks*, COLT 2016.
- Montúfar, Pascanu, Cho & Bengio, *On the number of linear regions of DNNs*, NeurIPS 2014.
- Lu et al., *The expressive power of neural networks: a view from the width*, NeurIPS 2017;
  Hanin & Sellke, *Approximating continuous functions by ReLU nets of minimal width*, 2017.
- Merrill & Sabharwal, *The parallelism tradeoff: limitations of log-precision transformers*, TACL 2023 (TC⁰).
- Merrill & Sabharwal, *The expressive power of transformers with chain of thought*, ICLR 2024.
- Dehghani et al., *Universal Transformers*, ICLR 2019; Lan et al., *ALBERT*, ICLR 2020.
- Giannou et al., *Looped transformers as programmable computers*, ICML 2023;
  Yang et al., *Looped transformers are better at learning learning algorithms*, 2024.
- Bai, Kolter & Koltun, *Deep equilibrium models*, NeurIPS 2019.
- Geiping et al., *Scaling up test-time compute with latent reasoning: a recurrent depth approach*, 2025.
- Schwarzschild/Bansal et al., *End-to-end algorithm synthesis with recurrent networks* ("deep thinking"), NeurIPS 2022.
- Snell et al., *Scaling LLM test-time compute optimally*, 2024.
- Brown et al., *Large language monkeys: scaling inference compute with repeated sampling*, 2024.
- This repo: `GRADIENT_ROUND_PROFILE.md`, `exp_gradient_stability.md`.
