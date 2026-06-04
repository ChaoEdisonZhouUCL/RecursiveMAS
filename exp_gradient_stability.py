#!/usr/bin/env python3
"""
Experiment: Gradient stability under outer-loop rollout training.

Simulates the recursive MAS forward pass:
    S(n)(S(n-1)(...S(1)(x)...))
where each S(i) is a simplified agent (Transformer block) connected to the
next via a CrossModelAdapter (outer link), exactly as in RecursiveMAS.

We measure:
  1. Loss convergence across training steps.
  2. Per-round gradient norms at the outer adapters (do early rounds vanish/explode?).
  3. Gradient norm ratio: round-1 / round-N (< 1 → vanishing, > 1 → exploding).
  4. Loss landscape smoothness (std of loss over a mini-batch sweep).

Usage:
    python exp_gradient_stability.py
    python exp_gradient_stability.py --n_rounds 6 --steps 300 --hidden 128

what this script is doing?
Core question: when you unroll the chain S(n)(S(n-1)(...S(1)(x)...)) and backprop a single CE loss from round-N back through round-1, do gradients vanish or explode?

The experiment builds a faithful miniaturized version of the real system:

Real RecursiveMAS	Experiment proxy
AgentModel (LLM)	2-layer Transformer block + LM head
CrossModelAdapter	OuterAdapter — identical architecture to modeling.py
Chain topology (planner→critic→solver)	RecursiveMASChain with n_agents nodes
n_rounds outer unrolling	same
CE loss on final round's last agent	compute_lm_loss on logits[-1]
Three things it records:

Loss curve — is training stable at all? Does it converge, oscillate, or diverge?
Gradient norm per round — using tensor.register_hook on the hidden state of the last agent in each round. This directly measures how much gradient signal survives the backward pass through each outer link back to earlier rounds.
Ratio round-1 / round-N — the key diagnostic:
< 0.1 → vanishing gradient in early rounds
> 10 → exploding gradient in early rounds
≈ 1 → stable propagation
How to run it

# Basic run (n=3 agents, 4 rounds, 200 steps):
python exp_gradient_stability.py

# Deeper unrolling (stress test):
python exp_gradient_stability.py --n_rounds 8 --steps 400 --hidden 128

# Sweep: compare stability across n_rounds = 1,2,3,4,6,8
python exp_gradient_stability.py --sweep --steps 200

# Larger model proxy:
python exp_gradient_stability.py --n_rounds 4 --hidden 256 --n_agents 3 --steps 300
Outputs a 3-panel PNG (gradient_stability.png):

Loss curve with smoothing
Gradient norm trajectories per round (log scale)
Bar chart of average grad norm by round
What to look for in results
The residual projections in OuterAdapter (x + residual_proj(x)) act like skip connections and are the main defense against vanishing. If you see the ratio r1/rN ≈ 1 across rounds, the architecture is doing its job. If ratio drops sharply with deeper --n_rounds, you have evidence that longer rollouts need gradient clipping, TBPTT, or per-round auxiliary losses.
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight stand-ins for the real RecursiveMAS components
# ---------------------------------------------------------------------------

class MiniTransformerBlock(nn.Module):
    """Single self-attention + FFN block; represents one agent's LM backbone."""
    def __init__(self, hidden: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class AgentModel(nn.Module):
    """Simplified agent: embedding + N transformer blocks + inner link + LM head.

    Each agent may have its own hidden dimension (heterogeneous agents, matching
    the real RecursiveMAS setup where e.g. Qwen-1.7B and LLaMA-1B have different
    d_h).  The prefix from the outer link is already projected to this agent's
    hidden dim before being passed in.
    """
    def __init__(self, vocab: int, hidden: int, seq_len: int, n_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(hidden) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.inner_link = InnerAdapter(hidden)  # R_in, Eq. (3)
        self.seq_len = seq_len

        # Freeze backbone; only inner_link trains (mirrors paper's training setup)
        for p in list(self.embed.parameters()) + \
                 list(self.pos_embed.parameters()) + \
                 list(self.blocks.parameters()) + \
                 list(self.ln_out.parameters()) + \
                 list(self.lm_head.parameters()):
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, prefix: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T)
            prefix:    (B, T, hidden_i) already projected to this agent's dim
        Returns:
            logits (B, T, V), last_hidden (B, T, hidden_i) after inner link
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        if prefix is not None:
            x = x + prefix
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_out(x)
        h_linked = self.inner_link(h)
        return self.lm_head(h_linked), h_linked


class InnerAdapter(nn.Module):
    """Inner RecursiveLink R_in: h -> h + W2*sigma(W1*h), matching Eq. (3) in paper.

    Used inside each agent to map last-layer hidden state back to input embedding
    space for the next auto-regressive step (Dense-to-Shallow Transition).
    The paper freezes LLM weights and only trains this adapter.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.proj1 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj2(self.act(self.proj1(h)))


class OuterAdapter(nn.Module):
    """Outer RecursiveLink R_out: h -> W3*h + W2*sigma(W1*h), matching Eq. (4) in paper.

    Bridges heterogeneous agents (Cross-Model Transition). W3 is the residual
    linear projection that maps source dim to target dim. The paper's modeling.py
    wraps this with ln_source/ln_target LayerNorms for training stability.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = out_dim * 2
        self.ln_source = nn.LayerNorm(in_dim)
        self.proj1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden, out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)  # W3 in the paper
        self.ln_target = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = out + self.residual_proj(x)
        return self.ln_target(out)


# ---------------------------------------------------------------------------
# Multi-agent system (chain topology, n agents, n-1 outer links)
# ---------------------------------------------------------------------------

class RecursiveMASChain(nn.Module):
    """
    Chain of N agents forming a closed loop, unrolled for n_rounds.
    Agents may have heterogeneous hidden dims: hidden_dims[i] is agent i's dim.
    Each outer link i projects from hidden_dims[src] to hidden_dims[dst].

    Link layout (1-indexed agents):
      outer_links[i]   : A_{i+1} -> A_{i+2},  i in [0, N-2]  (within-round)
      outer_links[N-1] : A_N     -> A_1        (cyclic, across rounds)
    """
    def __init__(self, n_agents: int, vocab: int, hidden_dims: List[int], seq_len: int):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dims = hidden_dims
        self.agents = nn.ModuleList(
            [AgentModel(vocab, hidden_dims[i], seq_len) for i in range(n_agents)]
        )
        # outer_links[i]: A_{i+1} -> A_{(i+2) % N}
        self.outer_links = nn.ModuleList([
            OuterAdapter(hidden_dims[i], hidden_dims[(i + 1) % n_agents])
            for i in range(n_agents)
        ])

    def forward_one_round(
        self,
        input_ids: torch.Tensor,
        cyclic_prefix: torch.Tensor | None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Run one complete forward pass through all agents in the chain.

        Args:
            input_ids:     (B, T) — same input fed to every agent
            cyclic_prefix: (B, T, H) or None — R_out(H_{A_N}) from previous round,
                           conditioning A_1's input embeddings

        Returns:
            logits_list:   list[N] of (B, T, V)
            hidden_list:   list[N] of (B, T, H)  — H_i after inner link
            next_cyclic:   R_out_{N->1}(H_{A_N}) ready for the next round
        """
        all_logits, all_hidden = [], []
        prefix = cyclic_prefix  # Only A_1 receives cross-round feedback

        for i, agent in enumerate(self.agents):
            logits, hidden = agent(input_ids, prefix=prefix)
            all_logits.append(logits)
            all_hidden.append(hidden)
            # Within-round: pass A_i's latent thoughts to A_{i+1}
            if i < self.n_agents - 1:
                prefix = self.outer_links[i](hidden)
            else:
                prefix = None  # will be overwritten below

        # Cyclic link: A_N -> A_1 for next round (outer_links[N-1])
        next_cyclic = self.outer_links[self.n_agents - 1](all_hidden[-1])
        return all_logits, all_hidden, next_cyclic


# ---------------------------------------------------------------------------
# Shared-Link model with Rotary Agent Encoding (RoAE)
# ---------------------------------------------------------------------------

def build_rope_cache(max_idx: int, dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for agent indices 0..max_idx (inclusive).

    Standard RoPE formula:
        theta_k = 1 / (10000 ^ (2k / dim))   for k = 0 .. dim//2 - 1
        angle(pos, k) = pos * theta_k
    Returns cos and sin tables of shape (max_idx+1, dim//2).
    """
    half = dim // 2
    k = torch.arange(half, dtype=torch.float32, device=device)
    theta = 1.0 / (10000.0 ** (2 * k / dim))           # (half,)
    idx = torch.arange(max_idx + 1, dtype=torch.float32, device=device)
    angles = torch.outer(idx, theta)                    # (max_idx+1, half)
    return angles.cos(), angles.sin()


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to the last dimension of x using scalar cos/sin tables.

    x:   (..., dim)   — any leading batch/seq dims
    cos: (half,)      — pre-looked-up for a single position index
    sin: (half,)
    Returns tensor of same shape as x.
    """
    dim = x.shape[-1]
    half = dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    # Rotate-half: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    x_rot = torch.cat([x1 * cos - x2 * sin,
                       x2 * cos + x1 * sin], dim=-1)
    return x_rot

class SharedRecursiveLink(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        shared_dim: int,
        max_agents: int,
        n_experts: int = 4,
        expert_dim_divisor: int = 4,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.shared_dim = shared_dim
        self.n_experts = n_experts
        self.max_agents = max_agents
        d = shared_dim
        N = len(hidden_dims)

        self.in_proj = nn.ModuleList(
            [nn.Identity()] +
            [nn.Linear(h, d) for h in hidden_dims]
        )

        # Learned update projection only.
        self.out_proj = nn.ModuleList(
            [nn.Identity()] +
            [nn.Linear(d, h) for h in hidden_dims]
        )

        # Direct gradient highway from src hidden space to dst hidden space.
        self.skip_proj = nn.ModuleList([
            nn.ModuleList([
                nn.Identity() if hidden_dims[i] == hidden_dims[j]
                else nn.Linear(hidden_dims[i], hidden_dims[j], bias=False)
                for j in range(N)
            ])
            for i in range(N)
        ])

        for i in range(N):
            for j in range(N):
                mod = self.skip_proj[i][j]
                if isinstance(mod, nn.Linear):
                    nn.init.orthogonal_(mod.weight)

        self.router = nn.Linear(3 * d, n_experts, bias=False)

        exp_hidden = max(1, d // expert_dim_divisor)
        self.expert_w1 = nn.ModuleList([
            nn.Linear(d, exp_hidden) for _ in range(n_experts)
        ])
        self.expert_w2 = nn.ModuleList([
            nn.Linear(exp_hidden, d) for _ in range(n_experts)
        ])

        self.act = nn.GELU()
        self.ln_in = nn.LayerNorm(d)

        # ReZero-style gate.
        self.alpha = nn.Parameter(torch.tensor(1e-3))

        # Start as almost pure skip connection.
        for w2 in self.expert_w2:
            nn.init.zeros_(w2.weight)
            nn.init.zeros_(w2.bias)

        self._rope_cos = None
        self._rope_sin = None
        self._rope_device = None

    def _get_rope(self, device):
        if self._rope_cos is None or device != self._rope_device:
            cos, sin = build_rope_cache(self.max_agents, self.shared_dim, device)
            self._rope_cos = cos
            self._rope_sin = sin
            self._rope_device = device
        return self._rope_cos, self._rope_sin

    def forward(self, h: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        src0 = src - 1
        dst0 = dst - 1

        # Direct highway in native hidden spaces.
        skip = self.skip_proj[src0][dst0](h)

        # Shared update branch.
        x = self.in_proj[src](h)

        cos_tab, sin_tab = self._get_rope(h.device)
        x_src = rope_rotate(x, cos_tab[src], sin_tab[src])
        x_dst = rope_rotate(x, cos_tab[dst], sin_tab[dst])

        gate_input = torch.cat([x_src, x_dst, x_dst - x_src], dim=-1)
        logits = self.router(gate_input.mean(dim=1))

        # I recommend soft routing first. Hard top-1 can starve experts early.
        w = torch.softmax(logits / 2.0, dim=-1).unsqueeze(1)

        normed = self.ln_in(x_src)

        mixture = torch.zeros_like(normed)
        for k in range(self.n_experts):
            expert_out = self.expert_w2[k](
                self.act(self.expert_w1[k](normed))
            )
            mixture = mixture + w[..., k:k+1] * expert_out

        update = self.out_proj[dst](mixture)

        return skip + self.alpha * update


class AgentModelShared(nn.Module):
    """Same as AgentModel but the inner link is the shared MoE link (src==dst)."""
    def __init__(self, vocab: int, hidden: int, seq_len: int,
                 shared_link: SharedRecursiveLink, agent_idx: int, n_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(hidden) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.shared_link = shared_link
        self.agent_idx = agent_idx   # 1-indexed

        for p in list(self.embed.parameters()) + \
                 list(self.pos_embed.parameters()) + \
                 list(self.blocks.parameters()) + \
                 list(self.ln_out.parameters()) + \
                 list(self.lm_head.parameters()):
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor,
                prefix: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        if prefix is not None:
            x = x + prefix
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_out(x)
        # Inner link: src==dst so RoAE offset = 0 (pure self-refinement role)
        h_linked = self.shared_link(h, src=self.agent_idx, dst=self.agent_idx)
        return self.lm_head(h_linked), h_linked


class SharedLinkMASChain(nn.Module):
    """
    Same closed-loop topology as RecursiveMASChain, but uses ONE SharedRecursiveLink
    for every transition (inner and outer), with:
      - Per-agent in/out projection layers for heterogeneous hidden dims
      - MoE core with K experts soft-dispatched by RoAE-conditioned router
      - Round-skip: next_cyclic += beta * initial_prefix so round-0 state is
        directly visible to every subsequent round without vanishing through
        the full chain depth (beta is a learnable scalar, init near zero).
    """
    def __init__(self, n_agents: int, vocab: int, hidden_dims: List[int],
                 seq_len: int, shared_dim: int, n_experts: int = 4,
                 expert_dim_divisor: int = 4):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dims = hidden_dims
        self.shared_link = SharedRecursiveLink(
            hidden_dims=hidden_dims,
            shared_dim=shared_dim,
            max_agents=n_agents,
            n_experts=n_experts,
            expert_dim_divisor=expert_dim_divisor,
        )
        self.agents = nn.ModuleList([
            AgentModelShared(vocab, hidden_dims[i], seq_len,
                             shared_link=self.shared_link,
                             agent_idx=i + 1)
            for i in range(n_agents)
        ])
        # Learnable round-skip gate: blends initial_prefix into every
        # subsequent round's cyclic input.  Init near zero so early training
        # behaves identically to the no-skip baseline.
        self.beta = nn.Parameter(torch.tensor(1e-3))

    def forward_one_round(
        self,
        input_ids: torch.Tensor,
        cyclic_prefix: torch.Tensor | None,
        initial_prefix: torch.Tensor | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        all_logits, all_hidden = [], []
        prefix = cyclic_prefix
        for i, agent in enumerate(self.agents):
            logits, hidden = agent(input_ids, prefix=prefix)
            all_logits.append(logits)
            all_hidden.append(hidden)
            if i < self.n_agents - 1:
                prefix = self.shared_link(hidden, src=i + 1, dst=i + 2)
            else:
                prefix = None
        next_cyclic = self.shared_link(all_hidden[-1], src=self.n_agents, dst=1)
        # Round-skip: add a direct highway from the very first cyclic prefix
        # so that round-0 state reaches every subsequent round without passing
        # through the full chain depth.  beta starts near 0 (ReZero-style) and
        # the model learns how much to blend in.
        if initial_prefix is not None:
            next_cyclic = next_cyclic + self.beta * initial_prefix
        return all_logits, all_hidden, next_cyclic

# ---------------------------------------------------------------------------
# Text-based baseline (Recursive-TextMAS analog)
# Theorem 4.1 claims text-based SFT suffers gradient vanishing because
# R_text(h) = W_in * softmax(W_out * h) has Jacobian norm O(epsilon) << 1
# when predictions are confident.  We simulate this by replacing the outer
# adapter with a softmax bottleneck (encode -> one-hot-like -> decode).
# ---------------------------------------------------------------------------

class TextBasedOuterAdapter(nn.Module):
    """
    Simulates text-mediated cross-agent communication:
      h  ->  logits (W_out h)  ->  softmax  ->  re-embed (W_in p)
    This is R_text(h) from Assumption A.1 of the paper.
    The softmax saturates gradients when confidence is high (entropy -> 0).
    """
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.to_logits = nn.Linear(hidden, vocab, bias=False)   # W_out
        self.to_embed = nn.Linear(vocab, hidden, bias=False)    # W_in

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        logits = self.to_logits(h)          # (B, T, V)
        p = torch.softmax(logits, dim=-1)   # differentiable but saturating
        return self.to_embed(p)             # (B, T, H)


class TextBasedOuterAdapterHetero(nn.Module):
    """Text bottleneck adapter that handles different in/out hidden dims."""
    def __init__(self, in_dim: int, out_dim: int, vocab: int):
        super().__init__()
        self.to_logits = nn.Linear(in_dim, vocab, bias=False)
        self.to_embed  = nn.Linear(vocab, out_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.to_embed(torch.softmax(self.to_logits(h), dim=-1))


class TextMASChain(nn.Module):
    """Same structure as RecursiveMASChain but outer links use text bottleneck."""
    def __init__(self, n_agents: int, vocab: int, hidden_dims: List[int], seq_len: int):
        super().__init__()
        self.n_agents = n_agents
        self.agents = nn.ModuleList(
            [AgentModel(vocab, hidden_dims[i], seq_len) for i in range(n_agents)]
        )
        # outer_links[i]: A_{i+1} -> A_{(i+2) % N}
        self.outer_links = nn.ModuleList([
            TextBasedOuterAdapterHetero(
                hidden_dims[i], hidden_dims[(i + 1) % n_agents], vocab
            )
            for i in range(n_agents)
        ])

    def forward_one_round(
        self,
        input_ids: torch.Tensor,
        cyclic_prefix: torch.Tensor | None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        all_logits, all_hidden = [], []
        prefix = cyclic_prefix
        for i, agent in enumerate(self.agents):
            logits, hidden = agent(input_ids, prefix=prefix)
            all_logits.append(logits)
            all_hidden.append(hidden)
            if i < self.n_agents - 1:
                prefix = self.outer_links[i](hidden)
            else:
                prefix = None
        next_cyclic = self.outer_links[self.n_agents - 1](all_hidden[-1])
        return all_logits, all_hidden, next_cyclic


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def make_batch(vocab: int, seq_len: int, batch: int, device: torch.device):
    """Random token sequences (language-modelling toy task)."""
    ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    return ids


def compute_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token prediction CE loss on last agent's output (L_out, Eq. 6)."""
    lm_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    targets = input_ids[:, 1:].contiguous().view(-1)
    return F.cross_entropy(lm_logits, targets)


def _run_one_model(model, cfg, device, label: str) -> Dict:
    """
    Train `model` for cfg.steps steps and record:
      - per-step loss
      - gradient norm through each round's last-agent hidden state
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01,
    )

    losses: List[float] = []
    grad_norms_per_round: Dict[int, List[float]] = {r: [] for r in range(cfg.n_rounds)}

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  n_agents={cfg.n_agents}  n_rounds={cfg.n_rounds}  "
          f"hidden={cfg.hidden}  steps={cfg.steps}")
    print(f"{'='*60}")

    for step in range(cfg.steps):
        model.train()
        input_ids = make_batch(cfg.vocab, cfg.seq_len, cfg.batch_size, device)

        cyclic_prefix = None
        initial_prefix = None   # captured after round 0 for the round-skip
        all_logits_list = []
        hook_handles = []
        round_hidden_grads: Dict[int, List[float]] = {r: [] for r in range(cfg.n_rounds)}

        is_shared = isinstance(model, SharedLinkMASChain)

        # Unroll n_rounds; attach gradient hooks to last-agent hidden per round
        for r in range(cfg.n_rounds):
            if is_shared:
                logits_list, hidden_list, next_cyclic = model.forward_one_round(
                    input_ids, cyclic_prefix, initial_prefix=initial_prefix
                )
            else:
                logits_list, hidden_list, next_cyclic = model.forward_one_round(
                    input_ids, cyclic_prefix
                )
            all_logits_list.append(logits_list)

            sentinel = hidden_list[-1]
            sentinel.retain_grad()

            def make_hook(ri, container):
                def _hook(grad):
                    container[ri].append(grad.detach().norm().item())
                return _hook

            h = sentinel.register_hook(make_hook(r, round_hidden_grads))
            hook_handles.append(h)

            # Feed cyclic output into next round (not the final round — no leak)
            if r < cfg.n_rounds - 1:
                cyclic_prefix = next_cyclic
                # Capture the first round's cyclic output as the skip baseline
                if r == 0:
                    initial_prefix = next_cyclic.detach()
            else:
                cyclic_prefix = None

        # L_out: CE on final round's last agent output only (Eq. 6)
        final_logits = all_logits_list[-1][-1]
        loss = compute_lm_loss(final_logits, input_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for h in hook_handles:
            h.remove()

        losses.append(loss.item())

        for r in range(cfg.n_rounds):
            vals = round_hidden_grads[r]
            if vals:
                grad_norms_per_round[r].append(float(np.mean(vals)))
            else:
                grad_norms_per_round[r].append(float("nan"))

        if step % max(1, cfg.steps // 10) == 0:
            norms_str = "  ".join(
                f"r{r+1}:{grad_norms_per_round[r][-1]:.3e}"
                for r in range(cfg.n_rounds)
            )
            print(f"  step {step:4d}  loss={loss.item():.4f}  grads [{norms_str}]")

    return {"losses": losses, "grad_norms_per_round": grad_norms_per_round}


def run_experiment(cfg) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def reset_seed():
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Heterogeneous hidden dims: agents get slightly different sizes to simulate
    # real RecursiveMAS (e.g. Qwen-1.7B d=2048, LLaMA-1B d=2048, Qwen2.5-1.5B d=1536).
    # We use multiples of cfg.hidden so heads still divide evenly.
    # Auto-extend hidden_mults to n_agents by cycling (e.g. [1,2,1] -> [1,2,1,1,2] for 5 agents).
    mults = [cfg.hidden_mults[i % len(cfg.hidden_mults)] for i in range(cfg.n_agents)]
    hidden_dims = [cfg.hidden * m for m in mults]
    # shared_dim for SharedLink: use the largest agent's dim as the common space
    shared_dim = max(hidden_dims)

    reset_seed()
    mas_model = RecursiveMASChain(
        n_agents=cfg.n_agents, vocab=cfg.vocab,
        hidden_dims=hidden_dims, seq_len=cfg.seq_len,
    ).to(device)
    results_mas = _run_one_model(mas_model, cfg, device,
        f"RecursiveMAS (independent OuterAdapters, dims={hidden_dims})")

    reset_seed()
    shared_model = SharedLinkMASChain(
        n_agents=cfg.n_agents, vocab=cfg.vocab,
        hidden_dims=hidden_dims, seq_len=cfg.seq_len,
        shared_dim=shared_dim, n_experts=cfg.n_experts,
        expert_dim_divisor=cfg.expert_dim_divisor,
    ).to(device)
    exp_hidden = max(1, shared_dim // cfg.expert_dim_divisor)
    results_shared = _run_one_model(shared_model, cfg, device,
        f"SharedLink MoE top-1 (K={cfg.n_experts}, exp_dim={exp_hidden}, RoAE, dims={hidden_dims})")

    reset_seed()
    text_model = TextMASChain(
        n_agents=cfg.n_agents, vocab=cfg.vocab,
        hidden_dims=hidden_dims, seq_len=cfg.seq_len,
    ).to(device)
    results_text = _run_one_model(text_model, cfg, device,
        "Text-based MAS (softmax bottleneck, Theorem 4.1)")

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    tail = max(1, cfg.steps // 5)
    print(f"\n{'='*60}")
    print("Analysis (last 20% of training)")
    print(f"{'='*60}")

    all_variants = [
        ("RecursiveMAS",  results_mas),
        ("SharedLink",    results_shared),
        ("Text-MAS",      results_text),
    ]
    for label, results in all_variants:
        avg_norms = {}
        for r in range(cfg.n_rounds):
            vals = [v for v in results["grad_norms_per_round"][r][-tail:] if not math.isnan(v)]
            avg_norms[r] = float(np.mean(vals)) if vals else float("nan")

        print(f"\n  [{label}] gradient norm per round:")
        max_norm = max((v for v in avg_norms.values() if not math.isnan(v)), default=1.0)
        for r in range(cfg.n_rounds):
            bar = "#" * max(1, int(40 * avg_norms[r] / (max_norm + 1e-12)))
            print(f"    Round {r+1:2d}: {avg_norms[r]:.4e}  {bar}")

        if cfg.n_rounds >= 2:
            ratio = avg_norms[0] / (avg_norms[cfg.n_rounds - 1] + 1e-12)
            verdict = (
                "VANISHING (ratio << 1)" if ratio < 0.1
                else "EXPLODING (ratio >> 1)" if ratio > 10
                else "STABLE (ratio ~= 1)"
            )
            print(f"    ratio round-1/round-{cfg.n_rounds}: {ratio:.4f}  -> {verdict}")

        loss_tail = results["losses"][-tail:]
        print(f"    loss:  mean={np.mean(loss_tail):.4f}  std={np.std(loss_tail):.4f}")

    # Param count comparison
    def trainable_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n  Trainable params:")
    print(f"    RecursiveMAS : {trainable_params(mas_model):>8,}")
    print(f"    SharedLink   : {trainable_params(shared_model):>8,}")
    print(f"    Text-MAS     : {trainable_params(text_model):>8,}")

    return {"mas": results_mas, "shared": results_shared, "text": results_text}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: Dict, cfg, out_path: str = "gradient_stability.png"):
    """
    5-panel figure:
      [0] Loss curves: all three models
      [1] RecursiveMAS grad norm per round over training
      [2] SharedLink   grad norm per round over training
      [3] Text-MAS     grad norm per round over training
      [4] Avg grad norm bar chart comparison (last 20%), all three side-by-side
    """
    n_rounds = cfg.n_rounds
    tail = max(1, cfg.steps // 5)
    cmap = plt.get_cmap("plasma")

    variants = [
        ("RecursiveMAS",  "mas",    "steelblue"),
        ("SharedLink",    "shared", "mediumseagreen"),
        ("Text-MAS",      "text",   "tomato"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    mults = [cfg.hidden_mults[i % len(cfg.hidden_mults)] for i in range(cfg.n_agents)]
    hidden_dims = [cfg.hidden * m for m in mults]
    fig.suptitle(
        f"Gradient Stability: RecursiveMAS vs SharedLink-MoE(K={cfg.n_experts}) vs Text-MAS\n"
        f"n_agents={cfg.n_agents}, n_rounds={cfg.n_rounds}, "
        f"hidden_dims={hidden_dims}  |  "
        f"Stable = flat bars across rounds; Vanish = bars shrink left->right",
        fontsize=10,
    )

    # Panel 0: Loss curves
    ax = axes[0]
    for label, key, color in variants:
        losses = results[key]["losses"]
        steps = len(losses)
        ax.plot(losses, color=color, linewidth=0.7, alpha=0.4)
        w = max(1, steps // 20)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, steps), smooth, color=color, linewidth=2, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("CE loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panels 1-3: Grad norm per round over training, one panel each
    for panel_idx, (label, key, _) in enumerate(variants):
        ax = axes[panel_idx + 1]
        grad_data = results[key]["grad_norms_per_round"]
        for r in range(n_rounds):
            vals = [v if not math.isnan(v) else 0.0 for v in grad_data[r]]
            if not any(v > 0 for v in vals):
                continue
            color = cmap(r / max(1, n_rounds - 1))
            ax.plot(vals, color=color, linewidth=1.1, alpha=0.85, label=f"Round {r+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad norm")
        ax.set_title(f"{label}\ngrad norm / round")
        ax.legend(fontsize=7)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Panel 4: Avg grad norm bar chart (last 20%), three groups side-by-side
    ax = axes[4]
    x = np.arange(n_rounds)
    n_groups = len(variants)
    width = 0.8 / n_groups
    for g_idx, (label, key, color) in enumerate(variants):
        grad_data = results[key]["grad_norms_per_round"]
        avg_norms = []
        for r in range(n_rounds):
            vals = [v for v in grad_data[r][-tail:] if not math.isnan(v)]
            avg_norms.append(float(np.mean(vals)) if vals else 1e-12)
        offset = (g_idx - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset, avg_norms, width,
            label=label, color=color, edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        for bar, val in zip(bars, avg_norms):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{val:.1e}", ha="center", va="bottom", fontsize=5, rotation=45,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r+1}" for r in range(n_rounds)])
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg gradient norm")
    ax.set_title(f"Avg grad norm (last {tail} steps)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Ablation: sweep n_rounds to see how stability changes
# ---------------------------------------------------------------------------

def run_rounds_sweep(cfg) -> None:
    """
    Sweep over multiple n_rounds values.
    For each depth, compare RecursiveMAS vs Text-MAS grad-norm ratio (round-1 / round-N).
    A vanishing baseline and stable RecursiveMAS confirms Theorem 4.1.
    """
    rounds_to_test = cfg.sweep_rounds
    summary: List[Tuple[int, float, float, float, float, float, float]] = []

    for n in rounds_to_test:
        print(f"\n{'='*40}  n_rounds={n}  {'='*40}")
        cfg_n = argparse.Namespace(**vars(cfg))
        cfg_n.n_rounds = n
        results = run_experiment(cfg_n)

        tail = max(1, cfg.steps // 5)

        def ratio_and_loss(key):
            gn = results[key]["grad_norms_per_round"]
            first = [v for v in gn[0][-tail:] if not math.isnan(v)]
            last = [v for v in gn[n - 1][-tail:] if not math.isnan(v)]
            r = (np.mean(first) if first else float("nan")) / \
                ((np.mean(last) if last else 0.0) + 1e-12)
            loss = float(np.mean(results[key]["losses"][-tail:]))
            return r, loss

        r_mas, l_mas = ratio_and_loss("mas")
        r_sh,  l_sh  = ratio_and_loss("shared")
        r_txt, l_txt = ratio_and_loss("text")
        summary.append((n, r_mas, l_mas, r_sh, l_sh, r_txt, l_txt))

    print(f"\n{'='*90}")
    print("Sweep summary  (ratio = grad_norm_round1 / grad_norm_roundN)")
    print(f"  {'n_rounds':>8}  {'MAS ratio':>10}  {'MAS loss':>9}  "
          f"{'Shared ratio':>12}  {'Shared loss':>11}  "
          f"{'Text ratio':>10}  {'Text loss':>9}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*12}  {'-'*11}  {'-'*10}  {'-'*9}")
    for n, r_mas, l_mas, r_sh, l_sh, r_txt, l_txt in summary:
        def v(r):
            return "VANISH" if r < 0.1 else "EXPLODE" if r > 10 else "STABLE"
        print(f"  {n:>8}  {r_mas:>10.4f}  {l_mas:>9.4f}  "
              f"{r_sh:>12.4f}  {l_sh:>11.4f}  "
              f"{r_txt:>10.4f}  {l_txt:>9.4f}"
              f"  MAS:{v(r_mas)}  Shared:{v(r_sh)}  Text:{v(r_txt)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gradient stability experiment for RecursiveMAS outer-loop training")
    p.add_argument("--n_agents", type=int, default=3,
                   help="Number of agents in chain (mirrors sequential MAS: planner/critic/solver)")
    p.add_argument("--n_rounds", type=int, default=4,
                   help="Number of recursive forward rounds (outer loop unrolling depth)")
    p.add_argument("--hidden", type=int, default=64,
                   help="Base hidden size; per-agent dims = hidden * hidden_mults[i]")
    p.add_argument("--hidden_mults", type=int, nargs="+", default=[1, 2, 1],
                   help="Per-agent hidden-dim multipliers (length must equal n_agents). "
                        "Default [1,2,1] gives dims [64,128,64] with --hidden 64, "
                        "simulating heterogeneous agents like Qwen-1.7B/LLaMA-3B/Qwen2.5-1.5B.")
    p.add_argument("--n_experts", type=int, default=4,
                   help="Number of MoE experts in SharedRecursiveLink")
    p.add_argument("--expert_dim_divisor", type=int, default=4,
                   help="Each expert's inner dim = shared_dim // expert_dim_divisor. "
                        "Default 4: expert bottleneck is 4x narrower than shared_dim.")
    p.add_argument("--vocab", type=int, default=256,
                   help="Vocabulary size")
    p.add_argument("--seq_len", type=int, default=32,
                   help="Sequence length")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=200,
                   help="Training steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over multiple n_rounds values to compare stability")
    p.add_argument("--sweep_rounds", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8],
                   help="Round counts to sweep (used with --sweep)")
    p.add_argument("--out", type=str, default="gradient_stability.png",
                   help="Output plot path")
    return p.parse_args()


def main():
    cfg = parse_args()

    if cfg.sweep:
        run_rounds_sweep(cfg)
    else:
        results = run_experiment(cfg)
        plot_results(results, cfg, out_path=cfg.out)


if __name__ == "__main__":
    main()
