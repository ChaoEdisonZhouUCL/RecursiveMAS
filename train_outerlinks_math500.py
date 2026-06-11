#!/usr/bin/env python3
"""
Outer-loop training of RecursiveMAS-Light outer RecursiveLinks on MATH-500.
Implements §4 "Learning to Recur as a Whole" (paper Eq. 6) with pipeline parallelism.

Architecture (Sequential-Light):
    Planner  (Qwen3-1.7B,        d=2048)  ← frozen backbone + frozen inner link
       ↓  outer_12  (2048→2048)            ← trained from random init
    Critic   (LLaMA-3.2-1B,     d=2048)  ← frozen backbone + frozen inner link
       ↓  outer_23  (2048→1536)            ← trained from random init
    Solver   (Qwen2.5-Math-1.5B, d=1536)  ← frozen backbone + frozen inner link
       ↓  outer_31  (1536→2048)  ──► back to Planner (next round)  ← trained from random init

Training follows the paper's outer-loop procedure:
  1. Unroll the full recursive loop for n rounds (forward pass, full graph preserved).
  2. Apply CE loss only at the final round on the Solver's teacher-forced output vs y.
  3. Back-propagate gradients through the full n-round graph jointly, assigning credit
     to each outer link according to its global contribution (paper Eq. 6, Theorem 4.1).
  4. All LLM backbone and inner-link parameters remain frozen throughout.
  Hyperparams (paper §5, App. B.3): AdamW lr=5e-4, cosine schedule, batch_size=4.

Two training modes (--mode):
  original    — three independent CrossModelAdapters, one per rank (paper default)
  shared_roae — one SharedRecursiveLink with MoE core + Rotary Agent Encoding
                loaded on ALL ranks; gradients allreduced after each backward step
  compare     — run both sequentially on the same problems and produce a side-by-side plot

Pipeline parallelism (multi-GPU)
---------------------------------
Launch with torchrun (one rank per agent):
    torchrun --standalone --nproc_per_node=3 train_outerlinks_math500.py

  rank 0 → Planner  + outer_12  (2048→2048)
  rank 1 → Critic   + outer_23  (2048→1536)
  rank 2 → Solver   + outer_31  (1536→2048)

Backward pass uses explicit, barrier-separated P2P grad exchanges (Steps A–D per round)
so that each rank accumulates gradients for only its own outer link.
retain_graph=True is used on all backward calls except the final one so the shared
n-round computation graph is not freed prematurely.

Single-GPU fallback
-------------------
    python train_outerlinks_math500.py
All three agents load onto cuda:0 (or cpu).

Usage
-----
  python  train_outerlinks_math500.py --n_rounds 3 --steps 200 --batch_size 4 --lr 5e-4
  torchrun --standalone --nproc_per_node=3 \\
           train_outerlinks_math500.py --n_rounds 3 --steps 200 --batch_size 4 --lr 5e-4
"""

import argparse
import copy
import math
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from modeling import CrossModelAdapter, OUTER_ADAPTER_TYPE, infer_outer_adapter_type_from_state_dict
from hf_resolver import resolve_inner_adapter, snapshot_repo, task_for_inner_repo
from prompts import (
    FEEDBACK_SLOT, PLANNER_SLOT, REFINED_SLOT,
    build_math_planner_prompt,
    build_math_planner_prompt_with_feedback_slot,
    build_math_refiner_prompt_with_slot,
    build_math_solver_prompt_with_slots,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

PLANNER_REPO = "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B"
CRITIC_REPO  = "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B"
SOLVER_REPO  = "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B"
OUTER_REPO   = "RecursiveMAS/Sequential-Light-Outerlinks"

PLANNER_DIM = 2048
CRITIC_DIM  = 2048
SOLVER_DIM  = 1536

# ── DDP helpers ──────────────────────────────────────────────────────────────

def _rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def _world() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def _barrier():
    if _is_dist():
        dist.barrier()

# ── P2P communication ─────────────────────────────────────────────────────────
# Use NCCL directly for GPU-to-GPU tensor transfers (fast path).
# Gloo group kept as fallback for shape-protocol exchanges only.
_GLOO_GROUP = None  # kept for API compatibility; no longer used for data transfers

def _init_gloo_group():
    pass  # P2P now uses NCCL directly; no separate gloo group needed


# ── Shared-Link with Rotary Agent Encoding (RoAE) ────────────────────────────
# Ported from exp_gradient_stability.py; adapted for real agent hidden dims.
# One module replaces all three CrossModelAdapters; loaded on every rank and
# kept in sync via allreduce after each backward step.

import torch.nn as nn
from typing import List as _List, Tuple as _Tuple


def _build_rope_cache(max_idx: int, dim: int, device: torch.device):
    half = dim // 2
    k = torch.arange(half, dtype=torch.float32, device=device)
    theta = 1.0 / (10000.0 ** (2 * k / dim))
    idx = torch.arange(max_idx + 1, dtype=torch.float32, device=device)
    angles = torch.outer(idx, theta)
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SharedRecursiveLink(nn.Module):
    """
    Single shared cross-model adapter for all pipeline transitions.

    Architecture:
      - Per-agent in/out linear projections to/from a shared latent space.
      - RoAE: apply rotary position encoding with agent index so the shared
        MoE core can distinguish "planner→critic" from "solver→planner" etc.
      - Soft-MoE with n_experts experts dispatched by an RoAE-conditioned router.
      - ReZero gate (alpha): starts as pure skip connection, learns to blend in
        the MoE update.
      - Direct skip_proj in native hidden spaces bypasses the shared bottleneck,
        providing a gradient highway with orthogonal-initialised weights.

    Agents are 1-indexed:  1=Planner, 2=Critic, 3=Solver.
    """

    N_AGENTS = 3

    def __init__(self, hidden_dims: _List[int], shared_dim: int,
                 n_experts: int = 4, expert_dim_divisor: int = 4):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.shared_dim  = shared_dim
        self.n_experts   = n_experts
        N = len(hidden_dims)
        d = shared_dim

        # Index 0 is a dummy (agents are 1-indexed); index i → agent i.
        self.in_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(h, d) for h in hidden_dims]
        )
        self.out_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(d, h) for h in hidden_dims]
        )
        # Direct skip in native hidden space for gradient highway.
        self.skip_proj = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dims[i], hidden_dims[j], bias=False)
                for j in range(N)
            ])
            for i in range(N)
        ])
        for i in range(N):
            for j in range(N):
                nn.init.orthogonal_(self.skip_proj[i][j].weight)

        self.router   = nn.Linear(3 * d, n_experts, bias=False)
        exp_hidden = max(1, d // expert_dim_divisor)
        # Fused weight tensors: one matmul covers all experts simultaneously.
        # Shape: (n_experts, d, exp_hidden) and (n_experts, exp_hidden, d)
        self.expert_W1 = nn.Parameter(torch.empty(n_experts, d, exp_hidden))
        self.expert_b1 = nn.Parameter(torch.zeros(n_experts, exp_hidden))
        self.expert_W2 = nn.Parameter(torch.zeros(n_experts, exp_hidden, d))
        self.expert_b2 = nn.Parameter(torch.zeros(n_experts, d))
        nn.init.kaiming_uniform_(self.expert_W1, a=math.sqrt(5))

        self.act   = nn.GELU()
        self.ln_in = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(1e-3))

        self._rope_cache: dict = {}   # keyed by (device, dtype)

    def _get_rope(self, device, dtype):
        key = (device, dtype)
        if key not in self._rope_cache:
            cos, sin = _build_rope_cache(self.N_AGENTS, self.shared_dim, device)
            self._rope_cache[key] = (cos.to(dtype), sin.to(dtype))
        return self._rope_cache[key]

    def forward(self, h: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        """h: (B, T, hidden_dims[src-1]);  src/dst are 1-indexed agent indices."""
        src0, dst0 = src - 1, dst - 1
        skip = self.skip_proj[src0][dst0](h)

        x = self.in_proj[src](h)
        cos_tab, sin_tab = self._get_rope(x.device, x.dtype)
        x_src = _rope_rotate(x, cos_tab[src], sin_tab[src])
        x_dst = _rope_rotate(x, cos_tab[dst], sin_tab[dst])

        gate_input = torch.cat([x_src, x_dst, x_dst - x_src], dim=-1)
        logits = self.router(gate_input.mean(dim=1))           # (B, n_experts)
        w = torch.softmax(logits / 2.0, dim=-1).unsqueeze(1)  # (B, 1, n_experts)

        normed = self.ln_in(x_src)           # (B, T, d)
        W1 = self.expert_W1.to(normed.dtype)  # (K, d, e)
        b1 = self.expert_b1.to(normed.dtype)  # (K, e)
        W2 = self.expert_W2.to(normed.dtype)  # (K, e, d)
        b2 = self.expert_b2.to(normed.dtype)  # (K, d)
        # h1: (B, T, K, e)  — all experts in one batched matmul
        h1 = self.act(torch.einsum("btd,kde->btke", normed, W1) + b1)
        h2 = torch.einsum("btke,ked->btd", h1 * w.unsqueeze(-1), W2) + (w @ b2).squeeze(1).unsqueeze(1)
        return skip + self.alpha * self.out_proj[dst](h2)


# ── Pipeline layout ──────────────────────────────────────────────────────────
# With world_size >= 3: rank 0 = Planner, rank 1 = Critic, rank 2 = Solver.
# With world_size < 3:  all agents on rank 0 (single-GPU path).

RANK_PLANNER = 0
RANK_CRITIC  = 1
RANK_SOLVER  = 2

def _pipeline_rank(agent: str) -> int:
    """Return the rank that owns a given agent (falls back to 0 if not enough GPUs)."""
    mapping = {"planner": RANK_PLANNER, "critic": RANK_CRITIC, "solver": RANK_SOLVER}
    r = mapping[agent]
    return r if r < _world() else 0

def _owns(agent: str) -> bool:
    return _rank() == _pipeline_rank(agent)

# ── Cross-rank tensor transfer ───────────────────────────────────────────────
#
# For the forward pass we just copy data between ranks.  The receiving rank
# creates a fresh leaf tensor with requires_grad=True so its local graph stays
# differentiable.
#
# For the backward pass we use EXPLICIT, barrier-separated exchanges (see
# pipeline_backward() in run_training).  Trying to piggy-back grad exchange
# onto autograd hooks or custom Functions fails in practice because the graphs
# on different ranks are disconnected: loss.backward() on rank 2 never triggers
# backward on rank 0 or 1.

def _send_tensor(t: torch.Tensor, dst: int):
    """Send tensor to dst via NCCL (GPU direct, no CPU round-trip)."""
    data = t.detach().contiguous()
    dist.send(data, dst=dst)


def _recv_tensor(src: int, device: torch.device, dtype: torch.dtype,
                 requires_grad: bool = False, shape: Optional[tuple] = None) -> torch.Tensor:
    """Receive tensor from src via NCCL. shape must be known by the receiver."""
    assert shape is not None, "_recv_tensor requires explicit shape with NCCL"
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src)
    if requires_grad:
        buf = buf.requires_grad_(True)
    return buf


# ── Model loading helpers ────────────────────────────────────────────────────

def load_model_and_tokenizer(repo_id: str, device: torch.device, dtype: torch.dtype):
    path = snapshot_repo(repo_id)
    tok  = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, token=HF_TOKEN)
    mdl  = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=dtype, device_map=str(device),
        trust_remote_code=True, token=HF_TOKEN,
    )
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return mdl, tok


def load_inner_adapter(repo_id: str, hidden: int, device: torch.device, dtype: torch.dtype):
    from modeling import Adapter, INNER_ADAPTER_TYPE
    repo_dir = snapshot_repo(repo_id)
    task     = task_for_inner_repo("math500")          # always "math" for this dataset
    pt_path  = resolve_inner_adapter(repo_dir, task)
    sd = torch.load(str(pt_path), map_location="cpu")
    a  = Adapter(hidden_size=hidden, adapter_type=INNER_ADAPTER_TYPE)
    a.load_state_dict(sd, strict=True)
    return a.to(device=device, dtype=dtype).eval()



# ── Dataset ──────────────────────────────────────────────────────────────────

def load_math500(n_samples: int = 0, seed: int = 42):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=HF_TOKEN)
    if n_samples > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [(row["problem"], row["answer"]) for row in ds]


def load_s1k(n_samples: int = 0, seed: int = 42):
    """s1K: reasoning-trace dataset — fields `question` (prompt) and `solution` (target)."""
    ds = load_dataset("simplescaling/s1K", split="train", token=HF_TOKEN)
    if n_samples > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [(row["question"], row["solution"]) for row in ds]


def load_m1k(n_samples: int = 0, seed: int = 42):
    """m1k-tokenized: medical-QA reasoning dataset — fields `prompt` and `distilled_answer_string`."""
    ds = load_dataset("UCSC-VLAA/m1k-tokenized", split="train", token=HF_TOKEN)
    if n_samples > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [(row["prompt"], row["distilled_answer_string"]) for row in ds]


DATASET_LOADERS = {
    "math500": load_math500,
    "s1k":     load_s1k,
    "m1k":     load_m1k,
}


def load_training_problems(name: str, n_samples: int = 0, seed: int = 42):
    """
    Dispatch to the right loader by name. `name` may be a single dataset
    ("s1k", "m1k", "math500") or a "+"-joined combination ("s1k+m1k"), in
    which case samples are pooled and the combined pool is shuffled.
    """
    keys = [k.strip().lower() for k in name.split("+") if k.strip()]
    pools = []
    for k in keys:
        if k not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset '{k}'. Choices: {sorted(DATASET_LOADERS)}")
        pools.append(DATASET_LOADERS[k](n_samples=0, seed=seed))   # load full pool, truncate after merge

    combined = [p for pool in pools for p in pool]
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    if n_samples > 0:
        combined = combined[:n_samples]
    return combined


def truncate_problems_by_tokens(problems, max_seq_len: int, chars_per_token: float = 4.0):
    """
    Drop / trim (question, answer) pairs so the combined length stays within
    `max_seq_len` tokens. Uses a character-count proxy (≈ chars_per_token chars
    per token) so the result is identical on every rank regardless of which
    agent tokenizer it happens to have loaded — exact tokenizer-based lengths
    would differ slightly per agent and could desync the shared problem list.
    Long answers are truncated; questions are kept whole (truncating the
    prompt would break the agent prompt templates).
    """
    budget_chars = int(max_seq_len * chars_per_token)
    out = []
    for q, a in problems:
        if len(q) >= budget_chars:
            continue   # question alone exceeds budget — skip
        out.append((q, a[:budget_chars - len(q)]))
    return out


def build_planner_prompt(question: str) -> Tuple[str, str]:
    """Round 1 planner: no feedback slot — returns ("full prompt", "")."""
    return build_math_planner_prompt(question), ""


def build_planner_feedback_prompt(question: str) -> Tuple[str, str]:
    """Rounds 2+: splits on FEEDBACK_SLOT so the latent is injected there."""
    full = build_math_planner_prompt_with_feedback_slot(question)
    pre, post = full.split(FEEDBACK_SLOT, 1)
    return pre, post


def build_critic_prompt(question: str) -> Tuple[str, str]:
    """Splits on PLANNER_SLOT so the planner latent is injected there."""
    full = build_math_refiner_prompt_with_slot(question)
    pre, post = full.split(PLANNER_SLOT, 1)
    return pre, post


def build_solver_prompt(question: str) -> Tuple[str, str]:
    """Splits on REFINED_SLOT so the refiner latent is injected there."""
    full = build_math_solver_prompt_with_slots(question)
    pre, post = full.split(REFINED_SLOT, 1)
    return pre, post


def _build_input_embeds(tokenizer, embed_fn, pre: str, post: str,
                        latent: Optional[torch.Tensor],
                        device: torch.device, dtype: torch.dtype):
    """Single-sample wrapper — delegates to the batched version."""
    ie, am = _build_input_embeds_batch(
        tokenizer, embed_fn, [pre], [post],
        latent,   # (1, L, d) or None
        device, dtype,
    )
    return ie, am


def _build_input_embeds_batch(
    tokenizer, embed_fn,
    pre_list: List[str], post_list: List[str],
    latent: Optional[torch.Tensor],   # (B, L_lat, d) or None
    device: torch.device, dtype: torch.dtype,
):
    """
    Batched version: tokenize B (pre, post) pairs, inject the shared latent block
    (same L_lat for every sample — it comes from a fixed latent_steps rollout),
    and left-pad all sequences to the same length.

    Layout per sample:  [PAD...] [pre tokens] [latent] [post tokens]
    Padding is on the left so that the last real token is always at position -1,
    which matches how decoder-only models handle variable-length prefixes.

    Returns:
        ie : (B, T_max, d)   — input embeddings, left-padded with zeros
        am : (B, T_max)      — attention mask, 0 for pad positions
    """
    B = len(pre_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Tokenize all pre/post strings (no batched call — they may have different lengths).
    pieces = []   # list of (emb_pre, am_pre, emb_post, am_post) per sample
    d_model = None
    for pre, post in zip(pre_list, post_list):
        enc_pre = tokenizer(pre, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            emb_pre = embed_fn(enc_pre["input_ids"]).to(dtype)   # (1, T_pre, d)
        if d_model is None:
            d_model = emb_pre.size(-1)

        if post:
            enc_post = tokenizer(post, return_tensors="pt",
                                 add_special_tokens=False).to(device)
            with torch.no_grad():
                emb_post = embed_fn(enc_post["input_ids"]).to(dtype)
            am_post = enc_post["attention_mask"]
        else:
            emb_post = emb_pre.new_zeros((1, 0, d_model))
            am_post  = enc_pre["attention_mask"].new_zeros((1, 0))

        pieces.append((emb_pre, enc_pre["attention_mask"], emb_post, am_post))

    L_lat = latent.size(1) if latent is not None else 0

    # Build unpadded sequences and find max length.
    seqs_emb = []
    seqs_am  = []
    for i, (emb_pre, am_pre, emb_post, am_post) in enumerate(pieces):
        ones_lat = am_pre.new_ones((1, L_lat)) if L_lat > 0 else am_pre.new_zeros((1, 0))
        if latent is not None:
            lat_i = latent[i:i+1].to(dtype)   # (1, L_lat, d)
            emb = torch.cat([emb_pre, lat_i, emb_post], dim=1)
            am  = torch.cat([am_pre,  ones_lat, am_post],  dim=1)
        else:
            emb = torch.cat([emb_pre, emb_post], dim=1)
            am  = torch.cat([am_pre,  am_post],  dim=1)
        seqs_emb.append(emb[0])   # (T_i, d)
        seqs_am.append(am[0])     # (T_i,)

    T_max = max(s.size(0) for s in seqs_emb)

    # Left-pad to T_max with F.pad + stack (differentiable) — in-place index
    # assignment into a fresh torch.zeros tensor (ie_batch[i, ...] = emb) would
    # sever the autograd graph between `latent` (the outer-link output) and the
    # returned ie_batch, since the destination tensor never requires_grad.
    padded_emb = []
    padded_am  = []
    for emb, am in zip(seqs_emb, seqs_am):
        T = emb.size(0)
        pad_len = T_max - T
        padded_emb.append(F.pad(emb, (0, 0, pad_len, 0)))   # pad along seq dim (left)
        padded_am.append(F.pad(am, (pad_len, 0)))

    ie_batch = torch.stack(padded_emb, dim=0)
    am_batch = torch.stack(padded_am, dim=0).long()

    return ie_batch, am_batch


# ── Latent rollout ────────────────────────────────────────────────────────────

def _detach_past_key_values(pkv):
    """Detach all tensors in a past_key_values structure (tuple-of-tuples or cache object)."""
    if pkv is None:
        return None
    if isinstance(pkv, tuple):
        return tuple(
            tuple(t.detach() for t in layer) if isinstance(layer, tuple) else layer.detach()
            for layer in pkv
        )
    # HuggingFace DynamicCache or similar object
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        for i in range(len(pkv.key_cache)):
            pkv.key_cache[i] = pkv.key_cache[i].detach()
            pkv.value_cache[i] = pkv.value_cache[i].detach()
        return pkv
    return pkv


def _get_last_layer(model) -> torch.nn.Module:
    """Return the final transformer layer so we can hook it instead of collecting all hidden states."""
    # Works for LlamaModel, Qwen2Model, and most HF decoder models.
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return layers[-1]
    # Fallback: return the norm — hook will fire after last layer.
    return getattr(inner, "norm", model)


def latent_rollout(
    model,
    inner_adapter,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    dtype: torch.dtype,
    agent_idx: int = 0,
    shared_link: Optional["SharedRecursiveLink"] = None,
) -> Tuple[torch.Tensor, object, torch.Tensor]:
    """
    Run `latent_steps` steps of inner-adapter auto-regression in latent space.

    Returns:
        hidden_states : (1, latent_steps, d_model)  — in autograd graph; used by outer link
                        and by lm_head for CE loss (paper Fig. 4, §4 Eq. 6).
        past_key_values                              — detached KV cache (unused by callers;
                        kept for API symmetry in case callers need it).
        grown_am      : (1, prompt_len+latent_steps) — attention mask after rollout.

    When shared_link is provided (shared_roae mode), the inner self-loop uses
    SharedRecursiveLink(h, src=agent_idx, dst=agent_idx) instead of inner_adapter.
    agent_idx is 1-indexed (1=Planner, 2=Critic, 3=Solver).
    """
    # Step 0's model forward keeps its autograd graph alive, so gradient can
    # flow from hidden_states[0] all the way back through `model` to
    # `input_embeds` — and from there to the previous agent's outer_link
    # output (critic_prefix / solver_prefix / feedback_prefix). Steps 1..47
    # run under no_grad with their last hidden state detached and re-enabled
    # as a leaf: their inputs come from inner_adapter(last_h.detach()), so
    # they are graph-independent from step 0 and from each other — only
    # outer_link(hidden_states) → loss needs gradient through these leaves.
    # This keeps at most one full model forward graph alive at a time instead
    # of `latent_steps` of them, while still letting gradient reach the
    # cross-agent outer-link adapters via step 0.
    _last_h: list = []

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out   # (B, T, d)
        _last_h.append(h[:, -1, :])

    last_layer = _get_last_layer(model)
    handle = last_layer.register_forward_hook(_hook)

    hidden_states = []
    past_key_values = None
    ie = input_embeds
    am = attention_mask
    try:
        for step in range(latent_steps):
            _last_h.clear()
            grad_ctx = torch.enable_grad() if step == 0 else torch.no_grad()
            with grad_ctx:
                try:
                    out = model(inputs_embeds=ie, attention_mask=am,
                                use_cache=True, past_key_values=past_key_values,
                                return_dict=True, logits_to_keep=1)
                except TypeError:
                    out = model(inputs_embeds=ie, attention_mask=am,
                                use_cache=True, past_key_values=past_key_values,
                                return_dict=True)

            if step == 0:
                last_h = _last_h[0]                      # (B, d) — keeps grad_fn back to ie
            else:
                last_h = _last_h[0].detach().requires_grad_(True)  # (B, d) — fresh leaf
            hidden_states.append(last_h.unsqueeze(1))

            # Detach cache and drop the previous step's cache immediately.
            prev_pkv = past_key_values
            past_key_values = _detach_past_key_values(out.past_key_values)
            del prev_pkv, out
            am = torch.cat([am, am.new_ones((am.size(0), 1))], dim=1)
            with torch.no_grad():
                if shared_link is not None:
                    ie = shared_link(last_h.detach().unsqueeze(1), src=agent_idx, dst=agent_idx).to(dtype)
                else:
                    ie = inner_adapter(last_h.detach()).unsqueeze(1).to(dtype)
    finally:
        handle.remove()

    # Free the final KV cache — callers never use it.
    del past_key_values
    # am has grown by latent_steps; return it so callers can build the right teacher-forcing mask.
    return torch.cat(hidden_states, dim=1), None, am


# ── Gradient monitor ─────────────────────────────────────────────────────────

class GradMonitor:
    """
    Records the incoming gradient norm at outer_12's output for each round.
    outer_12's output is the earliest in-graph tensor; its gradient measures
    how much of the final-round loss signal reaches round r's Planner stage.
    """

    def __init__(self, n_rounds: int):
        self.n_rounds  = n_rounds
        self.norms: Dict[int, List[float]] = {r: [] for r in range(n_rounds)}
        self._handles  = []
        self._step_buf: Dict[int, List[float]] = {r: [] for r in range(n_rounds)}

    def register_output(self, tensor: torch.Tensor, round_idx: int):
        if not tensor.requires_grad:
            raise RuntimeError(
                f"GradMonitor: tensor for round {round_idx} has requires_grad=False."
            )

        def _hook(grad):
            if grad is not None:
                self._step_buf[round_idx].append(grad.detach().float().norm().item())

        self._handles.append(tensor.register_hook(_hook))

    def commit_step(self):
        for r in range(self.n_rounds):
            vals = self._step_buf[r]
            self.norms[r].append(float(np.mean(vals)) if vals else float("nan"))
            self._step_buf[r] = []
        for h in self._handles:
            h.remove()
        self._handles = []



def _sync_norms_all_ranks(monitor: Optional["GradMonitor"], n_rounds: int):
    """Broadcast per-round grad norms from rank_planner to all ranks.
    Must be called by every rank every step so the broadcast collective is balanced."""
    for r in range(n_rounds):
        val = torch.tensor(
            [monitor.norms[r][-1] if (monitor is not None and monitor.norms[r]) else float("nan")],
            dtype=torch.float32, device=f"cuda:{_rank()}"
        )
        dist.broadcast(val, src=RANK_PLANNER)
        if monitor is not None and _rank() != RANK_PLANNER:
            monitor.norms[r].append(val.item())


# ── Single round (pipeline-aware) ────────────────────────────────────────────

def forward_one_round(
    planner_mdl, planner_tok, planner_inner,
    critic_mdl,  critic_tok,  critic_inner,
    solver_mdl,  solver_tok,  solver_inner,
    outer_12: Optional[CrossModelAdapter],
    outer_23: Optional[CrossModelAdapter],
    outer_31: Optional[CrossModelAdapter],
    questions: List[str],
    answers: List[str],
    feedback_prefix: Optional[torch.Tensor],   # (B, L, PLANNER_DIM) or None
    latent_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    monitor: Optional[GradMonitor] = None,
    round_idx: int = 0,
    shared_link: Optional["SharedRecursiveLink"] = None,
    debug: bool = False,
):
    """
    One Planner→Critic→Solver forward round (paper §3-4, Fig.1).

    Processes a batch of B samples jointly — all samples are left-padded to the
    same prompt length within each agent stage, so a single model forward pass
    covers the full batch.

    feedback_prefix is (B, latent_steps, PLANNER_DIM) on rounds 2+, None on round 1.

    Returns dict of tensors needed for backward, keyed by role.
    """
    rank     = _rank()
    rp       = _pipeline_rank("planner")
    rc       = _pipeline_rank("critic")
    rs       = _pipeline_rank("solver")
    pipeline = _is_dist() and (rp != rc or rc != rs)
    B        = len(questions)

    out = {}

    # ── Stage 1: Planner computes critic_prefix, sends to Critic ─────────────
    if rank == rp:
        if feedback_prefix is None:
            pre_list  = [build_planner_prompt(q)[0] for q in questions]
            post_list = [build_planner_prompt(q)[1] for q in questions]
        else:
            pre_list  = [build_planner_feedback_prompt(q)[0] for q in questions]
            post_list = [build_planner_feedback_prompt(q)[1] for q in questions]
        ie, am = _build_input_embeds_batch(
            planner_tok, planner_mdl.get_input_embeddings(),
            pre_list, post_list, feedback_prefix, device, dtype,
        )
        if debug:
            print(f"[dbg fwd r{round_idx} Planner] theoretical ie=(B={B}, T=?, d={PLANNER_DIM})  "
                  f"practical ie={tuple(ie.shape)}  am={tuple(am.shape)}  "
                  f"feedback_prefix={'None' if feedback_prefix is None else tuple(feedback_prefix.shape)}")
        planner_h, _, _ = latent_rollout(planner_mdl, planner_inner, ie, am, latent_steps, dtype,
                                         agent_idx=1, shared_link=shared_link)
        if debug:
            print(f"[dbg fwd r{round_idx} Planner] theoretical planner_h=(B={B}, latent_steps={latent_steps}, d={PLANNER_DIM})  "
                  f"practical planner_h={tuple(planner_h.shape)}  requires_grad={planner_h.requires_grad}")
        if shared_link is not None:
            critic_prefix = shared_link(planner_h, src=1, dst=2)
        else:
            critic_prefix = outer_12(planner_h)
        if monitor is not None:
            monitor.register_output(critic_prefix, round_idx)
        if critic_prefix.requires_grad:
            critic_prefix.retain_grad()
        if debug:
            print(f"[dbg fwd r{round_idx} Planner] theoretical critic_prefix=(B={B}, latent_steps={latent_steps}, d={CRITIC_DIM})  "
                  f"practical critic_prefix={tuple(critic_prefix.shape)}  requires_grad={critic_prefix.requires_grad}")
        out["critic_prefix"] = critic_prefix
        if pipeline:
            _send_tensor(critic_prefix, dst=rc)

    if pipeline and rp != rc and rank == rc:
        cp_recv = _recv_tensor(src=rp, device=device, dtype=dtype, requires_grad=True,
                               shape=(B, latent_steps, CRITIC_DIM))
        if cp_recv.requires_grad:
            cp_recv.retain_grad()
        out["cp_recv"] = cp_recv
        critic_prefix  = cp_recv

    # ── Stage 2: Critic computes solver_prefix, sends to Solver ──────────────
    if rank == rc:
        pre_c_list  = [build_critic_prompt(q)[0] for q in questions]
        post_c_list = [build_critic_prompt(q)[1] for q in questions]
        critic_input, critic_am = _build_input_embeds_batch(
            critic_tok, critic_mdl.get_input_embeddings(),
            pre_c_list, post_c_list, critic_prefix, device, dtype,
        )
        if debug:
            print(f"[dbg fwd r{round_idx} Critic] theoretical critic_input=(B={B}, T=?, d={CRITIC_DIM})  "
                  f"practical critic_input={tuple(critic_input.shape)}  am={tuple(critic_am.shape)}  "
                  f"critic_prefix={tuple(critic_prefix.shape)}  requires_grad={critic_prefix.requires_grad}")
        critic_h, _, _ = latent_rollout(critic_mdl, critic_inner, critic_input, critic_am,
                                        latent_steps, dtype, agent_idx=2, shared_link=shared_link)
        if shared_link is not None:
            solver_prefix = shared_link(critic_h, src=2, dst=3)
        else:
            solver_prefix = outer_23(critic_h)
        if solver_prefix.requires_grad:
            solver_prefix.retain_grad()
        if debug:
            print(f"[dbg fwd r{round_idx} Critic] theoretical solver_prefix=(B={B}, latent_steps={latent_steps}, d={SOLVER_DIM})  "
                  f"practical solver_prefix={tuple(solver_prefix.shape)}  requires_grad={solver_prefix.requires_grad}")
        out["solver_prefix_rc"] = solver_prefix
        if pipeline and rc != rs:
            _send_tensor(solver_prefix, dst=rs)

    if pipeline and rc != rs and rank == rs:
        sp_recv = _recv_tensor(src=rc, device=device, dtype=dtype, requires_grad=True,
                               shape=(B, latent_steps, SOLVER_DIM))
        if sp_recv.requires_grad:
            sp_recv.retain_grad()
        out["sp_recv"] = sp_recv
        solver_prefix  = sp_recv

    # ── Stage 3: Solver runs, produces feedback and logits ───────────────────
    if rank == rs:
        pre_s_list  = [build_solver_prompt(q)[0] for q in questions]
        post_s_list = [build_solver_prompt(q)[1] for q in questions]
        solver_input, solver_am = _build_input_embeds_batch(
            solver_tok, solver_mdl.get_input_embeddings(),
            pre_s_list, post_s_list, solver_prefix, device, dtype,
        )
        if debug:
            print(f"[dbg fwd r{round_idx} Solver] theoretical solver_input=(B={B}, T=?, d={SOLVER_DIM})  "
                  f"practical solver_input={tuple(solver_input.shape)}  am={tuple(solver_am.shape)}  "
                  f"solver_prefix={tuple(solver_prefix.shape)}  requires_grad={solver_prefix.requires_grad}")
        solver_h, _, _ = latent_rollout(
            solver_mdl, solver_inner, solver_input, solver_am, latent_steps, dtype,
            agent_idx=3, shared_link=shared_link,
        )
        if shared_link is not None:
            feedback = shared_link(solver_h, src=3, dst=1)
        else:
            feedback = outer_31(solver_h)
        if feedback.requires_grad:
            feedback.retain_grad()
        if debug:
            print(f"[dbg fwd r{round_idx} Solver] theoretical solver_h=(B={B}, latent_steps={latent_steps}, d={SOLVER_DIM})  "
                  f"practical solver_h={tuple(solver_h.shape)}  "
                  f"feedback theoretical=(B={B}, latent_steps={latent_steps}, d={PLANNER_DIM})  "
                  f"practical feedback={tuple(feedback.shape)}  requires_grad={feedback.requires_grad}")

        # CE loss: the LAST A predicted positions align with the A answer tokens
        # (solver_h[:, -A:, :] predicts answer token y[-A:]) — paper Eq.6.
        # Compute per-sample CE and average over the batch.
        final_norm    = getattr(getattr(solver_mdl, "model", solver_mdl), "norm", None)
        normed_h      = final_norm(solver_h.to(dtype)) if final_norm is not None else solver_h.to(dtype)
        solver_logits = solver_mdl.lm_head(normed_h)   # (B, latent_steps, vocab)
        if debug:
            print(f"[dbg fwd r{round_idx} Solver] theoretical solver_logits=(B={B}, latent_steps={latent_steps}, vocab=?)  "
                  f"practical solver_logits={tuple(solver_logits.shape)}")

        loss_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        for i, ans in enumerate(answers):
            ans_ids = solver_tok(ans, return_tensors="pt",
                                 add_special_tokens=False).to(device)["input_ids"]
            L = solver_logits.size(1)
            A = ans_ids.size(1)
            n = min(L, A)
            if debug:
                print(f"[dbg fwd r{round_idx} Solver] sample {i}: theoretical loss window n=min(L={L}, A={A})={n}  "
                      f"logits slice=[{L - n}:{L}]  answer slice=[{A - n}:{A}]  "
                      f"practical logits_slice_shape={tuple(solver_logits[i, L - n:].shape)}  "
                      f"answer_slice_shape={tuple(ans_ids[0, A - n:].shape)}")
            loss_sum = loss_sum + F.cross_entropy(
                solver_logits[i, L - n:].float(),
                ans_ids[0, A - n:],
            )
        solver_loss = loss_sum / B
        if debug:
            print(f"[dbg fwd r{round_idx} Solver] theoretical solver_loss=scalar mean-over-B(B={B})  "
                  f"practical solver_loss={solver_loss.item():.4f}  requires_grad={solver_loss.requires_grad}")

        out["feedback"]     = feedback
        out["solver_logits"] = solver_logits
        out["solver_loss"]   = solver_loss

    return out


# ── Pretrained inference probe ───────────────────────────────────────────────

@torch.no_grad()
def pretrained_inference_probe(
    question: str,
    answer: str,
    planner_mdl, planner_tok,
    critic_mdl,  critic_tok,
    solver_mdl,  solver_tok,
    planner_inner, critic_inner, solver_inner,
    outer_dir: Path,
    latent_steps: int,
    n_rounds: int,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    probe_log_file: Optional[str] = None,
):
    """
    Run the probe question through the pretrained (frozen, unmodified) RecursiveMAS
    for n_rounds recursive iterations, using proper chat-template prompts and
    model.generate() on the final round.

    Each round: Planner(+feedback) → outer_12 → Critic → outer_23 → Solver → outer_31 → feedback
    The feedback latent (outer_31 output) is fed back into the Planner on rounds 2+.
    On the final round the Solver calls generate() instead of producing a feedback latent.

    Works in both single-GPU and pipeline-parallel (3-GPU) mode via NCCL P2P.
    """
    import sys as _sys
    import argparse as _argparse
    _sys.path.insert(0, str(THIS_DIR))
    from inference_utils.inference_mas import (
        autoregressive_latent_rollout,
        render_chat_prompt_ids,
        split_prompt_ids_by_slots,
        token_ids_to_embeds,
        pad_left_ids,
        build_generation_kwargs,
    )

    def _load_pretrained_outer(path: str, in_dim: int, out_dim: int) -> CrossModelAdapter:
        sd = torch.load(path, map_location=device)
        adapter_type = infer_outer_adapter_type_from_state_dict(sd)
        adapter = CrossModelAdapter(in_dim=in_dim, out_dim=out_dim,
                                    adapter_type=adapter_type).to(device=device, dtype=dtype)
        adapter.load_state_dict(sd)
        return adapter.eval()

    rank = _rank()
    rp   = _pipeline_rank("planner")
    rc   = _pipeline_rank("critic")
    rs   = _pipeline_rank("solver")
    pipeline = _is_dist() and (rp != rc or rc != rs)

    # Load adapters and inner models once (only on the rank that needs them).
    if rank == rp:
        _p_inner = planner_inner if planner_inner is not None else \
            load_inner_adapter(PLANNER_REPO, PLANNER_DIM, device, dtype)
        _outer_12 = _load_pretrained_outer(
            str(outer_dir / "Planner-Critic-Outerlink(math).pt"),
            in_dim=PLANNER_DIM, out_dim=CRITIC_DIM,
        )
        _embed_p = planner_mdl.get_input_embeddings()
        feedback_prefix: Optional[torch.Tensor] = None   # None on round 1

    if rank == rc:
        _c_inner  = critic_inner if critic_inner is not None else \
            load_inner_adapter(CRITIC_REPO, CRITIC_DIM, device, dtype)
        _outer_23 = _load_pretrained_outer(
            str(outer_dir / "Critic-Solver-Outerlink(math).pt"),
            in_dim=CRITIC_DIM, out_dim=SOLVER_DIM,
        )
        _embed_c      = critic_mdl.get_input_embeddings()
        critic_prompt = build_math_refiner_prompt_with_slot(question)
        critic_segs   = split_prompt_ids_by_slots(critic_tok, critic_prompt, [PLANNER_SLOT],
                                                   enable_thinking=False)
        _pre_c  = token_ids_to_embeds(_embed_c, critic_segs[0], device, dtype)
        _post_c = token_ids_to_embeds(_embed_c, critic_segs[1], device, dtype)

    if rank == rs:
        _s_inner  = solver_inner if solver_inner is not None else \
            load_inner_adapter(SOLVER_REPO, SOLVER_DIM, device, dtype)
        _outer_31 = _load_pretrained_outer(
            str(outer_dir / "Solver-Planner-Outerlink(math).pt"),
            in_dim=SOLVER_DIM, out_dim=PLANNER_DIM,
        )
        _embed_s      = solver_mdl.get_input_embeddings()
        _dummy_args   = _argparse.Namespace(mas_shape="chain", solver_pre_question=0,
                                            enable_thinking=0, ans=False)
        solver_prompt = build_math_solver_prompt_with_slots(question, _dummy_args, mas_shape="chain")
        solver_segs   = split_prompt_ids_by_slots(solver_tok, solver_prompt, [REFINED_SLOT],
                                                   enable_thinking=False)
        _pre_s  = token_ids_to_embeds(_embed_s, solver_segs[0], device, dtype)
        _post_s = token_ids_to_embeds(_embed_s, solver_segs[1], device, dtype)

    # ── n_rounds recursive loop ───────────────────────────────────────────────
    for r in range(n_rounds):
        is_last = (r == n_rounds - 1)

        # Stage 1: Planner → outer_12 → send lat_12
        if rank == rp:
            if feedback_prefix is None:
                planner_ids = render_chat_prompt_ids(planner_tok,
                                                     build_math_planner_prompt(question),
                                                     enable_thinking=False)
                p_ids_t, p_am = pad_left_ids(
                    [planner_ids],
                    pad_id=planner_tok.pad_token_id or planner_tok.eos_token_id,
                    device=device,
                )
                p_embeds = _embed_p(p_ids_t).to(dtype)
            else:
                full_prompt = build_math_planner_prompt_with_feedback_slot(question)
                pre_fb, post_fb = full_prompt.split(FEEDBACK_SLOT, 1)
                p_embeds, p_am = _build_input_embeds(
                    planner_tok, _embed_p, pre_fb, post_fb, feedback_prefix, device, dtype,
                )
            planner_h = autoregressive_latent_rollout(planner_mdl, _p_inner, p_embeds, p_am,
                                                      latent_steps)
            lat_12 = _outer_12(planner_h).to(dtype)
            if pipeline:
                _send_tensor(lat_12, dst=rc)

        # Stage 2: Critic → outer_23 → send lat_23
        if rank == rc:
            if pipeline and rp != rc:
                lat_12 = _recv_tensor(src=rp, device=device, dtype=dtype,
                                      shape=(1, latent_steps, CRITIC_DIM))
            c_embeds = torch.cat([_pre_c, lat_12[0], _post_c], dim=0).unsqueeze(0)
            c_am     = torch.ones((1, c_embeds.size(1)), dtype=torch.long, device=device)
            critic_h = autoregressive_latent_rollout(critic_mdl, _c_inner, c_embeds, c_am,
                                                     latent_steps)
            lat_23   = _outer_23(critic_h).to(dtype)
            if pipeline and rc != rs:
                _send_tensor(lat_23, dst=rs)

        # Stage 3: Solver
        #   non-final round → latent rollout → outer_31 → feedback to Planner
        #   final round     → generate() text output
        if rank == rs:
            if pipeline and rc != rs:
                lat_23 = _recv_tensor(src=rc, device=device, dtype=dtype,
                                      shape=(1, latent_steps, SOLVER_DIM))
            s_embeds = torch.cat([_pre_s, lat_23[0], _post_s], dim=0).unsqueeze(0)
            s_am     = torch.ones((1, s_embeds.size(1)), dtype=torch.long, device=device)

            if is_last:
                gen_kwargs = build_generation_kwargs(solver_tok, max_new_tokens=max_new_tokens,
                                                     do_sample=False, temperature=1.0, top_p=1.0)
                generated = solver_mdl.generate(inputs_embeds=s_embeds, attention_mask=s_am,
                                                **gen_kwargs)
                sequences = generated.sequences if hasattr(generated, "sequences") else generated
                # inputs_embeds mode: generate() returns only new tokens, no prompt prefix.
                pretrained_output = solver_tok.decode(sequences[0], skip_special_tokens=True).strip()
            else:
                solver_h  = autoregressive_latent_rollout(solver_mdl, _s_inner, s_embeds, s_am,
                                                          latent_steps)
                feedback_out = _outer_31(solver_h).to(dtype)   # (1, latent_steps, PLANNER_DIM)
                if pipeline and rs != rp:
                    _send_tensor(feedback_out, dst=rp)
                else:
                    _feedback_nonpipeline = feedback_out

        # Inter-round: feedback from rs → rp
        if not is_last:
            if rank == rp:
                if pipeline and rs != rp:
                    feedback_prefix = _recv_tensor(src=rs, device=device, dtype=dtype,
                                                   shape=(1, latent_steps, PLANNER_DIM))
                else:
                    feedback_prefix = _feedback_nonpipeline  # noqa: F821

    # ── Write result (rank rs only) ──────────────────────────────────────────
    if rank == rs:
        lines = [
            f"\n{'=' * 64}",
            f"Pretrained RecursiveMAS baseline ({n_rounds}-round, greedy)",
            f"Q : {question}",
            f"GT: {answer}",
            f"→   {pretrained_output}",
            f"{'=' * 64}",
        ]
        for ln in lines:
            print(ln)
        if probe_log_file is not None:
            with open(probe_log_file, "a") as f:
                f.write("\n".join(lines) + "\n")

    # ── Release all probe-local GPU tensors before returning ─────────────────
    # These adapters and embeddings are only needed for the baseline probe;
    # leaving them alive causes OOM when the training forward pass starts.
    import gc as _gc
    if rank == rp:
        del _outer_12, _p_inner, _embed_p, feedback_prefix
    if rank == rc:
        del _outer_23, _c_inner, _embed_c, _pre_c, _post_c
    if rank == rs:
        del _outer_31, _s_inner, _embed_s, _pre_s, _post_s
    _gc.collect()
    torch.cuda.empty_cache()
    _barrier()  # ensure all ranks finish cleanup before training begins


# ── Eval probe ───────────────────────────────────────────────────────────────

def eval_probe(
    step: int,
    question: str,
    answer: str,
    planner_mdl, planner_tok, planner_inner,
    critic_mdl,  critic_tok,  critic_inner,
    solver_mdl,  solver_tok,  solver_inner,
    outer_12, outer_23, outer_31,
    shared_link,
    latent_steps: int,
    n_rounds: int,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    probe_log_file: Optional[str] = None,
):
    """
    Run the fixed probe example through all n_rounds (no grad), greedily
    decode from the solver's final hidden state, then print and optionally
    append the result to probe_log_file (written by rank==rs only).
    All ranks participate (pipeline collectives inside).
    """
    rank = _rank()
    rp   = _pipeline_rank("planner")
    rc   = _pipeline_rank("critic")
    rs   = _pipeline_rank("solver")

    with torch.no_grad():
        feedback_prefix = None
        for r in range(n_rounds):
            out = forward_one_round(
                planner_mdl, planner_tok, planner_inner,
                critic_mdl,  critic_tok,  critic_inner,
                solver_mdl,  solver_tok,  solver_inner,
                outer_12, outer_23, outer_31,
                questions=[question],
                answers=[answer],
                feedback_prefix=feedback_prefix,
                latent_steps=latent_steps,
                device=device,
                dtype=dtype,
                monitor=None,
                round_idx=r,
                shared_link=shared_link,
            )
            if r < n_rounds - 1:
                pipeline = _is_dist() and (rp != rc or rc != rs)
                if pipeline and rs != rp:
                    if rank == rs:
                        _send_tensor(out["feedback"], dst=rp)
                    if rank == rp:
                        feedback_prefix = _recv_tensor(
                            src=rs, device=device, dtype=dtype,
                            requires_grad=False,
                            shape=(1, latent_steps, PLANNER_DIM),
                        )
                else:
                    feedback_prefix = out.get("feedback")

        if rank == rs and solver_mdl is not None:
            solver_logits = out.get("solver_logits")   # (1, latent_steps, vocab)
            if solver_logits is not None:
                n_show   = min(latent_steps, max_new_tokens)
                pred_ids = solver_logits[0, :n_show].argmax(dim=-1).tolist()
                generated = solver_tok.decode(pred_ids, skip_special_tokens=True)
            else:
                generated = "<no output>"

            lines = [
                f"\n── Eval probe  [step {step}] {'─' * 44}",
                f"Q : {question}",
                f"GT: {answer}",
                f"→   {generated}",
                f"{'─' * 64}",
            ]
            for ln in lines:
                print(ln)
            if probe_log_file is not None:
                with open(probe_log_file, "a") as f:
                    f.write("\n".join(lines) + "\n")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(cfg, device: torch.device, mode: str = "original") -> Dict:
    rank  = _rank()
    world = _world()
    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = _dtype_map[getattr(cfg, "dtype", "float32")]

    if rank == 0:
        print(f"\nDevice: {device}  |  world={world}  |  n_rounds={cfg.n_rounds}"
              f"  latent_steps={cfg.latent_steps}  steps={cfg.steps}  mode={mode}")

    # ── Load models: each rank loads only what it owns ───────────────────────
    rp = _pipeline_rank("planner")
    rc = _pipeline_rank("critic")
    rs = _pipeline_rank("solver")

    planner_mdl = planner_tok = planner_inner = None
    critic_mdl  = critic_tok  = critic_inner  = None
    solver_mdl  = solver_tok  = solver_inner  = None

    if rank == rp:
        if rank == 0: print("Loading Planner...")
        planner_mdl, planner_tok = load_model_and_tokenizer(PLANNER_REPO, device, dtype)
        if mode != "shared_roae":
            planner_inner = load_inner_adapter(PLANNER_REPO, PLANNER_DIM, device, dtype)
            for p in planner_inner.parameters(): p.requires_grad_(False)

    if rank == rc:
        if rank == 0: print("Loading Critic...")
        critic_mdl, critic_tok = load_model_and_tokenizer(CRITIC_REPO, device, dtype)
        if mode != "shared_roae":
            critic_inner = load_inner_adapter(CRITIC_REPO, CRITIC_DIM, device, dtype)
            for p in critic_inner.parameters(): p.requires_grad_(False)

    if rank == rs:
        if rank == 0: print("Loading Solver...")
        solver_mdl, solver_tok = load_model_and_tokenizer(SOLVER_REPO, device, dtype)
        if mode != "shared_roae":
            solver_inner = load_inner_adapter(SOLVER_REPO, SOLVER_DIM, device, dtype)
            for p in solver_inner.parameters(): p.requires_grad_(False)

    _barrier()

    # ── Load outer-link adapters ─────────────────────────────────────────────
    outer_12 = outer_23 = outer_31 = None
    shared_link = None

    if mode == "shared_roae":
        hidden_dims = [PLANNER_DIM, CRITIC_DIM, SOLVER_DIM]
        shared_dim  = 512
        shared_link = SharedRecursiveLink(
            hidden_dims=hidden_dims,
            shared_dim=shared_dim,
            n_experts=getattr(cfg, "n_experts", 4),
            expert_dim_divisor=getattr(cfg, "expert_dim_divisor", 4),
        ).to(device=device, dtype=dtype).train()
        if rank == 0:
            n_sl = sum(p.numel() for p in shared_link.parameters() if p.requires_grad)
            print(f"SharedRecursiveLink trainable params: {n_sl:,}  (shared dim={shared_dim})")
    else:
        # Randomly initialise outer links — only these are trained; pretrained
        # weights are intentionally NOT loaded so the adapters start from scratch.
        if rank == rp:
            outer_12 = CrossModelAdapter(
                in_dim=PLANNER_DIM, out_dim=CRITIC_DIM, adapter_type=OUTER_ADAPTER_TYPE,
            ).to(device=device, dtype=dtype).train()
        if rank == rc:
            outer_23 = CrossModelAdapter(
                in_dim=CRITIC_DIM, out_dim=SOLVER_DIM, adapter_type=OUTER_ADAPTER_TYPE,
            ).to(device=device, dtype=dtype).train()
        if rank == rs:
            outer_31 = CrossModelAdapter(
                in_dim=SOLVER_DIM, out_dim=PLANNER_DIM, adapter_type=OUTER_ADAPTER_TYPE,
            ).to(device=device, dtype=dtype).train()

    # ── Optimiser: each rank optimises only its own outer link ───────────────
    local_params = []
    if shared_link is not None:
        local_params = list(shared_link.parameters())
    else:
        if outer_12 is not None: local_params += list(outer_12.parameters())
        if outer_23 is not None: local_params += list(outer_23.parameters())
        if outer_31 is not None: local_params += list(outer_31.parameters())

    n_trainable = sum(p.numel() for p in local_params)
    if rank == 0:
        print(f"Trainable outer-link parameters (rank {rank}): {n_trainable:,}")

    optimizer = AdamW(local_params, lr=cfg.lr, weight_decay=0.01) if local_params else None
    warmup_steps = max(1, cfg.steps // 10)
    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                   total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=max(1, cfg.steps - warmup_steps)),
            ],
            milestones=[warmup_steps],
        )
        if optimizer is not None else None
    )

    # ── Dataset (all ranks load, each picks the same sample via shared seed) ─
    dataset_name = getattr(cfg, "dataset", "math500")
    if rank == 0:
        print(f"Loading {dataset_name} ({cfg.n_samples} problems)...")
    problems = load_training_problems(dataset_name, n_samples=cfg.n_samples, seed=cfg.seed)

    max_seq_len = getattr(cfg, "max_seq_len", 0)
    if max_seq_len > 0:
        n_before = len(problems)
        problems = truncate_problems_by_tokens(problems, max_seq_len)
        if rank == 0:
            print(f"  Truncated to max_seq_len={max_seq_len}: {n_before} -> {len(problems)} problems.")

    if rank == 0:
        print(f"  Loaded {len(problems)} problems.")

    # Fix a single probe example for qualitative eval throughout training.
    # Use the first problem in the shuffled dataset (deterministic across runs).
    probe_question, probe_answer = problems[1]
    if rank == 0:
        print(f"\n  Probe question: {probe_question[:120]}{'...' if len(probe_question) > 120 else ''}")
        print(f"  Probe answer  : {probe_answer[:80]}{'...' if len(probe_answer) > 80 else ''}")

    _barrier()

    # ── Training ─────────────────────────────────────────────────────────────
    losses: List[float] = []
    monitor = GradMonitor(cfg.n_rounds) if rank == rp else None
    rng     = np.random.default_rng(cfg.seed)

    # Probe log: one file per (mode, n_rounds) run, written by rank==rs only.
    out_prefix     = getattr(cfg, "out_prefix", "outerlink_grad")
    probe_log_file = f"{out_prefix}_{mode}_r{cfg.n_rounds}_probe.txt"
    # Write header once (truncate any prior file from a rerun).
    if rank == rs:
        with open(probe_log_file, "w") as f:
            f.write(f"Eval probe log  |  mode={mode}  n_rounds={cfg.n_rounds}"
                    f"  latent_steps={cfg.latent_steps}  steps={cfg.steps}\n")
            f.write(f"Q : {probe_question}\n")
            f.write(f"GT: {probe_answer}\n")
            f.write("=" * 64 + "\n")

    # Pretrained baseline: run once before training using the original (unmodified)
    # outer adapters.  Works in both single-GPU and pipeline mode — each rank
    # executes only the stage it owns; tensors are passed via NCCL P2P inside.
    outer_dir_path = Path(snapshot_repo(OUTER_REPO))
    pretrained_inference_probe(
        question=probe_question,
        answer=probe_answer,
        planner_mdl=planner_mdl, planner_tok=planner_tok,
        critic_mdl=critic_mdl,   critic_tok=critic_tok,
        solver_mdl=solver_mdl,   solver_tok=solver_tok,
        planner_inner=planner_inner, critic_inner=critic_inner, solver_inner=solver_inner,
        outer_dir=outer_dir_path,
        latent_steps=cfg.latent_steps,
        n_rounds=cfg.n_rounds,
        max_new_tokens=256,
        device=device,
        dtype=dtype,
        probe_log_file=probe_log_file,
    )

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Training outer-links  —  {cfg.n_rounds} recursion round(s)"
              f"  world={world}")
        print(f"{'='*70}")

    pipeline = _is_dist() and (rp != rc or rc != rs)

    B = cfg.batch_size

    def _zero_like_grad(shape, dt, dev):
        return torch.zeros(shape, dtype=dt, device=dev)

    # ── Timing helper: wall-clock per phase, CUDA-synced for accuracy ────────
    _time_this_step = False
    _phase_times: Dict[str, float] = {}
    _t0 = [0.0]

    def _tic():
        if _time_this_step:
            torch.cuda.synchronize(device)
            _t0[0] = time.perf_counter()

    def _toc(label: str):
        if _time_this_step:
            torch.cuda.synchronize(device)
            dt_ = time.perf_counter() - _t0[0]
            _phase_times[label] = _phase_times.get(label, 0.0) + dt_
            _t0[0] = time.perf_counter()

    for step in range(cfg.steps):
        # Time the first few steps on every rank (step 0 includes warmup/compile
        # overhead, so step 1-2 are more representative of steady-state cost).
        _time_this_step = step in (0, 1, 2)
        _phase_times = {}
        _tic()

        if optimizer is not None:
            optimizer.zero_grad()

        # ── Sample a batch of B problems ─────────────────────────────────────
        idxs      = rng.integers(len(problems), size=B)
        questions = [problems[i][0] for i in idxs]
        answers   = [problems[i][1] for i in idxs]
        _toc("data_sampling")

        # ── Forward: all n rounds (paper Eq.6: L_out = CE(S^n(...S^1(x)), y)) ─
        feedback_prefix = None
        rounds = []

        for r in range(cfg.n_rounds):
            _tic()
            out = forward_one_round(
                planner_mdl, planner_tok, planner_inner,
                critic_mdl,  critic_tok,  critic_inner,
                solver_mdl,  solver_tok,  solver_inner,
                outer_12, outer_23, outer_31,
                questions=questions,
                answers=answers,
                feedback_prefix=feedback_prefix,
                latent_steps=cfg.latent_steps,
                device=device,
                dtype=dtype,
                monitor=monitor,
                round_idx=r,
                shared_link=shared_link,
                debug=(step == 0),
            )
            rounds.append(out)
            _toc(f"forward_round{r}")

            if r < cfg.n_rounds - 1:
                if pipeline and rs != rp:
                    if rank == rs:
                        _send_tensor(out["feedback"], dst=rp)
                    if rank == rp:
                        feedback_prefix = _recv_tensor(src=rs, device=device, dtype=dtype,
                                                       requires_grad=True,
                                                       shape=(B, cfg.latent_steps, PLANNER_DIM))
                        if feedback_prefix.requires_grad:
                            feedback_prefix.retain_grad()
                        out["fb_recv"] = feedback_prefix
                else:
                    feedback_prefix = out.get("feedback")
                _toc(f"feedback_exchange_round{r}")

        # ── Loss: already computed inside forward_one_round (mean over batch) ─
        last = rounds[-1]
        loss = last.get("solver_loss") if rank == rs else None

        # ── Backward: explicit, barrier-separated pipeline grad exchange ──────
        _tic()
        if not pipeline:
            if loss is not None:
                loss.backward()
        else:
            for r in range(cfg.n_rounds - 1, -1, -1):
                o = rounds[r]
                is_last_backward = (r == 0)

                # Step A ── rs: backward through local solver graph ────────────
                if rank == rs:
                    if r == cfg.n_rounds - 1:
                        if loss is not None:
                            loss.backward(retain_graph=not is_last_backward)
                    else:
                        g_fb = rounds[r + 1].get("g_fb_for_prev")
                        fb   = o.get("feedback")
                        if fb is not None and g_fb is not None:
                            fb.backward(g_fb, retain_graph=not is_last_backward)

                # Step B ── rs→rc grad ─────────────────────────────────────────
                if rank == rs:
                    sp_recv = o.get("sp_recv")
                    g = (sp_recv.grad if (sp_recv is not None and sp_recv.grad is not None)
                         else _zero_like_grad((B, cfg.latent_steps, SOLVER_DIM), dtype, device))
                    _send_tensor(g, dst=rc)

                if rank == rc:
                    sv_pfx = o.get("solver_prefix_rc")
                    g = _recv_tensor(src=rs, device=device, dtype=dtype,
                                     shape=(B, cfg.latent_steps, SOLVER_DIM))
                    if sv_pfx is not None:
                        sv_pfx.backward(g, retain_graph=not is_last_backward)

                # Step C ── rc→rp grad ─────────────────────────────────────────
                if rank == rc:
                    cp_recv = o.get("cp_recv")
                    g = (cp_recv.grad if (cp_recv is not None and cp_recv.grad is not None)
                         else _zero_like_grad((B, cfg.latent_steps, CRITIC_DIM), dtype, device))
                    _send_tensor(g, dst=rp)

                if rank == rp:
                    cr_pfx = o.get("critic_prefix")
                    g = _recv_tensor(src=rc, device=device, dtype=dtype,
                                     shape=(B, cfg.latent_steps, CRITIC_DIM))
                    if cr_pfx is not None:
                        cr_pfx.backward(g, retain_graph=not is_last_backward)

                # Step D ── rp→rs feedback grad (rounds r>0 only) ──────────────
                if r > 0 and pipeline and rs != rp:
                    if rank == rp:
                        fb_recv = rounds[r - 1].get("fb_recv")
                        g = (fb_recv.grad if (fb_recv is not None and fb_recv.grad is not None)
                             else _zero_like_grad((B, cfg.latent_steps, PLANNER_DIM), dtype, device))
                        _send_tensor(g, dst=rs)
                    if rank == rs:
                        o["g_fb_for_prev"] = _recv_tensor(src=rp, device=device, dtype=dtype,
                                                          shape=(B, cfg.latent_steps, PLANNER_DIM))

        _toc("backward_pipeline")

        # ── Debug: verify backward produced gradients on local trainable params ─
        if step == 0 and local_params:
            local_names = []
            if shared_link is not None:
                local_names = [n for n, _ in shared_link.named_parameters()]
            else:
                if outer_12 is not None: local_names += [f"outer_12.{n}" for n, _ in outer_12.named_parameters()]
                if outer_23 is not None: local_names += [f"outer_23.{n}" for n, _ in outer_23.named_parameters()]
                if outer_31 is not None: local_names += [f"outer_31.{n}" for n, _ in outer_31.named_parameters()]
            n_with_grad = sum(1 for p in local_params if p.grad is not None)
            grad_norm_total = sum(
                p.grad.detach().float().norm().item() ** 2 for p in local_params if p.grad is not None
            ) ** 0.5
            print(f"[dbg bwd rank {rank}] theoretical: all {len(local_params)} local trainable params "
                  f"({local_names[:1]}{'...' if len(local_names) > 1 else ''}) should have non-None .grad after backward  |  "
                  f"practical: {n_with_grad}/{len(local_params)} have grad, total_grad_norm={grad_norm_total:.4e}")

        _tic()
        # ── Sync loss scalar to rank 0 for logging ────────────────────────────
        if _is_dist():
            loss_val = torch.tensor(
                [loss.item() if loss is not None else float("nan")],
                dtype=torch.float32, device=device,
            )
            dist.broadcast(loss_val, src=rs)
            step_loss = loss_val.item()
        else:
            step_loss = loss.item() if loss is not None else float("nan")

        # ── Sync, clip, step ──────────────────────────────────────────────────
        _barrier()
        _toc("loss_sync_barrier")

        # Flattened allreduce for SharedRecursiveLink.
        if _is_dist() and shared_link is not None:
            grads = [p.grad for p in local_params if p.grad is not None]
            if grads:
                flat = torch.cat([g.reshape(-1) for g in grads])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(world)
                offset = 0
                for p in local_params:
                    if p.grad is not None:
                        n = p.grad.numel()
                        p.grad.copy_(flat[offset:offset + n].reshape(p.grad.shape))
                        offset += n

        if optimizer is not None:
            if step == 0 and local_params:
                _pre_step_snapshot = [p.detach().clone() for p in local_params]
            torch.nn.utils.clip_grad_norm_(local_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if step == 0 and local_params:
                total_delta = sum(
                    (p.detach() - p0).float().norm().item() ** 2
                    for p, p0 in zip(local_params, _pre_step_snapshot)
                ) ** 0.5
                print(f"[dbg opt rank {rank}] theoretical: optimizer.step() should change every local "
                      f"trainable param by a non-zero amount (lr={cfg.lr})  |  "
                      f"practical: total_param_delta_norm={total_delta:.4e}")
        _toc("grad_sync_and_optimizer")

        # ── Logging ──────────────────────────────────────────────────────────
        if rank == rp and monitor is not None:
            monitor.commit_step()
        if _is_dist():
            _sync_norms_all_ranks(monitor, cfg.n_rounds)
        _toc("logging_and_norm_sync")

        if _time_this_step:
            total = sum(_phase_times.values())
            ranked = sorted(_phase_times.items(), key=lambda kv: kv[1], reverse=True)
            breakdown = "  ".join(f"{name}={dt_:.3f}s ({100*dt_/total:.1f}%)" for name, dt_ in ranked)
            bottleneck = ranked[0]
            print(f"[dbg time rank {rank} step {step}] total={total:.3f}s  "
                  f"BOTTLENECK={bottleneck[0]} ({bottleneck[1]:.3f}s, {100*bottleneck[1]/total:.1f}%)\n"
                  f"    breakdown: {breakdown}")

        if rank == 0:
            losses.append(step_loss)

        log_this_step = (step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1)

        if log_this_step:
            if rank == 0:
                if monitor is not None and monitor.norms[0]:
                    norms_str = "  ".join(
                        f"r{r+1}:{monitor.norms[r][-1]:.3e}"
                        for r in range(cfg.n_rounds)
                    )
                    print(f"  step {step:4d}  loss={step_loss:.4f}"
                          f"  grad_norms [{norms_str}]")
                else:
                    print(f"  step {step:4d}  loss={step_loss:.4f}")

            # Eval probe — all ranks participate (pipeline collectives inside)
            eval_probe(
                step=step,
                question=probe_question,
                answer=probe_answer,
                planner_mdl=planner_mdl, planner_tok=planner_tok, planner_inner=planner_inner,
                critic_mdl=critic_mdl,   critic_tok=critic_tok,   critic_inner=critic_inner,
                solver_mdl=solver_mdl,   solver_tok=solver_tok,   solver_inner=solver_inner,
                outer_12=outer_12, outer_23=outer_23, outer_31=outer_31,
                shared_link=shared_link,
                latent_steps=cfg.latent_steps,
                n_rounds=cfg.n_rounds,
                max_new_tokens=cfg.latent_steps,
                device=device,
                dtype=dtype,
                probe_log_file=probe_log_file,
            )

    # Gather results: collect norm history from rank_planner to rank 0
    if _is_dist() and rp != 0:
        # rank_planner serialises norms and sends to rank 0
        import pickle
        if rank == rp:
            blob = pickle.dumps(monitor.norms)
            size = torch.tensor([len(blob)], dtype=torch.long, device=device)
            dist.send(size, dst=0)
            data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)
            dist.send(data, dst=0)
        if rank == 0:
            size = torch.zeros(1, dtype=torch.long, device=device)
            dist.recv(size, src=rp)
            data = torch.zeros(size.item(), dtype=torch.uint8, device=device)
            dist.recv(data, src=rp)
            import pickle
            grad_norms = pickle.loads(bytes(data.cpu().numpy()))
    else:
        grad_norms = monitor.norms if monitor is not None else {r: [] for r in range(cfg.n_rounds)}

    # ── Free this run's models/optimizer before returning ────────────────────
    # `compare` mode calls run_training twice in the same process (original,
    # then shared_roae); without releasing CUDA memory here, the second run's
    # models stack on top of the first's and the GPU runs out of memory.
    del optimizer, scheduler, local_params
    del planner_mdl, planner_tok, planner_inner
    del critic_mdl,  critic_tok,  critic_inner
    del solver_mdl,  solver_tok,  solver_inner
    del outer_12, outer_23, outer_31, shared_link
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {"losses": losses, "grad_norms_per_round": grad_norms}


# ── Analysis + plotting (rank 0 only) ────────────────────────────────────────

def analyse(results: Dict, cfg) -> None:
    n_rounds = cfg.n_rounds
    tail     = max(1, cfg.steps // 5)

    print(f"\n{'='*70}")
    print(f"  Gradient stability analysis  (last {tail} steps, {n_rounds} rounds)")
    print(f"{'='*70}")

    avg_norms = {}
    for r in range(n_rounds):
        vals = [v for v in results["grad_norms_per_round"][r][-tail:] if not math.isnan(v)]
        avg_norms[r] = float(np.mean(vals)) if vals else float("nan")

    max_norm = max((v for v in avg_norms.values() if not math.isnan(v)), default=1.0)
    for r in range(n_rounds):
        bar = "#" * max(1, int(40 * avg_norms.get(r, 0) / (max_norm + 1e-12)))
        print(f"  Round {r+1:2d}: grad_norm = {avg_norms.get(r, float('nan')):.4e}  {bar}")

    if n_rounds >= 2:
        ratio = avg_norms[0] / (avg_norms[n_rounds - 1] + 1e-12)
        verdict = (
            "VANISHING  (ratio << 1)"  if ratio < 0.1 else
            "EXPLODING  (ratio >> 1)"  if ratio > 10  else
            "STABLE     (ratio ≈ 1)"
        )
        print(f"\n  Ratio round-1 / round-{n_rounds}: {ratio:.4f}  →  {verdict}")

    loss_tail = results["losses"][-tail:]
    print(f"  Loss (last {tail} steps):  mean={np.mean(loss_tail):.4f}"
          f"  std={np.std(loss_tail):.4f}")


def plot_single(results: Dict, cfg, out_path: str) -> None:
    n_rounds = cfg.n_rounds
    tail     = max(1, cfg.steps // 5)
    cmap     = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"RecursiveMAS-Light  |  outer-link training on MATH-500\n"
        f"n_rounds={n_rounds}  latent_steps={cfg.latent_steps}  "
        f"steps={cfg.steps}  lr={cfg.lr}  world={_world()}",
        fontsize=10,
    )

    ax = axes[0]
    losses = results["losses"]
    w = max(1, len(losses) // 20)
    ax.plot(losses, alpha=0.35, color="steelblue", linewidth=0.8)
    if len(losses) >= w:
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(losses)), smooth, color="steelblue", linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("CE loss")
    ax.set_title("Training loss"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    gn = results["grad_norms_per_round"]
    for r in range(n_rounds):
        vals = [v if not math.isnan(v) else 0.0 for v in gn[r]]
        if not any(v > 0 for v in vals):
            continue
        color = cmap(r / max(1, n_rounds - 1))
        ax.plot(vals, color=color, linewidth=1.2, label=f"Round {r+1}")
    ax.set_xlabel("Step"); ax.set_ylabel("Grad norm (outer_12 output)")
    ax.set_title("Gradient norm per round"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    avg_norms = []
    for r in range(n_rounds):
        vals = [v for v in gn[r][-tail:] if not math.isnan(v)]
        avg_norms.append(float(np.mean(vals)) if vals else 1e-12)
    colors = [cmap(r / max(1, n_rounds - 1)) for r in range(n_rounds)]
    bars = ax.bar(range(1, n_rounds + 1), avg_norms, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, avg_norms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.1e}", ha="center", va="bottom", fontsize=7, rotation=30)
    ax.set_xlabel("Round"); ax.set_ylabel("Avg grad norm")
    ax.set_title(f"Avg grad norm (last {tail} steps)")
    ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(1, n_rounds + 1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


def plot_sweep(sweep_results: List[Tuple[int, Dict]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Sweep: gradient stability vs recursion depth  "
                 "(RecursiveMAS-Light, MATH-500)")

    round_counts = [n for n, _ in sweep_results]
    ratios, losses = [], []
    for n, res in sweep_results:
        gn   = res["grad_norms_per_round"]
        tail = max(1, len(res["losses"]) // 5)
        first = [v for v in gn[0][-tail:]     if not math.isnan(v)]
        last  = [v for v in gn[n - 1][-tail:] if not math.isnan(v)]
        ratios.append((np.mean(first) if first else float("nan")) /
                      ((np.mean(last) if last else 0.0) + 1e-12))
        losses.append(float(np.mean(res["losses"][-tail:])))

    ax = axes[0]
    colors = ["tomato" if r < 0.1 or r > 10 else "mediumseagreen" for r in ratios]
    ax.bar(round_counts, ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.1, color="red",    linewidth=1, linestyle="--", label="vanish (0.1)")
    ax.axhline(10,  color="orange", linewidth=1, linestyle="--", label="explode (10)")
    ax.axhline(1.0, color="green",  linewidth=1, linestyle=":",  label="ideal (1.0)")
    ax.set_xlabel("n_rounds"); ax.set_ylabel("Grad ratio  r1 / rN")
    ax.set_title("Gradient ratio vs depth"); ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3); ax.set_xticks(round_counts)

    ax = axes[1]
    ax.bar(round_counts, losses, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("n_rounds"); ax.set_ylabel("Mean loss (last 20%)")
    ax.set_title("Training loss vs depth")
    ax.grid(True, axis="y", alpha=0.3); ax.set_xticks(round_counts)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RecursiveMAS-Light outer-links on MATH-500 "
                    "and monitor gradient vanishing across recursion rounds.\n\n"
                    "Single-GPU:  python train_outerlinks_math500.py\n"
                    "Multi-GPU:   torchrun --standalone --nproc_per_node=3 "
                    "train_outerlinks_math500.py"
    )
    p.add_argument("--n_rounds", type=int, nargs="+", default=[3],
                   help="Recursion rounds. Multiple values → sweep.")
    p.add_argument("--latent_steps", type=int, default=48,
                   help="Latent rollout steps. run.py infers 48 for math500+sequential_light.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Problems per optimizer update (training batch size).")
    p.add_argument("--dataset", type=str, default="math500",
                   help="Training dataset: 'math500', 's1k', 'm1k', or a '+'-joined "
                        "combination such as 's1k+m1k' (pools and shuffles samples).")
    p.add_argument("--max_seq_len", type=int, default=0,
                   help="Max combined (question+answer) sequence length in tokens "
                        "(approximate, char-based truncation). 0 = no truncation.")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", type=str, default="outerlink_grad")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--mode", type=str, default="original",
                   choices=["original", "shared_roae", "compare"],
                   help="original: independent CrossModelAdapters (default); "
                        "shared_roae: single SharedRecursiveLink with MoE + RoAE; "
                        "compare: run both and plot side-by-side.")
    p.add_argument("--n_experts", type=int, default=4,
                   help="Number of MoE experts (shared_roae mode only).")
    p.add_argument("--expert_dim_divisor", type=int, default=4,
                   help="Expert inner dim = shared_dim // expert_dim_divisor.")
    return p.parse_args()


def plot_compare(results_orig: Dict, results_shared: Dict, cfg, out_path: str) -> None:
    """5-panel comparison: Original vs SharedLink-RoAE on real MATH-500 data."""
    n_rounds = cfg.n_rounds
    tail     = max(1, cfg.steps // 5)
    cmap     = plt.get_cmap("plasma")

    variants = [
        ("Original (CrossModelAdapter)", results_orig,   "steelblue"),
        ("SharedLink-RoAE (MoE+RoPE)",  results_shared, "mediumseagreen"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(
        f"RecursiveMAS-Light  |  Original vs SharedLink-RoAE  |  MATH-500\n"
        f"n_rounds={n_rounds}  latent_steps={cfg.latent_steps}  "
        f"steps={cfg.steps}  lr={cfg.lr}  world={_world()}\n"
        f"SharedLink: n_experts={cfg.n_experts}  expert_dim_divisor={cfg.expert_dim_divisor}",
        fontsize=9,
    )

    # Panel 0: loss curves
    ax = axes[0]
    for label, res, color in variants:
        losses = res["losses"]
        steps  = len(losses)
        ax.plot(losses, color=color, linewidth=0.7, alpha=0.4)
        w = max(1, steps // 20)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, steps), smooth, color=color, linewidth=2, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("CE loss")
    ax.set_title("Training loss"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panels 1-2: grad norm trajectories per model
    for panel_idx, (label, res, _) in enumerate(variants):
        ax = axes[panel_idx + 1]
        gn = res["grad_norms_per_round"]
        for r in range(n_rounds):
            vals = [v if not math.isnan(v) else 0.0 for v in gn[r]]
            if not any(v > 0 for v in vals):
                continue
            ax.plot(vals, color=cmap(r / max(1, n_rounds - 1)),
                    linewidth=1.1, alpha=0.85, label=f"Round {r + 1}")
        ax.set_xlabel("Step"); ax.set_ylabel("Grad norm")
        ax.set_title(f"{label}\ngrad norm / round")
        ax.legend(fontsize=7); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    # Panel 3: ratio r1/rN comparison bar
    ax = axes[3]
    for g_idx, (label, res, color) in enumerate(variants):
        gn = res["grad_norms_per_round"]
        first = [v for v in gn[0][-tail:] if not math.isnan(v)]
        last  = [v for v in gn[n_rounds - 1][-tail:] if not math.isnan(v)]
        ratio = (np.mean(first) if first else float("nan")) / \
                ((np.mean(last) if last else 0.0) + 1e-12)
        bar = ax.bar([g_idx], [ratio], color=color, edgecolor="black",
                     linewidth=0.5, alpha=0.85, label=label)
        ax.text(g_idx, ratio * 1.12, f"{ratio:.3f}", ha="center",
                va="bottom", fontsize=9)
    ax.axhline(0.1, color="red",    linewidth=1, linestyle="--", label="vanish (0.1)")
    ax.axhline(10,  color="orange", linewidth=1, linestyle="--", label="explode (10)")
    ax.axhline(1.0, color="green",  linewidth=1, linestyle=":",  label="ideal (1.0)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Original", "SharedLink-RoAE"])
    ax.set_ylabel("Grad ratio  r1 / rN"); ax.set_title("Grad ratio (lower=vanishing)")
    ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: avg grad norm per round, side-by-side bars
    ax = axes[4]
    x = np.arange(n_rounds)
    width = 0.35
    for g_idx, (label, res, color) in enumerate(variants):
        gn = res["grad_norms_per_round"]
        avg_norms = []
        for r in range(n_rounds):
            vals = [v for v in gn[r][-tail:] if not math.isnan(v)]
            avg_norms.append(float(np.mean(vals)) if vals else 1e-12)
        offset = (g_idx - 0.5) * width
        bars = ax.bar(x + offset, avg_norms, width, label=label,
                      color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, avg_norms):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12,
                    f"{val:.1e}", ha="center", va="bottom", fontsize=5, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels([f"R{r + 1}" for r in range(n_rounds)])
    ax.set_xlabel("Round"); ax.set_ylabel("Avg grad norm")
    ax.set_title(f"Avg grad norm (last {tail} steps)")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved → {out_path}")


def _print_compare_summary(results_orig: Dict, results_shared: Dict, cfg) -> None:
    rank  = _rank()
    if rank != 0:
        return
    n_rounds = cfg.n_rounds
    tail     = max(1, cfg.steps // 5)

    print(f"\n{'='*70}")
    print("Comparison: Original vs SharedLink-RoAE  (last 20% of training)")
    print(f"{'='*70}")
    for label, res in [("Original (CrossModelAdapter)", results_orig),
                       ("SharedLink-RoAE",              results_shared)]:
        print(f"\n  [{label}]")
        avg_norms = {}
        for r in range(n_rounds):
            vals = [v for v in res["grad_norms_per_round"][r][-tail:] if not math.isnan(v)]
            avg_norms[r] = float(np.mean(vals)) if vals else float("nan")
        max_norm = max((v for v in avg_norms.values() if not math.isnan(v)), default=1.0)
        for r in range(n_rounds):
            bar = "#" * max(1, int(40 * avg_norms.get(r, 0) / (max_norm + 1e-12)))
            print(f"    Round {r + 1:2d}: grad_norm = {avg_norms.get(r, float('nan')):.4e}  {bar}")
        if n_rounds >= 2:
            ratio = avg_norms[0] / (avg_norms[n_rounds - 1] + 1e-12)
            verdict = ("VANISHING" if ratio < 0.1 else "EXPLODING" if ratio > 10 else "STABLE")
            print(f"    ratio r1/r{n_rounds}: {ratio:.4f}  →  {verdict}")
        loss_tail = res["losses"][-tail:]
        print(f"    loss: mean={np.mean(loss_tail):.4f}  std={np.std(loss_tail):.4f}")


def main():
    cfg = parse_args()

    # ── DDP / device setup ────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        _init_gloo_group()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank       = _rank()
    round_list = cfg.n_rounds
    sweep      = len(round_list) > 1
    mode       = cfg.mode

    def _run_single(n_rounds_val: int, run_mode: str):
        """Helper: run one (n_rounds, mode) combination and return results."""
        cfg_n = copy.copy(cfg)
        cfg_n.n_rounds = n_rounds_val
        return run_training(cfg_n, device, mode=run_mode)

    if mode == "compare":
        # Run original then shared_roae and produce a side-by-side plot.
        if not sweep:
            cfg.n_rounds = round_list[0]
            if rank == 0:
                print("\n" + "=" * 70)
                print("  COMPARE mode — running Original first, then SharedLink-RoAE")
                print("=" * 70)
            results_orig   = _run_single(cfg.n_rounds, "original")
            results_shared = _run_single(cfg.n_rounds, "shared_roae")
            if rank == 0:
                analyse(results_orig,   cfg)
                analyse(results_shared, cfg)
                _print_compare_summary(results_orig, results_shared, cfg)
                out = f"{cfg.out_prefix}_compare_r{cfg.n_rounds}.png"
                plot_compare(results_orig, results_shared, cfg, out_path=out)
        else:
            for n in round_list:
                if rank == 0:
                    print(f"\n{'='*70}  n_rounds={n}  {'='*70}")
                r_orig   = _run_single(n, "original")
                r_shared = _run_single(n, "shared_roae")
                cfg_n = copy.copy(cfg); cfg_n.n_rounds = n
                if rank == 0:
                    _print_compare_summary(r_orig, r_shared, cfg_n)
                    plot_compare(r_orig, r_shared, cfg_n,
                                 out_path=f"{cfg.out_prefix}_compare_r{n}.png")

    elif not sweep:
        cfg.n_rounds = round_list[0]
        results = _run_single(cfg.n_rounds, mode)
        if rank == 0:
            analyse(results, cfg)
            plot_single(results, cfg,
                        out_path=f"{cfg.out_prefix}_{mode}_r{cfg.n_rounds}.png")
    else:
        sweep_results = []
        for n in round_list:
            results = _run_single(n, mode)
            cfg_n = copy.copy(cfg); cfg_n.n_rounds = n
            if rank == 0:
                analyse(results, cfg_n)
                plot_single(results, cfg_n,
                            out_path=f"{cfg.out_prefix}_{mode}_r{n}.png")
            sweep_results.append((n, results))

        if rank == 0:
            tail = max(1, cfg.steps // 5)
            print(f"\n{'='*80}")
            print(f"Sweep summary  [{mode}]  (ratio = grad_norm_round1 / grad_norm_roundN)")
            print(f"  {'rounds':>6}  {'ratio':>10}  {'loss':>8}  verdict")
            print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*20}")
            for n, res in sweep_results:
                gn    = res["grad_norms_per_round"]
                first = [v for v in gn[0][-tail:]     if not math.isnan(v)]
                last  = [v for v in gn[n - 1][-tail:] if not math.isnan(v)]
                ratio = (np.mean(first) if first else float("nan")) / \
                        ((np.mean(last) if last else 0.0) + 1e-12)
                avg_l = float(np.mean(res["losses"][-tail:]))
                verd  = ("VANISHING" if ratio < 0.1 else
                         "EXPLODING" if ratio > 10   else "STABLE")
                print(f"  {n:>6}  {ratio:>10.4f}  {avg_l:>8.4f}  {verd}")
            plot_sweep(sweep_results, out_path=f"{cfg.out_prefix}_{mode}_sweep.png")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
