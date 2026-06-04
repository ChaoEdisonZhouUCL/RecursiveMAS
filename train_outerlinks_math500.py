#!/usr/bin/env python3
"""
Train the outer-links of RecursiveMAS-Light on MATH-500 and monitor
gradient norms across recursion rounds to diagnose vanishing gradients.

Architecture (Sequential-Light):
    Planner  (Qwen3-1.7B,        d=2048)
       ↓  outer_12  (2048→2048)
    Critic   (LLaMA-3.2-1B,     d=2048)
       ↓  outer_23  (2048→1536)
    Solver   (Qwen2.5-Math-1.5B, d=1536)
       ↓  outer_31  (1536→2048)  ──► back to Planner (next round)

Two training modes (--mode):
  original    — three independent CrossModelAdapters, one per rank
  shared_roae — one SharedRecursiveLink with MoE core + Rotary Agent Encoding
                loaded on ALL ranks; gradients allreduced after each backward step
  compare     — run both sequentially on the same problems and produce a side-by-side plot

Multi-GPU (pipeline parallelism)
---------------------------------
Launch with torchrun:
    torchrun --standalone --nproc_per_node=3 train_outerlinks_math500.py

  rank 0 → Planner  + outer_12  (2048→2048)
  rank 1 → Critic   + outer_23  (2048→1536)
  rank 2 → Solver   + outer_31  (1536→2048)

Single-GPU fallback
-------------------
    python train_outerlinks_math500.py
All three agents load onto cuda:0 (or cpu).

Usage
-----
  python  train_outerlinks_math500.py --n_rounds 3 --steps 100
  torchrun --standalone --nproc_per_node=3 \\
           train_outerlinks_math500.py --n_rounds 3 --steps 100 --mode compare
"""

import argparse
import copy
import math
import os
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
from huggingface_hub import snapshot_download
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from modeling import CrossModelAdapter, infer_outer_adapter_type_from_state_dict

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
                nn.Identity() if hidden_dims[i] == hidden_dims[j]
                else nn.Linear(hidden_dims[i], hidden_dims[j], bias=False)
                for j in range(N)
            ])
            for i in range(N)
        ])
        for i in range(N):
            for j in range(N):
                m = self.skip_proj[i][j]
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)

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

def _resolve(repo_id: str) -> str:
    try:
        return snapshot_download(repo_id, local_files_only=True, token=HF_TOKEN)
    except Exception:
        return snapshot_download(repo_id, token=HF_TOKEN)


def _plain_view(model_dir: str) -> str:
    p = Path(model_dir)
    if not (p / "adapter_config.json").is_file():
        return model_dir
    view = p / "_plain_model_view"
    if view.is_dir():
        return str(view)
    view.mkdir(parents=True, exist_ok=True)
    skip = {"adapter_config.json", "innerlink_config.json", "README.md"}
    for item in p.iterdir():
        if item.name == view.name or item.name in skip:
            continue
        if item.name.startswith("adapter(") or item.suffix == ".pt":
            continue
        tgt = view / item.name
        if not tgt.exists():
            tgt.symlink_to(item.resolve())
    return str(view)


def load_model_and_tokenizer(repo_id: str, device: torch.device, dtype: torch.dtype):
    local = _resolve(repo_id)
    path  = _plain_view(local)
    tok   = AutoTokenizer.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    mdl   = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=str(device),
        trust_remote_code=True, token=HF_TOKEN,
    )
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return mdl, tok


def load_inner_adapter(repo_id: str, hidden: int, device: torch.device, dtype: torch.dtype):
    from modeling import Adapter, INNER_ADAPTER_TYPE
    local = _resolve(repo_id)
    pt = Path(local) / "adapter(math).pt"
    sd = torch.load(str(pt), map_location="cpu")
    a  = Adapter(hidden_size=hidden, adapter_type=INNER_ADAPTER_TYPE)
    a.load_state_dict(sd, strict=True)
    return a.to(device=device, dtype=dtype).eval()


def load_outer_adapter(pt_path: str, in_dim: int, out_dim: int,
                       device: torch.device, dtype: torch.dtype) -> CrossModelAdapter:
    sd    = torch.load(pt_path, map_location="cpu")
    atype = infer_outer_adapter_type_from_state_dict(sd)
    a     = CrossModelAdapter(in_dim=in_dim, out_dim=out_dim, adapter_type=atype)
    a.load_state_dict(sd, strict=True)
    return a.to(device=device, dtype=dtype)


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_math500(n_samples: int = 0, seed: int = 42):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=HF_TOKEN)
    if n_samples > 0:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [(row["problem"], row["answer"]) for row in ds]


def build_planner_prompt(question: str) -> Tuple[str, str]:
    """Returns (prefix, suffix) around the latent embedding slot."""
    pre  = "You are a planner agent in a recursive multi-agent system. Here is the latent information from previous round:\n"
    post = f"Given the latent information, you should output a step-by-step plan to solve the question: {question}"
    return pre, post


def build_critic_prompt(question: str) -> Tuple[str, str]:
    pre  = "You are a critic agent in a recursive multi-agent system. Here is the latent information from previous agent:\n"
    post = f"Given the latent information, you should critique the initial plan and output an improved plan to solve the question: {question}"
    return pre, post


def build_solver_prompt(question: str) -> Tuple[str, str]:
    pre  = "You are a solver agent in a recursive multi-agent system. Here is the latent information from previous agent:\n"
    post = (f"Given the latent information, you should solve the question and provide the final answer: {question}\n"
            "Solve the question and put the final answer inside \\boxed{}.")
    return pre, post


def _build_input_embeds(tokenizer, embed_fn, pre: str, post: str,
                        latent: Optional[torch.Tensor],
                        device: torch.device, dtype: torch.dtype):
    """
    Build input_embeds and attention_mask by injecting `latent` between the
    tokenized `pre` and `post` strings — matching the paper's prompt template:
        [pre tokens] [latent embeddings] [post tokens]

    When latent is None (first round, no prior agent), only [pre + post] is used.
    """
    enc_pre  = tokenizer(pre,  return_tensors="pt", add_special_tokens=True).to(device)
    enc_post = tokenizer(post, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        emb_pre  = embed_fn(enc_pre["input_ids"]).to(dtype)   # (1, T_pre, d)
        emb_post = embed_fn(enc_post["input_ids"]).to(dtype)  # (1, T_post, d)

    if latent is not None:
        lat = latent.to(dtype)                                 # (1, T_lat, d)
        ie  = torch.cat([emb_pre, lat, emb_post], dim=1)
        am  = torch.cat([
            enc_pre["attention_mask"],
            enc_pre["attention_mask"].new_ones((1, lat.size(1))),
            enc_post["attention_mask"],
        ], dim=1)
    else:
        ie = torch.cat([emb_pre, emb_post], dim=1)
        am = torch.cat([enc_pre["attention_mask"], enc_post["attention_mask"]], dim=1)

    return ie, am


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
) -> torch.Tensor:
    """
    Run `latent_steps` steps of inner-adapter auto-regression in latent space.
    Returns concatenated last-layer hidden states: (1, latent_steps, d_model).

    Uses a forward hook on the final transformer layer to capture only the
    last-position hidden state — avoids allocating all intermediate hidden
    states that output_hidden_states=True would return.
    KV-cache is used for O(1) per-step cost; cache is detached between steps
    so it never forms cycles in the autograd graph.
    """
    # Install a hook on the last layer to capture its output's last position.
    _last_h: list = []

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out   # (B, T, d)
        _last_h.append(h[:, -1, :])                     # (B, d)

    last_layer = _get_last_layer(model)
    handle = last_layer.register_forward_hook(_hook)

    hidden_states = []
    past_key_values = None
    ie = input_embeds
    am = attention_mask
    try:
        for _ in range(latent_steps):
            _last_h.clear()
            try:
                out = model(inputs_embeds=ie, attention_mask=am,
                            use_cache=True, past_key_values=past_key_values,
                            return_dict=True, logits_to_keep=1)
            except TypeError:
                out = model(inputs_embeds=ie, attention_mask=am,
                            use_cache=True, past_key_values=past_key_values,
                            return_dict=True)

            last_h = _last_h[0]                         # (1, d) — in autograd graph
            hidden_states.append(last_h.unsqueeze(1))

            # Detach cache to prevent autograd graph cycles across steps.
            past_key_values = _detach_past_key_values(out.past_key_values)
            am = torch.cat([am, am.new_ones((am.size(0), 1))], dim=1)
            with torch.no_grad():
                ie = inner_adapter(last_h.detach()).unsqueeze(1).to(dtype)
    finally:
        handle.remove()

    return torch.cat(hidden_states, dim=1)   # (1, latent_steps, d)


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
    question: str,
    feedback_prefix: Optional[torch.Tensor],
    latent_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    monitor: Optional[GradMonitor] = None,
    round_idx: int = 0,
    shared_link: Optional["SharedRecursiveLink"] = None,
):
    """
    One Planner→Critic→Solver forward round (paper §3-4, Fig.1).

    When shared_link is not None (mode='shared_roae'), all three outer
    transitions are computed with SharedRecursiveLink instead of the
    individual CrossModelAdapters (outer_12/23/31 are ignored).

    Forward data flow (sequential, barrier-separated):
      rp: input + feedback_prefix → latent_rollout → outer_12 → send critic_prefix to rc
      rc: recv critic_prefix      → latent_rollout → outer_23 → send solver_prefix to rs
      rs: recv solver_prefix      → latent_rollout → outer_31 → feedback
                                                   → lm_head(solver_prefix) → logits

    Returns dict of tensors needed for backward, keyed by role.
    """
    rank     = _rank()
    rp       = _pipeline_rank("planner")
    rc       = _pipeline_rank("critic")
    rs       = _pipeline_rank("solver")
    pipeline = _is_dist() and (rp != rc or rc != rs)

    out = {}

    # ── Stage 1: Planner computes critic_prefix, sends to Critic ─────────────
    if rank == rp:
        pre, post = build_planner_prompt(question)
        ie, am = _build_input_embeds(
            planner_tok, planner_mdl.get_input_embeddings(),
            pre, post, feedback_prefix, device, dtype,
        )
        planner_h = latent_rollout(planner_mdl, planner_inner, ie, am, latent_steps, dtype)
        if shared_link is not None:
            critic_prefix = shared_link(planner_h, src=1, dst=2)
        else:
            critic_prefix = outer_12(planner_h)
        if monitor is not None:
            monitor.register_output(critic_prefix, round_idx)
        out["critic_prefix"] = critic_prefix
        if pipeline:
            _send_tensor(critic_prefix, dst=rc)

    if pipeline and rp != rc and rank == rc:
        cp_recv = _recv_tensor(src=rp, device=device, dtype=dtype, requires_grad=True,
                               shape=(1, latent_steps, CRITIC_DIM))
        out["cp_recv"] = cp_recv
        critic_prefix  = cp_recv

    # ── Stage 2: Critic computes solver_prefix, sends to Solver ──────────────
    if rank == rc:
        pre_c, post_c = build_critic_prompt(question)
        critic_input, critic_am = _build_input_embeds(
            critic_tok, critic_mdl.get_input_embeddings(),
            pre_c, post_c, critic_prefix, device, dtype,
        )
        critic_h = latent_rollout(critic_mdl, critic_inner, critic_input, critic_am, latent_steps, dtype)
        if shared_link is not None:
            solver_prefix = shared_link(critic_h, src=2, dst=3)
        else:
            solver_prefix = outer_23(critic_h)
        out["solver_prefix_rc"] = solver_prefix
        if pipeline and rc != rs:
            _send_tensor(solver_prefix, dst=rs)

    if pipeline and rc != rs and rank == rs:
        sp_recv = _recv_tensor(src=rc, device=device, dtype=dtype, requires_grad=True,
                               shape=(1, latent_steps, SOLVER_DIM))
        out["sp_recv"] = sp_recv
        solver_prefix  = sp_recv

    # ── Stage 3: Solver runs, produces feedback and logits ───────────────────
    if rank == rs:
        pre_s, post_s = build_solver_prompt(question)
        solver_input, solver_am = _build_input_embeds(
            solver_tok, solver_mdl.get_input_embeddings(),
            pre_s, post_s, solver_prefix, device, dtype,
        )
        solver_h = latent_rollout(solver_mdl, solver_inner, solver_input, solver_am, latent_steps, dtype)
        if shared_link is not None:
            feedback = shared_link(solver_h, src=3, dst=1)
        else:
            feedback = outer_31(solver_h)
        lm_head          = solver_mdl.lm_head
        solver_logits    = lm_head(solver_prefix.to(lm_head.weight.dtype))
        solver_input_ids = solver_tok(post_s, return_tensors="pt",
                                      add_special_tokens=False).to(device)["input_ids"]
        out["feedback"]         = feedback
        out["solver_logits"]    = solver_logits
        out["solver_input_ids"] = solver_input_ids

    return out


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
        planner_inner = load_inner_adapter(PLANNER_REPO, PLANNER_DIM, device, dtype)
        for p in planner_inner.parameters(): p.requires_grad_(False)

    if rank == rc:
        if rank == 0: print("Loading Critic...")
        critic_mdl, critic_tok = load_model_and_tokenizer(CRITIC_REPO, device, dtype)
        critic_inner = load_inner_adapter(CRITIC_REPO, CRITIC_DIM, device, dtype)
        for p in critic_inner.parameters(): p.requires_grad_(False)

    if rank == rs:
        if rank == 0: print("Loading Solver...")
        solver_mdl, solver_tok = load_model_and_tokenizer(SOLVER_REPO, device, dtype)
        solver_inner = load_inner_adapter(SOLVER_REPO, SOLVER_DIM, device, dtype)
        for p in solver_inner.parameters(): p.requires_grad_(False)

    _barrier()

    # ── Load outer-link adapters ─────────────────────────────────────────────
    outer_dir = Path(_resolve(OUTER_REPO))
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
        if rank == rp:
            outer_12 = load_outer_adapter(
                str(outer_dir / "Planner-Critic-Outerlink(math).pt"),
                in_dim=PLANNER_DIM, out_dim=CRITIC_DIM, device=device, dtype=dtype,
            ).train()

        if rank == rc:
            outer_23 = load_outer_adapter(
                str(outer_dir / "Critic-Solver-Outerlink(math).pt"),
                in_dim=CRITIC_DIM, out_dim=SOLVER_DIM, device=device, dtype=dtype,
            ).train()

        if rank == rs:
            outer_31 = load_outer_adapter(
                str(outer_dir / "Solver-Planner-Outerlink(math).pt"),
                in_dim=SOLVER_DIM, out_dim=PLANNER_DIM, device=device, dtype=dtype,
            ).train()

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

    # ── Dataset (all ranks load, each picks the same sample via shared seed) ─
    if rank == 0:
        print(f"Loading MATH-500 ({cfg.n_samples} problems)...")
    problems = load_math500(n_samples=cfg.n_samples, seed=cfg.seed)
    if rank == 0:
        print(f"  Loaded {len(problems)} problems.")

    _barrier()

    # ── Training ─────────────────────────────────────────────────────────────
    losses: List[float] = []
    monitor = GradMonitor(cfg.n_rounds) if rank == rp else None
    rng     = np.random.default_rng(cfg.seed)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Training outer-links  —  {cfg.n_rounds} recursion round(s)"
              f"  world={world}")
        print(f"{'='*70}")

    for step in range(cfg.steps):
        question, _ = problems[rng.integers(len(problems))]

        # ── Forward: all n rounds (paper Eq.6: L_out = CE(S^n(...S^1(x)), y)) ─
        # Each rank runs only its own agent.  Cross-rank data is sent/recv'd as
        # plain copies; the receiving rank creates a requires_grad leaf so its
        # local graph is differentiable.  Backward is handled explicitly below.
        pipeline = _is_dist() and (rp != rc or rc != rs)

        feedback_prefix = None    # outer_31 output fed back to Planner each round
        rounds = []               # list of per-round output dicts

        for r in range(cfg.n_rounds):
            out = forward_one_round(
                planner_mdl, planner_tok, planner_inner,
                critic_mdl,  critic_tok,  critic_inner,
                solver_mdl,  solver_tok,  solver_inner,
                outer_12, outer_23, outer_31,
                question=question,
                feedback_prefix=feedback_prefix,
                latent_steps=cfg.latent_steps,
                device=device,
                dtype=dtype,
                monitor=monitor,
                round_idx=r,
                shared_link=shared_link,
            )
            rounds.append(out)

            if r < cfg.n_rounds - 1:
                # Send outer_31 feedback from rs to rp for the next round.
                # Barrier-separated so all ranks see sends/recvs in the same order.
                if pipeline and rs != rp:
                    if rank == rs:
                        _send_tensor(out["feedback"], dst=rp)
                    if rank == rp:
                        feedback_prefix = _recv_tensor(src=rs, device=device, dtype=dtype,
                                                       requires_grad=True,
                                                       shape=(1, cfg.latent_steps, PLANNER_DIM))
                        out["fb_recv"] = feedback_prefix
                else:
                    feedback_prefix = out.get("feedback")

        # ── Loss: CE on final-round Solver output (paper Eq.6) ───────────────
        last = rounds[-1]
        loss = None
        if rank == rs:
            solver_logits    = last["solver_logits"]
            solver_input_ids = last["solver_input_ids"]
            if solver_logits is not None:
                L          = solver_logits.size(1)
                target_ids = solver_input_ids[:, 1:L + 1]
                n          = min(L, target_ids.size(1))
                loss = F.cross_entropy(
                    solver_logits[:, :n].reshape(-1, solver_logits.size(-1)),
                    target_ids[:, :n].reshape(-1),
                ) if n > 0 else F.cross_entropy(
                    solver_logits.reshape(-1, solver_logits.size(-1)),
                    solver_input_ids[:, :solver_logits.size(1)].reshape(-1),
                )

        # ── Backward: explicit, barrier-separated pipeline grad exchange ──────
        #
        # The graph on each rank is local.  Gradients cross rank boundaries via
        # explicit send/recv, one boundary at a time, with a barrier ensuring
        # both sides of each exchange are ready before the next step.
        #
        # Per round (in reverse), the sequence is:
        #   Step A  rs: loss.backward() / feedback.backward(g_fb)
        #             → populates sp_recv.grad (grad w.r.t. solver_prefix input)
        #   BARRIER
        #   Step B  rs→rc: send sp_recv.grad  |  rc: recv → backward solver_prefix_rc
        #             → populates cp_recv.grad (grad w.r.t. critic_prefix input)
        #   BARRIER
        #   Step C  rc→rp: send cp_recv.grad  |  rp: recv → backward critic_prefix
        #             → populates critic_prefix.grad on rp (used by optimizer)
        #             → if r>0: also populates fb_recv.grad via autograd
        #   BARRIER
        #   Step D  (if r>0) rp→rs: send fb_recv.grad  |  rs: recv for next round

        if optimizer is not None:
            optimizer.zero_grad()

        if not pipeline:
            # Single-GPU: everything is in one graph, one backward call suffices.
            if loss is not None:
                loss.backward()
        else:
            for r in range(cfg.n_rounds - 1, -1, -1):
                o = rounds[r]

                # Step A ── rs: backward through local graph ───────────────────
                # No barrier needed: rs acts alone, B/C P2P below synchronise.
                if rank == rs:
                    if r == cfg.n_rounds - 1:
                        if loss is not None:
                            loss.backward()
                    else:
                        g_fb = rounds[r + 1].get("g_fb_for_prev")
                        fb   = o.get("feedback")
                        if fb is not None and g_fb is not None:
                            fb.backward(g_fb)

                # Step B ── rs→rc grad (P2P synchronises rs and rc; rp idles) ─
                if rank == rs:
                    sp_recv = o.get("sp_recv")
                    g = sp_recv.grad if (sp_recv is not None and sp_recv.grad is not None) \
                        else torch.zeros((1, cfg.latent_steps, SOLVER_DIM), dtype=dtype, device=device)
                    _send_tensor(g, dst=rc)

                if rank == rc:
                    sv_pfx = o.get("solver_prefix_rc")
                    g = _recv_tensor(src=rs, device=device, dtype=dtype,
                                     shape=(1, cfg.latent_steps, SOLVER_DIM))
                    if sv_pfx is not None:
                        sv_pfx.backward(g)

                # Step C ── rc→rp grad (P2P synchronises rc and rp; rs idles) ─
                if rank == rc:
                    cp_recv = o.get("cp_recv")
                    g = cp_recv.grad if (cp_recv is not None and cp_recv.grad is not None) \
                        else torch.zeros((1, cfg.latent_steps, CRITIC_DIM), dtype=dtype, device=device)
                    _send_tensor(g, dst=rp)

                if rank == rp:
                    cr_pfx = o.get("critic_prefix")
                    g = _recv_tensor(src=rc, device=device, dtype=dtype,
                                     shape=(1, cfg.latent_steps, CRITIC_DIM))
                    if cr_pfx is not None:
                        cr_pfx.backward(g)

                # Step D ── rp→rs feedback grad (only when r>0) ───────────────
                # Both sides execute before the loop's next iteration; no barrier needed.
                if r > 0 and pipeline and rs != rp:
                    if rank == rp:
                        fb_recv = rounds[r - 1].get("fb_recv")
                        g = fb_recv.grad if (fb_recv is not None and fb_recv.grad is not None) \
                            else torch.zeros((1, cfg.latent_steps, PLANNER_DIM), dtype=dtype, device=device)
                        _send_tensor(g, dst=rs)
                    if rank == rs:
                        o["g_fb_for_prev"] = _recv_tensor(src=rp, device=device, dtype=dtype,
                                                          shape=(1, cfg.latent_steps, PLANNER_DIM))

        # One barrier after all backward rounds to sync before optimizer step.
        _barrier()

        # Flattened allreduce for SharedRecursiveLink: pack all grads into one
        # buffer, allreduce once, unpack. Much cheaper than per-parameter calls.
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
            torch.nn.utils.clip_grad_norm_(local_params, max_norm=1.0)
            optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if rank == rp and monitor is not None:
            monitor.commit_step()
        if _is_dist():
            _sync_norms_all_ranks(monitor, cfg.n_rounds)

        # Broadcast loss scalar from rank_solver to rank 0 for printing
        if _is_dist():
            loss_val = torch.tensor(
                [loss.item() if loss is not None else float("nan")],
                dtype=torch.float32, device=device,
            )
            dist.broadcast(loss_val, src=rs)
            loss_scalar = loss_val.item()
        else:
            loss_scalar = loss.item() if loss is not None else float("nan")

        if rank == 0:
            losses.append(loss_scalar)

            if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1:
                if monitor is not None and monitor.norms[0]:
                    norms_str = "  ".join(
                        f"r{r+1}:{monitor.norms[r][-1]:.3e}"
                        for r in range(cfg.n_rounds)
                    )
                    print(f"  step {step:4d}  loss={loss_scalar:.4f}"
                          f"  grad_norms [{norms_str}]")
                else:
                    print(f"  step {step:4d}  loss={loss_scalar:.4f}")

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
    p.add_argument("--latent_steps", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
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
