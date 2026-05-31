#!/usr/bin/env python3
"""
Train the three outer-links of RecursiveMAS-Light on MATH-500 and monitor
gradient norms across recursion rounds to diagnose vanishing gradients.

Architecture (Sequential-Light):
    Planner  (Qwen3-1.7B,        d=2048)
       ↓  outer_12  (2048→2048)
    Critic   (LLaMA-3.2-1B,     d=2048)
       ↓  outer_23  (2048→1536)
    Solver   (Qwen2.5-Math-1.5B, d=1536)
       ↓  outer_31  (1536→2048)  ──► back to Planner (next round)

Multi-GPU (pipeline parallelism)
---------------------------------
Launch with torchrun:
    torchrun --standalone --nproc_per_node=3 train_outerlinks_math500.py

  rank 0 → Planner  + outer_12  (2048→2048)
  rank 1 → Critic   + outer_23  (2048→1536)
  rank 2 → Solver   + outer_31  (1536→2048)

Activations are passed between adjacent ranks via dist.send / dist.recv.
Gradients flow back through those same tensors automatically (autograd
tracks the send/recv as part of the graph via the gloo/nccl gradient hooks
built into dist.send — or via an explicit requires_grad buffer trick for
nccl, described in the code below).

Single-GPU fallback
-------------------
    python train_outerlinks_math500.py
All three agents load onto cuda:0 (or cpu).

Usage
-----
  python  train_outerlinks_math500.py --n_rounds 3 --steps 100
  torchrun --standalone --nproc_per_node=3 \\
           train_outerlinks_math500.py --n_rounds 3 --steps 100
  torchrun --standalone --nproc_per_node=3 \\
           train_outerlinks_math500.py --n_rounds 1 2 3 --steps 150
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

# ── DDP helpers (same pattern as pretrain.py) ────────────────────────────────

def _rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def _world() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def _barrier():
    if _is_dist():
        dist.barrier()

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

# ── Cross-rank tensor transfer (preserves autograd) ──────────────────────────

def _send_tensor(t: torch.Tensor, dst: int):
    """Send shape then data to dst rank."""
    shape = torch.tensor(list(t.shape), dtype=torch.long, device=t.device)
    dist.send(shape, dst=dst)
    dist.send(t.contiguous(), dst=dst)


def _recv_tensor(src: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Receive shape then data from src rank; returned tensor requires_grad=True."""
    shape_buf = torch.zeros(3, dtype=torch.long, device=device)
    dist.recv(shape_buf, src=src)
    shape = tuple(shape_buf.tolist())
    buf = torch.zeros(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src)
    # Re-attach to autograd so gradients can flow back through the pipeline.
    # The grad propagates back via a separate dist.send on the grad side
    # (handled in _pipeline_send_recv below).
    out = buf.requires_grad_(True)
    return out


def _pipeline_send_recv(
    tensor: Optional[torch.Tensor],
    src_rank: int,
    dst_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Move `tensor` from src_rank to dst_rank, preserving the autograd graph.

    On src_rank : sends the tensor data.
    On dst_rank : receives it and wraps it in a leaf tensor with requires_grad=True.
    Returns the received tensor on dst_rank, None elsewhere.

    Backward: dst_rank must call _pipeline_send_grad() and src_rank must call
    _pipeline_recv_grad() after loss.backward() to propagate gradients back.
    These are registered automatically as autograd hooks on the returned tensor.
    """
    if src_rank == dst_rank:
        return tensor  # same rank — no communication needed

    my_rank = _rank()

    if my_rank == src_rank:
        _send_tensor(tensor, dst=dst_rank)
        # Register a hook: when backward reaches the tensor on dst_rank,
        # dst_rank sends the grad back here.
        recv_buf = [None]

        def _recv_grad(grad):
            g = torch.zeros_like(tensor)
            dist.recv(g, src=dst_rank)
            recv_buf[0] = g
            return g

        tensor.register_hook(_recv_grad)
        return None

    elif my_rank == dst_rank:
        received = _recv_tensor(src=src_rank, device=device, dtype=dtype)

        def _send_grad(grad):
            if grad is None:
                grad = torch.zeros_like(received)
            dist.send(grad.contiguous(), dst=src_rank)

        received.register_hook(_send_grad)
        return received

    else:
        return None


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


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


# ── Latent rollout ────────────────────────────────────────────────────────────

def latent_rollout(
    model,
    inner_adapter,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Run `latent_steps` steps of inner-adapter auto-regression.
    Returns concatenated hidden states: (1, latent_steps, d_model).

    Backbone runs WITHOUT torch.no_grad() so activations are differentiable
    w.r.t. the outer-link prefix in input_embeds.  Backbone weights are frozen
    (requires_grad=False) so they accumulate no .grad.
    Inner adapter steps run under no_grad (frozen, constant Jacobian).
    """
    hidden_states = []
    ie = input_embeds
    am = attention_mask
    for _ in range(latent_steps):
        try:
            out = model(inputs_embeds=ie, attention_mask=am,
                        output_hidden_states=True, use_cache=False,
                        return_dict=True, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=ie, attention_mask=am,
                        output_hidden_states=True, use_cache=False,
                        return_dict=True)
        last_h = out.hidden_states[-1][:, -1, :]   # (1, d)
        hidden_states.append(last_h.unsqueeze(1))

        with torch.no_grad():
            next_emb = inner_adapter(last_h.detach()).unsqueeze(1).to(dtype)
        ie = torch.cat([ie, next_emb], dim=1)
        am = torch.cat([am, am.new_ones((am.size(0), 1))], dim=1)

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

    def sync_norms(self):
        """
        After commit_step(), broadcast norms from rank 0 (Planner) to all ranks
        so that every rank can print the same gradient statistics.
        Only called when distributed is active.
        """
        if not _is_dist():
            return
        for r in range(self.n_rounds):
            val = torch.tensor(
                [self.norms[r][-1] if self.norms[r] else float("nan")],
                dtype=torch.float32, device=f"cuda:{_rank()}"
            )
            dist.broadcast(val, src=RANK_PLANNER)
            if _rank() != RANK_PLANNER:
                self.norms[r].append(val.item())


# ── Single round (pipeline-aware) ────────────────────────────────────────────

def forward_one_round(
    # Each argument is None on ranks that don't own the agent.
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
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    One Planner→Critic→Solver round, distributed across ranks.

    Pipeline flow:
      rank_planner: planner hidden → outer_12 → send critic_prefix to rank_critic
      rank_critic:  recv critic_prefix → critic hidden → outer_23 → send solver_prefix
      rank_solver:  recv solver_prefix → solver hidden → outer_31 (feedback)
                    → lm_head(solver_prefix) for loss

    Returns (on relevant rank, else None):
      solver_h       (rank_solver)   — solver latent hidden states
      feedback       (rank_solver)   — outer_31 output for next round
      solver_logits  (rank_solver)   — lm_head(solver_prefix), in graph
      solver_input_ids (rank_solver) — target token ids
    """
    rank      = _rank()
    rp        = _pipeline_rank("planner")
    rc        = _pipeline_rank("critic")
    rs        = _pipeline_rank("solver")
    prompt    = build_prompt(question)
    pipeline  = _is_dist() and (rp != rc or rc != rs)  # True iff agents are on different ranks

    # ── Planner (rank rp) ────────────────────────────────────────────────────
    critic_prefix = None
    if rank == rp:
        embed = planner_mdl.get_input_embeddings()
        enc   = planner_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_embeds = embed(enc["input_ids"]).to(dtype)
        if feedback_prefix is not None:
            ie = torch.cat([feedback_prefix.to(dtype), base_embeds], dim=1)
            am = torch.cat([
                enc["attention_mask"].new_ones((1, feedback_prefix.size(1))),
                enc["attention_mask"],
            ], dim=1)
        else:
            ie, am = base_embeds, enc["attention_mask"]

        planner_h     = latent_rollout(planner_mdl, planner_inner, ie, am, latent_steps, dtype)
        critic_prefix = outer_12(planner_h)            # (1, L, 2048) — in graph

        if monitor is not None:
            monitor.register_output(critic_prefix, round_idx)

        if pipeline:
            _send_tensor(critic_prefix.detach(), dst=rc)
            # Register hook to receive grad back from rank_critic
            def _recv_grad_12(grad):
                g = torch.zeros_like(critic_prefix)
                dist.recv(g, src=rc)
                return g
            critic_prefix.register_hook(_recv_grad_12)

    # ── Critic (rank rc) ─────────────────────────────────────────────────────
    solver_prefix = None
    if rank == rc:
        if pipeline and rp != rc:
            # Receive critic_prefix from rank_planner
            cp_buf = _recv_tensor(src=rp, device=device, dtype=dtype)
            # Register hook to send grad back to rank_planner
            def _send_grad_12(grad):
                dist.send((grad if grad is not None else torch.zeros_like(cp_buf)).contiguous(), dst=rp)
            cp_buf.register_hook(_send_grad_12)
            critic_prefix = cp_buf
        # critic_prefix is already set if rp == rc

        critic_embed = critic_mdl.get_input_embeddings()
        enc_c        = critic_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            critic_base = critic_embed(enc_c["input_ids"]).to(dtype)
        critic_input = torch.cat([critic_prefix, critic_base], dim=1)
        critic_am    = torch.cat([
            enc_c["attention_mask"].new_ones((1, critic_prefix.size(1))),
            enc_c["attention_mask"],
        ], dim=1)

        critic_h      = latent_rollout(critic_mdl, critic_inner, critic_input, critic_am, latent_steps, dtype)
        solver_prefix = outer_23(critic_h)             # (1, L, 1536) — in graph

        if pipeline and rc != rs:
            _send_tensor(solver_prefix.detach(), dst=rs)
            def _recv_grad_23(grad):
                g = torch.zeros_like(solver_prefix)
                dist.recv(g, src=rs)
                return g
            solver_prefix.register_hook(_recv_grad_23)

    # ── Solver (rank rs) ─────────────────────────────────────────────────────
    solver_h = feedback = solver_logits = solver_input_ids = None
    if rank == rs:
        if pipeline and rc != rs:
            sp_buf = _recv_tensor(src=rc, device=device, dtype=dtype)
            def _send_grad_23(grad):
                dist.send((grad if grad is not None else torch.zeros_like(sp_buf)).contiguous(), dst=rc)
            sp_buf.register_hook(_send_grad_23)
            solver_prefix = sp_buf
        # solver_prefix already set if rc == rs

        solver_embed   = solver_mdl.get_input_embeddings()
        enc_s          = solver_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            solver_base = solver_embed(enc_s["input_ids"]).to(dtype)
        solver_input   = torch.cat([solver_prefix, solver_base], dim=1)
        solver_am      = torch.cat([
            enc_s["attention_mask"].new_ones((1, solver_prefix.size(1))),
            enc_s["attention_mask"],
        ], dim=1)

        solver_h       = latent_rollout(solver_mdl, solver_inner, solver_input, solver_am, latent_steps, dtype)
        feedback       = outer_31(solver_h)            # (1, L, 2048) — feedback for next round

        # Differentiable loss: frozen lm_head applied to solver_prefix (in graph)
        lm_head        = solver_mdl.lm_head
        solver_logits  = lm_head(solver_prefix.to(lm_head.weight.dtype))  # (1, L, V)
        solver_input_ids = enc_s["input_ids"]

    return solver_h, feedback, solver_logits, solver_input_ids


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(cfg, device: torch.device) -> Dict:
    rank  = _rank()
    world = _world()
    dtype = torch.float32

    if rank == 0:
        print(f"\nDevice: {device}  |  world={world}  |  n_rounds={cfg.n_rounds}"
              f"  latent_steps={cfg.latent_steps}  steps={cfg.steps}")

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

    # ── Load outer-link adapters (each rank loads only its own) ─────────────
    outer_dir = Path(_resolve(OUTER_REPO))
    outer_12 = outer_23 = outer_31 = None

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
        # All ranks draw the same question (same rng state from same seed)
        question, _ = problems[rng.integers(len(problems))]

        feedback_prefix = None

        for r in range(cfg.n_rounds):
            solver_h, feedback, solver_logits, solver_input_ids = forward_one_round(
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
            )

            if r < cfg.n_rounds - 1:
                # Paper §4 Eq.(6): full computation graph preserved across rounds.
                # If feedback is on rank_solver and next round needs it on rank_planner,
                # send it across.
                if _is_dist() and rs != rp:
                    if rank == rs:
                        _send_tensor(feedback.detach(), dst=rp)
                    if rank == rp:
                        fb_buf = _recv_tensor(src=rs, device=device, dtype=dtype)
                        feedback_prefix = fb_buf
                else:
                    feedback_prefix = feedback

        # ── Loss (rank_solver computes, others skip) ─────────────────────────
        loss = None
        if rank == rs and solver_logits is not None:
            L           = solver_logits.size(1)
            target_ids  = solver_input_ids[:, 1:L + 1]
            n           = min(L, target_ids.size(1))
            if n > 0:
                loss = F.cross_entropy(
                    solver_logits[:, :n].reshape(-1, solver_logits.size(-1)),
                    target_ids[:, :n].reshape(-1),
                )
            else:
                loss = F.cross_entropy(
                    solver_logits.reshape(-1, solver_logits.size(-1)),
                    solver_input_ids[:, :solver_logits.size(1)].reshape(-1),
                )

        # ── Backward + optimiser step ────────────────────────────────────────
        if optimizer is not None:
            optimizer.zero_grad()

        # Each rank calls backward on whatever loss it owns.
        # Gradients flow back through the pipeline via the registered hooks.
        if loss is not None:
            loss.backward()

        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(local_params, max_norm=1.0)
            optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if rank == rp and monitor is not None:
            monitor.commit_step()
            if _is_dist():
                monitor.sync_norms()

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
    return p.parse_args()


def main():
    cfg = parse_args()

    # ── DDP / device setup (same pattern as pretrain.py) ────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = _rank()

    round_list = cfg.n_rounds
    sweep      = len(round_list) > 1

    if not sweep:
        cfg.n_rounds = round_list[0]
        results      = run_training(cfg, device)
        if rank == 0:
            analyse(results, cfg)
            plot_single(results, cfg, out_path=f"{cfg.out_prefix}_r{cfg.n_rounds}.png")
    else:
        sweep_results = []
        for n in round_list:
            cfg_n         = copy.copy(cfg)
            cfg_n.n_rounds = n
            results       = run_training(cfg_n, device)
            if rank == 0:
                analyse(results, cfg_n)
                plot_single(results, cfg_n, out_path=f"{cfg.out_prefix}_r{n}.png")
            sweep_results.append((n, results))

        if rank == 0:
            tail = max(1, cfg.steps // 5)
            print(f"\n{'='*80}")
            print("Sweep summary  (ratio = grad_norm_round1 / grad_norm_roundN)")
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
            plot_sweep(sweep_results, out_path=f"{cfg.out_prefix}_sweep.png")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
