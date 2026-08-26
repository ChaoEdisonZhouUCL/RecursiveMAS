import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download



INNER_ADAPTER_TYPE = "ln_res_adapter"
OUTER_ADAPTER_TYPE = "outer_ln_res_adapter"


def infer_inner_adapter_type_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback: str = INNER_ADAPTER_TYPE,
) -> str:
    keys = set(state_dict.keys())
    expected = {
        "proj1.weight",
        "proj1.bias",
        "proj2.weight",
        "proj2.bias",
        "pre_ln.weight",
        "pre_ln.bias",
        "post_ln.weight",
        "post_ln.bias",
    }
    if not expected.issubset(keys):
        raise ValueError(
            "Release code only supports ln_res_adapter inner weights; "
            f"got keys={sorted(keys)}"
        )
    return normalize_inner_adapter_type(fallback)


def infer_outer_adapter_type_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback: str = OUTER_ADAPTER_TYPE,
) -> str:
    keys = set(state_dict.keys())
    required_prefixes = ("ln_source.", "ln_target.", "residual_proj.")
    if not all(any(k.startswith(prefix) for k in keys) for prefix in required_prefixes):
        raise ValueError(
            "Release code only supports outer_ln_res_adapter outer weights; "
            f"got keys={sorted(keys)}"
        )
    return normalize_outer_adapter_type(fallback)


def _maybe_build_plain_model_view(model_dir: str) -> str:
    path = Path(model_dir)
    if not path.is_dir():
        return model_dir

    # Released repos colocate adapter files with full-model weights. Recent
    # transformers versions may treat any directory containing adapter_config.json
    # as a PEFT adapter repo and fail before loading the full model. Build a
    # lightweight view that exposes only the base-model/tokenizer files.
    if not (path / "adapter_config.json").is_file():
        return str(path)
    if not ((path / "config.json").is_file() and any(path.glob("model*.safetensors"))):
        return str(path)

    view_dir = path / "_plain_model_view"
    if view_dir.is_dir():
        return str(view_dir)

    view_dir.mkdir(parents=True, exist_ok=True)
    excluded_names = {
        "adapter_config.json",
        "innerlink_config.json",
        "README.md",
    }
    excluded_suffixes = (".pt",)
    for item in path.iterdir():
        if item.name == view_dir.name:
            continue
        if item.name in excluded_names:
            continue
        if item.name.startswith("adapter("):
            continue
        if item.suffix in excluded_suffixes:
            continue
        target = view_dir / item.name
        if target.exists():
            continue
        try:
            target.symlink_to(item.resolve())
        except OSError:
            if item.is_file():
                target.write_bytes(item.read_bytes())
            elif item.is_dir():
                target.symlink_to(item.resolve(), target_is_directory=True)
    return str(view_dir)


def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return _maybe_build_plain_model_view(model_name_or_path)
    try:
        resolved = snapshot_download(model_name_or_path, local_files_only=True)
        return _maybe_build_plain_model_view(resolved)
    except Exception:
        return model_name_or_path


def normalize_inner_adapter_type(adapter_type: str) -> str:
    if adapter_type != INNER_ADAPTER_TYPE:
        raise ValueError(f"Unsupported adapter_type: {adapter_type}")
    return INNER_ADAPTER_TYPE


def normalize_outer_adapter_type(adapter_type: str) -> str:
    if adapter_type != OUTER_ADAPTER_TYPE:
        raise ValueError(f"Unsupported outer adapter_type: {adapter_type}")
    return OUTER_ADAPTER_TYPE


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_inner_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        return self.post_ln(out)


def _module_param_dtype(module: nn.Module) -> torch.dtype:
    """The dtype this module's weights are stored in.

    A trainable link may be held in float32 while the frozen backbones around it
    run in bfloat16 — see `run_at_param_dtype`.  Falls back to float32 for a
    module with no parameters.
    """
    for prm in module.parameters():
        return prm.dtype
    return torch.float32


def run_at_param_dtype(module: nn.Module, x: torch.Tensor, fn) -> torch.Tensor:
    """Evaluate `fn(x)` in the module's parameter dtype, return in x's dtype.

    Lets a link keep float32 weights, gradients and optimizer state inside an
    otherwise-bfloat16 pipeline without any call site having to know: the tensors
    crossing the module boundary keep the ambient dtype, so the P2P sends between
    pipeline stages are unchanged.

    This matters for more than round-off.  A recursive link accumulates one
    gradient contribution per round into a single `.grad` buffer, and those
    contributions span many orders of magnitude (GRADIENT_ROUND_PROFILE.md §2-3).
    In bfloat16 — 8 mantissa bits, eps 7.8e-3 — adding a contribution 1e-3 or more
    below the running total discards most of it, so the early rounds are dropped
    from the update before the optimizer ever sees them.  float32 (eps 1.2e-7)
    keeps them.  See ROUND_GRADIENT_WEIGHTING.md.
    """
    in_dtype = x.dtype
    p_dtype = _module_param_dtype(module)
    if in_dtype == p_dtype:
        return fn(x)
    return fn(x.to(p_dtype)).to(in_dtype)


class CrossModelAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_outer_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_dim = out_dim * 2
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.ln_source = nn.LayerNorm(in_dim)
        self.ln_target = nn.LayerNorm(out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_at_param_dtype(self, x, self._forward)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = out + self.residual_proj(x)
        return self.ln_target(out)


# ── SharedRecursiveLink (shared_roae mode) ────────────────────────────────────

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
      - Per-agent in_proj  (hidden_dims[src] → shared_dim, with bias).
      - Per-agent skip_proj (hidden_dims[src] → shared_dim, no bias) — gradient
        highway that lives entirely in latent space, shared across all dst agents.
      - RoAE: rotary position encoding per agent index so the shared MoE core
        can distinguish "planner→critic" from "solver→planner", etc.
      - Soft-MoE with n_experts experts dispatched by an RoAE-conditioned router.
      - ReZero gate (alpha): MoE output starts near zero, skip dominates early.
      - Per-agent out_proj (shared_dim → hidden_dims[dst]) applied to the sum
        (skip + alpha * MoE), so a single out_proj handles all src agents.

    Forward: out_proj[dst]( skip_proj[src](h) + alpha * MoE(in_proj[src](h)) )

    This moves the skip connection into latent space (N projections, not N×N),
    reducing parameters while keeping a gradient highway through every transition.

    Agents are 1-indexed: 1=Planner, 2=Critic, 3=Solver.
    """

    N_AGENTS = 3

    def __init__(self, hidden_dims: List[int], shared_dim: int,
                 n_experts: int = 4, expert_dim_divisor: int = 4):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.shared_dim  = shared_dim
        self.n_experts   = n_experts
        N = len(hidden_dims)
        d = shared_dim

        # index 0 unused (Identity placeholder keeps 1-based agent indexing)
        self.in_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(h, d) for h in hidden_dims]
        )
        self.out_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(d, h) for h in hidden_dims]
        )
        # One projection per source agent: hidden_dims[src] → d, no bias.
        # Shared across all dst agents; dst-specificity comes from out_proj.
        self.skip_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(h, d, bias=False) for h in hidden_dims]
        )
        for proj in self.skip_proj[1:]:
            nn.init.orthogonal_(proj.weight)

        self.router    = nn.Linear(3 * d, n_experts, bias=False)
        exp_hidden = max(1, d // expert_dim_divisor)
        self.expert_W1 = nn.Parameter(torch.empty(n_experts, d, exp_hidden))
        self.expert_b1 = nn.Parameter(torch.zeros(n_experts, exp_hidden))
        self.expert_W2 = nn.Parameter(torch.zeros(n_experts, exp_hidden, d))
        self.expert_b2 = nn.Parameter(torch.zeros(n_experts, d))
        nn.init.kaiming_uniform_(self.expert_W1, a=math.sqrt(5))

        self.act   = nn.GELU()
        self.ln_in = nn.LayerNorm(d)
        self.alpha = nn.Parameter(torch.tensor(1e-3))
        # Round-skip gate: blends round-0 cyclic output into every subsequent
        # round's feedback prefix.  ReZero-init so early training is identical
        # to the no-skip baseline; the model learns how much to blend in.
        self.beta  = nn.Parameter(torch.tensor(1e-3))

        self._rope_cache: dict = {}

    def _get_rope(self, device, dtype):
        key = (device, dtype)
        if key not in self._rope_cache:
            cos, sin = _build_rope_cache(self.N_AGENTS, self.shared_dim, device)
            self._rope_cache[key] = (cos.to(dtype), sin.to(dtype))
        return self._rope_cache[key]

    def _latent_core(self, h: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        """Transition output in the shared latent space (before out_proj).

        h: (B, T, hidden_dims[src-1]);  src/dst are 1-indexed agent indices.
        Returns (B, T, shared_dim): skip + alpha * MoE.
        """
        # Skip connection in latent space: hidden_dims[src] → d
        skip = self.skip_proj[src](h)

        x = self.in_proj[src](h)
        cos_tab, sin_tab = self._get_rope(x.device, x.dtype)
        x_src = _rope_rotate(x, cos_tab[src], sin_tab[src])
        x_dst = _rope_rotate(x, cos_tab[dst], sin_tab[dst])

        gate_input = torch.cat([x_src, x_dst, x_dst - x_src], dim=-1)
        logits = self.router(gate_input.mean(dim=1))
        w = torch.softmax(logits / 2.0, dim=-1).unsqueeze(1)

        normed = self.ln_in(x_src)
        W1 = self.expert_W1.to(normed.dtype)
        b1 = self.expert_b1.to(normed.dtype)
        W2 = self.expert_W2.to(normed.dtype)
        b2 = self.expert_b2.to(normed.dtype)
        h1 = self.act(torch.einsum("btd,kde->btke", normed, W1) + b1)
        h2 = (torch.einsum("btke,ked->btd", h1 * w.unsqueeze(-1), W2)
              + (w @ b2).squeeze(1).unsqueeze(1))
        return skip + self.alpha * h2

    def _project_out(self, x: torch.Tensor, dst: int) -> torch.Tensor:
        """Project a shared-latent tensor to dst's hidden dim (variants override)."""
        return self.out_proj[dst](x)

    def forward(self, h: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        """h: (B, T, hidden_dims[src-1]);  src/dst are 1-indexed agent indices."""
        # Combine in latent space, then project to dst hidden dim.  Runs in the
        # link's own parameter dtype so float32 link weights work unchanged inside
        # a bfloat16 pipeline; the returned tensor keeps h's dtype.
        return run_at_param_dtype(
            self, h, lambda t: self._project_out(self._latent_core(t, src, dst), dst))


def load_shared_recursive_link(
    ckpt_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> "SharedRecursiveLink":
    """
    Load a SharedRecursiveLink from a checkpoint directory produced by
    train_outerlinks_math500.py (shared_roae mode).

    The directory must contain:
        shared_recursive_link.pt   — state dict
        shared_roae_config.json    — architecture hyperparams
    """
    config_path = Path(ckpt_dir) / "shared_roae_config.json"
    weights_path = Path(ckpt_dir) / "shared_recursive_link.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"shared_roae_config.json not found in {ckpt_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"shared_recursive_link.pt not found in {ckpt_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    module = SharedRecursiveLink(
        hidden_dims=cfg["hidden_dims"],
        shared_dim=cfg["shared_dim"],
        n_experts=cfg["n_experts"],
        expert_dim_divisor=cfg.get("expert_dim_divisor", 4),
    )
    sd = torch.load(str(weights_path), map_location="cpu")
    # Old checkpoints (before round-skip) lack the beta parameter; fill with
    # the same ReZero init so behaviour is identical to the pre-skip baseline.
    if "beta" not in sd:
        sd["beta"] = torch.tensor(1e-3)
    module.load_state_dict(sd, strict=True)
    module.to(device=device, dtype=dtype).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class SharedRecursiveStateLink(SharedRecursiveLink):
    """
    SharedRecursiveLink plus a shared recursive state z that bypasses one
    complete MAS recursion round.

    Pairwise transitions (planner→critic, critic→solver, agent self-loops)
    behave exactly like SharedRecursiveLink.  The solver→planner round
    boundary is replaced by a residual update of a persistent state in the
    shared latent space:

        f^(r)   = skip_proj[3](h_solver) + alpha * MoE(...)   (latent feedback)
        z^(1)   = f^(1)                                       (round-1 init)
        z^(r+1) = z^(r) + gamma * f^(r+1)
        p^(r+1) = out_proj[1](z^(r+1))                        (planner prefix)

    so dz^(r+1)/dz^(r) = I + gamma * J_F.  With gamma ReZero-initialised
    (~1e-3), the product prod_r (I + gamma J_r) over R recursion rounds stays
    close to identity while gamma * ||J_r|| is small, keeping cross-round
    gradients stable; the model learns how far to move away from identity.

    Usage per round (replaces `shared_link(solver_h, src=3, dst=1)`):

        feedback, state = link.round_feedback(solver_h, state)   # state=None on round 1
    """

    def __init__(self, hidden_dims: List[int], shared_dim: int,
                 n_experts: int = 4, expert_dim_divisor: int = 4,
                 gamma_init: float = 1e-3):
        super().__init__(hidden_dims, shared_dim,
                         n_experts=n_experts, expert_dim_divisor=expert_dim_divisor)
        # Round-bypass gate: near-identity round-to-round Jacobian at init.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Populated by round_feedback(); diagnostics only, not part of the graph.
        self.last_f_norm = float("nan")
        self.last_state_norm = float("nan")
        self.last_write_ratio = float("nan")
        # One (|f|, |z|, write_ratio) triple per round_feedback() call, in round
        # order.  The training loop drains it each logging step.
        self.write_log: List[Tuple[float, float, float]] = []

    def round_feedback(
        self,
        solver_h: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solver→planner round boundary with residual state update.

        solver_h: (B, T, hidden_dims[2]) solver latent rollout output.
        state:    (B, T, shared_dim) recursive state z from the previous
                  round, or None on round 1.

        Returns (feedback_prefix, new_state):
          feedback_prefix: (B, T, hidden_dims[0]) planner prefix for the next
                           round — out_proj[1](z_new).
          new_state:       (B, T, shared_dim) z to carry into the next round.
        """
        # The recursive state is kept in the link's parameter dtype for the whole
        # chain, never the ambient pipeline dtype.  The update is z + gamma*f with
        # gamma ReZero-initialised at ~1e-3, which sits below bfloat16's eps of
        # 7.8e-3.  Measured at the production state shape (1x48x1024 = 49,152
        # coords), bfloat16 lands 48-81% of the correct norm change while moving
        # only 7-22% of the coordinates: z receives a sparsified, biased version of
        # the update rather than the update.  float32 moves all of them.
        # See ROUND_GRADIENT_WEIGHTING.md.
        in_dtype = solver_h.dtype
        p_dtype = _module_param_dtype(self)
        f = self._latent_core(solver_h.to(p_dtype), src=3, dst=1)
        # Diagnostic (no autograd effect): how much this round actually writes into
        # the state.  gamma gates every round after the first, so `write_ratio` is
        # the forward-pass counterpart of the per-round gradient profile -- if it is
        # ~1e-3 then the state is still, to three significant figures, round 1's.
        with torch.no_grad():
            prev_norm = None if state is None else float(state.to(p_dtype).norm())
            self.last_f_norm = float(f.norm())
        state = f if state is None else state.to(p_dtype) + self.gamma * f
        with torch.no_grad():
            self.last_state_norm = float(state.norm())
            self.last_write_ratio = (
                float("nan") if prev_norm is None
                else float((self.gamma * f).norm()) / max(prev_norm, 1e-30)
            )
            if len(self.write_log) < 64:   # bounded; drained by the training loop
                self.write_log.append(
                    (self.last_f_norm, self.last_state_norm, self.last_write_ratio)
                )
        return self._project_out(state, 1).to(in_dtype), state


def load_shared_recursive_state_link(
    ckpt_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> "SharedRecursiveStateLink":
    """
    Load a SharedRecursiveStateLink from a checkpoint directory.

    Accepts both native shared_state checkpoints and plain shared_roae
    checkpoints (warm start): missing beta/gamma are filled with their
    ReZero inits, which reproduces the parent's behaviour up to the
    round-boundary state rewiring.
    """
    config_path = Path(ckpt_dir) / "shared_roae_config.json"
    weights_path = Path(ckpt_dir) / "shared_recursive_link.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"shared_roae_config.json not found in {ckpt_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"shared_recursive_link.pt not found in {ckpt_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    module = SharedRecursiveStateLink(
        hidden_dims=cfg["hidden_dims"],
        shared_dim=cfg["shared_dim"],
        n_experts=cfg["n_experts"],
        expert_dim_divisor=cfg.get("expert_dim_divisor", 4),
        gamma_init=cfg.get("gamma_init", 1e-3),
    )
    sd = torch.load(str(weights_path), map_location="cpu")
    if "beta" not in sd:
        sd["beta"] = torch.tensor(1e-3)
    if "gamma" not in sd:
        sd["gamma"] = torch.tensor(float(cfg.get("gamma_init", 1e-3)))
    module.load_state_dict(sd, strict=True)
    module.to(device=device, dtype=dtype).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module



class SharedRecursiveTiedLink(SharedRecursiveLink):
    """
    SharedRecursiveLink with tied source/destination projections: O_i = S_i^T.

    The parent decodes the shared latent with an independently *randomly*
    initialised out_proj, so the init-time transition Jacobian is
    J_{s->d}^(0) ~= O_d S_s — even with orthogonal S_s, the random O_d can
    shrink or distort gradients.  Here the decoder is tied to the (semi-)
    orthogonal skip encoder of the destination agent:

        S_i : R^{h_i} -> R^d   (skip_proj, orthogonal init)
        O_i = S_i^T : R^d -> R^{h_i}   (no independent parameters)

    so a same-agent round trip is O_i S_i = S_i^T S_i — exactly the identity
    when d >= h_i, and a rank-d orthogonal projector with unit singular values
    in the retained subspace when d < h_i.  Cross-agent transfer starts at
    R_{i->j}^(0) = S_j^T S_i, whose singular values are likewise close to one
    within the retained subspace — preserving both forward norms and backward
    gradients at init.  A free zero-init bias per destination agent keeps the
    decoder affine without touching the Jacobian.

    State dict: no out_proj weights (Identity placeholders); adds out_bias.
    """

    def __init__(self, hidden_dims: List[int], shared_dim: int,
                 n_experts: int = 4, expert_dim_divisor: int = 4):
        super().__init__(hidden_dims, shared_dim,
                         n_experts=n_experts, expert_dim_divisor=expert_dim_divisor)
        # Drop the independent decoders; decoding reuses skip_proj transposed.
        self.out_proj = nn.ModuleList(
            [nn.Identity() for _ in range(len(hidden_dims) + 1)]
        )
        # Free affine offset per destination agent (index dst-1), zero-init.
        self.out_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(h)) for h in hidden_dims]
        )

    def _project_out(self, x: torch.Tensor, dst: int) -> torch.Tensor:
        w = self.skip_proj[dst].weight          # (d, h_dst): S_dst
        return torch.nn.functional.linear(x, w.t(), self.out_bias[dst - 1])

    @torch.no_grad()
    def re_orthonormalize_(self) -> None:
        """Retract each skip encoder back to the Stiefel manifold (QR, float32).

        Every transition Jacobian is S_dst^T S_src — quadratic in S's singular
        values — and the latent-rollout self-loop applies S_i^T S_i once per
        latent step, so any singular value drifting above 1 during training is
        amplified as sigma^(2*latent_steps) and overflows bf16 within a few
        steps.  Call this after every optimizer step: it projects S back to
        exact (semi-)orthogonality, i.e. projected gradient descent on the
        manifold the tied design assumes.
        """
        for proj in self.skip_proj[1:]:
            W = proj.weight
            rows, cols = W.shape
            A = W.float().t() if rows < cols else W.float()   # tall (m >= n)
            Q, R = torch.linalg.qr(A, mode="reduced")
            s = torch.sign(torch.diagonal(R))                 # continuous retraction
            s[s == 0] = 1.0
            Q = Q * s
            W.copy_((Q.t() if rows < cols else Q).to(W.dtype))

    @torch.no_grad()
    def max_singular_value(self) -> float:
        """Largest singular value across the skip encoders (drift diagnostic)."""
        return max(
            torch.linalg.svdvals(proj.weight.float())[0].item()
            for proj in self.skip_proj[1:]
        )


def load_shared_recursive_tied_link(
    ckpt_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> "SharedRecursiveTiedLink":
    """
    Load a SharedRecursiveTiedLink from a checkpoint directory.

    Accepts native shared_tied checkpoints, and warm starts from plain
    shared_roae checkpoints: independent out_proj weights are dropped (the
    tied decoder replaces them) and missing out_bias/beta entries are filled
    with their inits.
    """
    config_path = Path(ckpt_dir) / "shared_roae_config.json"
    weights_path = Path(ckpt_dir) / "shared_recursive_link.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"shared_roae_config.json not found in {ckpt_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"shared_recursive_link.pt not found in {ckpt_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    module = SharedRecursiveTiedLink(
        hidden_dims=cfg["hidden_dims"],
        shared_dim=cfg["shared_dim"],
        n_experts=cfg["n_experts"],
        expert_dim_divisor=cfg.get("expert_dim_divisor", 4),
    )
    sd = torch.load(str(weights_path), map_location="cpu")
    sd = adapt_state_dict_for_tied_link(sd, cfg["hidden_dims"])
    module.load_state_dict(sd, strict=True)
    module.to(device=device, dtype=dtype).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def adapt_state_dict_for_tied_link(sd: dict, hidden_dims: List[int]) -> dict:
    """Convert a shared_roae/shared_tied state dict to SharedRecursiveTiedLink's
    layout: drop independent out_proj weights, carry out_proj biases into
    out_bias, fill anything still missing with its init."""
    sd = {k: v for k, v in sd.items() if not (k.startswith("out_proj.") and k.endswith(".weight"))}
    for i, h in enumerate(hidden_dims):
        key = f"out_bias.{i}"
        if key not in sd:
            legacy = sd.pop(f"out_proj.{i + 1}.bias", None)
            sd[key] = legacy if legacy is not None else torch.zeros(h)
    sd = {k: v for k, v in sd.items() if not k.startswith("out_proj.")}
    if "beta" not in sd:
        sd["beta"] = torch.tensor(1e-3)
    return sd
