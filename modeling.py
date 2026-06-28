import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

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
      - Per-agent in/out linear projections to/from a shared latent space.
      - RoAE: rotary position encoding per agent index so the shared MoE core
        can distinguish "planner→critic" from "solver→planner", etc.
      - Soft-MoE with n_experts experts dispatched by an RoAE-conditioned router.
      - ReZero gate (alpha): starts as a pure skip connection.
      - Direct skip_proj in native hidden spaces for a gradient highway.

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

        self.in_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(h, d) for h in hidden_dims]
        )
        self.out_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(d, h) for h in hidden_dims]
        )
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

        self._rope_cache: dict = {}

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
        return skip + self.alpha * self.out_proj[dst](h2)


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
    module.load_state_dict(sd, strict=True)
    module.to(device=device, dtype=dtype).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module

