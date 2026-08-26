"""
Reconstruct the two reference gradient-profile figures whose PNGs were
overwritten (plot_single writes to a fixed path per mode/n_rounds, so the
200-step sweep runs clobbered the 3000-step reference runs).

Sources, both fully preserved:
  * loss curve            -- resume_global.pt['losses'] in the run's checkpoint
  * per-round grad norms  -- the every-300-step tables in the SLURM log
  * final averages        -- the analyse() block at the end of the SLURM log

The middle panel is at 300-step resolution (that is what the log records); the
originals were per-step.  Left and right panels are exact.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "slurm_logs"
CKPT = ROOT / "outputs" / "checkpoints"

RUNS = [
    dict(job="15518361", mode="shared_roae",
         ckpt="outerlink_grad_shared_roae_r3_noskip_20260811_145707",
         lr="0.001", out="reference_shared_roae_r3_lr1e-3_s3000.png"),
    dict(job="15518359", mode="shared_state",
         ckpt="outerlink_grad_shared_state_r3_noskip_20260811_145708",
         lr="0.0001", out="reference_shared_state_r3_lr1e-4_s3000.png"),
]

STEP_RE = re.compile(r"^\s*step\s+(\d+)\s+loss=")
G12_RE = re.compile(r"grad\[\s*12\]\s*\[(.*?)\]")
FINAL_RE = re.compile(r"link\s+12:\s+(.*?)\s+ratio")
CELL_RE = re.compile(r"r(\d+):([0-9.eE+-]+)")


def parse_log(job):
    """-> (steps, {round: [norms]}, {round: final_avg})"""
    text = (LOGS / f"job-{job}" / f"job-{job}.out").read_text(errors="ignore")
    steps, series, pending = [], {}, None
    for line in text.splitlines():
        m = STEP_RE.match(line)
        if m:
            pending = int(m.group(1))
            continue
        m = G12_RE.search(line)
        if m and pending is not None:
            cells = CELL_RE.findall(m.group(1))
            steps.append(pending)
            for r, v in cells:
                series.setdefault(int(r) - 1, []).append(float(v))
            pending = None
    final = {}
    m = FINAL_RE.search(text)
    if m:
        for r, v in CELL_RE.findall(m.group(1)):
            final[int(r) - 1] = float(v)
    return steps, series, final


def build(run):
    steps, series, final = parse_log(run["job"])
    losses = torch.load(CKPT / run["ckpt"] / "step_3000" / "resume_global.pt",
                        map_location="cpu", weights_only=False)["losses"]
    n_rounds = len(final) or len(series)
    cmap = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"RecursiveMAS-Light  |  outer-link training  [RECONSTRUCTED]\n"
        f"mode={run['mode']}  n_rounds={n_rounds}  latent_steps=48  "
        f"steps=3000  lr={run['lr']}  world=3   (job {run['job']})",
        fontsize=10,
    )

    ax = axes[0]
    w = max(1, len(losses) // 20)
    ax.plot(losses, alpha=0.35, color="steelblue", linewidth=0.8)
    ax.plot(range(w - 1, len(losses)),
            np.convolve(losses, np.ones(w) / w, mode="valid"),
            color="steelblue", linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("CE loss")
    ax.set_title("Training loss  (exact)"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in sorted(series):
        ax.plot(steps, series[r], marker="o", markersize=3.5, linewidth=1.3,
                color=cmap(r / max(1, n_rounds - 1)), label=f"Round {r+1}")
    ax.set_xlabel("Step"); ax.set_ylabel("Grad norm (outer_12 output)")
    ax.set_title("Gradient norm per round  (300-step samples)")
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    vals = [final[r] for r in sorted(final)]
    colors = [cmap(r / max(1, n_rounds - 1)) for r in sorted(final)]
    bars = ax.bar(range(1, len(vals) + 1), vals, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{v:.1e}", ha="center", va="bottom", fontsize=7, rotation=30)
    ax.set_xlabel("Round"); ax.set_ylabel("Avg grad norm")
    ax.set_title("Avg grad norm (last 600 steps)  (exact)")
    ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(1, len(vals) + 1))

    plt.tight_layout()
    out = Path(__file__).resolve().parent / run["out"]
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{run['mode']:>13s}  final={[f'{v:.3e}' for v in vals]}  ->  {out.name}")


for r in RUNS:
    build(r)
