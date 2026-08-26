"""Plot analysis figures from eval_results.md.

Parses the markdown tables (reference table + one table per train_steps
section) and writes three figures to eval_figures/:

  fig1_accuracy_vs_steps.png   accuracy vs training steps per dataset, across
                               all methods (outer_link, shared link,
                               shared_state, shared_tied, shared_state (full)),
                               with released-ckpt and paper reference lines
  fig2_method_gaps.png         heatmaps of pairwise method gaps in percentage
                               points per dataset and step (shared - outer,
                               shared_state - shared, shared_tied - shared,
                               shared_state(full) - shared_state)
  fig3_final_vs_references.png final-step methods vs released ckpt vs paper
  fig4_lr_sweep.png             grid of small heatmaps: rows = method, cols =
                               dataset (+ average); each cell is accuracy over
                               (learning rate x training step), for every
                               method with an explicit "(lr=...)" tag

Usage:
    python plot_eval_results.py [--results eval_results.md] [--out eval_figures]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Validated categorical palette (dataviz reference instance, light mode).
C_OUTER = "#2a78d6"     # blue    — outer_link
C_SHARED = "#008300"    # green   — shared link
C_STATE = "#d55181"     # magenta — shared_state (dark-mode step: 3:1 on light surface)
C_TIED = "#c98500"      # yellow  — shared_tied (dark-mode step: 3:1 on light surface)
C_STATE_FULL = "#1baf7a"  # aqua  — shared_state (full training samples)
C_RELEASED = "#a8a7a0"  # gray bar — released-checkpoint reference
C_PAPER = "#52514e"     # dark gray — paper reference
METHODS = [
    ("outer_link", C_OUTER),
    ("shared link", C_SHARED),
    ("shared_state", C_STATE),
    ("shared_tied", C_TIED),
    ("shared_state (full)", C_STATE_FULL),
]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"
# Diverging pair blue <-> red with neutral midpoint (for the difference heatmap).
DIV_CMAP = LinearSegmentedColormap.from_list(
    "shared_vs_outer", ["#e34948", "#f0efec", "#2a78d6"]
)

DATASETS = ["math500", "medqa", "aime2025", "aime2026", "gpqa", "mbppplus", "livecodebench"]


def _cell_to_float(cell: str):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%?", cell)
    return float(m.group(1)) if m else None


def _label_to_method(low: str) -> str:
    if "state" in low and "full" in low:   # "shared-state(full training samples)"
        return "shared_state (full)"
    if "state" in low:          # "shared_state" / "shared-state"
        return "shared_state"
    if "tied" in low:           # "shared_tied" / "shared-tied"
        return "shared_tied"
    if "shared" in low:
        return "shared link"
    return "outer_link"


def parse_results(path: Path):
    """Return (references, sweep, lr_sweep) where
    references = {row_label: {dataset: value}}
    sweep      = {method: {dataset: {step: value}}}  (last lr wins per cell,
                 matching each figure's original single-line-per-method view)
    lr_sweep   = {method: {lr: {dataset: {step: value}}}}  only rows with an
                 explicit "(lr=...)" tag in their label
    """
    text = path.read_text(encoding="utf-8")
    references: dict[str, dict[str, float]] = {}
    sweep: dict[str, dict[str, dict[int, float]]] = {}
    lr_sweep: dict[str, dict[float, dict[str, dict[int, float]]]] = {}

    current_steps = None
    header: list[str] = []
    for line in text.splitlines():
        m = re.search(r"train_steps\s*=\s*(\d+)", line)
        if m:
            current_steps = int(m.group(1))
            continue
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= {"-", " ", ":"} for c in cells):
            continue  # separator row
        if cells[0] == "":  # header row
            header = [c.lower() for c in cells[1:]]
            continue
        label, values = cells[0], cells[1:]
        row = {
            ds: _cell_to_float(v)
            for ds, v in zip(header, values)
            if _cell_to_float(v) is not None
        }
        if current_steps is None:
            references[label] = row
            continue

        low = label.lower()
        method = _label_to_method(low)
        for ds, v in row.items():
            sweep.setdefault(method, {}).setdefault(ds, {})[current_steps] = v

        lr_m = re.search(r"lr\s*=\s*([0-9.eE+-]+)", low)
        if lr_m:
            lr_val = float(lr_m.group(1))
            for ds, v in row.items():
                by_ds = lr_sweep.setdefault(method, {}).setdefault(lr_val, {}).setdefault(ds, {})
                by_ds[current_steps] = v
    return references, sweep, lr_sweep


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_accuracy_vs_steps(references, sweep, out: Path):
    released = next((v for k, v in references.items() if "ckpt" in k.lower()), {})
    paper = next((v for k, v in references.items() if "paper" in k.lower()), {})

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), facecolor=SURFACE)
    for ax, ds in zip(axes.flat, DATASETS):
        style_axis(ax)
        for method, color in METHODS:
            pts = sorted(sweep.get(method, {}).get(ds, {}).items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5)
        if ds in paper:
            ax.axhline(paper[ds], color=C_PAPER, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(0.02, paper[ds], "paper", transform=ax.get_yaxis_transform(),
                    color=C_PAPER, fontsize=7, va="bottom")
        if ds in released:
            ax.axhline(released[ds], color=C_RELEASED, linewidth=1.2, linestyle=(0, (1, 2)))
            ax.text(0.98, released[ds], "released ckpt", transform=ax.get_yaxis_transform(),
                    color=INK_2, fontsize=7, va="bottom", ha="right")
        ax.set_title(ds, fontsize=10, color=INK)
        ax.set_xlabel("train steps", fontsize=8, color=INK_2)
        ax.set_ylabel("accuracy (%)", fontsize=8, color=INK_2)

    # Hide any unused slots between the last dataset and the average panel.
    for ax in axes.flat[len(DATASETS):-1]:
        ax.axis("off")

    # Last slot: macro-average over all tasks (equal weight per dataset).
    ax = axes.flat[-1]
    style_axis(ax)
    for method, color in METHODS:
        per_ds = sweep.get(method, {})
        steps = sorted({s for d in per_ds.values() for s in d})
        xs, ys = [], []
        for s in steps:
            vals = [per_ds[ds][s] for ds in DATASETS if s in per_ds.get(ds, {})]
            if vals:
                xs.append(s)
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5)
    paper_avg = [paper[ds] for ds in DATASETS if ds in paper]
    released_avg = [released[ds] for ds in DATASETS if ds in released]
    if paper_avg:
        ax.axhline(np.mean(paper_avg), color=C_PAPER, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(0.02, np.mean(paper_avg), "paper", transform=ax.get_yaxis_transform(),
                color=C_PAPER, fontsize=7, va="bottom")
    if released_avg:
        ax.axhline(np.mean(released_avg), color=C_RELEASED, linewidth=1.2, linestyle=(0, (1, 2)))
        ax.text(0.98, np.mean(released_avg), "released ckpt", transform=ax.get_yaxis_transform(),
                color=INK_2, fontsize=7, va="bottom", ha="right")
    ax.set_title(f"average ({len(DATASETS)} tasks)", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("train steps", fontsize=8, color=INK_2)
    ax.set_ylabel("mean accuracy (%)", fontsize=8, color=INK_2)

    handles = [
        plt.Line2D([], [], color=color, linewidth=2, marker="o", markersize=5, label=method)
        for method, color in METHODS
    ] + [
        plt.Line2D([], [], color=C_PAPER, linewidth=1.2, linestyle=(0, (4, 3)), label="paper"),
        plt.Line2D([], [], color=C_RELEASED, linewidth=1.2, linestyle=(0, (1, 2)), label="released ckpt"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Eval accuracy vs training steps (r=3, latent=48)", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out / "fig1_accuracy_vs_steps.png", dpi=160)
    plt.close(fig)


def fig_method_gaps(sweep, out: Path):
    steps = sorted({s for m in sweep.values() for d in m.values() for s in d})
    pairs = [
        ("shared link", "outer_link", "shared link - outer_link"),
        ("shared_state", "shared link", "shared_state - shared link"),
        ("shared_tied", "shared link", "shared_tied - shared link"),
        ("shared_state (full)", "shared_state", "shared_state(full) - shared_state"),
    ]
    diffs = []
    for a_key, b_key, _ in pairs:
        diff = np.full((len(DATASETS), len(steps)), np.nan)
        for i, ds in enumerate(DATASETS):
            for j, s in enumerate(steps):
                a = sweep.get(a_key, {}).get(ds, {}).get(s)
                b = sweep.get(b_key, {}).get(ds, {}).get(s)
                if a is not None and b is not None:
                    diff[i, j] = a - b
        diffs.append(diff)

    lim = max(np.nanmax(np.abs(d)) for d in diffs)  # shared scale across panels
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs) + 1, 4.8),
                             facecolor=SURFACE, gridspec_kw={"wspace": 0.45})
    for ax, diff, (_, _, label) in zip(axes, diffs, pairs):
        im = ax.imshow(diff, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        ax.set_xticks(range(len(steps)), [str(s) for s in steps], fontsize=9, color=INK_2)
        ax.set_yticks(range(len(DATASETS)), DATASETS, fontsize=9, color=INK_2)
        ax.set_xlabel("train steps", fontsize=9, color=INK_2)
        for i in range(len(DATASETS)):
            for j in range(len(steps)):
                v = diff[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.1f}",
                        ha="center", va="center", fontsize=8,
                        color=INK if np.isnan(v) or abs(v) < 0.7 * lim else "white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{label} (pp)", fontsize=10, color=INK)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("gap (pp, blue = first method ahead)", fontsize=9, color=INK_2)
    cbar.ax.tick_params(labelsize=8, colors=INK_2)
    fig.suptitle("Pairwise method gaps by dataset and training step", fontsize=12, color=INK)
    fig.savefig(out / "fig2_method_gaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_final_vs_references(references, sweep, out: Path):
    released = next((v for k, v in references.items() if "ckpt" in k.lower()), {})
    paper = next((v for k, v in references.items() if "paper" in k.lower()), {})
    last_step = {
        m: {ds: d[max(d)] for ds, d in per.items()} for m, per in sweep.items()
    }
    series = [
        (f"{m} (final step)", last_step.get(m, {}), c) for m, c in METHODS
    ] + [
        ("released ckpt", released, C_RELEASED),
        ("paper", paper, C_PAPER),
    ]
    x = np.arange(len(DATASETS))
    width = 0.12
    fig, ax = plt.subplots(figsize=(15, 4.8), facecolor=SURFACE)
    style_axis(ax)
    offset0 = (len(series) - 1) / 2
    for k, (label, vals, color) in enumerate(series):
        ys = [vals.get(ds, np.nan) for ds in DATASETS]
        ax.bar(x + (k - offset0) * width, ys, width * 0.92, color=color, label=label,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, DATASETS, fontsize=9, color=INK_2)
    ax.set_ylabel("accuracy (%)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper right", labelcolor=INK)
    ax.set_title("Final-step checkpoints vs released weights vs paper", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_final_vs_references.png", dpi=160)
    plt.close(fig)


def fig_lr_sweep(lr_sweep, out: Path):
    """Small multiples: one row per dataset (+ average), one mini line-chart
    per learning rate within the row. Each mini chart plots accuracy vs
    training step with one line per method, so the two methods are compared
    directly by line shape/crossings rather than scanned numbers."""
    if not lr_sweep:
        print("no '(lr=...)' tagged rows found; skipping fig4_lr_sweep.png")
        return

    methods = list(lr_sweep.keys())
    if len(methods) < 2:
        print(f"only one lr-tagged method ({methods}); need >= 2 to compare, skipping fig4_lr_sweep.png")
        return

    rows = DATASETS + ["average"]
    all_lrs = sorted({lr for per_lr in lr_sweep.values() for lr in per_lr})
    all_steps = sorted({s for per_lr in lr_sweep.values()
                        for per_ds in per_lr.values() for d in per_ds.values() for s in d})
    method_colors = dict(METHODS)

    def series(method, lr, row):
        per_ds = lr_sweep[method].get(lr, {})
        xs, ys = [], []
        for s in all_steps:
            if row == "average":
                vals = [per_ds[ds][s] for ds in DATASETS if s in per_ds.get(ds, {})]
                if vals:
                    xs.append(s)
                    ys.append(float(np.mean(vals)))
            elif s in per_ds.get(row, {}):
                xs.append(s)
                ys.append(per_ds[row][s])
        return xs, ys

    # Shared y-range per row (dataset) so the two methods and all three LR
    # panels in that row are visually comparable; datasets differ wildly in
    # baseline difficulty (math500 ~75-80%, aime2026 ~13-27%), so ranges are
    # NOT shared across rows.
    row_range = {}
    for row in rows:
        vals = [y for m in methods for lr in all_lrs for y in series(m, lr, row)[1]]
        if vals:
            pad = 0.08 * (max(vals) - min(vals) + 1e-9)
            row_range[row] = (min(vals) - pad, max(vals) + pad)
        else:
            row_range[row] = (0.0, 100.0)

    ncols = len(all_lrs)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.9 * nrows + 0.7),
                             facecolor=SURFACE, squeeze=False)

    for i, row in enumerate(rows):
        ymin, ymax = row_range[row]
        for j, lr in enumerate(all_lrs):
            ax = axes[i][j]
            style_axis(ax)
            ax.set_ylim(ymin, ymax)
            for method in methods:
                xs, ys = series(method, lr, row)
                if xs:
                    ax.plot(xs, ys, color=method_colors.get(method, INK_2), linewidth=2,
                           marker="o", markersize=4)
            ax.set_xticks(all_steps, [str(s) for s in all_steps], fontsize=6,
                         rotation=45, color=INK_2)
            ax.tick_params(axis="y", labelsize=6.5)
            if j == 0:
                ax.set_ylabel(row, fontsize=8.5, color=INK,
                            fontweight="bold" if row == "average" else "normal")
            if i == 0:
                ax.set_title(f"lr={lr:g}", fontsize=9, color=INK)

    handles = [
        plt.Line2D([], [], color=method_colors.get(m, INK_2), linewidth=2,
                   marker="o", markersize=5, label=m)
        for m in methods
    ]
    fig.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False,
               fontsize=9.5, labelcolor=INK)
    fig.suptitle("Accuracy vs training step, by learning rate — methods compared directly\n"
                "(rows: dataset; columns: LR; y-axis range shared within each row)",
                color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fname = out / "fig4_lr_sweep.png"
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="eval_results.md")
    p.add_argument("--out", default="eval_figures")
    p.add_argument("--exclude", default="livecodebench",
                   help="Comma-separated datasets to drop from all figures "
                        "(default: livecodebench, whose paper number is unverified; "
                        "pass --exclude '' to include everything).")
    args = p.parse_args()

    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}
    global DATASETS
    DATASETS = [ds for ds in DATASETS if ds not in excluded]
    if excluded:
        print(f"excluding: {', '.join(sorted(excluded))}")

    references, sweep, lr_sweep = parse_results(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_accuracy_vs_steps(references, sweep, out)
    fig_method_gaps(sweep, out)
    fig_final_vs_references(references, sweep, out)
    fig_lr_sweep(lr_sweep, out)

    n_pts = sum(len(d) for m in sweep.values() for d in m.values())
    n_lr_pts = sum(len(d) for m in lr_sweep.values() for lrd in m.values() for d in lrd.values())
    print(f"parsed {n_pts} sweep points ({n_lr_pts} lr-tagged), references: {list(references)}")
    for f in sorted(out.glob("fig*.png")):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
