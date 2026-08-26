#!/usr/bin/env python3
"""
Attach eval results to the W&B run that produced the checkpoint.

Compute nodes on this cluster have no internet, so training logs to W&B in
offline mode and eval jobs cannot log at all.  The flow is:

  1. training  (compute) writes  <run_dir>/wandb_run.json   — the W&B run id
  2. eval      (compute) writes  <run_dir>/eval/<dataset>.json  via run.py --result_json
  3. this script (LOGIN NODE, has internet) syncs the offline training run,
     then re-opens it online and logs the eval metrics into it.

Usage:
    # sync the offline training runs first, then attach eval metrics
    python sync_eval_to_wandb.py --run_dir outputs/checkpoints/<run> [...] --sync-offline

    # or point it at every run that has eval results
    python sync_eval_to_wandb.py --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

THIS_DIR = Path(__file__).resolve().parent
CKPT_ROOT = THIS_DIR / "outputs" / "checkpoints"


def find_run_dirs(explicit: List[str], scan_all: bool) -> List[Path]:
    if explicit:
        return [Path(d).resolve() for d in explicit]
    if not scan_all:
        return []
    return sorted(d for d in CKPT_ROOT.iterdir()
                  if (d / "wandb_run.json").is_file()
                  and (any(d.glob("*/eval_*.json")) or (d / "eval").is_dir()))


def sync_offline_run(meta: dict) -> None:
    """`wandb sync` the offline directory so the run exists server-side."""
    offline_dir = meta.get("offline_dir")
    if not offline_dir:
        print("    (no offline_dir recorded; skipping wandb sync)")
        return
    # meta['offline_dir'] points at <offline-run-.../files>; sync the run dir.
    d = Path(offline_dir)
    if d.name == "files":
        d = d.parent
    if not d.is_dir():
        print(f"    (offline dir {d} is gone; skipping wandb sync)")
        return
    print(f"    wandb sync {d}")
    r = subprocess.run(["wandb", "sync", str(d)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        print("    WARNING: wandb sync failed; the run may not exist online yet.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", nargs="*", default=[],
                    help="Training run directory/directories under outputs/checkpoints/.")
    ap.add_argument("--all", action="store_true",
                    help="Every run directory that has both wandb_run.json and eval/.")
    ap.add_argument("--sync-offline", action="store_true",
                    help="Run `wandb sync` on the offline training run first.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be logged without touching W&B.")
    args = ap.parse_args()

    run_dirs = find_run_dirs(args.run_dir, args.all)
    if not run_dirs:
        print("No run directories given. Use --run_dir <path> ... or --all.")
        return 1

    import wandb

    rc = 0
    for run_dir in run_dirs:
        meta_path = run_dir / "wandb_run.json"
        eval_dir = run_dir / "eval"
        print(f"\n=== {run_dir.name}")
        if not meta_path.is_file():
            print(f"    no wandb_run.json — training ran without --wandb? skipping.")
            rc = 1
            continue
        meta = json.loads(meta_path.read_text())

        # run.py writes <run_dir>/step_N/eval_<dataset>.json
        results = sorted(run_dir.glob("*/eval_*.json"))
        if eval_dir.is_dir():
            results += sorted(eval_dir.glob("*.json"))
        if not results:
            print("    no eval/*.json yet — eval jobs still running? skipping.")
            continue

        payloads = [json.loads(p.read_text()) for p in results]
        print(f"    W&B run: {meta['entity'] or '-'}/{meta['project']}/{meta['id']}"
              f"  ({meta.get('name')})")
        for p in payloads:
            print(f"      {p['dataset']:>14s}  {p['metric_name']}={p['metric_value']:.2f}%"
                  f"  (ls={p['latent_steps']})")
        if args.dry_run:
            continue

        if args.sync_offline:
            sync_offline_run(meta)

        run = wandb.init(project=meta["project"], entity=meta["entity"] or None,
                         id=meta["id"], resume="allow")
        table = wandb.Table(columns=["ckpt_step", "dataset", "metric", "value_pct",
                                     "latent_steps", "n_rounds", "ckpt_dir"])
        for p in payloads:
            step = Path(p.get("ckpt_dir") or "").name or "-"
            run.summary[f"eval/{p['dataset']}/{p['metric_name']}"] = p["metric_value"]
            run.summary[f"eval/{p['dataset']}/latent_steps"] = p["latent_steps"]
            table.add_data(step, p["dataset"], p["metric_name"], p["metric_value"],
                           p["latent_steps"], p["n_rounds"], p.get("ckpt_dir") or "")
        vals = [p["metric_value"] for p in payloads]
        run.summary["eval/mean_over_datasets"] = sum(vals) / len(vals)
        run.summary["eval/n_datasets"] = len(vals)
        run.log({"eval/results": table})
        run.finish()
        print(f"    attached {len(payloads)} dataset result(s).")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
