#!/usr/bin/env python3
"""Compare self-inject vs control eval runs on the same checkpoint.

Reads the eval result_json files written next to a checkpoint and reports the
per-seed accuracies, the paired difference, and whether that difference clears
the sampling-noise floor -- which on math500 at one seed is around 7.5 points
(see ROUND_GRADIENT_WEIGHTING.md section 5).

    python analyze_self_inject.py <ckpt_dir> [dataset]
"""
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

N_PROBLEMS = {"math500": 500, "medqa": 300, "gpqa": 198, "mbppplus": 378,
              "aime2025": 30, "aime2026": 30, "livecodebench": 1055}


def load(ckpt: Path, dataset: str):
    arms = {"control": {}, "self_inject": {}}
    for f in sorted(ckpt.glob(f"eval_{dataset}*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        if d.get("dataset", dataset) != dataset:
            continue
        arm = "self_inject" if d.get("self_inject") else "control"
        arms[arm][int(d.get("seed", -1))] = (float(d["metric_value"]), f.name)
    return arms


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ckpt = Path(sys.argv[1])
    dataset = sys.argv[2] if len(sys.argv) > 2 else "math500"
    arms = load(ckpt, dataset)

    seeds = sorted(set(arms["control"]) & set(arms["self_inject"]))
    print(f"dataset={dataset}  n={N_PROBLEMS.get(dataset, '?')}  ckpt={ckpt.name}\n")
    print(f"{'seed':>6} {'control':>10} {'self_inject':>12} {'delta':>9}")
    print("-" * 42)
    deltas = []
    for s in seeds:
        c, si = arms["control"][s][0], arms["self_inject"][s][0]
        deltas.append(si - c)
        print(f"{s:>6} {c:>9.2f}% {si:>11.2f}% {si - c:>+8.2f}")
    for arm in ("control", "self_inject"):
        only = sorted(set(arms[arm]) - set(seeds))
        for s in only:
            print(f"{s:>6}  {arm} only: {arms[arm][s][0]:.2f}%  ({arms[arm][s][1]})")

    if not deltas:
        print("\nNo paired seeds yet -- jobs still running?")
        return 0

    n = N_PROBLEMS.get(dataset)
    print("-" * 42)
    print(f"{'mean':>6} {mean(x[0] for x in arms['control'].values()):>9.2f}% "
          f"{mean(x[0] for x in arms['self_inject'].values()):>11.2f}% "
          f"{mean(deltas):>+8.2f}")
    if len(deltas) > 1:
        print(f"\npaired delta: mean {mean(deltas):+.2f}, sd {stdev(deltas):.2f}, "
              f"n={len(deltas)} seeds")
    if n:
        # single-seed binomial sd at the observed rate, and the paired MDE
        p = mean(x[0] for x in arms["control"].values()) / 100
        sd1 = 100 * math.sqrt(max(p * (1 - p), 1e-9) / n)
        mde = 2.8 * sd1 * math.sqrt(2 / max(len(deltas), 1))
        print(f"binomial sd at 1 seed: {sd1:.2f} pt")
        print(f"approx MDE at {len(deltas)} seed(s), 80% power: {mde:.1f} pt")
        verdict = ("ABOVE the noise floor -- worth taking seriously"
                   if abs(mean(deltas)) > mde else
                   "INSIDE the noise floor -- not resolvable at this seed count")
        print(f"observed |delta| = {abs(mean(deltas)):.2f} pt -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
