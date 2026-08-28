#!/usr/bin/env python3
"""Compare math500 accuracy across the training-time self-injection arms.

Each arm is a checkpoint trained under a different injection policy and evaluated
train/test matched (arms trained with injection are evaluated with --self_inject).
Pairs by seed so the per-seed differences carry the comparison rather than the
across-seed spread.

    python analyze_train_self_inject_eval.py \\
        control=outputs/checkpoints/si_control_.../step_3000 \\
        detached=outputs/checkpoints/si_detached_.../step_3000 \\
        grad=outputs/checkpoints/si_grad_.../step_3000

The noise floor is the thing to read first: math500 is 500 items, so one seed
resolves about 1.8 pt and three seeds about 4.2 pt at 80% power. A delta inside
that band is not a result, whichever way it points.
"""
import json
import math
import sys
from pathlib import Path

N_ITEMS = 500


def load_arm(d: Path):
    """seed -> (accuracy, self_inject_flag) for every eval json in this dir."""
    out = {}
    for f in sorted(d.glob("eval_math500*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception as e:                      # noqa: BLE001
            print(f"  !! unreadable {f.name}: {e}")
            continue
        if j.get("dataset") != "math500":
            continue
        out[int(j["seed"])] = (float(j["metric_value"]),
                               bool(j.get("self_inject", False)),
                               f.name)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    arms = []
    for a in args:
        label, _, path = a.partition("=")
        if not path:
            print(f"!! expected label=path, got {a!r}")
            return 1
        d = Path(path)
        if not d.is_dir():
            print(f"!! not a directory: {d}")
            return 1
        arms.append((label, load_arm(d)))

    seeds = sorted(set.intersection(*[set(a[1]) for a in arms])) if arms else []
    if not seeds:
        print("!! no seeds common to every arm yet")
        for label, res in arms:
            print(f"   {label}: seeds {sorted(res) or 'none'}")
        return 1

    # Every arm should be evaluated the way it was trained.
    print("\ntrain/test matching:")
    for label, res in arms:
        flags = {res[s][1] for s in seeds}
        expect = (label != "control")
        ok = flags == {expect}
        print(f"  {'OK ' if ok else 'BAD'} {label:>10}: self_inject={flags} "
              f"(expected {{{expect}}})")
        if not ok:
            print("      ^ arm is not evaluated the way it was trained; "
                  "the comparison below is not valid")

    head = "  ".join(f"{label:>10}" for label, _ in arms)
    print(f"\n{'seed':>6}  {head}")
    print("-" * (8 + 12 * len(arms)))
    for s in seeds:
        row = "  ".join(f"{res[s][0]:>9.2f}%" for _, res in arms)
        print(f"{s:>6}  {row}")
    print("-" * (8 + 12 * len(arms)))
    means = [mean([res[s][0] for s in seeds]) for _, res in arms]
    print(f"{'mean':>6}  " + "  ".join(f"{m:>9.2f}%" for m in means))

    base_label, base_res = arms[0]
    base = means[0]
    p = base / 100.0
    binom_sd = 100.0 * math.sqrt(p * (1 - p) / N_ITEMS)
    mde = 2.8 * binom_sd * math.sqrt(2.0 / len(seeds))

    print(f"\nnoise floor: binomial sd at one seed {binom_sd:.2f} pt; "
          f"approx MDE at {len(seeds)} seeds, 80% power {mde:.2f} pt")
    print(f"\npaired deltas vs {base_label} (positive = better):")
    for (label, res), m in zip(arms[1:], means[1:]):
        deltas = [res[s][0] - base_res[s][0] for s in seeds]
        verdict = ("OUTSIDE the noise floor" if abs(mean(deltas)) > mde
                   else "inside the noise floor -- not resolvable")
        print(f"  {label:>10}  {mean(deltas):+6.2f} pt  (per-seed "
              f"{', '.join(f'{d:+.1f}' for d in deltas)}; sd {sd(deltas):.2f})"
              f"  -> {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
