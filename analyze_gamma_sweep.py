#!/usr/bin/env python3
"""Summarise the gamma_init sweep from SLURM logs.

Reads the per-step traces that train_outerlinks_math500.py prints and reports,
per run: the gamma trajectory (does the write gate stay where it was put?), the
forward write_ratio, the per-round parameter-gradient profile, and the tail loss.

    python analyze_gamma_sweep.py                      # the launched sweep
    python analyze_gamma_sweep.py 15556711 15562451    # explicit job ids
"""
import re
import sys
from pathlib import Path
from statistics import mean

LOG_DIR = Path("outputs/slurm_logs")
# job id -> gamma_init it was launched with
SWEEP = {"15556711": 1e-3, "15562451": 1e-2,
         "15562452": 1e-1, "15562453": 3e-1, "15562454": 1.0}

RE_STEP = re.compile(r"step\s+(\d+)\s+loss=([-\d.eE+]+)")
RE_GATE = re.compile(r"gamma=([-\d.eE+]+)")
RE_ALPHA = re.compile(r"alpha=([-\d.eE+]+)")
RE_WRITE = re.compile(r"write\[r(\d+)\].*?write_ratio=([-\d.eEna+]+)")
RE_GRAD = re.compile(r"grad\[prm\]\[\s*(\S+)\]\s+\[(.*?)\]")
RE_RND = re.compile(r"r(\d+):([-\d.eE+]+|--)")


def parse(job):
    f = LOG_DIR / f"job-{job}" / f"job-{job}.out"
    if not f.is_file():                      # some clusters write flat files
        alt = list(LOG_DIR.glob(f"job-{job}*.out"))
        if not alt:
            return None
        f = alt[0]
    steps, gammas, alphas, losses, writes, grads = [], [], [], [], {}, {}
    for line in f.read_text(errors="replace").splitlines():
        m = RE_STEP.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            g = RE_GATE.search(line)
            a = RE_ALPHA.search(line)
            gammas.append(float(g.group(1)) if g else float("nan"))
            alphas.append(float(a.group(1)) if a else float("nan"))
        m = RE_WRITE.search(line)
        if m:
            writes.setdefault(int(m.group(1)), []).append(m.group(2))
        m = RE_GRAD.search(line)
        if m:
            per = {int(r): (float(v) if v != "--" else None)
                   for r, v in RE_RND.findall(m.group(2))}
            grads.setdefault(m.group(1), []).append(per)
    return dict(steps=steps, gammas=gammas, alphas=alphas,
                losses=losses, writes=writes, grads=grads, path=f)


def main():
    jobs = sys.argv[1:] or list(SWEEP)
    print(f"{'job':>10} {'gamma_init':>11} {'gamma_final':>12} {'sign flips':>11} "
          f"{'|gamma| max':>12} {'alpha_final':>12} {'loss(tail)':>11}  steps")
    print("-" * 96)
    parsed = {}
    for j in jobs:
        d = parse(j)
        parsed[j] = d
        if d is None or not d["steps"]:
            print(f"{j:>10} {'':>11}  no log yet")
            continue
        g = [x for x in d["gammas"] if x == x]
        flips = sum(1 for a, b in zip(g, g[1:]) if a * b < 0)
        tail = d["losses"][-3:] or d["losses"]
        init = SWEEP.get(j)
        print(f"{j:>10} {init if init is not None else '?':>11} "
              f"{g[-1] if g else float('nan'):>12.3e} {flips:>11} "
              f"{max(abs(x) for x in g) if g else float('nan'):>12.3e} "
              f"{d['alphas'][-1]:>12.3e} {mean(tail):>11.4f}  {d['steps'][-1]}")

    for j in jobs:
        d = parsed.get(j)
        if not d or not d["steps"]:
            continue
        print(f"\n=== job {j}  (gamma_init={SWEEP.get(j, '?')}) ===")
        print("  gamma:", "  ".join(f"{s}:{v:.2e}" for s, v in
                                    zip(d["steps"], d["gammas"])))
        if d["writes"]:
            for r in sorted(d["writes"]):
                print(f"  write_ratio r{r}:", "  ".join(d["writes"][r][-6:]))
        else:
            print("  write_ratio: not logged (run predates --gamma_init)")
        for link, hist in d["grads"].items():
            if not hist:
                continue
            last = hist[-1]
            cells = "  ".join(f"r{r}:{('%.3e' % v) if v is not None else '--'}"
                              for r, v in sorted(last.items()))
            r1, r2 = last.get(1), last.get(2)
            ratio = f"  r1/r2={r1/r2:.3g}" if r1 and r2 else ""
            print(f"  grad[prm][{link:>6}] {cells}{ratio}")


if __name__ == "__main__":
    main()
