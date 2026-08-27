#!/usr/bin/env python3
"""Compare training arms of the self-injection A/B.

Reads the per-step traces train_outerlinks_math500.py prints and reports, per
arm: the tail loss (the script's own 600-step accounting, which uses every step
rather than the sparse logged ones), the per-round parameter-gradient profile,
and the gate trajectories.

    python analyze_train_self_inject.py control=15570001 inject=15570002 grad=15570003

A caveat the tool cannot check for you: with --self_inject_grad the per-round
gradient split is not trustworthy.  The backward loop attributes everything a
rank accumulates during round r's backward to round r, which holds only while
"the graph is cut at the received tensors".  A gradient-carrying injection adds
an edge that reaches round r-1's activations, so round r's column absorbs some
of round r-1's contribution.  Totals stay correct; the split does not.
"""
import re
import sys
from pathlib import Path

LOG_DIR = Path("outputs/slurm_logs")

RE_STEP = re.compile(r"step\s+(\d+)\s+loss=([-\d.eE+]+)")
RE_GAMMA = re.compile(r"gamma=([-\d.eE+]+)")
RE_ALPHA = re.compile(r"alpha=([-\d.eE+]+)")
RE_TAIL = re.compile(r"Loss \(last (\d+) steps\):\s+mean=([-\d.eE+]+)\s+std=([-\d.eE+]+)")
RE_GRAD = re.compile(r"grad\[prm\]\[\s*(\S+)\]\s+\[(.*?)\]")
RE_RND = re.compile(r"r(\d+):([-\d.eE+]+|--)")
RE_SI = re.compile(r"\[self_inject\] ON.*?block is (.+)")


def parse(job):
    f = LOG_DIR / f"job-{job}" / f"job-{job}.out"
    if not f.is_file():
        alt = sorted(LOG_DIR.glob(f"job-{job}*.out"))
        if not alt:
            return None
        f = alt[0]
    txt = f.read_text(errors="replace")
    d = {"path": f, "steps": [], "loss": [], "gamma": [], "alpha": [],
         "grads": {}, "tail": None, "self_inject": None}
    for line in txt.splitlines():
        m = RE_STEP.search(line)
        if m:
            d["steps"].append(int(m.group(1)))
            d["loss"].append(float(m.group(2)))
            g, a = RE_GAMMA.search(line), RE_ALPHA.search(line)
            d["gamma"].append(float(g.group(1)) if g else float("nan"))
            d["alpha"].append(float(a.group(1)) if a else float("nan"))
        m = RE_TAIL.search(line)
        if m:
            d["tail"] = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        m = RE_GRAD.search(line)
        if m:
            per = {int(r): (float(v) if v != "--" else None)
                   for r, v in RE_RND.findall(m.group(2))}
            d["grads"].setdefault(m.group(1), []).append(per)
        m = RE_SI.search(line)
        if m:
            d["self_inject"] = m.group(1).strip()
    return d


def last_grads(d, link):
    seq = d["grads"].get(link) or []
    return seq[-1] if seq else {}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    arms = []
    for a in args:
        label, _, job = a.partition("=")
        if not job:
            label, job = f"job{a}", a
        d = parse(job)
        if d is None:
            print(f"!! no log found for job {job} ({label})")
            continue
        arms.append((label, job, d))
    if not arms:
        return 1

    print(f"\n{'arm':>10} {'job':>10} {'injection':>22} {'tail loss':>12} "
          f"{'sd':>8} {'n_steps':>8}")
    print("-" * 76)
    base = None
    for label, job, d in arms:
        t = d["tail"]
        inj = d["self_inject"] or "off (control)"
        if t:
            if base is None:
                base = t[1]
            print(f"{label:>10} {job:>10} {inj:>22} {t[1]:>12.4f} {t[2]:>8.4f} "
                  f"{t[0]:>8}")
        else:
            n = d['steps'][-1] if d['steps'] else 0
            print(f"{label:>10} {job:>10} {inj:>22} {'(running)':>12} {'':>8} {n:>8}")

    if base is not None and len(arms) > 1:
        print(f"\ndelta vs {arms[0][0]} (negative = better):")
        for label, job, d in arms[1:]:
            if d["tail"]:
                print(f"  {label:>10}  {d['tail'][1] - base:+.4f} nats")

    for label, job, d in arms:
        print(f"\n=== {label}  (job {job}) ===")
        if d["gamma"] and d["gamma"][-1] == d["gamma"][-1]:
            print(f"  gamma: {d['gamma'][0]:.3e} -> {d['gamma'][-1]:.3e}"
                  f"   alpha: {d['alpha'][0]:.3e} -> {d['alpha'][-1]:.3e}")
        for link in ("12", "23", "31", "shared", "state"):
            g = last_grads(d, link)
            if not g:
                continue
            cells = "  ".join(
                f"r{r}:{'--' if g[r] is None else f'{g[r]:.3e}'}" for r in sorted(g))
            r1, r2 = g.get(1), g.get(2)
            ratio = f"  r1/r2={r1 / r2:.3g}" if (r1 and r2) else ""
            print(f"  grad[prm][{link:>6}] {cells}{ratio}")
        if d["self_inject"] and "grad" in d["self_inject"]:
            print("  NOTE: per-round split is not trustworthy for this arm "
                  "(see module docstring).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
