"""
CPU-only replication of the RecursiveMAS round recursion, using the REAL
SharedRecursiveLink / SharedRecursiveStateLink modules from modeling.py.

Each agent's latent rollout (which in the real system contributes exactly one
in-graph model forward -- step 0 -- see latent_rollout()) is stood in for by a
fixed random linear map with a sub-unit spectral scale.  Everything else is the
real module.

Measures ||dL/d(critic_prefix^(r))||, i.e. the "12" link that both of the
user's plots report, for every round r.
"""
import sys, math
import torch, torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from modeling import SharedRecursiveLink, SharedRecursiveStateLink

torch.manual_seed(0)

H, D, T, B = 64, 64, 8, 2
HID = [H, H, H]


def agent_maps(scale):
    """Stand-in for each agent's in-graph latent-rollout Jacobian."""
    ms = []
    for _ in range(3):
        W = torch.randn(H, H)
        W *= scale / torch.linalg.svdvals(W)[0]
        ms.append(W)
    return ms


def run(mode, n_rounds, gamma=1e-3, alpha=1e-3, scale=0.3, seed=0):
    torch.manual_seed(seed)
    if mode == "shared_state":
        link = SharedRecursiveStateLink(HID, D, n_experts=3, expert_dim_divisor=3,
                                        gamma_init=gamma)
    else:
        link = SharedRecursiveLink(HID, D, n_experts=3, expert_dim_divisor=3)
    with torch.no_grad():
        link.alpha.fill_(alpha)
    A = agent_maps(scale)

    x0 = torch.randn(B, T, H)
    fb, z = None, None
    cps = []

    for r in range(n_rounds):
        p_in = x0 if fb is None else fb
        h1 = p_in @ A[0].T
        cp = link(h1, src=1, dst=2)          # <-- the monitored "12" tensor
        cp.retain_grad()
        cps.append(cp)

        h2 = cp @ A[1].T
        sp = link(h2, src=2, dst=3)
        h3 = sp @ A[2].T

        if r == n_rounds - 1:
            loss = (h3 ** 2).mean()          # stands in for the CE head
        elif mode == "shared_state":
            fb, z = link.round_feedback(h3, z)
        else:
            fb = link(h3, src=3, dst=1)

    loss.backward()
    return [c.grad.norm().item() for c in cps]


def show(mode, n_rounds, **kw):
    g = run(mode, n_rounds, **kw)
    cells = "  ".join(f"r{i+1}:{v:.3e}" for i, v in enumerate(g))
    rat = "  ".join(f"r{i+1}/r{i+2}={g[i]/g[i+1]:.3g}" for i in range(len(g) - 1))
    print(f"  {mode:>13s} R={n_rounds}   {cells}")
    print(f"  {'':>13s}        {rat}")


print("=" * 78)
print("  d(loss)/d(critic_prefix) per round   [gamma=1e-3, alpha=1e-3]")
print("=" * 78)
for R in (3, 4, 5, 7):
    show("shared_roae", R)
    show("shared_state", R)
    print()

print("=" * 78)
print("  shared_state: sweep gamma  (R=5)   -- prediction: r1/r2 ~ 1/gamma")
print("=" * 78)
for gm in (1e-1, 1e-2, 1e-3, 1e-4):
    g = run("shared_state", 5, gamma=gm)
    print(f"  gamma={gm:.0e}   " + "  ".join(f"r{i+1}:{v:.3e}" for i, v in enumerate(g))
          + f"    r1/r2={g[0]/g[1]:.4g}  (1/gamma={1/gm:.0f})")

print()
print("=" * 78)
print("  Plateau test: shared_state middle rounds should be FLAT (R=7)")
print("=" * 78)
g = run("shared_state", 7)
print("  middle-round ratios r2..r6:",
      "  ".join(f"{g[i]/g[i+1]:.3g}" for i in range(1, 5)))
g = run("shared_roae", 7)
print("  shared_roae   ratios r1..r6:",
      "  ".join(f"{g[i]/g[i+1]:.3g}" for i in range(0, 6)))
