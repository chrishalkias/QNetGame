"""Empirical policy decision map over (occupancy, fidelity), split by can_swap.

Bins the agent's real interior-node decisions by observed occupancy (x) and
fidelity (y); each cell is coloured by the dominant action actually taken there,
and greyed where too few states were visited. Two panels: can_swap = 0 (left) and
can_swap = 1 (right). Purely empirical — no synthetic states.

  PYTHONPATH=. python diagnostics/policy_probes/decision_map.py --ckpt <path>
"""
from __future__ import annotations
import argparse, os
import numpy as np
from diagnostics.policy_probes import _collect as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--min_count", type=int, default=5,
                    help="cells with fewer visited decisions are greyed out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def dominant_action_grid(occ, fid, act, nbins, min_count):
    """Per (occ,fid) bin: dominant action index, or -1 if under-sampled."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    grid = np.full((nbins, nbins), -1, dtype=int)
    count = np.zeros((nbins, nbins), dtype=int)
    oi = np.clip(np.digitize(occ, edges) - 1, 0, nbins - 1)
    fi = np.clip(np.digitize(fid, edges) - 1, 0, nbins - 1)
    for bx in range(nbins):
        for by in range(nbins):
            sel = (oi == bx) & (fi == by)
            n = int(sel.sum())
            count[by, bx] = n
            if n >= min_count:
                grid[by, bx] = int(np.bincount(act[sel], minlength=3).argmax())
    return grid, count, edges


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)

    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed)
    occ, fid, can_swap, act = d["X"][:, 0], d["X"][:, 1], d["X"][:, 4], d["A"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    cmap = ListedColormap(C.ACTION_COLORS)           # 0=noop,1=swap,2=purify
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    for ax, cs in zip(axes, (0, 1)):
        sel = np.round(can_swap).astype(int) == cs
        grid, count, _ = dominant_action_grid(occ[sel], fid[sel], act[sel],
                                               a.bins, a.min_count)
        disp = np.ma.masked_where(grid < 0, grid)
        ax.set_facecolor("0.9")                      # grey = under-sampled
        ax.imshow(disp, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                  cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"can_swap = {cs}   ({int(sel.sum())} decisions)")
        ax.set_xlabel("occupancy")
    axes[0].set_ylabel("fidelity")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in
                        zip(C.ACTION_COLORS, C.ACTION_NAMES)],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Dominant action over real rollouts (grey = under-sampled)", y=1.02)
    stem = os.path.join(out, "decision_map")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")


if __name__ == "__main__":
    main()
