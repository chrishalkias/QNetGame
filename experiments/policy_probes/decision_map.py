"""
--------------------------------------------------------------------------------
Empirical action-eagerness maps over (occupancy, fidelity), tiled, paper figure.

2x3 panel grid over the agent's real greedy rollouts, one row per --n_chs value:
  col 1 = P(PURIFY | can_purify=1) per (occupied count, fidelity) tile
  col 2 = P(SWAP   | can_swap=1)
  col 3 = swap-vs-purify preference where BOTH are available:
          P(SWAP|both) - P(PURIFY|both) in [-1,1] (NOOP absorbs the rest)
Rollouts run at a single n_ch per row so the occupancy axis is the exact number
of occupied qubits (n_ch+1 discrete tiles); fidelity is binned. Tiles are
outlined, colour-coded, greyed where under-sampled. Purely empirical.

Dual-mode so the figure re-renders without recomputing:
  compute: PYTHONPATH=src:. python experiments/policy_probes/decision_map.py --ckpt <path>
  plot:    PYTHONPATH=src:. python experiments/policy_probes/decision_map.py --plot --save_dir <dir>
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np

# y-axis node feature -> (obs column, default lower edge, axis label)
YFEATS = {
    "fidelity": (1, 0.5, "mean fidelity of available qubits"),
    "urgency":  (8, 0.0, "mean link urgency"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="re-render from the JSON")
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--n_chs", type=int, nargs="+", default=[4, 6],
                    help="one panel row per memory size (occupancy axis = "
                         "occupied count 0..n_ch)")
    ap.add_argument("--fid_bins", type=int, default=20)
    ap.add_argument("--yfeat", choices=list(YFEATS), default="fidelity",
                    help="y-axis feature: fidelity (X[:,1]) or urgency (X[:,8])")
    ap.add_argument("--fid_lo", type=float, default=None,
                    help="lower edge of the y-axis (default: 0.5 for fidelity "
                         "= Werner separability, 0.0 for urgency)")
    ap.add_argument("--min_count", type=int, default=5,
                    help="tiles with fewer eligible decisions are greyed out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def eagerness_grid(occ_count, fid, took, n_occ, edges, min_count):
    """Per (occupied-count, fidelity-bin) tile: mean of `took` over eligible
    decisions (NaN = under-sampled). `took` may be a bool or signed float."""
    fid_bins = len(edges) - 1
    frac = np.full((fid_bins, n_occ), np.nan)
    fi = np.clip(np.digitize(fid, edges) - 1, 0, fid_bins - 1)
    for bx in range(n_occ):
        for by in range(fid_bins):
            sel = (occ_count == bx) & (fi == by)
            if int(sel.sum()) >= min_count:
                frac[by, bx] = float(np.asarray(took, float)[sel].mean())
    return frac


def _jsonable(grid):
    return [[None if np.isnan(v) else round(float(v), 4) for v in row]
            for row in grid]


def _np(grid):
    return np.array([[np.nan if v is None else v for v in row] for row in grid])


def run_compute(a, out_json):
    from experiments.policy_probes import _collect as C
    col, lo_default, ylabel = YFEATS[a.yfeat]
    fid_lo = a.fid_lo if a.fid_lo is not None else lo_default
    data = dict(ckpt=a.ckpt, episodes=a.episodes, seed=a.seed,
                fid_bins=a.fid_bins, fid_lo=fid_lo, min_count=a.min_count,
                ylabel=ylabel, rows={})
    edges = np.linspace(fid_lo, 1.0, a.fid_bins + 1)
    for n_ch in a.n_chs:
        d = C.collect(a.ckpt, episodes=a.episodes, n_chs=(n_ch,), seed=a.seed)
        X, act = d["X"], d["A"]
        occ = np.rint(X[:, 0] * n_ch).astype(int)   # occ feature = count/n_ch
        fid = X[:, col]
        cs = np.rint(X[:, 4]).astype(bool)
        cp = np.rint(X[:, 5]).astype(bool)
        both = cs & cp
        n_occ = n_ch + 1
        g = lambda el, took: _jsonable(eagerness_grid(
            occ[el], fid[el], took[el], n_occ, edges, a.min_count))
        data["rows"][str(n_ch)] = {
            "purify": g(cp, act == 2),
            "swap": g(cs, act == 1),
            "both_pref": g(both, (act == 1).astype(float) - (act == 2).astype(float)),
            "n_purify": int(cp.sum()), "n_swap": int(cs.sum()),
            "n_both": int(both.sum()),
        }
    json.dump(data, open(out_json, "w"), indent=1)
    print(f"saved -> {out_json}")
    return data


def run_plot(data, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    usetex = shutil.which("latex") is not None
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                         "text.usetex": usetex, "font.family": "serif"})

    pref_cmap = LinearSegmentedColormap.from_list(
        "purify_swap", ["#1ba31b", "#ffffff", "#1f63d6"])
    cols = [("purify", r"$P(\mathrm{PURIFY} \mid \mathrm{can\,purify})$",
             "Greens", 0.0, 1.0, "n_purify"),
            ("swap", r"$P(\mathrm{SWAP} \mid \mathrm{can\,swap})$",
             "Blues", 0.0, 1.0, "n_swap"),
            ("both_pref",
             r"$P(\mathrm{SWAP}) - P(\mathrm{PURIFY})$, both available",
             pref_cmap, -1.0, 1.0, "n_both")]
    n_chs = sorted(data["rows"], key=int)
    fid_edges = np.linspace(data.get("fid_lo", 0.0), 1.0, data["fid_bins"] + 1)

    fig, axes = plt.subplots(len(n_chs), 3, figsize=(12.5, 4.1 * len(n_chs)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    letters = "ABCDEFGH"
    for r, n_ch in enumerate(n_chs):
        row = data["rows"][n_ch]
        n_occ = int(n_ch) + 1
        x_edges = np.arange(n_occ + 1) - 0.5
        for c, (key, title, cmap_name, vmin, vmax, nkey) in enumerate(cols):
            ax = axes[r, c]
            cmap = (plt.get_cmap(cmap_name) if isinstance(cmap_name, str)
                    else cmap_name).copy()
            cmap.set_bad("0.88")
            pm = ax.pcolormesh(x_edges, fid_edges, np.ma.masked_invalid(_np(row[key])),
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               edgecolors="0.55", linewidth=0.6)
            fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.03)
            ax.set_xticks(range(n_occ))
            ax.set_title(f"{title}\n({row[nkey]} eligible decisions, "
                         rf"$n_\mathrm{{ch}}={n_ch}$)", fontsize=10)
            lab = f"({letters[r * 3 + c]})"
            ax.text(-0.12, 1.12, rf"\textbf{{{lab}}}" if usetex else lab,
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=13, fontweight="bold")
            if r == len(n_chs) - 1:
                ax.set_xlabel("occupied qubits")
        axes[r, 0].set_ylabel(data.get("ylabel", "mean fidelity of available qubits"))
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf (usetex={usetex})")


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)
    stem = "decision_map" if a.yfeat == "fidelity" else f"decision_map_{a.yfeat}"
    out_json = os.path.join(out, f"{stem}.json")
    data = json.load(open(out_json)) if a.plot else run_compute(a, out_json)
    run_plot(data, os.path.join(out, stem))


if __name__ == "__main__":
    main()
