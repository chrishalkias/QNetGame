"""
--------------------------------------------------------------------------------
Average agent decisions over the operation-quality plane (p_swap x p_gen).

Top row: three panels over a p_s (x) by p_e (y) grid, 0.1-wide bins, aggregated
across the whole training distribution (all sizes / n_ch):
  (A) P(PURIFY | can_purify)   (B) P(SWAP | can_swap)   (C) P(SWAP)-P(PURIFY), both
Bottom row: the SAME three panels, each split into a 2x2 block conditioned on link
urgency u in {0.0, 0.2, 0.4, 0.6} (reading order TL, TR, BL, BR), so the p_e/p_s
maps are resolved by how close the node's links are to the cutoff.

p_e = per-repeater generation prob (feature 6), p_s = per-repeater BSM prob
(feature 7); their per-node spread comes from the inhomogeneity std, so the grid
fills beyond the [0.4,0.9] episode-mean range. Tiles greyed where under-sampled.

  compute: PYTHONPATH=src:. python experiments/policy_probes/quality_map.py --ckpt <path>
  plot:    PYTHONPATH=src:. python experiments/policy_probes/quality_map.py --plot --save_dir <dir>
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np

URG_BINS = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65)]   # 4 urgency slices


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="re-render from the JSON")
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--n_ch", type=int, nargs="+", default=[2, 3, 4],
                    help="n_ch pool for rollouts (match the ckpt's training n_ch)")
    ap.add_argument("--p_lo", type=float, default=0.4,
                    help="lower edge of the per-episode p_gen/p_swap MEAN draw "
                         "(collect() default 0.4 = training range; lower this to "
                         "densify the low-p_gen/p_swap corner of the grid)")
    ap.add_argument("--p_hi", type=float, default=0.9)
    ap.add_argument("--bin", type=float, default=0.1, help="p_s / p_e grid bin width")
    ap.add_argument("--min_count", type=int, default=20,
                    help="aggregate tiles with fewer decisions are greyed out")
    ap.add_argument("--min_count_cond", type=int, default=8,
                    help="urgency-conditioned tiles: sparser, lower threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def grid2d(px, py, took, edges, min_count):
    """Mean of `took` per (p_s bin, p_e bin) tile; NaN where under-sampled.
    Shape (n_bins_y, n_bins_x) so pcolormesh puts p_e on y."""
    n = len(edges) - 1
    out = np.full((n, n), np.nan)
    xi = np.clip(np.digitize(px, edges) - 1, 0, n - 1)
    yi = np.clip(np.digitize(py, edges) - 1, 0, n - 1)
    for by in range(n):
        for bx in range(n):
            sel = (xi == bx) & (yi == by)
            if int(sel.sum()) >= min_count:
                out[by, bx] = float(np.asarray(took, float)[sel].mean())
    return out


def _jsonable(g):
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in g]


def _np(g):
    return np.array([[np.nan if v is None else v for v in row] for row in g])


def run_compute(a, out_json):
    from experiments.policy_probes import _collect as C
    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed, p_lo=a.p_lo, p_hi=a.p_hi,
                  n_chs=tuple(a.n_ch))
    X, act = d["X"], d["A"]
    p_e, p_s, urg = X[:, 6], X[:, 7], X[:, 8]
    cs = np.rint(X[:, 4]).astype(bool)
    cp = np.rint(X[:, 5]).astype(bool)
    both = cs & cp
    edges = np.arange(0.0, 1.0 + 1e-9, a.bin)

    def slc(s):
        lo, hi = URG_BINS[s]
        return (urg >= lo) & (urg < hi)

    def panel(el, took):
        agg = _jsonable(grid2d(p_s[el], p_e[el], took[el], edges, a.min_count))
        cond = [_jsonable(grid2d(p_s[el & slc(s)], p_e[el & slc(s)],
                                 took[el & slc(s)], edges, a.min_count_cond))
                for s in range(4)]
        return dict(agg=agg, cond=cond)

    tookP = (act == 2).astype(float)
    tookS = (act == 1).astype(float)
    pref = tookS - tookP
    data = dict(ckpt=a.ckpt, episodes=a.episodes, seed=a.seed, bin=a.bin,
                p_lo=a.p_lo, p_hi=a.p_hi,
                min_count=a.min_count, min_count_cond=a.min_count_cond,
                urg_bins=URG_BINS,
                panels=dict(
                    purify=dict(cmap="Greens", vmin=0.0, vmax=1.0, n=int(cp.sum()),
                                title=r"$P(\mathrm{PURIFY} \mid \mathrm{can\,purify})$",
                                **panel(cp, tookP)),
                    swap=dict(cmap="Blues", vmin=0.0, vmax=1.0, n=int(cs.sum()),
                              title=r"$P(\mathrm{SWAP} \mid \mathrm{can\,swap})$",
                              **panel(cs, tookS)),
                    both_pref=dict(cmap="pref", vmin=-1.0, vmax=1.0, n=int(both.sum()),
                                   title=r"$P(\mathrm{SWAP}) - P(\mathrm{PURIFY})$, both available",
                                   **panel(both, pref)),
                ))
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

    def cmap_for(tag):
        c = (pref_cmap if tag == "pref" else plt.get_cmap(tag)).copy()
        c.set_bad("0.88")
        return c

    edges = np.arange(0.0, 1.0 + 1e-9, data["bin"])
    urg_bins = data["urg_bins"]
    order = ["purify", "swap", "both_pref"]
    letters = "ABC"

    fig = plt.figure(figsize=(13.5, 9.5))
    outer = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.14, wspace=0.28)

    for c, key in enumerate(order):
        P = data["panels"][key]
        cmap = cmap_for(P["cmap"])

        # --- top: aggregate over all urgency ---
        axt = fig.add_subplot(outer[0, c])
        pm = axt.pcolormesh(edges, edges, np.ma.masked_invalid(_np(P["agg"])),
                            cmap=cmap, vmin=P["vmin"], vmax=P["vmax"],
                            edgecolors="0.55", linewidth=0.5)
        fig.colorbar(pm, ax=axt, fraction=0.046, pad=0.03)
        axt.set_aspect("equal")
        axt.set_title(f"{P['title']}\n({P['n']} decisions, all urgency)", fontsize=9.5)
        lab = f"({letters[c]})"
        axt.text(-0.10, 1.16, rf"\textbf{{{lab}}}" if usetex else lab,
                 transform=axt.transAxes, va="top", ha="left", fontsize=12)
        axt.set_xlabel(r"$p_s$ (swap quality)", fontsize=9)
        if c == 0:
            axt.set_ylabel(r"$p_e$ (generation quality)", fontsize=9)

        # --- bottom: 2x2 block conditioned on urgency (TL,TR,BL,BR = u 0,.2,.4,.6) ---
        sub = outer[1, c].subgridspec(2, 2, hspace=0.30, wspace=0.12)
        for k in range(4):
            axs = fig.add_subplot(sub[k // 2, k % 2])
            axs.pcolormesh(edges, edges, np.ma.masked_invalid(_np(P["cond"][k])),
                           cmap=cmap, vmin=P["vmin"], vmax=P["vmax"],
                           edgecolors="0.6", linewidth=0.25)
            axs.set_aspect("equal")
            lo, hi = urg_bins[k]
            axs.set_title(rf"${lo:g} \leq u < {hi:g}$", fontsize=8, pad=2)
            axs.set_xticks([0, 0.5, 1]); axs.set_yticks([0, 0.5, 1])
            axs.tick_params(labelsize=9)
            if k // 2 != 1:
                axs.set_xticklabels([])
            if k % 2 != 0:
                axs.set_yticklabels([])
            # block label (D/E/F) at the top-left corner of the block's TL tile
            if k == 0:
                blab = f"({'DEF'[c]})"
                axs.text(-0.30, 1.16, rf"\textbf{{{blab}}}" if usetex else blab,
                         transform=axs.transAxes, va="top", ha="left", fontsize=12)
            # label axes only on the first group's outer edges
            if c == 0 and k in (2, 3):
                axs.set_xlabel(r"$p_s$", fontsize=9)
            if c == 0 and k in (0, 2):
                axs.set_ylabel(r"$p_e$", fontsize=9)

    fig.text(0.5, 0.06, r"(D,E,F): same panels conditioned on link urgency "
             r"$u=\langle\mathrm{age/cutoff}\rangle$ in disjoint bins "
             r"(TL,TR,BL,BR $= [0,0.2),[0.2,0.35),[0.35,0.5),[0.5,0.65)$)",
             ha="center", fontsize=9)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf (usetex={usetex})")


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)
    out_json = os.path.join(out, "quality_map.json")
    data = json.load(open(out_json)) if a.plot else run_compute(a, out_json)
    run_plot(data, os.path.join(out, "quality_map"))


if __name__ == "__main__":
    main()
