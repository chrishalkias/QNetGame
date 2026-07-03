"""Two-panel gap-to-optimal heatmap (swap-only vs purify-enabled agent).

Color = gap% = 100 * (T_agent - T_opt) / T_opt over the (p_gen, p_swap) grid.
Diverging RdBu_r centered at 0: blue (<0) = agent beats the swap-only DP optimum
(purification frees memory); white (=0) = matches optimal; red (>0) = slower.

  PYTHONPATH=. python experiments/heatmap/plot_heatmap_gap.py
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def to_grid(rows, key, pgs, pss):
    ig = {v: i for i, v in enumerate(pgs)}
    js = {v: j for j, v in enumerate(pss)}
    M = np.full((len(pgs), len(pss)), np.nan)
    for r in rows:
        M[ig[round(r["p_gen"], 2)], js[round(r["p_swap"], 2)]] = r[key]
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="results/heatmaps/heatmap_gap_N4_9x9.json")
    ap.add_argument("--out", default="results/figures/heatmap_gap_N4")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--vmax", type=float, default=45.0,
                    help="symmetric color cap in %% (cells beyond saturate)")
    ap.add_argument("--title_a", default="Without purification (swap-only)")
    ap.add_argument("--title_b", default="With purification")
    ap.add_argument("--suptitle",
                    default=r"Trained agent vs swap-only optimum  ($N={N}$, "
                            r"$n_\mathrm{ch}=2$, cutoff $=5$)")
    ap.add_argument("--cbar_label", default="Gap to optimal delivery time (%)")
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(args.data))
    pgs = sorted({round(r["p_gen"], 2) for r in rows})
    pss = sorted({round(r["p_swap"], 2) for r in rows})
    G_swo = to_grid(rows, "gap_swaponly_pct", pgs, pss)
    G_pur = to_grid(rows, "gap_purify_pct", pgs, pss)

    vmax = float(args.vmax)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_over(cmap(1.0))   # cells above +vmax saturate to deep red
    cmap.set_under(cmap(0.0))
    n_clip = int(np.nansum(G_swo > vmax) + np.nansum(G_pur > vmax)
                 + np.nansum(G_swo < -vmax) + np.nansum(G_pur < -vmax))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    dpg = (pgs[1] - pgs[0]) / 2 if len(pgs) > 1 else 0.05
    dps = (pss[1] - pss[0]) / 2 if len(pss) > 1 else 0.05
    extent = [pss[0] - dps, pss[-1] + dps, pgs[0] - dpg, pgs[-1] + dpg]
    im = None
    for ax, (title, M, lab) in zip(
            axes,
            [(args.title_a, G_swo, "A"),
             (args.title_b, G_pur, "B")]):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                       norm=norm, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(r"$p_\mathrm{swap}$")
        ax.set_xticks(pss)
        ax.set_yticks(pgs)
        ax.text(-0.18, 1.05, lab, transform=ax.transAxes, fontweight="bold",
                fontsize=11, va="top")
        # label the two extreme cells: most-negative (bluest) + most-positive
        # (reddest). White text on saturated cells, black on near-white ones.
        if not np.all(np.isnan(M)):
            for idx in (np.nanargmin(M), np.nanargmax(M)):
                ei, ej = np.unravel_index(idx, M.shape)
                v = M[ei, ej]
                tc = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(pss[ej], pgs[ei], f"{v:.0f}", ha="center",
                        va="center", fontsize=8, fontweight="bold", color=tc)
        if args.annotate:
            for i, pg in enumerate(pgs):
                for j, ps in enumerate(pss):
                    if not np.isnan(M[i, j]):
                        ax.text(ps, pg, f"{M[i, j]:.0f}", ha="center",
                                va="center", fontsize=5, color="k")
    axes[0].set_ylabel(r"$p_\mathrm{gen}$")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend="both")
    cbar.set_label(args.cbar_label)
    fig.suptitle(args.suptitle.replace("{N}", str(args.N)), fontsize=10)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.pdf", dpi=300, bbox_inches="tight")
    print(f"saved -> {args.out}.pdf   ({n_clip} cells saturate |gap|>{vmax:.0f}%)")
    print(f"swap-only gap%: mean={np.nanmean(G_swo):+.2f} "
          f"range=[{np.nanmin(G_swo):+.1f}, {np.nanmax(G_swo):+.1f}]")
    print(f"purify    gap%: mean={np.nanmean(G_pur):+.2f} "
          f"range=[{np.nanmin(G_pur):+.1f}, {np.nanmax(G_pur):+.1f}]  "
          f"cells beating optimal: {int(np.nansum(G_pur < 0))}/{G_pur.size}")


if __name__ == "__main__":
    main()
