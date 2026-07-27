"""
--------------------------------------------------------------------------------
#4  Delivery time T vs p_swap, one line per p_gen (5 lines), single panel.

Agent (solid) and purify-then-swap (dashed) overlaid on one axis, coloured by
p_gen, over the (p_swap, p_gen) operating regime at fixed N, cutoff, n_ch.

  eval:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_pswap.py --ckpt ...
  plot:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_pswap.py --plot
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
from experiments.comparisons import _common as C
from experiments.mc_eval import mc_eval_stats


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--policies", nargs="+", default=["agent", "purify_swap"],
                    choices=["agent", "swap_asap", "purify_swap"],
                    help="policies to evaluate; swap-ASAP dropped from the "
                         "default roster (paper decision 2026-07-13). A single "
                         "policy lets SLURM array tasks split cells x policies")
    ap.add_argument("--logy", action="store_true", help="log-scale y axis")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--p_gens", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--p_swaps", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_pswap.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_pswap")
    return ap.parse_args()


def run_eval(a):
    pols = {k: v for k, v in C.build_policies(a.ckpt).items()
            if k in a.policies}
    C.write_meta(a)
    print(f"N={a.N} p_gens={a.p_gens} p_swaps={a.p_swaps} n_ch={a.n_ch} "
          f"cutoff={a.cutoff} H={a.horizon} mc_eps={a.mc_eps} pols={list(pols)}")
    rows = []
    for pg in a.p_gens:
        for ps in a.p_swaps:
            row = dict(p_gen=pg, p_swap=ps, N=a.N, n_ch=a.n_ch, cutoff=a.cutoff,
                       horizon=a.horizon, mc_eps=a.mc_eps)
            for name, fn in pols.items():
                s = mc_eval_stats(fn, a.N, a.n_ch, pg, ps, a.cutoff,
                                  a.horizon, a.mc_eps)
                row[f"T_{name}"], row[f"se_{name}"] = s["T"], s["se"]
                row[f"conn_rate_{name}"] = s["conn_rate"]
                row[f"F_{name}"] = s["mean_F_conn"]
                print(f"  p_gen={pg:.1f} p_swap={ps:.1f} {name:<12} "
                      f"T={s['T']:9.2f}  conn_rate={s['conn_rate']:.3f}", flush=True)
            rows.append(row)
            C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = C.load_json(a.out)
    pgs = sorted({r["p_gen"] for r in rows})
    cmap = plt.get_cmap("viridis")
    cols = [cmap(i / max(len(pgs) - 1, 1)) for i in range(len(pgs))]
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    # agent solid, purify-then-swap dashed; both coloured by p_gen
    styles = [("agent", "-", "o"), ("purify_swap", "--", "s")]
    for i, pg in enumerate(pgs):
        sub = sorted([r for r in rows if r["p_gen"] == pg], key=lambda r: r["p_swap"])
        xs = [r["p_swap"] for r in sub]
        for key, ls, mk in styles:
            T = np.array([r[f"T_{key}"] for r in sub])
            se = np.array([r[f"se_{key}"] for r in sub])
            ax.plot(xs, T, marker=mk, ls=ls, color=cols[i], lw=1.7, ms=4)
            ax.fill_between(xs, T - se, T + se, color=cols[i], alpha=0.12, lw=0)
    ax.set_xlabel(r"$p_\mathrm{swap}$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    if a.logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    # two legends: p_gen colour + policy line-style
    pg_handles = [Line2D([], [], color=cols[i], lw=2, label=rf"${pg}$")
                  for i, pg in enumerate(pgs)]
    style_handles = [Line2D([], [], color="grey", ls="-", marker="o", label="Agent"),
                     Line2D([], [], color="grey", ls="--", marker="s", label="Purify-then-swap")]
    leg1 = ax.legend(handles=pg_handles, frameon=False, title=r"$p_\mathrm{gen}$",
                     fontsize=8, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, frameon=False, fontsize=8, loc="lower left")
    ax.set_title(rf"Delivery time vs $p_\mathrm{{swap}}$ "
                 rf"($N={rows[0]['N']}$, $n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                 rf"cutoff $={rows[0]['cutoff']}$)", fontsize=11)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    if a.plot:
        run_plot(a)
    else:
        run_eval(a)
