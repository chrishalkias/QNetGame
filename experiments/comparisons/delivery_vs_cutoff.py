"""
--------------------------------------------------------------------------------
#3  Delivery time T vs memory cutoff tau, at fixed (p_gen=p_swap=0.5, N=10).

Two policies (agent / purify-then-swap). Solid = delivery time T (left axis);
dashed = mean terminal fidelity of delivered links (right axis). Shows behaviour
under memory pressure (small cutoff = links expire fast + decohere faster, since
werner ~ exp(-age/cutoff)) where purification should pay off.

  eval:  PYTHONPATH=src:. python experiments/comparisons/delivery_vs_cutoff.py --ckpt ...
  plot:  PYTHONPATH=src:. python experiments/comparisons/delivery_vs_cutoff.py --plot
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
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--p_gen", type=float, default=0.5)
    ap.add_argument("--p_swap", type=float, default=0.5)
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoffs", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30, 35, 40, 50])
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_cutoff.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_cutoff")
    return ap.parse_args()


POLICIES = ("agent", "purify_swap")   # swap-ASAP dropped from this figure


def run_eval(a):
    allpols = C.build_policies(a.ckpt)
    pols = {k: allpols[k] for k in POLICIES}
    print(f"N={a.N} p_gen={a.p_gen} p_swap={a.p_swap} n_ch={a.n_ch} "
          f"cutoffs={a.cutoffs} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for ct in a.cutoffs:
        row = dict(cutoff=ct, N=a.N, p_gen=a.p_gen, p_swap=a.p_swap,
                   n_ch=a.n_ch, horizon=a.horizon, mc_eps=a.mc_eps)
        for name, fn in pols.items():
            s = mc_eval_stats(fn, a.N, a.n_ch, a.p_gen, a.p_swap,
                              ct, a.horizon, a.mc_eps)
            T, se, Fse = s["T"], s["se"], s["seF_conn"]
            # nan (not None) when nothing delivered: the figure plots F as a
            # float array and a gap is the honest rendering of "no data"
            F = s["mean_F_conn"] if s["mean_F_conn"] is not None else float("nan")
            row[f"T_{name}"], row[f"se_{name}"] = T, se
            row[f"F_{name}"], row[f"Fse_{name}"] = F, Fse
            print(f"  cutoff={ct:>2} {name:<12} T={T:7.3f} ± {se:.3f}  "
                  f"F={F:.3f} ± {Fse:.3f}", flush=True)
        rows.append(row)
        C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    from matplotlib.lines import Line2D
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(C.load_json(a.out), key=lambda r: r["cutoff"])
    xs = [r["cutoff"] for r in rows]
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axf = ax.twinx()   # right axis: terminal fidelity (dashed)
    for key in POLICIES:
        T = np.array([r[f"T_{key}"] for r in rows])
        se = np.array([r[f"se_{key}"] for r in rows])
        ax.plot(xs, T, marker="o", color=C.COLORS[key], label=C.LABELS[key], lw=1.6, ms=5)
        ax.fill_between(xs, T - se, T + se, color=C.COLORS[key], alpha=0.18, lw=0)
        F = np.array([r[f"F_{key}"] for r in rows])
        Fse = np.array([r[f"Fse_{key}"] for r in rows])
        axf.plot(xs, F, marker="s", ls="--", color=C.COLORS[key], lw=1.4, ms=4, alpha=0.9)
        axf.fill_between(xs, F - Fse, F + Fse, color=C.COLORS[key], alpha=0.10, lw=0)
    ax.set_xlabel(r"memory coherence time $\tau$  ($1/e$ discard)")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    axf.set_ylabel(r"mean terminal fidelity $\bar{F}$ (dashed)")
    axf.set_ylim(top=1.0)
    ax.set_title(rf"Delivery time & delivered fidelity vs memory coherence time "
                 rf"($N={rows[0]['N']}$, $p_\mathrm{{gen}}=p_\mathrm{{swap}}={rows[0]['p_gen']}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$)")
    ax.set_xticks(xs); ax.grid(alpha=0.3)
    # legend: policy colours + line-style key (solid=T, dashed=F)
    handles = [Line2D([], [], color=C.COLORS[k], marker="o", lw=1.6, label=C.LABELS[k])
               for k in POLICIES]
    handles += [Line2D([], [], color="grey", ls="-", label=r"delivery time $T$"),
                Line2D([], [], color="grey", ls="--", label=r"fidelity $\bar{F}$")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center right")
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
