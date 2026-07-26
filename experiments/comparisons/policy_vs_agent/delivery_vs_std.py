"""
--------------------------------------------------------------------------------
#7  Delivery time T vs inhomogeneity strength sigma, at fixed
(N=10, p_gen=0.5, p_swap=0.5, n_ch=4).

sigma sets BOTH p_gen_std and p_swap_std: per-repeater rates are drawn uniform
with std=sigma, centred on the mean, clipped to [0.05, 1.0] (see
network._sample_matched_uniform). sigma=0 is the homogeneous chain; the omni
agent was trained at sigma=0.15 (marked on the plot).

Three policies: agent / swap-ASAP / purify-then-swap.

  eval:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_std.py --ckpt ...
  plot:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_std.py --plot
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
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--sigma_train", type=float, default=0.15,
                    help="inhomogeneity the agent was trained at (dotted marker)")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_std.json")
    ap.add_argument("--fig", default="results/figures/inhom/delivery_vs_std")
    return ap.parse_args()


def run_eval(a):
    pols = C.build_policies(a.ckpt)
    print(f"N={a.N} p_gen={a.p_gen} p_swap={a.p_swap} n_ch={a.n_ch} "
          f"cutoff={a.cutoff} sigmas={a.sigmas} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for sg in a.sigmas:
        row = dict(sigma=sg, N=a.N, p_gen=a.p_gen, p_swap=a.p_swap, n_ch=a.n_ch,
                   cutoff=a.cutoff, sigma_train=a.sigma_train)
        for name, fn in pols.items():
            # sigma spreads BOTH p_gen and p_swap per repeater
            s = mc_eval_stats(fn, a.N, a.n_ch, a.p_gen, a.p_swap, a.cutoff,
                              a.horizon, a.mc_eps, p_gen_std=sg, p_swap_std=sg)
            T, se = s["T"], s["se"]
            row[f"T_{name}"], row[f"se_{name}"] = T, se
            print(f"  sigma={sg:.2f} {name:<12} T={T:7.3f} ± {se:.3f}", flush=True)
        rows.append(row)
        C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(C.load_json(a.out), key=lambda r: r["sigma"])
    xs = [r["sigma"] for r in rows]
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key in ("agent", "swap_asap", "purify_swap"):
        T = np.array([r[f"T_{key}"] for r in rows])
        se = np.array([r[f"se_{key}"] for r in rows])
        ax.plot(xs, T, marker="o", color=C.COLORS[key], label=C.LABELS[key], lw=1.6, ms=5)
        ax.fill_between(xs, T - se, T + se, color=C.COLORS[key], alpha=0.18, lw=0)
    st = rows[0].get("sigma_train")
    if st is not None:
        ax.axvline(st, color="grey", ls=":", lw=1.3)
        ax.text(st + 0.004, ax.get_ylim()[1], r" trained ($\sigma=0.15$)",
                color="grey", fontsize=8, va="top")
    ax.set_xlabel(r"inhomogeneity $\sigma$ (on $p_\mathrm{gen}$ and $p_\mathrm{swap}$)")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs inhomogeneity "
                 rf"($N={rows[0]['N']}$, $p_\mathrm{{gen}}={rows[0]['p_gen']}$, "
                 rf"$p_\mathrm{{swap}}={rows[0]['p_swap']}$, $n_\mathrm{{ch}}={rows[0]['n_ch']}$)")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
