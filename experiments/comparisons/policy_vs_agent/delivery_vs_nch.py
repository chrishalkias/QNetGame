"""
--------------------------------------------------------------------------------
#6  Delivery time T vs memory size n_ch, at fixed (N=10, p_swap=0.8, p_gen=0.3).

Grouped bars per n_ch: agent / swap-ASAP / purify-then-swap. n_ch>4 is past the
training set {2,3,4} (zero-shot in memory size).

  eval:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_nch.py --ckpt ...
  plot:  PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_nch.py --plot
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse
from experiments.comparisons import _common as C
from experiments.mc_eval import mc_eval_stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--p_gen", type=float, default=0.3)
    p.add_argument("--p_swap", type=float, default=0.8)
    p.add_argument("--n_chs", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--n_ch_train_max", type=int, default=4,
                   help="training ceiling for memory size; n_ch>this is zero-shot")
    p.add_argument("--cutoff", type=int, default=20)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--mc_eps", type=int, default=2000)
    p.add_argument("--out", default="results/comparisons/delivery_vs_nch.json")
    p.add_argument("--fig", default="results/figures/delivery_vs_nch")
    return p.parse_args()


def run_eval(a):
    pols = C.build_policies(a.ckpt)
    print(f"N={a.N} p_gen={a.p_gen} p_swap={a.p_swap} n_chs={a.n_chs} "
          f"cutoff={a.cutoff} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for nch in a.n_chs:
        row = dict(n_ch=nch, N=a.N, p_gen=a.p_gen, p_swap=a.p_swap,
                   cutoff=a.cutoff, n_ch_train_max=a.n_ch_train_max)
        for name, fn in pols.items():
            s = mc_eval_stats(fn, a.N, nch, a.p_gen, a.p_swap, a.cutoff,
                              a.horizon, a.mc_eps)
            T, se = s["T"], s["se"]
            row[f"T_{name}"], row[f"se_{name}"] = T, se
            print(f"  n_ch={nch} {name:<12} T={T:7.3f} ± {se:.3f}", flush=True)
        rows.append(row)
        C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(C.load_json(a.out), key=lambda r: r["n_ch"])
    nchs = [r["n_ch"] for r in rows]
    keys = ("agent", "swap_asap", "purify_swap")
    x = np.arange(len(nchs)); w = 0.26
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for j, key in enumerate(keys):
        T = [r[f"T_{key}"] for r in rows]
        se = [r[f"se_{key}"] for r in rows]
        ax.bar(x + (j - 1) * w, T, w, yerr=se, capsize=2,
               color=C.COLORS[key], label=C.LABELS[key])
    ntm = rows[0].get("n_ch_train_max", a.n_ch_train_max)
    # shade the zero-shot memory region (n_ch > training max)
    oos = [i for i, n in enumerate(nchs) if n > ntm]
    if oos:
        ax.axvspan(min(oos) - 0.5, len(nchs) - 0.5, color="grey", alpha=0.08, lw=0)
        ax.text(min(oos) - 0.45, ax.get_ylim()[1], " zero-shot $n_\\mathrm{ch}$",
                color="grey", fontsize=8, va="top")
    ax.set_xticks(x); ax.set_xticklabels(nchs)
    ax.set_xlabel(r"memory per node $n_\mathrm{ch}$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs memory size "
                 rf"($N={rows[0]['N']}$, $p_\mathrm{{gen}}={rows[0]['p_gen']}$, "
                 rf"$p_\mathrm{{swap}}={rows[0]['p_swap']}$, cutoff $={rows[0]['cutoff']}$)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
