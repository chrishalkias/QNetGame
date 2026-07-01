"""#3  Delivery time T vs memory cutoff tau, at fixed (p_gen=p_swap=0.5, N=10).

Three lines: agent / swap-ASAP / purify-then-swap. Shows behaviour under memory
pressure (small cutoff = links expire fast) where purification should pay off.

  eval:  PYTHONPATH=. python experiments/comparisons/delivery_vs_cutoff.py --ckpt ...
  plot:  PYTHONPATH=. python experiments/comparisons/delivery_vs_cutoff.py --plot
"""
from __future__ import annotations
import argparse
from experiments.comparisons import _common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_15k/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--p_gen", type=float, default=0.5)
    ap.add_argument("--p_swap", type=float, default=0.5)
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoffs", type=int, nargs="+",
                    default=[10, 15, 20, 25, 30, 35, 40])
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_cutoff.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_cutoff")
    return ap.parse_args()


def run_eval(a):
    pols = C.build_policies(a.ckpt, hidden=a.hidden)
    print(f"N={a.N} p_gen={a.p_gen} p_swap={a.p_swap} n_ch={a.n_ch} "
          f"cutoffs={a.cutoffs} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for ct in a.cutoffs:
        row = dict(cutoff=ct, N=a.N, p_gen=a.p_gen, p_swap=a.p_swap,
                   n_ch=a.n_ch, horizon=a.horizon, mc_eps=a.mc_eps)
        for name, fn in pols.items():
            T, se = C.eval_T(fn, a.N, a.n_ch, a.p_gen, a.p_swap, ct, a.horizon, a.mc_eps)
            row[f"T_{name}"], row[f"se_{name}"] = T, se
            print(f"  cutoff={ct:>2} {name:<12} T={T:7.3f} ± {se:.3f}", flush=True)
        rows.append(row)
        C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = C.load_json(a.out)
    xs = [r["cutoff"] for r in rows]
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key in ("agent", "swap_asap", "purify_swap"):
        T = np.array([r[f"T_{key}"] for r in rows])
        se = np.array([r[f"se_{key}"] for r in rows])
        ax.plot(xs, T, marker="o", color=C.COLORS[key], label=C.LABELS[key], lw=1.6, ms=5)
        ax.fill_between(xs, T - se, T + se, color=C.COLORS[key], alpha=0.18, lw=0)
    ax.set_xlabel(r"memory cutoff $\tau$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs memory cutoff "
                 rf"($N={rows[0]['N']}$, $p_\mathrm{{gen}}=p_\mathrm{{swap}}={rows[0]['p_gen']}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$)")
    ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend(frameon=False)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
