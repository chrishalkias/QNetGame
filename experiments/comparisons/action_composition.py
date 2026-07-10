"""#5  Agent action composition vs p_swap, at fixed (N=10, p_gen=0.5, cutoff).

Stacked bars: for each p_swap, the fraction of interior-node decisions that are
NOOP / SWAP / PURIFY (sums to 1). Explains the mechanism behind the agent's
delivery times (e.g. more purification under low p_swap). Agent only.

  eval:  PYTHONPATH=. python experiments/comparisons/action_composition.py --ckpt ...
  plot:  PYTHONPATH=. python experiments/comparisons/action_composition.py --plot
"""
from __future__ import annotations
import argparse
from experiments.comparisons import _common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--p_gen", type=float, default=0.5)
    ap.add_argument("--p_swaps", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/action_composition.json")
    ap.add_argument("--fig", default="results/figures/action_composition")
    return ap.parse_args()


def run_eval(a):
    pols = C.build_policies(a.ckpt, hidden=a.hidden)
    agent = pols["agent"]
    print(f"N={a.N} p_gen={a.p_gen} p_swaps={a.p_swaps} n_ch={a.n_ch} "
          f"cutoff={a.cutoff} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for ps in a.p_swaps:
        f_noop, f_swap, f_purify = C.action_fractions(
            agent, a.N, a.n_ch, a.p_gen, ps, a.cutoff, a.horizon, a.mc_eps)
        rows.append(dict(p_swap=ps, N=a.N, p_gen=a.p_gen, n_ch=a.n_ch,
                         cutoff=a.cutoff, f_noop=f_noop, f_swap=f_swap, f_purify=f_purify))
        print(f"  p_swap={ps:.1f}  noop={f_noop:.3f} swap={f_swap:.3f} "
              f"purify={f_purify:.3f}", flush=True)
        C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(C.load_json(a.out), key=lambda r: r["p_swap"])
    xs = [f"{r['p_swap']:.1f}" for r in rows]
    fnoop = np.array([r["f_noop"] for r in rows])
    fswap = np.array([r["f_swap"] for r in rows])
    fpur = np.array([r["f_purify"] for r in rows])
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.bar(xs, fnoop, color=C.ACTION_COLORS[0], label=C.ACTION_LABELS[0])
    ax.bar(xs, fswap, bottom=fnoop, color=C.ACTION_COLORS[1], label=C.ACTION_LABELS[1])
    ax.bar(xs, fpur, bottom=fnoop + fswap, color=C.ACTION_COLORS[2], label=C.ACTION_LABELS[2])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p_\mathrm{swap}$")
    ax.set_ylabel("fraction of interior-node decisions")
    ax.set_title(rf"Agent action composition "
                 rf"($N={rows[0]['N']}$, $p_\mathrm{{gen}}={rows[0]['p_gen']}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$, cutoff $={rows[0]['cutoff']}$)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
