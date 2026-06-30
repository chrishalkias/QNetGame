"""#4  Agent delivery time T vs p_swap, one line per p_gen (5 lines).

Fixed N, cutoff, n_ch. Maps the agent's delivery-time surface over the
(p_swap, p_gen) operating regime (agent only).

  eval:  PYTHONPATH=. python experiments/comparisons/delivery_vs_pswap.py --ckpt ...
  plot:  PYTHONPATH=. python experiments/comparisons/delivery_vs_pswap.py --plot
"""
from __future__ import annotations
import argparse
from experiments.comparisons import _common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
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
    pols = C.build_policies(a.ckpt, hidden=a.hidden)
    agent = pols["agent"]
    print(f"N={a.N} p_gens={a.p_gens} p_swaps={a.p_swaps} n_ch={a.n_ch} "
          f"cutoff={a.cutoff} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for pg in a.p_gens:
        for ps in a.p_swaps:
            T, se = C.eval_T(agent, a.N, a.n_ch, pg, ps, a.cutoff, a.horizon, a.mc_eps)
            rows.append(dict(p_gen=pg, p_swap=ps, N=a.N, n_ch=a.n_ch,
                             cutoff=a.cutoff, T_agent=T, se_agent=se))
            print(f"  p_gen={pg:.1f} p_swap={ps:.1f}  T={T:7.3f} ± {se:.3f}", flush=True)
            C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = C.load_json(a.out)
    pgs = sorted({r["p_gen"] for r in rows})
    cmap = plt.get_cmap("viridis")
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for i, pg in enumerate(pgs):
        sub = sorted([r for r in rows if r["p_gen"] == pg], key=lambda r: r["p_swap"])
        xs = [r["p_swap"] for r in sub]
        T = np.array([r["T_agent"] for r in sub])
        se = np.array([r["se_agent"] for r in sub])
        col = cmap(i / max(len(pgs) - 1, 1))
        ax.plot(xs, T, marker="o", color=col, lw=1.6, ms=4,
                label=rf"$p_\mathrm{{gen}}={pg}$")
        ax.fill_between(xs, T - se, T + se, color=col, alpha=0.15, lw=0)
    ax.set_xlabel(r"$p_\mathrm{swap}$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Agent delivery time vs $p_\mathrm{{swap}}$ "
                 rf"($N={rows[0]['N']}$, $n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                 rf"cutoff $={rows[0]['cutoff']}$)")
    ax.grid(alpha=0.3); ax.legend(frameon=False, title=None)
    C.savefig(fig, a.fig)


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
