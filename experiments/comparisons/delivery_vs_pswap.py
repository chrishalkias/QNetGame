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
    ap.add_argument("--augment_swapasap", action="store_true",
                    help="backfill the swap-ASAP baseline into an existing --out json "
                         "(MC-evals only swap-ASAP, keeps the agent column as-is)")
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
    print(f"N={a.N} p_gens={a.p_gens} p_swaps={a.p_swaps} n_ch={a.n_ch} "
          f"cutoff={a.cutoff} H={a.horizon} mc_eps={a.mc_eps}")
    rows = []
    for pg in a.p_gens:
        for ps in a.p_swaps:
            Ta, sea = C.eval_T(pols["agent"], a.N, a.n_ch, pg, ps, a.cutoff, a.horizon, a.mc_eps)
            Ts, ses = C.eval_T(pols["swap_asap"], a.N, a.n_ch, pg, ps, a.cutoff, a.horizon, a.mc_eps)
            rows.append(dict(p_gen=pg, p_swap=ps, N=a.N, n_ch=a.n_ch, cutoff=a.cutoff,
                             T_agent=Ta, se_agent=sea, T_swap_asap=Ts, se_swap_asap=ses))
            print(f"  p_gen={pg:.1f} p_swap={ps:.1f}  agent={Ta:7.3f}±{sea:.3f}  "
                  f"swapASAP={Ts:7.3f}±{ses:.3f}", flush=True)
            C.save_json(rows, a.out)
    print(f"saved -> {a.out}")


def run_plot(a):
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = C.load_json(a.out)
    pgs = sorted({r["p_gen"] for r in rows})
    has_swap = "T_swap_asap" in rows[0]
    cmap = plt.get_cmap("viridis")
    plt.rcParams.update(C.PLOT_RC)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    color_handles = []
    for i, pg in enumerate(pgs):
        sub = sorted([r for r in rows if r["p_gen"] == pg], key=lambda r: r["p_swap"])
        xs = [r["p_swap"] for r in sub]
        col = cmap(i / max(len(pgs) - 1, 1))
        T = np.array([r["T_agent"] for r in sub]); se = np.array([r["se_agent"] for r in sub])
        ax.plot(xs, T, marker="o", color=col, lw=1.7, ms=4)
        ax.fill_between(xs, T - se, T + se, color=col, alpha=0.15, lw=0)
        if has_swap:                                   # swap-ASAP overlay, dashed, same colour
            Ts = np.array([r["T_swap_asap"] for r in sub])
            ax.plot(xs, Ts, color=col, lw=1.3, ls="--", marker="x", ms=4)
        color_handles.append(Line2D([], [], color=col, lw=2, label=rf"$p_\mathrm{{gen}}={pg}$"))
    ax.set_xlabel(r"$p_\mathrm{swap}$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Agent (solid) vs swap-ASAP (dashed) "
                 rf"($N={rows[0]['N']}$, $n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                 rf"cutoff $={rows[0]['cutoff']}$)")
    ax.grid(alpha=0.3)
    leg1 = ax.legend(handles=color_handles, frameon=False, loc="upper right",
                     title=r"$p_\mathrm{gen}$", fontsize=8)
    ax.add_artist(leg1)
    if has_swap:
        style = [Line2D([], [], color="k", lw=1.7, label="Agent"),
                 Line2D([], [], color="k", lw=1.3, ls="--", marker="x", ms=4, label="Swap-ASAP")]
        ax.legend(handles=style, frameon=False, loc="center right", fontsize=8)
    C.savefig(fig, a.fig)


def run_augment(a):
    """Add swap-ASAP T to an existing agent-only json without re-evaluating the
    (expensive) agent column. swap-ASAP needs no checkpoint."""
    from rl_stack import strategies
    swap_fn = lambda env, obs: strategies.swap_asap(env)
    rows = C.load_json(a.out)
    todo = [r for r in rows if r.get("T_swap_asap") is None]
    print(f"augmenting {len(todo)}/{len(rows)} cells with swap-ASAP "
          f"({len(rows) - len(todo)} already done; H={a.horizon}, mc_eps={a.mc_eps})")
    for r in todo:                                   # resumable: skip done cells
        Ts, ses = C.eval_T(swap_fn, r["N"], r["n_ch"], r["p_gen"], r["p_swap"],
                           r["cutoff"], a.horizon, a.mc_eps)
        r["T_swap_asap"], r["se_swap_asap"] = Ts, ses
        print(f"  p_gen={r['p_gen']:.1f} p_swap={r['p_swap']:.1f}  "
              f"agent={r['T_agent']:7.2f}  swapASAP={Ts:7.2f}", flush=True)
        C.save_json(rows, a.out)
    print(f"augmented -> {a.out}")


if __name__ == "__main__":
    a = parse_args()
    if a.augment_swapasap:
        run_augment(a)
    elif a.plot:
        run_plot(a)
    else:
        run_eval(a)
