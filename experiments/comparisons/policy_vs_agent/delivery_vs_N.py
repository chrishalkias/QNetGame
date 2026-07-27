"""
--------------------------------------------------------------------------------
Delivery time vs chain size N, at a fixed (p_gen, p_swap).

Metric: delivery time T = avg steps until the source holds a topological link
to the dest (source-dest connected), censored at the horizon. Delivery is
topological, not fidelity-gated: the cutoff already bounds how decohered a
surviving link can be. Three policies: agent (trained generalist), swap-ASAP,
purify-then-swap.

N scans an in-distribution range and beyond; a dotted line marks the training
ceiling (N_train_max) past which the agent extrapolates (zero-shot).

Two modes (one file):
  eval (default, for the cluster) -> MC-evaluates and writes a JSON
      PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_N.py \
          --ckpt checkpoints/sota/policy.pth
  plot (--plot, local) -> reads the JSON and renders the lineplot
      PYTHONPATH=src:. python experiments/comparisons/policy_vs_agent/delivery_vs_N.py --plot
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", action="store_true",
                   help="plot from --out json instead of evaluating")
    p.add_argument("--metric", choices=["T", "delta"], default="T",
                   help="T = delivery-time lines; delta = %% reduction of agent "
                   "vs swap-ASAP (headline generalization plot, #2)")
    p.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    p.add_argument("--p_gen", type=float, default=0.4)
    p.add_argument("--p_swap", type=float, default=0.8)
    p.add_argument("--n_lo", type=int, default=10)
    p.add_argument("--n_hi", type=int, default=15)
    p.add_argument("--n_train_max", type=int, default=12,
                   help="training ceiling; dotted line, N>this is out-of-distribution")
    p.add_argument("--n_ch", type=int, default=4)
    p.add_argument("--p_gen_std", type=float, default=0.0,
                   help="per-repeater inhomogeneity spread on p_gen (0 = homogeneous)")
    p.add_argument("--p_swap_std", type=float, default=0.0,
                   help="per-repeater inhomogeneity spread on p_swap (0 = homogeneous)")
    p.add_argument("--cutoff", type=int, default=20)
    p.add_argument("--agent_only", action="store_true",
                   help="evaluate only the agent policy (skip the ckpt-independent "
                   "heuristics); used for the seed-sweep curves")
    p.add_argument("--policies", nargs="+", default=["agent", "purify_swap"],
                   choices=["agent", "swap_asap", "purify_swap"],
                   help="policies to evaluate; swap-ASAP dropped from the "
                   "default roster (paper decision 2026-07-13). A single "
                   "policy lets SLURM array tasks split N x policies")
    p.add_argument("--logy", action="store_true", help="log-scale y axis")
    p.add_argument("--legend_loc", default="upper left",
                   help="matplotlib legend location for the T-metric plot")
    p.add_argument("--fidelity", action="store_true",
                   help="retained for back-compat: mean end-to-end fidelity is "
                   "now always recorded (free from the gated pass) and "
                   "overlaid when present")
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--mc_eps", type=int, default=2000)
    p.add_argument("--out", default="results/comparisons/delivery_vs_N.json")
    p.add_argument("--fig", default="results/figures/delivery_vs_N")
    return p.parse_args()


def run_eval(args):
    from experiments import mc_eval as ob
    from experiments.mc_eval import mc_eval_stats
    from rl_stack import policies

    wanted = ["agent"] if args.agent_only else args.policies
    # NOT named `policies`: that would shadow the rl_stack.policies module the
    # two baseline lambdas close over (it did, and both raised AttributeError)
    pols = {}
    if "agent" in wanted:
        pols["agent"] = ob.make_agent_fn(args.ckpt)
    if "swap_asap" in wanted:
        pols["swap_asap"] = lambda env, obs: policies.swap_asap(env)
    if "purify_swap" in wanted:
        pols["purify_swap"] = lambda env, obs: policies.purify_then_swap(env)
    from experiments.comparisons import _common as C
    C.write_meta(args)
    Ns = list(range(args.n_lo, args.n_hi + 1))
    print(f"N={Ns} p_gen={args.p_gen} p_swap={args.p_swap} n_ch={args.n_ch} "
          f"cutoff={args.cutoff} H={args.horizon} mc_eps={args.mc_eps}")

    rows = []
    for N in Ns:
        row = dict(N=N, p_gen=args.p_gen, p_swap=args.p_swap, n_ch=args.n_ch,
                   cutoff=args.cutoff, horizon=args.horizon, mc_eps=args.mc_eps,
                   p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std)
        for name, fn in pols.items():
            s = mc_eval_stats(fn, N, args.n_ch, args.p_gen, args.p_swap,
                              args.cutoff, args.horizon, args.mc_eps,
                              p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std)
            row[f"T_{name}"] = s["T"]
            row[f"se_{name}"] = s["se"]
            row[f"conn_rate_{name}"] = s["conn_rate"]
            row[f"F_{name}"] = s["mean_F_conn"]        # mean F over connected eps
            row[f"seF_{name}"] = s["seF_conn"]
            fstr = "nan" if s["mean_F_conn"] is None else f"{s['mean_F_conn']:.3f}"
            print(f"  N={N:>2} {name:<12} T={s['T']:7.3f}±{s['se']:.3f}"
                  f"  conn_rate={s['conn_rate']:.3f}  F_conn={fstr}", flush=True)
        rows.append(row)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)   # incremental save
    print(f"saved -> {args.out}")


def _primary_T(row, key):
    """(T, se) for a policy at this N; NaN if the policy is absent (ragged tail)."""
    if f"T_{key}" in row:
        return row[f"T_{key}"], row.get(f"se_{key}", 0.0)
    return float("nan"), 0.0


def run_plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.out))
    Ns = [r["N"] for r in rows]
    sig = rows[0].get("p_gen_std", 0.0)
    inh = rf", $\sigma_\mathrm{{inh}}={sig:g}$" if sig else ""   # inhomogeneity tag

    if args.metric == "delta":
        # #2 headline: % delivery-time reduction of agent vs swap-ASAP.
        Ta = np.array([_primary_T(r, "agent")[0] for r in rows])
        Ts = np.array([_primary_T(r, "swap_asap")[0] for r in rows])
        delta = 100.0 * (Ts - Ta) / Ts
        plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        ax.plot(Ns, delta, marker="o", color="tab:blue", lw=1.8, ms=5)
        ax.axhline(0, color="k", lw=0.8, ls="-")
        ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
        ax.text(args.n_train_max + 0.08, ax.get_ylim()[1], " out-of-distribution →",
                color="grey", fontsize=8, va="top")
        pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
        cut = rows[0].get("cutoff", args.cutoff)
        hor = rows[0].get("horizon", args.horizon)
        ax.set_xlabel("chain size $N$")
        ax.set_ylabel("delivery-time reduction vs swap-ASAP (%)")
        ax.set_title(rf"Agent generalization "
                     rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                     rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                     rf"$\tau=\mathrm{{cutoff}}={cut}$, $H={hor}${inh})",
                     fontsize=9)
        ax.set_xticks(Ns); ax.grid(alpha=0.3)
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        fig.savefig(f"{args.fig}_delta.pdf", bbox_inches="tight")
        print(f"saved -> {args.fig}_delta.pdf")
        return
    series = [("agent", "Agent", "tab:blue", "o"),
              ("swap_asap", "Swap-ASAP", "tab:orange", "s"),
              ("purify_swap", "Purify-then-swap", "tab:green", "^")]
    # tolerate JSONs evaluated with a reduced policy roster (no swap-ASAP)
    series = [s for s in series if f"T_{s[0]}" in rows[0]]

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key, label, color, mk in series:
        T = np.array([_primary_T(r, key)[0] for r in rows])
        se = np.array([_primary_T(r, key)[1] for r in rows])
        ax.plot(Ns, T, marker=mk, color=color, label=label, lw=1.6, ms=5)
        ax.fill_between(Ns, T - se, T + se, color=color, alpha=0.18, lw=0)

    ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
    # label hugs the bottom of the dividing line (x in data coords, y in axes
    # fraction) so it cannot collide with curves on either linear or log axes
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(args.n_train_max + 0.12, 0.04, "out-of-distribution →",
            color="grey", fontsize=9, va="bottom", transform=trans)

    pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
    cut = rows[0].get("cutoff", args.cutoff)
    hor = rows[0].get("horizon", args.horizon)
    # a cutoff far beyond any horizon is operationally infinite (perfect memory)
    cut_lab = r"\infty" if cut >= 10**6 else str(cut)
    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs chain size "
                 rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                 rf"$\tau=\mathrm{{cutoff}}={cut_lab}$, $H={hor}${inh})",
                 fontsize=9)
    ax.set_xticks(Ns)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # twin axis: mean end-to-end fidelity F over delivered episodes (dashed).
    have_F = rows[0].get("F_agent") is not None
    if have_F:
        ax2 = ax.twinx()
        for key, color, mk in (("agent", "tab:blue", "o"),
                               ("purify_swap", "tab:green", "^")):
            F = np.array([np.nan if r.get(f"F_{key}") is None else r[f"F_{key}"]
                          for r in rows])
            seF = np.array([r.get(f"seF_{key}") or 0.0 for r in rows])
            lab = dict(agent="Agent", purify_swap="Purify-then-swap")[key]
            ax2.plot(Ns, F, marker=mk, color=color, ls="--", lw=1.4, ms=4,
                     label=rf"{lab} $\overline{{F}}$")
            ax2.fill_between(Ns, F - seF, F + seF, color=color, alpha=0.12, lw=0)
        ax2.set_ylabel(r"mean end-to-end fidelity $\overline{F}$ (delivered eps)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8,
                  loc=args.legend_loc)
    else:
        ax.legend(frameon=False, loc=args.legend_loc)

    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(f"{args.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {args.fig}.pdf")


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
