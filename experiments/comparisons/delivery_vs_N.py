"""Delivery time vs chain size N, at a fixed (p_gen, p_swap).

Primary metric: time to end-to-end ENTANGLEMENT T_ent = avg steps until the
source holds a link to the dest with delivered fidelity F > 1/2 (a two-qubit
Werner state is separable at F <= 1/2), censored at the horizon. Because the env
terminates on the FIRST topological connection, a separable first delivery is a
failure for T_ent (censored at H, no retry). The legacy time-to-connection
T_conn (any F > 0) is recorded from the same rollouts so older numbers stay
derivable. Three policies: agent (trained generalist), swap-ASAP,
purify-then-swap.

N scans an in-distribution range and beyond; a dotted line marks the training
ceiling (N_train_max) past which the agent extrapolates (zero-shot).

Two modes (one file):
  eval (default, for the cluster) -> MC-evaluates and writes a JSON
      PYTHONPATH=. python experiments/comparisons/delivery_vs_N.py \
          --ckpt checkpoints/sota/policy.pth
  plot (--plot, local) -> reads the JSON and renders the lineplot
      PYTHONPATH=. python experiments/comparisons/delivery_vs_N.py --plot
"""
from __future__ import annotations
import argparse, json, os


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="plot from --out json instead of evaluating")
    ap.add_argument("--metric", choices=["T", "delta"], default="T",
                    help="T = delivery-time lines; delta = %% reduction of agent "
                         "vs swap-ASAP (headline generalization plot, #2)")
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--p_gen", type=float, default=0.4)
    ap.add_argument("--p_swap", type=float, default=0.8)
    ap.add_argument("--n_lo", type=int, default=10)
    ap.add_argument("--n_hi", type=int, default=15)
    ap.add_argument("--n_train_max", type=int, default=12,
                    help="training ceiling; dotted line, N>this is out-of-distribution")
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--p_gen_std", type=float, default=0.0,
                    help="per-repeater inhomogeneity spread on p_gen (0 = homogeneous)")
    ap.add_argument("--p_swap_std", type=float, default=0.0,
                    help="per-repeater inhomogeneity spread on p_swap (0 = homogeneous)")
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--agent_only", action="store_true",
                    help="evaluate only the agent policy (skip the ckpt-independent "
                         "heuristics); used for the seed-sweep curves")
    ap.add_argument("--fidelity", action="store_true",
                    help="retained for back-compat: mean end-to-end fidelity is "
                         "now always recorded (free from the gated pass) and "
                         "overlaid when present")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_N.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_N")
    return ap.parse_args()


def run_eval(args):
    from experiments.heatmap import optimal_baseline as ob
    from experiments.comparisons._common import eval_gated
    from rl_stack import strategies

    agent_fn = ob.make_agent_fn(args.ckpt, hidden=args.hidden)
    policies = {"agent": agent_fn}
    if not args.agent_only:   # heuristics are ckpt-independent -> skip in seed sweeps
        policies["swap_asap"] = lambda env, obs: strategies.swap_asap(env)
        policies["purify_swap"] = lambda env, obs: strategies.purify_then_swap(env)
    Ns = list(range(args.n_lo, args.n_hi + 1))
    print(f"N={Ns} p_gen={args.p_gen} p_swap={args.p_swap} n_ch={args.n_ch} "
          f"cutoff={args.cutoff} H={args.horizon} mc_eps={args.mc_eps} "
          f"gate=F>1/2 (T_ent primary)")

    rows = []
    for N in Ns:
        row = dict(N=N, p_gen=args.p_gen, p_swap=args.p_swap, n_ch=args.n_ch,
                   cutoff=args.cutoff, horizon=args.horizon, mc_eps=args.mc_eps,
                   p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std)
        for name, fn in policies.items():
            # One paired pass yields the primary T_ent and the legacy T_conn.
            s = eval_gated(fn, N, args.n_ch, args.p_gen, args.p_swap,
                           args.cutoff, args.horizon, args.mc_eps,
                           p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std)
            # Existing key T_{name} keeps its connection meaning (schema additive).
            row[f"T_{name}"] = s["T_conn"]
            row[f"se_{name}"] = s["se_conn"]
            row[f"T_ent_{name}"] = s["T_ent"]
            row[f"se_ent_{name}"] = s["se_ent"]
            row[f"conn_rate_{name}"] = s["conn_rate"]
            row[f"ent_rate_{name}"] = s["ent_rate"]
            row[f"F_{name}"] = s["mean_F_conn"]        # mean F over connected eps
            row[f"seF_{name}"] = s["seF_conn"]
            row[f"Fent_{name}"] = s["mean_F_ent"]
            fstr = "nan" if s["mean_F_conn"] is None else f"{s['mean_F_conn']:.3f}"
            print(f"  N={N:>2} {name:<12} T_ent={s['T_ent']:7.3f}±{s['se_ent']:.3f}"
                  f"  T_conn={s['T_conn']:7.3f}  ent_rate={s['ent_rate']:.3f}"
                  f"  F_conn={fstr}", flush=True)
        rows.append(row)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)   # incremental save
    print(f"saved -> {args.out}")


def _primary_T(row, key):
    """(T, se, gated?) for a policy: entanglement-gated T_ent when present,
    otherwise the legacy connection T (old JSONs), flagged so labels can say so."""
    if f"T_ent_{key}" in row:
        return row[f"T_ent_{key}"], row.get(f"se_ent_{key}", 0.0), True
    return row[f"T_{key}"], row.get(f"se_{key}", 0.0), False


def run_plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.out))
    Ns = [r["N"] for r in rows]
    sig = rows[0].get("p_gen_std", 0.0)
    inh = rf", $\sigma_\mathrm{{inh}}={sig:g}$" if sig else ""   # inhomogeneity tag
    gated = "T_ent_agent" in rows[0]
    gate_tag = "" if gated else " (connection semantics)"

    if args.metric == "delta":
        # #2 headline: % delivery-time reduction of agent vs swap-ASAP, computed
        # from the gated T when available (falls back to connection for old JSONs).
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
        metric_name = "entanglement" if gated else "connection"
        ax.set_xlabel("chain size $N$")
        ax.set_ylabel(f"{metric_name}-time reduction vs swap-ASAP (%)")
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

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key, label, color, mk in series:
        T = np.array([_primary_T(r, key)[0] for r in rows])
        se = np.array([_primary_T(r, key)[1] for r in rows])
        ax.plot(Ns, T, marker=mk, color=color, label=label, lw=1.6, ms=5)
        ax.fill_between(Ns, T - se, T + se, color=color, alpha=0.18, lw=0)

    ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
    y1, y0 = ax.get_ylim()[1], ax.get_ylim()[0]
    drop = 0.10 * (y1 - y0)   # ~1cm lower on the 4in-tall axis
    ax.text(args.n_train_max + 0.08, y1 - drop, " out-of-distribution →",
            color="grey", fontsize=12, va="top")

    pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
    cut = rows[0].get("cutoff", args.cutoff)
    hor = rows[0].get("horizon", args.horizon)
    ylab = ("time to end-to-end entanglement $T_\\mathrm{ent}$ (steps)"
            if gated else "delivery time $T$ (avg steps to termination)")
    ax.set_xlabel("chain size $N$")
    ax.set_ylabel(ylab)
    ax.set_title(rf"Time to end-to-end entanglement vs chain size{gate_tag} "
                 rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$, "
                 rf"$\tau=\mathrm{{cutoff}}={cut}$, $H={hor}${inh})",
                 fontsize=9)
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)

    # twin axis: mean end-to-end fidelity F over CONNECTED episodes (dashed),
    # with the F=1/2 Werner separability line: below it a delivered link is
    # separable and does NOT count toward T_ent.
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
        ax2.axhline(0.5, color="firebrick", lw=1.0, ls=":")
        ax2.text(Ns[0], 0.5, " separable $F=1/2$", color="firebrick",
                 fontsize=7, va="bottom", ha="left")
        ax2.set_ylabel(r"mean end-to-end fidelity $\overline{F}$ (connected eps)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    else:
        ax.legend(frameon=False)

    # entangled-delivery rate for the agent, annotated per N (reliability that
    # the winning connection is actually entangled), when the gated data exists.
    if "ent_rate_agent" in rows[0]:
        er = [r["ent_rate_agent"] for r in rows]
        txt = "agent entangled-delivery rate: " + ", ".join(
            f"N{n}:{e:.2f}" for n, e in zip(Ns, er))
        fig.text(0.5, -0.02, txt, ha="center", va="top", fontsize=6.5,
                 color="tab:blue")

    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(f"{args.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {args.fig}.pdf")


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
