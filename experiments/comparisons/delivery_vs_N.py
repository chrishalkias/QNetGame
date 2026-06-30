"""Delivery time T vs chain size N, at a fixed (p_gen, p_swap).

T = avg steps to end-to-end delivery (censored at the horizon, the same metric
as optimal_baseline.mc_eval / batch_validate). Three policies are compared:
agent (trained generalist), swap-ASAP, purify-then-swap.

N scans an in-distribution range and beyond; a dotted line marks the training
ceiling (N_train_max) past which the agent extrapolates (zero-shot).

Two modes (one file):
  eval (default, for the cluster) -> MC-evaluates and writes a JSON
      PYTHONPATH=. python experiments/comparisons/delivery_vs_N.py \
          --ckpt checkpoints/omni_nopen_3k/policy.pth
  plot (--plot, local) -> reads the JSON and renders the lineplot
      PYTHONPATH=. python experiments/comparisons/delivery_vs_N.py --plot
"""
from __future__ import annotations
import argparse, json, math, os


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="plot from --out json instead of evaluating")
    ap.add_argument("--metric", choices=["T", "delta"], default="T",
                    help="T = delivery-time lines; delta = %% reduction of agent "
                         "vs swap-ASAP (headline generalization plot, #2)")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--p_gen", type=float, default=0.4)
    ap.add_argument("--p_swap", type=float, default=0.8)
    ap.add_argument("--n_lo", type=int, default=10)
    ap.add_argument("--n_hi", type=int, default=15)
    ap.add_argument("--n_train_max", type=int, default=12,
                    help="training ceiling; dotted line, N>this is out-of-distribution")
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=2000)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_N.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_N")
    return ap.parse_args()


def run_eval(args):
    from experiments.heatmap import optimal_baseline as ob
    from rl_stack import strategies

    agent_fn = ob.make_agent_fn(args.ckpt, hidden=args.hidden)
    policies = {
        "agent":       agent_fn,
        "swap_asap":   lambda env, obs: strategies.swap_asap(env),
        "purify_swap": lambda env, obs: strategies.purify_then_swap(env),
    }
    Ns = list(range(args.n_lo, args.n_hi + 1))
    print(f"N={Ns} p_gen={args.p_gen} p_swap={args.p_swap} n_ch={args.n_ch} "
          f"cutoff={args.cutoff} H={args.horizon} mc_eps={args.mc_eps}")

    rows = []
    for N in Ns:
        row = dict(N=N, p_gen=args.p_gen, p_swap=args.p_swap, n_ch=args.n_ch,
                   cutoff=args.cutoff, horizon=args.horizon, mc_eps=args.mc_eps)
        for name, fn in policies.items():
            T, sd = ob.mc_eval(fn, N, args.n_ch, args.p_gen, args.p_swap,
                               args.cutoff, args.horizon, args.mc_eps)
            row[f"T_{name}"] = T
            row[f"se_{name}"] = sd / math.sqrt(args.mc_eps)
            print(f"  N={N:>2} {name:<12} T={T:7.3f} ± {row[f'se_{name}']:.3f}",
                  flush=True)
        rows.append(row)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)   # incremental save
    print(f"saved -> {args.out}")


def run_plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.out))
    Ns = [r["N"] for r in rows]

    if args.metric == "delta":
        # #2 headline: % delivery-time reduction of agent vs swap-ASAP.
        Ta = np.array([r["T_agent"] for r in rows])
        Ts = np.array([r["T_swap_asap"] for r in rows])
        delta = 100.0 * (Ts - Ta) / Ts
        plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        ax.plot(Ns, delta, marker="o", color="tab:blue", lw=1.8, ms=5)
        ax.axhline(0, color="k", lw=0.8, ls="-")
        ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
        ax.text(args.n_train_max + 0.08, ax.get_ylim()[1], " out-of-distribution →",
                color="grey", fontsize=8, va="top")
        pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
        ax.set_xlabel("chain size $N$")
        ax.set_ylabel("delivery-time reduction vs swap-ASAP (%)")
        ax.set_title(rf"Agent generalization "
                     rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                     rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$)")
        ax.set_xticks(Ns); ax.grid(alpha=0.3)
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.fig}_delta.{ext}", bbox_inches="tight")
        print(f"saved -> {args.fig}_delta.png / .pdf")
        return
    series = [("agent", "Agent", "tab:blue", "o"),
              ("swap_asap", "Swap-ASAP", "tab:orange", "s"),
              ("purify_swap", "Purify-then-swap", "tab:green", "^")]

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key, label, color, mk in series:
        T = np.array([r[f"T_{key}"] for r in rows])
        se = np.array([r[f"se_{key}"] for r in rows])
        ax.plot(Ns, T, marker=mk, color=color, label=label, lw=1.6, ms=5)
        ax.fill_between(Ns, T - se, T + se, color=color, alpha=0.18, lw=0)

    ntm = rows[0].get("n_train_max", args.n_train_max) if rows else args.n_train_max
    ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
    ax.text(args.n_train_max + 0.08, ax.get_ylim()[1], " out-of-distribution →",
            color="grey", fontsize=8, va="top")

    pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs chain size "
                 rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$)")
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.fig}.{ext}", bbox_inches="tight")
    print(f"saved -> {args.fig}.png / .pdf")


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
