"""TEMP experiment: delivery time T vs chain size N WITH classical-communication
(CC) delays turned on. Same setup as results/figures/delivery_vs_N_15k.pdf, but
every deferred event (swap/purify) resolves only after a classical signal has
traversed the physical distance between the involved nodes.

Timing calibration ("1 step per hop"):
    delay_steps(d_km) = ceil(d_km / (c_fiber * dt_seconds))   [network.py]
    chain nodes sit at i*spacing, so distance(i,j) = |i-j| * spacing.
    Choosing dt_seconds = spacing / c_fiber makes a k-hop span resolve after
    exactly k steps (adjacent = 1 step). With spacing=50 km, c_fiber=2e5 km/s
    -> dt_seconds = 2.5e-4 s.

channel_loss is kept at 0, so distance affects ONLY the CC delay (not p_gen or
fidelity) -- this isolates the delay effect. The 15k agent was trained with
dt_seconds=0, so this is a zero-shot robustness probe.

  eval:  PYTHONPATH=. python experiments/temp/delivery_vs_N_ccdelay.py --ckpt ...
  plot:  PYTHONPATH=. python experiments/temp/delivery_vs_N_ccdelay.py --plot
"""
from __future__ import annotations
import argparse, json, math, os
import numpy as np

C_FIBER_KM_S = 200_000.0   # must match RepeaterNetwork.c_fiber


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/omni_initial/omni_nopen_15k/policy.pth")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--p_gen", type=float, default=0.4)
    ap.add_argument("--p_swap", type=float, default=0.8)
    ap.add_argument("--n_lo", type=int, default=5)
    ap.add_argument("--n_hi", type=int, default=12)
    ap.add_argument("--n_train_max", type=int, default=12,
                    help="training ceiling; dotted line, N>this is out-of-distribution")
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--spacing", type=float, default=50.0)
    ap.add_argument("--dt_seconds", type=float, default=-1.0,
                    help="<0 -> derive spacing/c_fiber (1 step/hop); 0 -> no CC delay")
    ap.add_argument("--horizon", type=int, default=800,
                    help="larger than the no-delay figure since CC delays slow delivery")
    ap.add_argument("--mc_eps", type=int, default=500)
    ap.add_argument("--agent_only", action="store_true",
                    help="eval only the agent (skip the checkpoint-independent heuristics)")
    ap.add_argument("--drop_swap_asap", action="store_true",
                    help="skip swap-ASAP (near-fully censored at large N in CC env -> "
                         "very slow); keep agent + purify-then-swap only")
    ap.add_argument("--out", default="results/comparisons/delivery_vs_N_ccdelay.json")
    ap.add_argument("--fig", default="results/figures/temp/delivery_vs_N_ccdelay")
    return ap.parse_args()


def dt_for(args):
    return args.spacing / C_FIBER_KM_S if args.dt_seconds < 0 else args.dt_seconds


def eval_T_cc(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, mc_eps, dt_seconds,
              spacing, seed=42):
    """(mean T, se): delivery time censored at H, on a chain with CC delays.
    Mirrors optimal_baseline.mc_eval but threads dt_seconds + spacing through."""
    from rl_stack.env_wrapper import QRNEnv
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(mc_eps):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     spacing=spacing, F0=1.0, channel_loss=0.0,
                     dt_seconds=dt_seconds, max_steps=H, topology="chain",
                     rng=np.random.default_rng(int(rng.integers(2**32))))
        obs = env.reset()
        step, done, info = 0, False, {}
        for step in range(H):
            obs, _, done, info = env.step(policy_fn(env, obs))
            if done:
                break
        times.append(step + 1 if (done and info.get("fidelity", 0.0) > 0) else H)
    T = np.asarray(times, float)
    return float(T.mean()), float(T.std() / math.sqrt(mc_eps))


def run_eval(args):
    from experiments.comparisons import _common as C
    dt = dt_for(args)
    pols = C.build_policies(args.ckpt, hidden=args.hidden)
    if args.agent_only:
        pols = {"agent": pols["agent"]}
    elif args.drop_swap_asap:
        pols.pop("swap_asap", None)
    Ns = list(range(args.n_lo, args.n_hi + 1))
    steps_per_hop = args.spacing / (C_FIBER_KM_S * dt) if dt > 0 else 0.0
    print(f"N={Ns} p_gen={args.p_gen} p_swap={args.p_swap} n_ch={args.n_ch} "
          f"cutoff={args.cutoff} H={args.horizon} mc_eps={args.mc_eps}")
    print(f"dt_seconds={dt:.3e}  spacing={args.spacing}  -> {steps_per_hop:.2f} step(s)/hop")
    rows = []
    for N in Ns:
        row = dict(N=N, p_gen=args.p_gen, p_swap=args.p_swap, n_ch=args.n_ch,
                   cutoff=args.cutoff, horizon=args.horizon, mc_eps=args.mc_eps,
                   dt_seconds=dt, spacing=args.spacing, steps_per_hop=steps_per_hop)
        for name, fn in pols.items():
            T, se = eval_T_cc(fn, N, args.n_ch, args.p_gen, args.p_swap, args.cutoff,
                              args.horizon, args.mc_eps, dt, args.spacing)
            row[f"T_{name}"], row[f"se_{name}"] = T, se
            print(f"  N={N:>2} {name:<12} T={T:8.3f} ± {se:.3f}", flush=True)
        rows.append(row)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(json.load(open(args.out)), key=lambda r: r["N"])
    Ns = [r["N"] for r in rows]
    sph = rows[0].get("steps_per_hop", 1.0)
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
    if args.n_train_max <= max(Ns):
        ax.axvline(args.n_train_max, color="grey", ls=":", lw=1.3)
        ax.text(args.n_train_max + 0.05, ax.get_ylim()[1], " OOD $N$ →",
                color="grey", fontsize=8, va="top")
    pg, ps = rows[0]["p_gen"], rows[0]["p_swap"]
    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title(rf"Delivery time vs $N$ with CC delays "
                 rf"($p_\mathrm{{gen}}={pg}$, $p_\mathrm{{swap}}={ps}$, "
                 rf"$n_\mathrm{{ch}}={rows[0]['n_ch']}$, {sph:g} step/hop)")
    ax.set_xticks(Ns); ax.grid(alpha=0.3); ax.legend(frameon=False)
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(f"{args.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {args.fig}.pdf")


if __name__ == "__main__":
    a = parse_args()
    (run_plot if a.plot else run_eval)(a)
