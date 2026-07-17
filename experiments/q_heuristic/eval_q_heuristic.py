"""Evaluate the stochastic q-heuristic against purify_then_swap (and, optionally,
the trained agents) over a range of chain sizes N.

Question under test: does a single scalar q (the both-legal purify probability
inside the purify_then_swap skeleton) recover the trained GNN agent's edge over
purify_then_swap? If yes, the benefit is "don't always purify" (a rate), not
smart state-selection. See hybrid_policy.py for the policy and the RNG-
independence invariant.

Everything reuses the repo untouched:
  - experiments.heatmap.optimal_baseline.mc_eval : THE canonical delivery-time
    evaluator (censored at H, seeded), used with return_stats=True.
  - experiments.heatmap.optimal_baseline.make_agent_fn / swap_asap_fn.
  - experiments.q_heuristic.hybrid_policy : the q-heuristic and the ptswap wrapper.

Dual mode (one file), mirroring experiments/comparisons house style:
  eval (default, cluster) -> MC-evaluates and writes JSON incrementally,
      resuming from any existing --out (walltime kills lose nothing).
  plot (--plot, local)    -> reads one or more JSONs (glob in --out) and renders
      T-vs-N per policy, no recompute.
  sanity (--sanity)       -> the q=1.0 bit-identity self-test; exits nonzero on
      failure.

Only purify_then_swap and the hybrid q list are torch-free; pass --agents ''
(empty) to keep the whole run numpy-only. torch is imported lazily by
make_agent_fn, only when --agents is non-empty.

Examples (from repo root, PYTHONPATH=.):
  python experiments/q_heuristic/eval_q_heuristic.py --sanity
  python experiments/q_heuristic/eval_q_heuristic.py --N 8 13 --agents ''
  python experiments/q_heuristic/eval_q_heuristic.py --plot --out 'results/comparisons/q_heuristic/eval_*.json'
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys


DEFAULT_AGENTS = ["checkpoints/omni_v3_20k_s1/policy.pth",
                  "checkpoints/omni_v3_20k_s3/policy.pth"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="render T-vs-N from --out (may be a glob) instead of evaluating")
    ap.add_argument("--sanity", action="store_true",
                    help="run the q=1.0 bit-identity self-test and exit (nonzero on failure)")
    ap.add_argument("--N", type=int, nargs="+", default=[8, 13],
                    help="chain sizes to evaluate")
    ap.add_argument("--q", type=float, nargs="+", default=[0.215, 0.369, 0.5],
                    help="both-legal purify probabilities for the hybrid roster")
    ap.add_argument("--agents", nargs="*", default=DEFAULT_AGENTS,
                    help="trained-agent checkpoints (torch); empty list = "
                         "heuristics+hybrids only, fully numpy-only")
    ap.add_argument("--include_swap_asap", action="store_true",
                    help="also evaluate swap_asap alongside purify_then_swap")
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--p_gen", type=float, default=0.4)
    ap.add_argument("--p_swap", type=float, default=0.8)
    ap.add_argument("--cutoff", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="JSON path; default results/comparisons/q_heuristic/eval_{tag}.json")
    ap.add_argument("--fig", default=None,
                    help="plot output stem (PDF); default derived from the first --out")
    ap.add_argument("--logy", action="store_true", help="log-scale y axis in --plot")
    ap.add_argument("--chunk", type=int, default=0,
                    help="this task's index in [0, nchunks) for SLURM arrays")
    ap.add_argument("--nchunks", type=int, default=1,
                    help="split the (N x policy) work list round-robin; >1 forces "
                         "a per-chunk output file (never a shared JSON)")
    return ap.parse_args(argv)


# --------------------------- naming / io helpers -----------------------------
def _tag(args):
    return f"nch{args.n_ch}_pg{args.p_gen:g}_ps{args.p_swap:g}_co{args.cutoff}"


def _default_out(args):
    return f"results/comparisons/q_heuristic/eval_{_tag(args)}.json"


def _chunked_path(path, chunk, nchunks):
    """Insert _chunk{chunk} before the extension when splitting the work list."""
    if nchunks <= 1:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_chunk{chunk}{ext}"


def _hybrid_seed(q):
    """Deterministic per-q seed for the hybrid coin, independent of env.rng and
    of ordering (so the same q reproduces the same coin stream every run)."""
    return 1000 + int(round(q * 1000))


def _agent_name(ckpt):
    """agent_<run-dir> from checkpoints/<run-dir>/policy.pth."""
    return "agent_" + os.path.basename(os.path.dirname(ckpt))


def _load_existing(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ------------------------------ policy roster --------------------------------
def build_policies(args):
    """Ordered {name: policy_fn(env, obs)}. purify_then_swap first, then the
    hybrid q list, then agents (torch, only if requested), then swap_asap. The
    order fixes the (N x policy) work list so chunk assignment is stable."""
    from experiments.q_heuristic.hybrid_policy import (
        make_hybrid_fn, purify_then_swap_fn)

    policies = {"purify_then_swap": purify_then_swap_fn}
    for q in args.q:
        policies[f"hybrid_q{q:g}"] = make_hybrid_fn(q=q, seed=_hybrid_seed(q))

    agents = [a for a in (args.agents or []) if a]   # drop the empty-string sentinel
    if agents:
        from experiments.heatmap.optimal_baseline import make_agent_fn
        for ckpt in agents:
            policies[_agent_name(ckpt)] = make_agent_fn(ckpt, hidden=64)

    if args.include_swap_asap:
        from experiments.heatmap.optimal_baseline import swap_asap_fn
        policies["swap_asap"] = swap_asap_fn
    return policies, [_agent_name(a) for a in agents]


def _recovery(cell, agent_names):
    """Per-agent recovery fraction (T_pts - T_x) / (T_pts - T_agent) for each
    hybrid present in this cell (None when the denominator is 0)."""
    if "purify_then_swap" not in cell:
        return None
    T_pts = cell["purify_then_swap"]["T"]
    hybrid_names = [n for n in cell if n.startswith("hybrid")]
    rec = {}
    for aname in agent_names:
        if aname not in cell:
            continue
        T_agent = cell[aname]["T"]
        denom = T_pts - T_agent
        rec[aname] = dict(
            T_agent=T_agent, denom_T_pts_minus_T_agent=denom,
            recovery_by_hybrid={
                hn: (None if denom == 0 else (T_pts - cell[hn]["T"]) / denom)
                for hn in hybrid_names},
        )
    return rec or None


# --------------------------------- eval mode ---------------------------------
def run_eval(args):
    from experiments.heatmap.optimal_baseline import mc_eval
    from experiments.comparisons import _common as C

    out = _chunked_path(args.out or _default_out(args), args.chunk, args.nchunks)
    args.out = out                                   # for write_meta / plot fallback
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    policies, agent_names = build_policies(args)
    C.write_meta(args, extra=dict(
        evaluator="optimal_baseline.mc_eval(return_stats=True, seed=%d)" % args.seed,
        policies=list(policies)))

    # (N, policy) work list, round-robin over chunks.
    worklist = [(N, name) for N in args.N for name in policies]
    mine = [w for i, w in enumerate(worklist) if i % args.nchunks == args.chunk]

    payload = _load_existing(out) or {}
    payload["config"] = dict(
        N_list=args.N, q_list=args.q, n_ch=args.n_ch, p_gen=args.p_gen,
        p_swap=args.p_swap, cutoff=args.cutoff, H=args.horizon,
        n_episodes=args.episodes, seed=args.seed, agents=agent_names,
        chunk=args.chunk, nchunks=args.nchunks)
    results = payload.setdefault("results", {})

    def flush():
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

    hdr = f"{'N':>3} {'policy':>20} {'T':>9} {'T_std':>8} {'conn':>6} {'mean_F':>8}"
    print(hdr)
    print("-" * len(hdr))

    touched_cells = set()
    for N, name in mine:
        cell_key = f"N={N}"
        cell = results.setdefault(cell_key, {})
        if name in cell:
            print(f"{N:>3} {name:>20}  (already done, skipped)", flush=True)
            touched_cells.add(cell_key)
            continue
        stats = mc_eval(policies[name], N, args.n_ch, args.p_gen, args.p_swap,
                        args.cutoff, args.horizon, args.episodes, seed=args.seed,
                        return_stats=True)
        cell[name] = stats
        touched_cells.add(cell_key)
        mf = stats["mean_F_conn"]
        mf_s = f"{mf:.4f}" if mf is not None else "  None"
        print(f"{N:>3} {name:>20} {stats['T']:>9.3f} {stats['T_std']:>8.3f} "
              f"{stats['conn_rate']:>6.3f} {mf_s:>8}", flush=True)
        flush()                                      # incremental save every point

    # recompute recovery for every cell this task touched (guarded on presence).
    for cell_key in touched_cells:
        rec = _recovery(results[cell_key], agent_names)
        if rec is not None:
            results[cell_key]["_recovery"] = rec
    flush()
    print(f"\nsaved -> {out}")


# --------------------------------- plot mode ---------------------------------
def _merge_results(paths):
    """Union of results dicts across JSONs (chunk files); config from the first."""
    merged, config = {}, None
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        config = config or payload.get("config")
        for cell_key, cell in payload.get("results", {}).items():
            dst = merged.setdefault(cell_key, {})
            for name, stats in cell.items():
                dst.setdefault(name, stats)
    return merged, (config or {})


def run_plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = args.out or _default_out(args)
    paths = sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else [])
    if not paths:
        print(f"no JSON matching {pattern}", file=sys.stderr)
        sys.exit(1)
    merged, config = _merge_results(paths)
    n_eps = config.get("n_episodes", args.episodes)

    cells = sorted(merged, key=lambda k: int(k.split("=")[1]))
    Ns = [int(k.split("=")[1]) for k in cells]
    names = []                                       # policy names, first-seen order
    for k in cells:
        for name in merged[k]:
            if not name.startswith("_") and name not in names:
                names.append(name)

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for name in names:
        T = np.array([merged[k].get(name, {}).get("T", np.nan) for k in cells])
        sd = np.array([merged[k].get(name, {}).get("T_std", 0.0) for k in cells])
        se = sd / math.sqrt(max(n_eps, 1))
        ax.plot(Ns, T, marker="o", lw=1.6, ms=4, label=name)
        ax.fill_between(Ns, T - se, T + se, alpha=0.15, lw=0)

    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (steps, censored at $H$)")
    ax.set_title(rf"q-heuristic vs purify-then-swap "
                 rf"($p_\mathrm{{gen}}={config.get('p_gen')}$, "
                 rf"$p_\mathrm{{swap}}={config.get('p_swap')}$, "
                 rf"$n_\mathrm{{ch}}={config.get('n_ch')}$, "
                 rf"$\tau={config.get('cutoff')}$, $H={config.get('H')}$)",
                 fontsize=8)
    ax.set_xticks(Ns)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    stem = args.fig or os.path.splitext(paths[0])[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")


# -------------------------------- sanity mode --------------------------------
SANITY = dict(N=6, n_ch=4, p_gen=0.4, p_swap=0.8, cutoff=30, H=500,
              n_episodes=40, seed=42)


def run_sanity():
    """q=1.0 must be bit-identical to purify_then_swap (the graft plumbing check)."""
    from experiments.heatmap.optimal_baseline import mc_eval
    from experiments.q_heuristic.hybrid_policy import (
        make_hybrid_fn, purify_then_swap_fn)
    c = SANITY
    print("=" * 78)
    print(f"SANITY  q=1.0 == purify_then_swap  (N={c['N']} n_ch={c['n_ch']} "
          f"p_gen={c['p_gen']} p_swap={c['p_swap']} cutoff={c['cutoff']} "
          f"H={c['H']} eps={c['n_episodes']} seed={c['seed']})")
    print("=" * 78)

    def ev(fn):
        return mc_eval(fn, c["N"], c["n_ch"], c["p_gen"], c["p_swap"],
                       c["cutoff"], c["H"], c["n_episodes"], seed=c["seed"],
                       return_stats=True)

    pts = ev(purify_then_swap_fn)
    hyb = ev(make_hybrid_fn(q=1.0, seed=123))
    identical = (pts["T"] == hyb["T"]
                 and pts["conn_rate"] == hyb["conn_rate"]
                 and pts["mean_F_conn"] == hyb["mean_F_conn"]
                 and pts["T_std"] == hyb["T_std"])
    print(f"  purify_then_swap : T={pts['T']:.6f} T_std={pts['T_std']:.6f} "
          f"conn={pts['conn_rate']:.6f}")
    print(f"  hybrid q=1.0     : T={hyb['T']:.6f} T_std={hyb['T_std']:.6f} "
          f"conn={hyb['conn_rate']:.6f}")
    print("=" * 78)
    print(f"IDENTICAL (T, conn_rate, mean_F_conn, T_std): {identical}")
    print("=" * 78)
    return identical


def main(argv=None):
    args = parse_args(argv)
    if args.sanity:
        sys.exit(0 if run_sanity() else 1)
    if args.plot:
        run_plot(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
