"""Pre-computed heuristic statistics over the homogeneous chain domain.

Gather-once baseline: delivery-rate, steps-to-end-to-end, and end-to-end fidelity
(mean + std) for the random / swap-asap / purify-then-swap heuristics, so that
agent-validation runs compare against a *fixed* table instead of re-Monte-Carlo-ing
the heuristics on every run.

Two decoupled sweeps
--------------------
  physics  : p_gen x p_swap x cutoff(tau)  at fixed N=5, n_ch in {2,4}
  nscaling : N in {3..20}                  at fixed (p_gen,p_swap,tau)=(0.1,0.9,20),
                                           n_ch in {2,4}

Physics is the CLEAN config (F0=1, channel_loss=0, dt=0, chain) -- identical to the
optimal-baseline / heatmap pipeline, so the numbers line up apples-to-apples with
the rest of the paper. Fidelity is still non-trivial: a link's Werner parameter
decays with age as p0*exp(-age/cutoff) (simulator/repeater.py), so the
cutoff tau directly shapes delivered fidelity.

All three policies are evaluated at a COMMON horizon (--horizon) so that
delivery_rate / steps-to-e2e / fidelity are directly comparable across policies.
A policy that delivers fast just breaks early, so the horizon only costs its full
length in the degenerate corner (low p_gen / small tau) where nothing delivers --
those cells are recorded with delivery_rate < 1 and nan step/fidelity stats.

  PYTHONPATH=. python experiments/baselines/sweep_heuristics.py --smoke
  PYTHONPATH=. python experiments/baselines/sweep_heuristics.py --sweep nscaling
  PYTHONPATH=. python experiments/baselines/sweep_heuristics.py --sweep physics \
      --chunk K --nchunks M --out results/baselines/heuristics_physics.csv
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import strategies


# ───────────────────────── argparse (top of file, per repo convention) ─────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=["physics", "nscaling", "both"], default="both")
    p.add_argument("--episodes", type=int, default=300,
                   help="MC episodes per policy per cell")
    p.add_argument("--horizon", type=int, default=20000,
                   help="common max_steps for ALL policies (delivery censored beyond this)")
    p.add_argument("--pilot", type=int, default=50,
                   help="abort a cell after this many 0-delivery episodes (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    # physics-sweep axes
    p.add_argument("--n_p", type=int, default=13, help="points on p_gen and p_swap axes")
    p.add_argument("--n_t", type=int, default=13, help="points on the cutoff (tau) axis")
    p.add_argument("--pgen_log", action="store_true",
                   help="log-space p_gen in [1e-3, 1] instead of linear")
    p.add_argument("--n_list", type=str, default="5",
                   help="chain N for the physics sweep (comma-separated)")
    p.add_argument("--nch_list", type=str, default="2,4",
                   help="qubits-per-repeater values to sweep")
    # nscaling-sweep fixed operating point
    p.add_argument("--ns_pgen", type=float, default=0.1)
    p.add_argument("--ns_pswap", type=float, default=0.9)
    p.add_argument("--ns_tau", type=int, default=20)
    p.add_argument("--ns_nmin", type=int, default=3)
    p.add_argument("--ns_nmax", type=int, default=20)
    # chunking / output / self-check
    p.add_argument("--chunk", type=int, default=0, help="this process's index")
    p.add_argument("--nchunks", type=int, default=1, help="total processes")
    p.add_argument("--out", type=str, default=None,
                   help="CSV path; default results/baselines/heuristics_<sweep>.csv")
    p.add_argument("--smoke", action="store_true", help="quick self-check, then exit")
    return p.parse_args(argv)


# Clean physics: matches optimal_baseline / heatmap. Connectivity + timing identical
# across configs; fidelity varies only through age-decay exp(-age/cutoff).
CLEAN = dict(F0=1.0, channel_loss=0.0, dt_seconds=0.0, topology="chain")

CSV_COLUMNS = ["sweep", "policy", "N", "n_ch", "p_gen", "p_swap", "cutoff",
               "max_steps_used", "n_episodes", "delivery_rate", "n_delivered",
               "mean_steps", "std_steps", "mean_fid", "std_fid"]


# ───────────────────────── evaluator ─────────────────────────
def eval_policy_stats(policy_fn, N, n_ch, p_gen, p_swap, cutoff, max_steps,
                      n_episodes, seed=42, pilot=None):
    """MC-evaluate one policy on the clean homogeneous chain.

    Mirrors optimal_baseline.mc_eval's env construction, but records delivery
    *fidelity* and *delivery rate* instead of returning censored time. Steps and
    fidelity are aggregated over the DELIVERED episodes only ("steps until e2e" is
    undefined otherwise); both are nan when nothing delivers.

    `pilot`: if set, abort the cell after this many episodes when NOTHING has
    delivered -- a degenerate cell (low p_gen / small tau) would otherwise burn
    the full budget at the horizon to confirm delivery_rate~=0. The returned
    `n_episodes` reflects how many were actually run, so an aborted cell is
    transparent in the CSV. (Bias: a genuinely rare-delivery cell can abort on a
    zero-streak; widen --pilot to trade compute for resolution there.)
    """
    rng = np.random.default_rng(seed)
    steps, fids, n_delivered, ep = [], [], 0, 0
    for ep in range(n_episodes):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     max_steps=max_steps,
                     rng=np.random.default_rng(rng.integers(2**32)), **CLEAN)
        obs = env.reset()
        done, info, step = False, {"fidelity": 0.0}, 0
        for step in range(max_steps):
            obs, _, done, info = env.step(policy_fn(env, obs))
            if done:
                break
        if done and info.get("fidelity", 0.0) > 0:
            n_delivered += 1
            steps.append(step + 1)
            fids.append(info["fidelity"])
        if pilot and ep + 1 >= pilot and n_delivered == 0:
            break
    n_run = ep + 1
    nan = float("nan")
    return dict(
        n_episodes=n_run,
        delivery_rate=n_delivered / n_run,
        n_delivered=n_delivered,
        mean_steps=float(np.mean(steps)) if steps else nan,
        std_steps=float(np.std(steps)) if steps else nan,
        mean_fid=float(np.mean(fids)) if fids else nan,
        std_fid=float(np.std(fids)) if fids else nan,
    )


# ───────────────────────── per-cell driver ─────────────────────────
def run_cell(sweep, N, n_ch, p_gen, p_swap, cutoff, args, cell_seed):
    """Evaluate all three heuristics at a COMMON horizon -> list of CSV rows.

    One shared horizon (args.horizon) keeps delivery_rate / steps / fidelity
    directly comparable across policies. Fast cells break early, so the horizon
    only costs its full length where nothing delivers (degenerate corner)."""
    # random's action RNG must be independent of the env RNG -- drawing random
    # actions from env.rng would perturb link gen / swap outcomes and invalidate
    # the comparison (see strategies.random_policy).
    action_rng = np.random.default_rng(cell_seed + 777)
    policies = (
        ("random", lambda env, obs: strategies.random_policy(env, action_rng)),
        ("swap_asap", lambda env, obs: strategies.swap_asap(env)),
        ("purify_swap", lambda env, obs: strategies.purify_then_swap(env)),
    )
    rows = []
    for name, fn in policies:
        r = eval_policy_stats(fn, N, n_ch, p_gen, p_swap, cutoff, args.horizon,
                              args.episodes, seed=cell_seed,
                              pilot=(args.pilot or None))
        rows.append(_row(sweep, name, N, n_ch, p_gen, p_swap, cutoff,
                         args.horizon, r))
    return rows


def _row(sweep, policy, N, n_ch, p_gen, p_swap, cutoff, max_steps, stats):
    return dict(sweep=sweep, policy=policy, N=int(N), n_ch=int(n_ch),
                p_gen=round(float(p_gen), 6), p_swap=round(float(p_swap), 6),
                cutoff=int(cutoff), max_steps_used=int(max_steps), **stats)


# ───────────────────────── cell enumeration ─────────────────────────
def physics_cells(args):
    pgens = (np.logspace(-3, 0, args.n_p) if args.pgen_log
             else np.linspace(0.001, 1.0, args.n_p))
    pswaps = np.linspace(0.1, 1.0, args.n_p)
    taus = np.unique(np.linspace(3, 50, args.n_t).round().astype(int))
    Ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    nchs = [int(x) for x in args.nch_list.split(",") if x.strip()]
    return [("physics", N, nch, float(pg), float(ps), int(tau))
            for N in Ns for nch in nchs
            for pg in pgens for ps in pswaps for tau in taus]


def nscaling_cells(args):
    nchs = [int(x) for x in args.nch_list.split(",") if x.strip()]
    return [("nscaling", N, nch, args.ns_pgen, args.ns_pswap, args.ns_tau)
            for N in range(args.ns_nmin, args.ns_nmax + 1) for nch in nchs]


def cell_seed(base, N, n_ch, p_gen, p_swap, cutoff):
    """Deterministic, process-independent per-cell seed (no salted str hashing),
    so results don't depend on how cells are chunked across the SLURM array."""
    h = (N * 1_000_003 + n_ch * 10_007
         + int(round(p_gen * 1e6)) * 31 + int(round(p_swap * 1e6)) * 17 + cutoff * 7)
    return base + (h % 2_000_000)


# ───────────────────────── output ─────────────────────────
def resolve_out(args, sweep):
    out = args.out or f"results/baselines/heuristics_{sweep}.csv"
    if args.out and args.sweep == "both":            # don't clobber across sweeps
        base, ext = os.path.splitext(out)
        out = f"{base}_{sweep}{ext or '.csv'}"
    if args.nchunks > 1:                             # one file per chunk, no race
        base, ext = os.path.splitext(out)
        out = f"{base}.chunk{args.chunk:02d}of{args.nchunks:02d}{ext or '.csv'}"
    return out


def _open_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    w.writeheader()
    f.flush()
    return f, w


def _fmt(r):
    ms = f"{r['mean_steps']:7.1f}" if r["mean_steps"] == r["mean_steps"] else "    nan"
    mf = f"{r['mean_fid']:.3f}" if r["mean_fid"] == r["mean_fid"] else "  nan"
    return (f"  {r['policy']:>11} N={r['N']:>2} nch={r['n_ch']} pg={r['p_gen']:.3f} "
            f"ps={r['p_swap']:.2f} tau={r['cutoff']:>2} | deliv={r['delivery_rate']:.2f} "
            f"steps={ms} fid={mf} eps={r['n_episodes']:>3} H={r['max_steps_used']}")


# ───────────────────────── smoke self-check ─────────────────────────
def _smoke():
    action_rng = np.random.default_rng(0)
    rand_fn = lambda env, obs: strategies.random_policy(env, action_rng)
    easy = eval_policy_stats(lambda e, o: strategies.swap_asap(e),
                             N=3, n_ch=2, p_gen=0.9, p_swap=0.9, cutoff=20,
                             max_steps=200, n_episodes=20, seed=1)
    hard = eval_policy_stats(rand_fn, N=6, n_ch=2, p_gen=0.001, p_swap=0.1,
                             cutoff=3, max_steps=200, n_episodes=30, seed=1, pilot=5)
    keys = {"n_episodes", "delivery_rate", "n_delivered",
            "mean_steps", "std_steps", "mean_fid", "std_fid"}
    for r in (easy, hard):
        assert set(r) == keys, r.keys()
        assert 0.0 <= r["delivery_rate"] <= 1.0
        assert (r["mean_fid"] != r["mean_fid"]) or (0.0 <= r["mean_fid"] <= 1.0 + 1e-9)
        assert (r["mean_steps"] != r["mean_steps"]) or r["mean_steps"] > 0
    assert easy["delivery_rate"] > 0.5, f"swap-asap should deliver easily: {easy}"
    if hard["n_delivered"] == 0:                       # nan + pilot-abort must trigger
        assert hard["mean_steps"] != hard["mean_steps"]
        assert hard["n_episodes"] == 5, hard["n_episodes"]
    print(f"smoke OK | easy(swap_asap pg.9): deliv={easy['delivery_rate']:.2f} "
          f"steps={easy['mean_steps']:.1f} fid={easy['mean_fid']:.3f} | "
          f"hard(random pg.001): deliv={hard['delivery_rate']:.2f}")
    return 0


# ───────────────────────── driver ─────────────────────────
def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        return _smoke()

    sweeps = ["physics", "nscaling"] if args.sweep == "both" else [args.sweep]
    for sweep in sweeps:
        cells = physics_cells(args) if sweep == "physics" else nscaling_cells(args)
        mine = [c for i, c in enumerate(cells) if i % args.nchunks == args.chunk]
        out = resolve_out(args, sweep)
        print(f"[{sweep}] {len(mine)}/{len(cells)} cells "
              f"(chunk {args.chunk}/{args.nchunks}) episodes={args.episodes} "
              f"horizon={args.horizon} pilot={args.pilot} -> {out}", flush=True)
        f, w = _open_csv(out)
        try:
            for (sw, N, nch, pg, ps, tau) in mine:
                cs = cell_seed(args.seed, N, nch, pg, ps, tau)
                for row in run_cell(sw, N, nch, pg, ps, tau, args, cs):
                    w.writerow(row)
                    print(_fmt(row), flush=True)
                f.flush()                              # kill-safe: completed cells persist
        finally:
            f.close()
        print(f"[{sweep}] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
