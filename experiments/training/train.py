"""
--------------------------------------------------------------------------------
Train an RL agent on a specified system
--------------------------------------------------------------------------------
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
import numpy as np
import torch
from rl_stack import QRNAgent

def parse_args():
    p = argparse.ArgumentParser(description="Train QRNAgent")
    # Algorithm Variables
    p.add_argument("--run_id", type=str, default="xxx")
    p.add_argument("--seed", type=int, default=0,
                   help="master seed: seeds torch (net init), the agent RNG "
                   "(eps-greedy + per-episode domain draws), the replay "
                   "sampler, AND the per-episode env/network physics RNG "
                   "(entangle/swap/purify coin flips, auto-entangle "
                   "shuffle, inhomogeneity). A given seed makes the whole "
                   "training trajectory + metrics.json bit-reproducible on "
                   "cpu. Vary for independent agents.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--episodes", type=int, default=300)

    # System Variables
    p.add_argument("--n_lo", type=int, default=5)
    p.add_argument("--n_hi", type=int, default=8)
    p.add_argument("--curriculum", action='store_false')
    p.add_argument("--n_ch", type=int, nargs="+", default=[4],
                   help="n_ch pool, e.g. --n_ch 2 3 4 (per-episode draw)")
    p.add_argument("--p_gen", type=float, nargs="+", default=[0.60],
                   help="MEAN p_gen; pass two values for a (lo,hi) per-episode range")
    p.add_argument("--p_swap", type=float, nargs="+", default=[0.85],
                   help="MEAN p_swap; pass two values for a (lo,hi) per-episode range")
    p.add_argument("--p_gen_std", type=float, default=0.0,
                   help="per-repeater spread of p_gen (0 = homogeneous)")
    p.add_argument("--p_swap_std", type=float, default=0.0,
                   help="per-repeater spread of p_swap (0 = homogeneous)")
    p.add_argument("--cutoff", type=int, default=6)
    p.add_argument("--cutoff_lo", type=int, default=None,
                   help="with --cutoff_hi, sample cutoff per episode in [lo,hi]")
    p.add_argument("--cutoff_hi", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.995,
                   help="DQN discount AND env PBRS gamma (kept matched)")
    p.add_argument("--eps_schedule", choices=["linear", "cosine"],
                   default="linear",
                   help="ε annealing shape over first 90%% of episodes "
                   "(linear = Mnih/SB3 standard; cosine = legacy)")

    p.add_argument("--channel_loss", type=float, default=0.00)
    p.add_argument("--F0", type=float, default=1.0)
    
    p.add_argument("--compare", action="store_true",
                   help="log sampled greedy-agent vs swap-asap vs random "
                   "returns on a shared seeded net (+ training_compare.png)")
    p.add_argument("--compare_every", type=int, default=10,
                   help="run the paired comparison rollouts every K episodes "
                   "(each sample costs 3 extra greedy rollouts; K=1 "
                   "reproduces the old every-episode behaviour)")
    p.add_argument("--save_base_dir", type=str, default="checkpoints")
    p.add_argument("--prune_unwinnable", action="store_true",
                   help="skip cells swap-asap can't deliver (winnability oracle)")
    p.add_argument("--no_eval_ckpt", action="store_true",
                   help="disable the delivery-time eval probe for best-checkpoint "
                   "selection (auto-enabled for episodes >= 5000); falls back "
                   "to rolling-mean reward selection")
    p.add_argument("--force_eval_ckpt", action="store_true",
                   help="build the delivery-time probe even below the "
                   "episodes >= 5000 auto threshold (smoke tests, "
                   "short diagnostic runs)")
    p.add_argument("--ckpt_pool", action="store_true",
                   help="save EVERY eval-probe checkpoint to <save>/pool/ "
                   "and, after training, re-score the whole pool at "
                   "--runoff_episodes and copy the winner to policy.pth")
    p.add_argument("--runoff_episodes", type=int, default=400,
                   help="episodes per cell in the final pool runoff (the "
                   "in-training probe uses 40; the runoff is the "
                   "honest one)")
    p.add_argument("--disable_purify", action="store_true",
                   help="mask PURIFY in BOTH selection and the DQN target -> "
                   "train a pure swap-scheduler")
    return p.parse_args()


# -- Run manifest ----------------------------------------------------------
def _git_info():
    """(commit, dirty) from git; ('unknown', False) if git is unavailable so a
    missing repo never crashes a training run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
            text=True)
        dirty = bool(porcelain.strip())
    except Exception:
        dirty = False
    return commit, dirty


def _pkg_versions():
    vers = {}
    for name in ("numpy", "torch", "torch_geometric"):
        try:
            vers[name] = __import__(name).__version__
        except Exception:
            vers[name] = "unknown"
    return vers


def write_run_manifest(save_path, args):
    """Dump the full resolved config + provenance so a run is reproducible from
    (commit, seed, config). Best-effort git info: never crash training."""
    commit, dirty = _git_info()
    manifest = {
        "args": vars(args),
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": _pkg_versions(),
    }
    with open(os.path.join(save_path, "run_config.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# -- Eval probe (delivery-time best-checkpoint selection) -------------------
def build_eval_probe(args, hard_cells, probe_seed=12345, n_episodes=40):
    """Return a deterministic greedy-policy delivery-time probe for train()'s
    eval_fn hook. Reuses the agent's own paired-rollout machinery (no dependency
    on experiments/), so best-checkpoint selection tracks censored delivery
    steps (lower = better) instead of rolling reward. `hard_cells`: list of
    dicts with (n_repeaters, n_ch, p_gen, p_swap, cutoff) at the hard end of the
    training ranges."""
    max_steps = args.max_steps

    def probe(agent):
        all_steps = []
        for cell in hard_cells:
            env_args = {
                "n_repeaters": cell["n_repeaters"],
                "n_ch": cell["n_ch"],
                "spacing": 50,
                "p_gen": cell["p_gen"],
                "p_swap": cell["p_swap"],
                "p_gen_std": args.p_gen_std,
                "p_swap_std": args.p_swap_std,
                "cutoff": cell["cutoff"],
                "gamma": agent.gamma,
                "F0": args.F0,
                "channel_loss": args.channel_loss,
                "max_steps": max_steps,
            }
            for k in range(n_episodes):
                # greedy rollout (select_action training=False ignores epsilon);
                # deterministic per (cell, k); steps censored at max_steps.
                _, steps, _ = agent._cmp_rollout(
                    env_args, probe_seed + 1000 * k, "agent", max_steps)
                all_steps.append(steps)
        return float(np.mean(all_steps))

    return probe


def _pilot_delivery_rate(cell, args, n_episodes=20, seed=999):
    """Fraction of purify-then-swap pilot episodes that deliver end-to-end
    within max_steps. Post cutoff-fix, every delivery is entangled by
    construction, so plain delivery rate is the right calibration signal."""
    from rl_stack.env_wrapper import QRNEnv
    from rl_stack import policies
    wins = 0
    for k in range(n_episodes):
        env = QRNEnv(
            n_repeaters=cell["n_repeaters"], n_ch=cell["n_ch"], spacing=50,
            p_gen=cell["p_gen"], p_swap=cell["p_swap"],
            p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std,
            cutoff=cell["cutoff"], F0=args.F0,
            channel_loss=args.channel_loss,
            max_steps=args.max_steps,
            rng=np.random.default_rng(seed + k))
        env.reset()
        # Serialized sweep: one micro-decision at env.active_node per
        # env.step call; the env self-truncates at max_steps ticks.
        while True:
            a = policies.purify_then_swap(env)
            _, _, done, info = env.step(a)
            if done:
                wins += int(bool(info["terminated"]))
                break
    return wins / n_episodes


def make_calibrated_cells(args, lo=0.30, hi=0.70, n_cells=2, n_episodes=20):
    """Probe cells calibrated to an informative regime (~lo-hi delivery rate
    under the purify-then-swap pilot). Replaces the hard-corner cells that
    pinned at the censoring ceiling and made best-checkpoint selection noise
    (tracked defect: omni_v2_15k_s2 policy.pth). Candidates span the run's
    own ranges hardest-first; first n_cells in band win; otherwise fall back
    to the closest rates with a printed warning. Deterministic given
    args.seed."""
    p_gen_lo, p_gen_hi = min(args.p_gen), max(args.p_gen)
    p_swap_lo, p_swap_hi = min(args.p_swap), max(args.p_swap)
    p_gen_mid = (p_gen_lo + p_gen_hi) / 2
    p_swap_mid = (p_swap_lo + p_swap_hi) / 2
    cutoff_lo = args.cutoff_lo if args.cutoff_lo is not None else args.cutoff
    cutoff_hi = args.cutoff_hi if args.cutoff_hi is not None else args.cutoff
    cutoff_mid = (cutoff_lo + cutoff_hi) // 2
    n_mid = (args.n_lo + args.n_hi) // 2
    n_ch_lo = min(args.n_ch)

    candidates = [
        {"n_repeaters": n, "n_ch": n_ch_lo, "p_gen": pg, "p_swap": ps,
         "cutoff": cut}
        for n in (args.n_hi, n_mid)
        for cut in (cutoff_lo, cutoff_mid, cutoff_hi)
        for pg, ps in ((p_gen_lo, p_swap_lo), (p_gen_mid, p_swap_mid))
    ]
    # Dedupe (hardest-first order preserved; identity = full parameter tuple).
    # Degenerate ranges (lo==hi on p_gen/p_swap/cutoff/n) otherwise leave
    # literal duplicate candidates, which can silently return the SAME cell
    # twice below and defeat "probe at two distinct difficulty points".
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["n_repeaters"], c["n_ch"], c["p_gen"], c["p_swap"],
               c["cutoff"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    candidates = deduped
    if len(candidates) < n_cells:
        print(f"[eval-ckpt] WARNING: only {len(candidates)} distinct "
              f"candidate cell(s) from the run's ranges (requested "
              f"{n_cells}); ranges are too narrow to probe distinct "
              f"difficulty points")
    pilot_seed = args.seed * 7919 + 13
    scored = [(c, _pilot_delivery_rate(c, args, n_episodes=n_episodes,
                                       seed=pilot_seed))
              for c in candidates]
    in_band = [(c, r) for c, r in scored if lo <= r <= hi]
    if len(in_band) >= n_cells:
        chosen = in_band[:n_cells]
    else:
        target = (lo + hi) / 2
        by_dist = sorted(scored, key=lambda cr: abs(cr[1] - target))
        chosen = (in_band + [cr for cr in by_dist if cr not in in_band])[:n_cells]
        print(f"[eval-ckpt] WARNING: only {len(in_band)} candidate cells in "
              f"[{lo},{hi}]; falling back to closest rates "
              f"{[round(r, 2) for _, r in chosen]}")
    print("[eval-ckpt] calibrated probe cells (N, cutoff, pilot rate): "
          f"{[(c['n_repeaters'], c['cutoff'], round(r, 2)) for c, r in chosen]}")
    return [c for c, _ in chosen]


def resolve_eval_probe(args):
    """Decide whether this run gets a delivery-time eval probe, and build it.

    Returns (eval_fn, eval_every, cells); (None, 0, []) when the run has no
    probe. Rolling-mean reward demonstrably mis-picked long runs (the 35k
    overtraining regression, and CC best < final), and raw return is not even
    comparable across a curriculum run because N grows while STEP_COST
    accumulates against a SUCCESS_REWARD capped at 1.0. So real runs select
    policy.pth by a held-out censored delivery time (lower = better).

    The probe is auto-enabled for episodes >= 5000; --force_eval_ckpt turns it
    on below that threshold (smoke tests) and --no_eval_ckpt turns it off
    entirely. --ckpt_pool without a probe is rejected here rather than after
    hours of training against an empty pool.
    """
    want = (not args.no_eval_ckpt) and (args.episodes >= 5000
                                        or args.force_eval_ckpt)
    if not want:
        if args.ckpt_pool:
            raise ValueError(
                "--ckpt_pool needs the delivery-time eval probe (the pool is "
                "written at each probe): drop --no_eval_ckpt, or add "
                "--force_eval_ckpt for a run below the 5000-episode auto "
                "threshold")
        return None, 0, []
    cells = make_calibrated_cells(args)
    eval_fn = build_eval_probe(args, cells)
    # episodes // 20 gives ~20 probes per run; the max(1, .) floor only ever
    # binds for the short forced runs (>= 5000 episodes always yields >= 250).
    eval_every = max(1, args.episodes // 20)
    print(f"[eval-ckpt] delivery-time probe every {eval_every} eps "
          f"({len(cells)} cells x 40 eps); early-stop off")
    return eval_fn, eval_every, cells


if __name__ == "__main__":
    args = parse_args()

    # Reproducibility: the master seed drives torch (network init), the agent RNG
    # (action sampling + per-episode cell draws), the replay sampler, and the
    # per-episode env/network physics RNG (threaded via env_seed below). Distinct
    # seeds -> independent agents; a given seed bit-reproduces metrics.json on cpu.
    torch.manual_seed(args.seed)

    # Generate unique save directory to prevent checkpoint overwriting
    run_name = str(args.run_id)
    save_path = os.path.join(args.save_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    # Run manifest: full resolved config + git commit + dirty flag + versions, so
    # the run is reconstructable from (commit, seed, config). Written up-front.
    write_run_manifest(save_path, args)

    agent = QRNAgent(lr=args.lr,
                     hidden=args.hidden,
                     batch_size=args.batch_size,
                     buffer_size=80_000,
                     gamma=args.gamma,
                     tau=0.005,
                     epsilon=1,
                     rng=np.random.default_rng(args.seed),
                     seed=args.seed,)

    # Eval-probe best-checkpoint selection (default for real runs). Early
    # stopping is OFF (eval_patience=0): we want better checkpoint selection,
    # not early termination. See resolve_eval_probe for the flag semantics.
    eval_fn, eval_every, probe_cells = resolve_eval_probe(args)

    metrics = agent.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        n_range=list(range(args.n_lo, args.n_hi+1)),
        curriculum=args.curriculum,
        p_gen=(args.p_gen[0] if len(args.p_gen) == 1 else tuple(args.p_gen[:2])),
        p_swap=(args.p_swap[0] if len(args.p_swap) == 1 else tuple(args.p_swap[:2])),
        p_gen_std=args.p_gen_std,
        p_swap_std=args.p_swap_std,
        n_ch=args.n_ch,
        cutoff=((args.cutoff_lo, args.cutoff_hi)
                if args.cutoff_lo is not None else args.cutoff),
        channel_loss=args.channel_loss,
        F0=args.F0,
        env_seed=args.seed,
        eps_schedule=args.eps_schedule,
        eval_fn=eval_fn,
        eval_every=eval_every,
        eval_patience=0,      # checkpoint selection only, no early termination
        eval_mode='min',
        save_path=save_path,
        save_best=True,
        ckpt_pool=args.ckpt_pool,
        prune_unwinnable=args.prune_unwinnable,
        disable_actions=((2,) if args.disable_purify else ()),
        compare=args.compare,
        compare_every=args.compare_every,
        plot=True,)

    # Final runoff: the in-training probe is cheap (40 eps/cell) so its running
    # argmin is within noise of several candidates. Re-score the whole pool once
    # at --runoff_episodes and promote the true winner to policy.pth. The
    # comparison is PAIRED: build_eval_probe seeds rollout k from
    # (probe_seed, k) alone, over the SAME cells the in-training probe used, so
    # every candidate meets a bit-identical episode set. policy.pth still means
    # BEST and policy_final.pth (already written by train) still means LAST.
    if args.ckpt_pool:
        runoff_probe = build_eval_probe(args, probe_cells,
                                        n_episodes=args.runoff_episodes)
        best, score = agent.runoff(os.path.join(save_path, "pool"),
                                   runoff_probe)
        shutil.copyfile(best, os.path.join(save_path, "policy.pth"))
        print(f"[runoff] policy.pth <- {os.path.basename(best)} "
              f"(T={score:.3f})")