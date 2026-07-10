"""
Train an RL agent on a specified system topology
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
import numpy as np
import torch
from rl_stack import QRNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train QRNAgent")
    # Algorithm Variables
    parser.add_argument("--run_id", type=str, default="xxx")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed: seeds torch (net init), the agent RNG "
                             "(eps-greedy + per-episode domain draws), the replay "
                             "sampler, AND the per-episode env/network physics RNG "
                             "(entangle/swap/purify coin flips, auto-entangle "
                             "shuffle, inhomogeneity). A given seed makes the whole "
                             "training trajectory + metrics.json bit-reproducible on "
                             "cpu. Vary for independent agents.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=300)

    # System Variables
    parser.add_argument("--n_lo", type=int, default=5)
    parser.add_argument("--n_hi", type=int, default=8)
    parser.add_argument("--curriculum", action='store_false')
    parser.add_argument("--n_ch", type=int, nargs="+", default=[4],
                        help="n_ch pool, e.g. --n_ch 2 3 4 (per-episode draw)")
    parser.add_argument("--topology", type=str, default='chain')
    parser.add_argument("--p_gen", type=float, nargs="+", default=[0.60],
                        help="MEAN p_gen; pass two values for a (lo,hi) per-episode range")
    parser.add_argument("--p_swap", type=float, nargs="+", default=[0.85],
                        help="MEAN p_swap; pass two values for a (lo,hi) per-episode range")
    parser.add_argument("--p_gen_std", type=float, default=0.0,
                        help="per-repeater spread of p_gen (0 = homogeneous)")
    parser.add_argument("--p_swap_std", type=float, default=0.0,
                        help="per-repeater spread of p_swap (0 = homogeneous)")
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--cutoff_lo", type=int, default=None,
                        help="with --cutoff_hi, sample cutoff per episode in [lo,hi]")
    parser.add_argument("--cutoff_hi", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="DQN discount AND env PBRS gamma (kept matched)")

    # CC Variables
    parser.add_argument("--dt_seconds", type=float, default=0.00) #1e-4 for CC
    parser.add_argument("--channel_loss", type=float, default=0.00)
    parser.add_argument("--F0", type=float, default=1.0)
    
    parser.add_argument("--compare", action="store_true",
                        help="log per-episode greedy-agent vs swap-asap vs random "
                             "returns on a shared seeded net (+ training_compare.png)")
    parser.add_argument("--save_base_dir", type=str, default="checkpoints")
    parser.add_argument("--prune_unwinnable", action="store_true",
                        help="skip cells swap-asap can't deliver (winnability oracle)")
    parser.add_argument("--no_eval_ckpt", action="store_true",
                        help="disable the delivery-time eval probe for best-checkpoint "
                             "selection (auto-enabled for episodes >= 5000); falls back "
                             "to rolling-mean reward selection")
    return parser.parse_args()


# ── Run manifest ──────────────────────────────────────────────────────────
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


# ── Eval probe (delivery-time best-checkpoint selection) ───────────────────
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
                "dt_seconds": args.dt_seconds,
                "max_steps": max_steps,
                "topology": args.topology,
            }
            for k in range(n_episodes):
                # greedy rollout (select_actions training=False ignores epsilon);
                # deterministic per (cell, k); steps censored at max_steps.
                _, steps, _ = agent._cmp_rollout(
                    env_args, probe_seed + 1000 * k, "agent", max_steps)
                all_steps.append(steps)
        return float(np.mean(all_steps))

    return probe


def make_hard_cells(args):
    """A small fixed grid at the HARD end of the run's own training ranges
    (largest N, lowest rates, tightest cutoff/memory) so the probe stresses the
    regime where checkpoint quality actually differs."""
    p_gen_lo = min(args.p_gen)
    p_swap_lo = min(args.p_swap)
    n_hi = args.n_hi
    n_mid = (args.n_lo + args.n_hi) // 2
    n_ch_lo = min(args.n_ch)
    cutoff_lo = args.cutoff_lo if args.cutoff_lo is not None else args.cutoff
    cells = [
        {"n_repeaters": n_hi, "n_ch": n_ch_lo,
         "p_gen": p_gen_lo, "p_swap": p_swap_lo, "cutoff": cutoff_lo},
        {"n_repeaters": max(n_mid, args.n_lo), "n_ch": n_ch_lo,
         "p_gen": p_gen_lo, "p_swap": p_swap_lo, "cutoff": cutoff_lo},
    ]
    return cells

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

    # Eval-probe best-checkpoint selection (default for real runs): rolling-mean
    # reward demonstrably mis-picked long runs (35k overtraining; CC best<final),
    # so for episodes >= 5000 select policy.pth by a held-out delivery-time probe
    # (lower = better). Early stopping is OFF (eval_patience=0): we want better
    # checkpoint selection, not early termination. --no_eval_ckpt reverts.
    eval_fn = None
    eval_every = 0
    if args.episodes >= 5000 and not args.no_eval_ckpt:
        eval_fn = build_eval_probe(args, make_hard_cells(args))
        eval_every = max(250, args.episodes // 20)
        print(f"[eval-ckpt] delivery-time probe every {eval_every} eps "
              f"({len(make_hard_cells(args))} cells x 40 eps); early-stop off")

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
        dt_seconds=args.dt_seconds,
        env_seed=args.seed,
        eval_fn=eval_fn,
        eval_every=eval_every,
        eval_patience=0,      # checkpoint selection only, no early termination
        eval_mode='min',
        save_path=save_path,
        topology=args.topology,
        prune_unwinnable=args.prune_unwinnable,
        compare=args.compare,
        plot=True,)