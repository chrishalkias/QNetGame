"""CLI entry point for Phase 1: train small-chain curriculum, then report
gap-to-optimal. Run from repo root with PYTHONPATH=. and .venv311 active."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os

import numpy as np

from rl_stack import QRNAgent
from game.phases import PHASE1
from game.runner import run_phase
from game.compare_optimal import compare_to_optimal
from game.report import format_report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="game Phase 1: curriculum train + optimal compare")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override PHASE1.episodes (default: PHASE1 value)")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="override PHASE1.max_steps")
    ap.add_argument("--save_dir", type=str, default="checkpoints/cluster/game/phase1")
    ap.add_argument("--policy_dir", type=str, default="results/optimal_policies")
    ap.add_argument("--mc_eps", type=int, default=2000,
                    help="Monte-Carlo episodes per policy in the comparison")
    ap.add_argument("--horizon", type=int, default=30,
                    help="MC rollout / DP horizon; must match the pickle's h")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plot", dest="plot", action="store_false", default=True)
    ap.add_argument("--skip_compare", action="store_true",
                    help="train + save only; skip the optimal-comparison report")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {}
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    cfg = dataclasses.replace(PHASE1, **overrides) if overrides else PHASE1

    agent = QRNAgent(rng=np.random.default_rng(args.seed))
    run_phase(agent, cfg, args.save_dir, plot=args.plot)

    if args.skip_compare:
        print(f"[phase1] checkpoint saved -> {os.path.join(args.save_dir, 'policy.pth')}")
        return

    ckpt = os.path.join(args.save_dir, "policy.pth")
    report = compare_to_optimal(ckpt, cfg, args.policy_dir,
                                mc_eps=args.mc_eps, horizon=args.horizon)
    print(format_report(report))
    out = os.path.join(args.save_dir, "optimal_comparison.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase1] report saved -> {out}")


if __name__ == "__main__":
    main()
