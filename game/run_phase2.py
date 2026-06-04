"""CLI entry point for Phase 2 (grid). Trains the agent with a topology-general
PBRS reward and a greedy grid probe for early-stopping/best-checkpoint, then
evaluates vs swap-asap on grids. Cluster entry: python -m game.run_phase2."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os

import numpy as np

from rl_stack import QRNAgent
from rl_stack.env_wrapper import QRNEnv
from game.phases import PHASE2
from game.runner import run_phase
from game.grid_eval import evaluate_on_grids


def _greedy_grid_delivery(agent, side, n_ch, p_gen, p_swap, cutoff, max_steps,
                          n, seed=7):
    master = np.random.default_rng(seed)
    times = []
    for _ in range(n):
        env = QRNEnv(n_repeaters=side, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap,
                     cutoff=cutoff, F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                     max_steps=max_steps, topology="grid",
                     rng=np.random.default_rng(master.integers(2**32)))
        obs = env.reset()
        done, step, info = False, 0, {}
        for step in range(max_steps):
            a = agent.select_actions(obs, env.get_action_mask(), training=False)
            obs, _, done, info = env.step(a)
            if done:
                break
        times.append(step + 1 if (done and info.get("fidelity", 0.0) > 0) else max_steps)
    return float(np.mean(times))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="game Phase 2: grid PBRS train + eval")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default="checkpoints/cluster/game/phase2")
    ap.add_argument("--eval_episodes", type=int, default=500)
    ap.add_argument("--probe_episodes", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plot", dest="plot", action="store_false", default=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {}
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    cfg = dataclasses.replace(PHASE2, **overrides) if overrides else PHASE2

    agent = QRNAgent(rng=np.random.default_rng(args.seed))

    def probe(ag):
        return _greedy_grid_delivery(ag, side=4, n_ch=2, p_gen=0.7, p_swap=0.75,
                                     cutoff=cfg.cutoff, max_steps=cfg.max_steps,
                                     n=args.probe_episodes)

    run_phase(agent, cfg, args.save_dir, plot=args.plot,
              eval_fn=probe, eval_every=args.eval_every,
              eval_patience=args.eval_patience, eval_mode="min")

    import torch
    agent.policy_net.load_state_dict(torch.load(
        os.path.join(args.save_dir, "policy.pth"), map_location="cpu",
        weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0
    agent_fn = lambda env, obs: agent.select_actions(obs, env.get_action_mask(),
                                                     training=False)
    report = evaluate_on_grids(agent_fn, grid_sides=(3, 4), n_ch=2,
                               p_gen=0.7, p_swap=0.8, cutoff=cfg.cutoff,
                               max_steps=cfg.max_steps, n_episodes=args.eval_episodes)
    out = os.path.join(args.save_dir, "grid_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    for r in report["rows"]:
        print(f"grid {r['grid']}x{r['grid']}: T_agent={r['T_agent']:.2f} "
              f"T_swap_asap={r['T_swap_asap']:.2f} "
              f"agent_beats_swap={r['agent_beats_swap_pct']:+.1f}%")
    print(f"[phase2] saved -> {out}")


if __name__ == "__main__":
    main()
