"""Train a homogeneous-chain agent at n_ch=2 for the gap-to-optimal heatmap.

Matches the exact-DP config (N, n_ch=2, cutoff=5, horizon=30, F0=1, no loss):
each episode draws (p_gen, p_swap) from a shared discrete grid. Run twice:

  purify-enabled:  python experiments/training/train_homog_nch2.py --run_id heat_purify
  swap-only:       python experiments/training/train_homog_nch2.py --run_id heat_swaponly --disable_purify

The swap-only agent (PURIFY masked in training) is the fair learner against the
swap-only DP optimum; the purify-enabled agent can beat it by freeing memory.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from rl_stack import QRNAgent
from rl_stack.env_wrapper import PURIFY


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", required=True)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--n_ch", type=int, default=2)
    p.add_argument("--cutoff", type=int, default=5)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--grid_n", type=int, default=9,
                   help="grid points per axis; values = linspace(0.1, 0.9, grid_n)")
    p.add_argument("--disable_purify", action="store_true",
                   help="mask PURIFY during training (swap-only agent)")
    p.add_argument("--save_base_dir", type=str, default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    grid = set(np.round(np.linspace(0.1, 0.9, args.grid_n), 2).tolist())
    save_path = os.path.join(args.save_base_dir, args.run_id)
    os.makedirs(save_path, exist_ok=True)

    agent = QRNAgent(lr=args.lr, hidden=args.hidden, buffer_size=80_000,
                     gamma=0.99, tau=0.005, epsilon=1.0)

    agent.train(
        episodes=args.episodes,
        max_steps=args.horizon,
        n_range=[args.N],
        n_ch=args.n_ch,
        p_gen=grid, p_swap=grid,
        p_gen_std=0.0, p_swap_std=0.0,
        cutoff=args.cutoff,
        F0=1.0, channel_loss=0.0, dt_seconds=0.0,
        curriculum=False,
        topology="chain",
        disable_actions=(PURIFY,) if args.disable_purify else (),
        compare=False,
        save_path=save_path,
        plot=True,
    )
    print(f"[done] {save_path}/policy.pth "
          f"({'swap-only' if args.disable_purify else 'purify-enabled'})")
