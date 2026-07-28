"""
--------------------------------------------------------------------------------
Validate ONE checkpoint against the heuristics at ONE parameter cell.

Prints the paired-seed table (avg steps, avg fidelity, success%) across
Agent / SwapASAP / PurifySwap / Random. Seconds to run: this is the checkpoint
sanity check, not a paper figure. For the Delta% parameter SWEEP and its
heatmaps, see experiments/training/batch_validate.py (hours, needs pandas +
seaborn).

Delivery is topological (source connects to dest); the cutoff already bounds
delivered-link decoherence, so no fidelity threshold gates the win condition.

  PYTHONPATH=src:. python experiments/training/validation.py \
      --path checkpoints/sota/ --dict policy.pth --episodes 100
--------------------------------------------------------------------------------
"""

import argparse
import os

from rl_stack import QRNAgent


def parse_args():
    p = argparse.ArgumentParser(description="Validate a QRNAgent checkpoint against the heuristics")
    # checkpoint
    p.add_argument("--path", type=str, default="checkpoints/sota/", help="run directory holding the state dict")
    p.add_argument("--dict", type=str, default="policy.pth", help="state dict filename inside --path")
    # validation run
    p.add_argument("--episodes", type=int, default=100, help="paired episodes per strategy")
    p.add_argument("--steps", type=int, default=200, help="episode horizon in ticks")
    # network
    p.add_argument("--nodes", type=int, default=6, help="chain length N")
    p.add_argument("--n_ch", type=int, default=4, help="qubits per side, per repeater")
    p.add_argument("--p_gen", type=float, default=0.6, help="per-network MEAN link-generation prob.")
    p.add_argument("--p_swap", type=float, default=0.85, help="per-network MEAN BSM success prob.")
    p.add_argument("--p_gen_std", type=float, default=0.0, help="per-repeater spread of p_gen (0 = homogeneous)")
    p.add_argument("--p_swap_std", type=float, default=0.0, help="per-repeater spread of p_swap (0 = homogeneous)")
    p.add_argument("--cutoff", type=int, default=20, help="memory cutoff age in ticks")
    # physics
    p.add_argument("--F0", type=float, default=1.0, help="fidelity of a freshly generated link")
    p.add_argument("--channel_loss", type=float, default=0.0, help="attenuation alpha per km")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = os.path.join(args.path, args.dict)

    agent = QRNAgent()
    agent.validate(
        model_path=model_path,
        n_episodes=args.episodes,
        max_steps=args.steps,
        n_ch=args.n_ch,
        n_repeaters=args.nodes,
        p_gen=args.p_gen,
        p_swap=args.p_swap,
        p_gen_std=args.p_gen_std,
        p_swap_std=args.p_swap_std,
        cutoff=args.cutoff,
        F0=args.F0,
        channel_loss=args.channel_loss,
    )


if __name__ == "__main__":
    main()
