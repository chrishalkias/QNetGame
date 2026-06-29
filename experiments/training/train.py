"""
Train an RL agent on a specified system topology
"""

import argparse
import os
from rl_stack import QRNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train QRNAgent")
    # Algorithm Variables
    parser.add_argument("--run_id", type=str, default="xxx")
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
    parser.add_argument("--p_gen", type=float, default=0.60,
                        help="per-network MEAN link-generation prob.")
    parser.add_argument("--p_swap", type=float, default=0.85,
                        help="per-network MEAN BSM success prob.")
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Generate unique save directory to prevent checkpoint overwriting
    run_name = str(args.run_id)
    save_path = os.path.join(args.save_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    

    agent = QRNAgent(lr=args.lr,
                     hidden=args.hidden,
                     batch_size=args.batch_size,
                     buffer_size=80_000,
                     gamma=args.gamma,
                     tau=0.005,
                     epsilon=1,)

    metrics = agent.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        n_range=list(range(args.n_lo, args.n_hi+1)),
        curriculum=args.curriculum,
        p_gen=args.p_gen,
        p_swap=args.p_swap,
        p_gen_std=args.p_gen_std,
        p_swap_std=args.p_swap_std,
        n_ch=args.n_ch,
        cutoff=((args.cutoff_lo, args.cutoff_hi)
                if args.cutoff_lo is not None else args.cutoff),
        channel_loss=args.channel_loss,
        F0=args.F0,
        dt_seconds=args.dt_seconds,
        save_path=save_path,
        topology=args.topology,
        compare=args.compare,
        plot=True,)