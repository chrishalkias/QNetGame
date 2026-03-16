import argparse
import os
from rl_stack import QRNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train QRNAgent")
    parser.add_argument("--run_id", type=float, default="004")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--n_lo", type=int, default=6)
    parser.add_argument("--n_hi", type=int, default=10)
    parser.add_argument("--disable_curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--p_gen", type=float, default=0.80)
    parser.add_argument("--p_swap", type=float, default=0.85)
    parser.add_argument("--cutoff", type=int, default=15)
    parser.add_argument("--save_base_dir", type=str, default="checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Generate unique save directory to prevent checkpoint overwriting
    run_name = str(args.run_id)
    save_path = args.save_base_dir + run_name
    os.makedirs(save_path, exist_ok=True)
    

    agent = QRNAgent(lr=args.lr, 
                     hidden=args.hidden,
                     batch_size=args.batch_size, 
                     buffer_size=10_000,
                     gamma=0.99, 
                     tau=0.005,
                     epsilon=1,)

    metrics = agent.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        n_range=list(range(args.n_lo, args.n_hi+1)),
        curriculum=args.curriculum,
        heterogeneous=False,
        p_gen=args.p_gen, 
        p_swap=args.p_swap,
        cutoff=args.cutoff,
        channel_loss=0.0,
        F0=1.0,
        dt_seconds=0,
        save_path=save_path,
        plot=True,)