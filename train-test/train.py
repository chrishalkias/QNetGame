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
    parser.add_argument("--n_ch", type=int, default=4)
    parser.add_argument("--topology", type=str, default='chain')
    parser.add_argument("--heterogeneous", action="store_false")
    parser.add_argument("--p_gen", type=float, default=0.60)
    parser.add_argument("--p_swap", type=float, default=0.85)
    parser.add_argument("--cutoff", type=int, default=6)

    # CC Variables
    parser.add_argument("--dt_seconds", type=float, default=0.00) #1e-4 for CC
    parser.add_argument("--channel_loss", type=float, default=0.00)
    parser.add_argument("--F0", type=float, default=1.0)
    
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
                     gamma=0.99, 
                     tau=0.005,
                     epsilon=1,)

    metrics = agent.train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        n_range=list(range(args.n_lo, args.n_hi+1)),
        curriculum=args.curriculum,
        heterogeneous=args.heterogeneous,
        p_gen=args.p_gen, 
        p_swap=args.p_swap,
        cutoff=args.cutoff,
        channel_loss=args.channel_loss,
        F0=args.F0,
        dt_seconds=args.dt_seconds,
        save_path=save_path,
        topology=args.topology,
        plot=True,)