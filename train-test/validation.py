import argparse
from rl_stack import QRNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Test QRNAgent")
    parser.add_argument("--run_id", type=str, default="v003")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--p_gen", type=float, default=1.0)
    parser.add_argument("--p_swap", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=20)
    parser.add_argument("--F0", type=float, default=1.0)
    parser.add_argument("--channel_loss", type=float, default=0.0)
    parser.add_argument("--path", type=str, default="checkpoints/003/")
    parser.add_argument("--dict", type=str, default="policy.pth")
    
    return parser.parse_args()


if __name__ == "__main__":
    agent = QRNAgent()
    args = parse_args()
    model_path = args.path + args.dict

    results = agent.validate(
        model_path=model_path,
        n_episodes=args.episodes, 
        max_steps=args.steps,
        n_repeaters=args.nodes,           
        p_gen=args.p_gen, 
        p_swap=args.p_swap,  
        cutoff=args.cutoff, 
        F0=args.F0, 
        channel_loss=args.channel_loss,
        dt_seconds=0, 
        plot_actions=True,
        save_dir=args.path,
        ee=True
    )