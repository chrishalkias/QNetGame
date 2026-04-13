import argparse
from rl_stack import QRNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Test QRNAgent")
    #Validation variables
    parser.add_argument("--run_id", type=str, default="v006")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)

    #System variables
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--n_ch", type=int, default=4)
    parser.add_argument("--p_gen", type=float, default=0.1)
    parser.add_argument("--p_swap", type=float, default=0.85)
    parser.add_argument("--cutoff", type=int, default=100)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--topology", type=str, default='chain')

    # CC variables
    parser.add_argument("--F0", type=float, default=1.0)
    parser.add_argument("--channel_loss", type=float, default=0.0)
    parser.add_argument("--dt_seconds", type=float, default=0.00) #1e-4 for CC

    parser.add_argument("--path", type=str, default="checkpoints/cluster_004/")
    parser.add_argument("--dict", type=str, default="policy.pth")
    parser.add_argument("--no_plot_actions", dest="plot_actions", action="store_false", default=True)
    parser.add_argument("--verbose", type=int, default=0)
    
    return parser.parse_args()


if __name__ == "__main__":
    agent = QRNAgent()
    args = parse_args()
    model_path = args.path + args.dict

    results = agent.validate(
        model_path=model_path,
        n_episodes=args.episodes, 
        max_steps=args.steps,
        n_ch=args.n_ch,
        n_repeaters=args.nodes,           
        p_gen=args.p_gen, 
        p_swap=args.p_swap,  
        cutoff=args.cutoff, 
        F0=args.F0, 
        channel_loss=args.channel_loss,
        dt_seconds=args.dt_seconds, 
        plot_actions=args.plot_actions,
        save_dir=args.path,
        topology=args.topology,
        verbose=args.verbose,
    )