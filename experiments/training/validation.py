"""
Validate a trained checkpoint against the heuristics.

Reporting is entanglement-gated: a win counts only when the delivered end-to-end
fidelity exceeds 1/2 (a two-qubit Werner state is separable at F <= 1/2). The
table reports the time to end-to-end entanglement T_ent (separable deliveries
censored at the horizon, exactly like a never-connected episode) next to the old
time-to-connection T_conn and the mean delivered fidelity. The env terminates on
the FIRST connection, so T_ent measures whether that first connection is
entangled; a policy cannot retry after a separable delivery.
"""

import argparse
import numpy as np
from rl_stack import QRNAgent
from rl_stack import strategies
from rl_stack.env_wrapper import QRNEnv

F_ENT = 0.5   # Werner entanglement threshold on delivered fidelity


def gated_report(agent, args):
    """Paired-seed gated table: for Agent / SwapASAP / PurifySwap / Random print
    conn% (any F>0), ent% (F>1/2), T_conn, T_ent, and mean delivered F over the
    connected and the entangled subsets. Uses the same env config as validate()
    and the same seed_rng(42) episode seeds, so all policies see one network per
    episode (paired). This does not touch the training/env code paths."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    env_args = dict(n_repeaters=args.nodes, n_ch=args.n_ch, spacing=50,
                    p_gen=args.p_gen, p_swap=args.p_swap,
                    p_gen_std=args.p_gen_std, p_swap_std=args.p_swap_std,
                    cutoff=args.cutoff, F0=args.F0, channel_loss=args.channel_loss,
                    dt_seconds=args.dt_seconds, max_steps=args.steps,
                    topology=args.topology)
    seed_rng = np.random.default_rng(42)
    ep_seeds = seed_rng.integers(0, 2**32, size=args.episodes)
    action_rng = np.random.default_rng()      # Random baseline: independent stream
    H = args.steps

    def act(name, env, obs):
        if name == "Agent":
            return agent.select_actions(obs, env.get_action_mask(), training=False)
        if name == "Random":
            return strategies.random_policy(env, action_rng)
        return {"SwapASAP": strategies.swap_asap,
                "PurifySwap": strategies.purify_then_swap}[name](env)

    print(f"\nEntanglement-gated (F>{F_ENT}) validation  "
          f"N={args.nodes} n_ch={args.n_ch} p_gen={args.p_gen} "
          f"p_swap={args.p_swap} cutoff={args.cutoff} H={H} eps={args.episodes}")
    hdr = (f"{'strategy':<11}{'conn%':>7}{'ent%':>7}{'T_conn':>9}{'T_ent':>9}"
           f"{'F_conn':>8}{'F_ent':>8}")
    print(hdr); print("-" * len(hdr))
    for name in ("Agent", "SwapASAP", "PurifySwap", "Random"):
        t_conn, t_ent, f_conn, f_ent = [], [], [], []
        for ep in range(args.episodes):
            env = QRNEnv(**env_args, rng=np.random.default_rng(int(ep_seeds[ep])))
            obs = env.reset()
            step, done, info = 0, False, {}
            for step in range(H):
                obs, _, done, info = env.step(act(name, env, obs))
                if done:
                    break
            F = info.get("fidelity", 0.0)
            connected = bool(done) and F > 0
            entangled = connected and F > F_ENT
            t_conn.append(step + 1 if connected else H)
            t_ent.append(step + 1 if entangled else H)
            if connected:
                f_conn.append(float(F))
            if entangled:
                f_ent.append(float(F))
        n = float(args.episodes)
        mf_conn = f"{np.mean(f_conn):.3f}" if f_conn else "  -  "
        mf_ent = f"{np.mean(f_ent):.3f}" if f_ent else "  -  "
        print(f"{name:<11}{100*len(f_conn)/n:>6.1f} {100*len(f_ent)/n:>6.1f} "
              f"{np.mean(t_conn):>8.3f} {np.mean(t_ent):>8.3f} "
              f"{mf_conn:>7} {mf_ent:>7}")
    agent.epsilon = old_eps

def parse_args():
    parser = argparse.ArgumentParser(description="Test QRNAgent")
    #Validation variables
    parser.add_argument("--run_id", type=str, default="v006")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)

    #System variables
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--n_ch", type=int, default=4)
    parser.add_argument("--p_gen", type=float, default=0.1,
                        help="per-network MEAN link-generation prob.")
    parser.add_argument("--p_swap", type=float, default=0.85,
                        help="per-network MEAN BSM success prob.")
    parser.add_argument("--p_gen_std", type=float, default=0.0,
                        help="per-repeater spread of p_gen (0 = homogeneous)")
    parser.add_argument("--p_swap_std", type=float, default=0.0,
                        help="per-repeater spread of p_swap (0 = homogeneous)")
    parser.add_argument("--cutoff", type=int, default=100)
    parser.add_argument("--topology", type=str, default='chain')

    # CC variables
    parser.add_argument("--F0", type=float, default=1.0)
    parser.add_argument("--channel_loss", type=float, default=0.0)
    parser.add_argument("--dt_seconds", type=float, default=0.00) #1e-4 for CC

    parser.add_argument("--path", type=str, default="checkpoints/sota/")
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
        p_gen_std=args.p_gen_std,
        p_swap_std=args.p_swap_std,
        cutoff=args.cutoff,
        F0=args.F0, 
        channel_loss=args.channel_loss,
        dt_seconds=args.dt_seconds, 
        plot_actions=args.plot_actions,
        save_dir=args.path,
        topology=args.topology,
        verbose=args.verbose,
    )

    # validate() loaded the checkpoint into agent.policy_net; reuse it for the
    # entanglement-gated table (F>1/2 win condition) on the same config.
    gated_report(agent, args)