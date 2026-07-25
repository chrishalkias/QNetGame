"""
--------------------------------------------------------------------------------
THE canonical delivery-time evaluator (censored Monte-Carlo) + policy wrappers.

Extracted verbatim from experiments/heatmap/optimal_baseline.py (2026-07-18)
when the exact-DP machinery was retired to .local/legacy/optimal_dp/. Every
experiment measures delivery time through mc_eval so numbers stay comparable:
T = mean steps to end-to-end delivery, censored at the horizon H (undelivered
episodes count as H). Delivery is topological (source holds a link to dest);
the cutoff already bounds how decohered a surviving link can be, so no extra
fidelity threshold is imposed on the terminal state (dropped 2026-07-21: the
F>1/2 gate this file used to carry never actually fired in practice).

    from experiments.mc_eval import mc_eval, make_agent_fn, swap_asap_fn
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import strategies


def mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes, seed=42,
            p_gen_std=0.0, p_swap_std=0.0, return_stats=False):
    # p_gen_std/p_swap_std > 0 -> per-repeater inhomogeneity (fresh draw each
    # episode); =0 keeps the homogeneous RNG stream bit-for-bit.
    # return_stats=True returns a richer dict; the default is a (mean, std) tuple.
    rng = np.random.default_rng(seed)
    times = []
    conn_fids = []
    for _ in range(n_episodes):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     p_gen_std=p_gen_std, p_swap_std=p_swap_std,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(rng.integers(2**32)))
        obs = env.reset()
        info = {}
        while not env.done and env.steps < H:
            a = policy_fn(env, obs)
            obs, _, done, info = env.step(a)
            if done:
                break
        F = info.get("fidelity", 0.0)
        connected = bool(info.get("terminated")) and F > 0
        if connected:
            conn_fids.append(float(F))
        times.append(info["ticks"] if connected else H)
    T, T_std = float(np.mean(times)), float(np.std(times))
    if not return_stats:
        return T, T_std
    n = float(n_episodes)
    return dict(
        T=T, T_std=T_std,
        conn_rate=(len(conn_fids) / n),
        mean_F_conn=(float(np.mean(conn_fids)) if conn_fids else None),
    )


def swap_asap_fn(env, obs):
    return strategies.swap_asap(env)


def make_agent_fn(ckpt, hidden=64, disable_actions=None):
    """Policy fn for a trained checkpoint. `disable_actions` masks the given
    action columns at inference (e.g. (PURIFY,) for a swap-only evaluation)."""
    import torch
    from rl_stack.agent import QRNAgent
    agent = QRNAgent(hidden=hidden)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(sd)
    agent.policy_net.eval()

    def fn(env, obs):
        mask_row = env.get_action_mask()[env.active_node].copy()
        if disable_actions:
            for a in disable_actions:
                mask_row[a] = False
        return agent.select_action(obs, mask_row, env.active_node, training=False)
    return fn
