"""THE canonical delivery-time evaluator (censored Monte-Carlo) + policy wrappers.

Extracted verbatim from experiments/heatmap/optimal_baseline.py (2026-07-18)
when the exact-DP machinery was retired to .local/legacy/optimal_dp/. Every
experiment measures delivery time through mc_eval so numbers stay comparable:
T = mean steps to end-to-end delivery, censored at the horizon H (undelivered
episodes count as H).

    from experiments.mc_eval import mc_eval, make_agent_fn, swap_asap_fn
"""
from __future__ import annotations
import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import strategies

# A two-qubit Werner state is entangled iff its fidelity exceeds 1/2 (separable
# at F <= 1/2). The env terminates on the FIRST topological connection and puts
# the delivered end-to-end F in info["fidelity"]; a policy cannot retry after a
# separable delivery, so T_ent measures whether that first connection is
# entangled, not the time to eventually reach an entangled link.
ENT_THRESHOLD = 0.5


def mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes, seed=42,
            p_gen_std=0.0, p_swap_std=0.0, f_min=None, return_stats=False):
    # p_gen_std/p_swap_std > 0 -> per-repeater inhomogeneity (fresh draw each
    # episode); =0 keeps the homogeneous RNG stream bit-for-bit.
    #
    # f_min gates what counts as a delivery for the censored time T:
    #   f_min is None -> time-to-connection T_conn (delivered iff topologically
    #                    connected, i.e. F > 0), the original semantics.
    #   f_min = 0.5   -> time-to-entanglement T_ent (a connected-but-separable
    #                    episode is censored at H, exactly like never connecting).
    # return_stats=True returns a richer dict; the default (mean, std) tuple and
    # the default f_min=None keep every existing caller bit-compatible.
    rng = np.random.default_rng(seed)
    times = []
    n_delivered = 0                   # episodes counting as a delivery under the gate
    conn_fids, ent_fids = [], []      # F over connected / over entangled episodes
    for _ in range(n_episodes):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     p_gen_std=p_gen_std, p_swap_std=p_swap_std,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0, max_steps=H,
                     topology="chain", rng=np.random.default_rng(rng.integers(2**32)))
        obs = env.reset()
        step = 0
        for step in range(H):
            a = policy_fn(env, obs)
            obs, _, done, info = env.step(a)
            if done:
                break
        F = info.get("fidelity", 0.0)
        connected = bool(done) and F > 0
        if connected:
            conn_fids.append(float(F))
            if F > ENT_THRESHOLD:
                ent_fids.append(float(F))
        delivered = connected if f_min is None else (connected and F > f_min)
        n_delivered += int(delivered)
        times.append(step + 1 if delivered else H)
    T, T_std = float(np.mean(times)), float(np.std(times))
    if not return_stats:
        return T, T_std
    n = float(n_episodes)
    return dict(
        T=T, T_std=T_std,
        delivery_rate=(n_delivered / n),
        conn_rate=(len(conn_fids) / n),
        mean_F_conn=(float(np.mean(conn_fids)) if conn_fids else None),
        mean_F_ent=(float(np.mean(ent_fids)) if ent_fids else None),
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
        mask = env.get_action_mask()
        if disable_actions:
            mask = mask.copy()
            for a in disable_actions:
                mask[:, a] = False
        return agent.select_actions(obs, mask, training=False)
    return fn
