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

    from experiments.mc_eval import mc_eval, mc_eval_stats, make_agent_fn

`_rollout` is the ONE rollout loop in the project. mc_eval and mc_eval_stats
are two projections of it: mc_eval returns the golden-pinned (T, T_std) tuple
or the golden-pinned four-key dict, mc_eval_stats returns the same numbers plus
the standard errors the comparisons suite plots. The four rival copies that
used to live in experiments/comparisons/_common.py (eval_T, eval_stats,
eval_T_and_F) and experiments/training/batch_validate.py (run_comparison) were
proved bit-identical to this loop and deleted (2026-07-27).
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import math
import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import policies


def _rollout(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes, seed,
             p_gen_std, p_swap_std):
    """(delivery times censored at H, terminal fidelities of delivered eps).

    p_gen_std/p_swap_std > 0 -> per-repeater inhomogeneity (fresh draw each
    episode); =0 keeps the homogeneous RNG stream bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    times = []
    conn_fids = []
    for _ in range(n_episodes):
        env = QRNEnv(N, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                     p_gen_std=p_gen_std, p_swap_std=p_swap_std,
                     F0=1.0, channel_loss=0.0, max_steps=H,
                     rng=np.random.default_rng(rng.integers(2**32)))
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
    return times, conn_fids


def _se(xs):
    """Standard error of the mean; 0.0 on an empty sample."""
    return float(np.std(xs) / math.sqrt(len(xs))) if len(xs) else 0.0


def mc_eval(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes, seed=42,
            p_gen_std=0.0, p_swap_std=0.0, return_stats=False):
    # return_stats=True returns a richer dict; the default is a (mean, std)
    # tuple. Both returns are pinned by tests/golden_numbers.json (which also
    # pins the exact key SET of the dict), so add nothing here: the richer
    # projection with standard errors is mc_eval_stats below.
    times, conn_fids = _rollout(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H,
                                n_episodes, seed, p_gen_std, p_swap_std)
    T, T_std = float(np.mean(times)), float(np.std(times))
    if not return_stats:
        return T, T_std
    n = float(n_episodes)
    return dict(
        T=T, T_std=T_std,
        conn_rate=(len(conn_fids) / n),
        mean_F_conn=(float(np.mean(conn_fids)) if conn_fids else None),
    )


def mc_eval_stats(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H, n_episodes,
                  seed=42, p_gen_std=0.0, p_swap_std=0.0):
    """Same rollout as mc_eval, plus the standard errors the figures need.

    Returns dict(T, T_std, se, conn_rate, mean_F_conn, seF_conn):
      T, T_std     delivery time censored at H, mean and population std
      se           standard error of T
      conn_rate    fraction of episodes that delivered end-to-end
      mean_F_conn  mean terminal fidelity over delivered episodes (None if
                   nothing delivered)
      seF_conn     standard error of that fidelity (0.0 if nothing delivered)
    """
    times, conn_fids = _rollout(policy_fn, N, n_ch, p_gen, p_swap, cutoff, H,
                                n_episodes, seed, p_gen_std, p_swap_std)
    return dict(
        T=float(np.mean(times)), T_std=float(np.std(times)), se=_se(times),
        conn_rate=(len(conn_fids) / float(n_episodes)),
        mean_F_conn=(float(np.mean(conn_fids)) if conn_fids else None),
        seF_conn=_se(conn_fids),
    )


def swap_asap_fn(env, obs):
    return policies.swap_asap(env)


def make_agent_fn(ckpt, disable_actions=None):
    """Policy fn for a trained checkpoint. `disable_actions` masks the given
    action columns at inference (e.g. (PURIFY,) for a swap-only evaluation).

    node_dim and hidden are inferred from the checkpoint by load_qnet, so a
    caller can never pass a width that silently disagrees with the weights.
    Inference only: the QRNAgent wrapper is here for select_action, its
    optimizer is never stepped."""
    from rl_stack.agent import QRNAgent
    from rl_stack.model import load_qnet
    agent = QRNAgent()
    net = load_qnet(ckpt, device=str(agent.device))
    agent.policy_net = net

    def fn(env, obs):
        mask_row = env.action_mask(env.active_node)
        if disable_actions:
            for a in disable_actions:
                mask_row[a] = False
        return agent.select_action(obs, mask_row, env.active_node, training=False)
    return fn
