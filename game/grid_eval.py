"""Evaluate a policy against swap-asap on square grids (the Phase-2 success bar:
beat swap-asap on grid delivery time)."""
from __future__ import annotations
from typing import Dict, Sequence

import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import strategies


def _mean_delivery(policy_fn, side, n_ch, p_gen, p_swap, cutoff, max_steps,
                   n_episodes, seed):
    master = np.random.default_rng(seed)
    times = []
    for _ in range(n_episodes):
        env = QRNEnv(n_repeaters=side, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap,
                     cutoff=cutoff, F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                     max_steps=max_steps, topology="grid",
                     rng=np.random.default_rng(master.integers(2**32)))
        obs = env.reset()
        done, step, info = False, 0, {}
        for step in range(max_steps):
            obs, _, done, info = env.step(policy_fn(env, obs))
            if done:
                break
        times.append(step + 1 if (done and info.get("fidelity", 0.0) > 0) else max_steps)
    return float(np.mean(times))


def evaluate_on_grids(agent_fn, grid_sides: Sequence[int] = (3, 4), n_ch: int = 2,
                      p_gen: float = 0.7, p_swap: float = 0.8, cutoff: int = 20,
                      max_steps: int = 60, n_episodes: int = 500,
                      seed: int = 42) -> Dict:
    """Agent vs swap-asap mean delivery time on each grid side. Positive
    `agent_beats_swap_pct` = agent faster (the Phase-2 goal)."""
    swap_fn = lambda env, obs: strategies.swap_asap(env)
    rows = []
    for side in grid_sides:
        ta = _mean_delivery(agent_fn, side, n_ch, p_gen, p_swap, cutoff,
                            max_steps, n_episodes, seed)
        ts = _mean_delivery(swap_fn, side, n_ch, p_gen, p_swap, cutoff,
                            max_steps, n_episodes, seed)
        rows.append({
            "grid": side, "T_agent": ta, "T_swap_asap": ts,
            "agent_beats_swap_pct": 100.0 * (ts - ta) / ts if ts else float("nan"),
        })
    return {"config": {"n_ch": n_ch, "p_gen": p_gen, "p_swap": p_swap,
                       "cutoff": cutoff, "n_episodes": n_episodes}, "rows": rows}
