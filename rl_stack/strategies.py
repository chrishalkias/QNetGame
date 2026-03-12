"""Heuristic strategies for baseline comparison against the RL agent.

Each strategy takes a QRNEnv and returns an (N,) action array
containing only NOOP, SWAP, or PURIFY.  All strategies respect the
action mask (source/dest are always NOOP).

Entanglement is handled automatically by the environment step.
"""

from __future__ import annotations
import numpy as np
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY



def swap_asap(env: QRNEnv) -> np.ndarray:
    """Swap at every interior node that can, immediately.

    If a node has ≥2 available qubits linked to distinct partners,
    assign SWAP.  The network's swap function itself handles
    contention gracefully (returns failure if qubits became
    locked by an earlier swap in the same timestep).
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if mask[i, SWAP]:
            actions[i] = SWAP
    return actions


def purify_then_swap(env: QRNEnv) -> np.ndarray:
    """Purify if possible, otherwise swap if possible, else noop."""
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if mask[i, PURIFY]:
            actions[i] = PURIFY
        elif mask[i, SWAP]:
            actions[i] = SWAP
    return actions


def random_policy(env: QRNEnv, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random valid action per node."""
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        valid = np.flatnonzero(mask[i])
        actions[i] = rng.choice(valid) if len(valid) > 0 else NOOP
    return actions

STRATEGY_MAP = {
    "SwapASAP":        swap_asap,
    "PurifyThenSwap":  purify_then_swap,
}
