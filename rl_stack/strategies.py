'''
Heuristic strategies for baseline comparison against the RL agent.

Each strategy takes a QRNEnv and returns an (N,) action array
containing only NOOP, SWAP, or PURIFY.  All strategies respect the
action mask (source/dest are always NOOP).

Entanglement is handled automatically by the environment step.
'''

from __future__ import annotations
import numpy as np
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY



def swap_asap(env: QRNEnv) -> np.ndarray:
    """Swap at every interior node that can, immediately.

    If a node has =>2 available qubits linked to distinct partners,
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


def fidelity_gated_swap(env: QRNEnv, f_threshold: float = 0.5) -> np.ndarray:
    """Swap only when the node's mean link fidelity exceeds a threshold.

    Approximates the learned RL policy from cluster_004: the agent
    waits for fresh, high-quality links before swapping and never
    purifies.  This single-threshold rule captures ~80 % of the
    agent's advantage over swap-ASAP in most regimes.
    """
    from quantum_repeater_sim.repeater import werner_to_fidelity

    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if not mask[i, SWAP]:
            continue
        rep = env.net.repeaters[i]
        occ = rep.occupied_indices()
        if len(occ) == 0:
            continue
        mean_f = float(np.mean(werner_to_fidelity(rep.werner_param[occ])))
        if mean_f >= f_threshold:
            actions[i] = SWAP
    return actions


def random_policy(env: QRNEnv, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random valid action per node.

    IMPORTANT: *rng* must be independent of env.rng, otherwise drawing
    action choices perturbs the environment's own random stream
    (link generation, swap outcomes) and invalidates the comparison.
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        valid = np.flatnonzero(mask[i])
        actions[i] = rng.choice(valid) if len(valid) > 0 else NOOP
    return actions

STRATEGY_MAP = {
    "SwapASAP":        swap_asap,
    "PurifyThenSwap":  purify_then_swap,
    "FidGatedSwap":    fidelity_gated_swap,
}
