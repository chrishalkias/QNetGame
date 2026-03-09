"""Heuristic strategies for baseline comparison against the RL agent.

Each strategy takes a QRNEnv and returns an (N,) action array.
All strategies respect action masks.
"""

from __future__ import annotations
import numpy as np
from rl_stack.env_wrapper import QRNEnv, NOOP, ENTANGLE, SWAP, PURIFY


def _select_non_conflicting_swaps(env: QRNEnv, mask: np.ndarray) -> set:
    """
    Return a set of node indices that can swap without conflicting with each other.

    When node r swaps qubits linked to ra and rb, it locks remote qubits at ra
    and rb for the duration of the classical round-trip delay.  If a second node
    also tries to swap in the same timestep and one of *its* partners is ra or
    rb, that second swap will find the remote qubit locked and silently fail.

    Strategy: greedily assign SWAP to nodes in order; track which nodes have a
    qubit that will be locked; skip any node whose partners are already claimed.
    """
    locked: set = set()
    swapping: set = set()

    for i in range(env.N):
        if not mask[i, SWAP]:
            continue
        if i in locked:
            continue
        rep = env.net.repeaters[i]
        pair = rep.select_swap_pair(env.net._positions)
        if pair is None:
            continue
        qa, qb = pair
        ra = int(rep.partner_repeater[qa])
        rb = int(rep.partner_repeater[qb])
        # Skip if either remote endpoint is already being locked
        if ra in locked or rb in locked:
            continue
        swapping.add(i)
        # Mark i and both remote endpoints as locked so no other swap touches them
        locked.update({i, ra, rb})

    return swapping


def swap_asap(env: QRNEnv) -> np.ndarray:
    """
    Swap at every node that can, respecting qubit-locking conflicts.

    Naively assigning SWAP to every node that has ≥2 available qubits breaks
    down because the BSM at node r immediately locks remote qubits at its two
    partner nodes.  If two nodes share a partner (common in chains), the second
    swap finds its remote qubit locked and does nothing — wasting the action.

    Fix: choose a maximal conflict-free set of swapping nodes (greedy over node
    index), then entangle at the remaining nodes.
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    swapping = _select_non_conflicting_swaps(env, mask)
    for i in range(env.N):
        if i in swapping:
            actions[i] = SWAP
        elif mask[i, ENTANGLE]:
            actions[i] = ENTANGLE
    return actions


def purify_then_swap(env: QRNEnv) -> np.ndarray:
    """
    Purify if possible, then swap (conflict-free), then entangle.

    Applies the same conflict-aware swap selection as swap_asap so that
    simultaneous swaps never silently clobber each other's remote qubits.
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    # First pass: assign purify
    for i in range(env.N):
        if mask[i, PURIFY]:
            actions[i] = PURIFY
    # Second pass: conflict-free swaps at nodes not already purifying
    purifying = set(np.flatnonzero(actions == PURIFY))
    # Temporarily mask out purifying nodes so they aren't considered for swap
    swap_mask = mask.copy()
    for i in purifying:
        swap_mask[i, SWAP] = False
    swapping = _select_non_conflicting_swaps(env, swap_mask)
    for i in range(env.N):
        if i in swapping:
            actions[i] = SWAP
        elif actions[i] == NOOP and mask[i, ENTANGLE]:
            actions[i] = ENTANGLE
    return actions


def entangle_only(env: QRNEnv) -> np.ndarray:
    """Only entangle, never swap or purify (lower bound)."""
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if mask[i, ENTANGLE]:
            actions[i] = ENTANGLE
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
    "SwapASAP": swap_asap,
    "PurifyThenSwap": purify_then_swap,
    "EntangleOnly": entangle_only,
}