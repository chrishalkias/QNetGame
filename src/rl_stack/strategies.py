'''
Heuristic strategies for baseline comparison against the RL agent.

Each strategy takes a QRNEnv and returns a SCALAR action (NOOP/SWAP/PURIFY)
for env.active_node -- the one node deciding this micro-step under the
serialized sweep. All strategies respect that node's action mask.

Entanglement is handled automatically by the environment step.
'''

from __future__ import annotations
import numpy as np
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY


def swap_asap(env: QRNEnv) -> int:
    """Swap at env.active_node if it can, immediately.

    The engine's swap function handles same-tick contention gracefully
    (a beaten swap simply fails: its qubits are no longer available once an
    earlier micro-step this tick has consumed them).
    """
    r = env.active_node
    return SWAP if env.get_action_mask()[r, SWAP] else NOOP


def purify_then_swap(env: QRNEnv) -> int:
    """Purify env.active_node if possible, otherwise swap if possible, else noop."""
    r = env.active_node
    m = env.get_action_mask()[r]
    return PURIFY if m[PURIFY] else (SWAP if m[SWAP] else NOOP)


def random_policy(env: QRNEnv, rng: np.random.Generator) -> int:
    """Uniformly random valid action for env.active_node.

    IMPORTANT: *rng* must be independent of env.rng, otherwise drawing
    action choices perturbs the environment's own random stream
    (link generation, swap outcomes) and invalidates the comparison.
    """
    r = env.active_node
    valid = np.flatnonzero(env.get_action_mask()[r])
    return int(rng.choice(valid)) if len(valid) else NOOP
