"""Stochastic q-heuristic policy for the purify-selectivity control experiment.

The rule search found the trained agent's purify selectivity is DIFFUSE (no
compact both-legal rule; ceiling AUC ~0.63-0.71). This module packages the
STOCHASTIC hybrid that tests whether a single scalar recovers the agent's edge:
inside the purify_then_swap skeleton, when BOTH swap and purify are legal at a
node, purify with a FIXED probability q; otherwise act deterministically
(purify-only -> PURIFY, swap-only -> SWAP). The both-legal coin is the ONLY
free parameter.

  - q = 1.0 is action-identical to purify_then_swap (PURIFY always wins the
    both-legal branch), which is the plumbing sanity check.
  - q measured from the trained agents is ~0.215 (omni_v3_20k_s1) and
    ~0.369 (omni_v3_20k_s3).

RNG discipline (mirrors rl_stack.strategies.random_policy): the hybrid's own
rng is a fresh np.random.default_rng(seed), INDEPENDENT of env.rng. Consuming
draws from it therefore never perturbs the environment's stream (link
generation, swap coin flips). That independence is exactly what makes the
q=1.0 identity hold bit-for-bit.

Numpy-only: this module must import without torch. Trained-agent policies come
from experiments.heatmap.optimal_baseline at the eval layer, not here.
"""
from __future__ import annotations

import numpy as np

from rl_stack.env_wrapper import NOOP, SWAP, PURIFY
from rl_stack import strategies


def make_hybrid_fn(q, seed):
    """Stochastic hybrid on the purify_then_swap skeleton.

    The deterministic branches (purify-only -> PURIFY, swap-only -> SWAP)
    mirror purify_then_swap; the both-legal branch flips a coin with
    P(PURIFY) = q. q=1.0 reproduces purify_then_swap exactly.

    `rng` is independent of env.rng (see module docstring). The returned
    signature matches mc_eval's policy_fn(env, obs).
    """
    rng = np.random.default_rng(seed)

    def policy(env, obs=None):
        mask = env.get_action_mask()
        actions = np.full(env.N, NOOP, dtype=np.int32)
        for i in range(env.N):
            if mask[i, PURIFY] and mask[i, SWAP]:          # both-legal
                actions[i] = PURIFY if rng.random() < q else SWAP
            elif mask[i, PURIFY]:                          # purify-only
                actions[i] = PURIFY
            elif mask[i, SWAP]:                            # swap-only
                actions[i] = SWAP
        return actions

    return policy


def purify_then_swap_fn(env, obs=None):
    """purify_then_swap baseline wrapped to the mc_eval policy_fn(env, obs)
    signature (the repo heuristic takes only env)."""
    return strategies.purify_then_swap(env)
