"""
--------------------------------------------------------------------------------
Stochastic q-heuristic policy for the purify-selectivity control experiment.

The rule search found the trained agent's purify selectivity is DIFFUSE (no
compact both-legal rule; ceiling AUC ~0.63-0.71). This module packages the
STOCHASTIC hybrid that tests whether a single scalar recovers the agent's edge:
inside the purify_then_swap skeleton, when BOTH swap and purify are legal at
env.active_node (the serialized sweep decides one node per env.step call),
purify with a FIXED probability q; otherwise act deterministically
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
q=1.0 identity hold bit-for-bit. Exactly one rng.random() draw is consumed
per both-legal MICRO-STEP (never zero, never two), since each env.step call
decides exactly one node.

Numpy-only: this module must import without torch. Trained-agent policies come
from experiments.mc_eval at the eval layer, not here.
--------------------------------------------------------------------------------
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
    signature matches mc_eval's policy_fn(env, obs) -> int for env.active_node.
    """
    rng = np.random.default_rng(seed)

    def policy(env, obs=None):
        i = env.active_node
        mask = env.get_action_mask()[i]
        if mask[PURIFY] and mask[SWAP]:          # both-legal
            return PURIFY if rng.random() < q else SWAP
        elif mask[PURIFY]:                       # purify-only
            return PURIFY
        elif mask[SWAP]:                         # swap-only
            return SWAP
        return NOOP

    return policy


def purify_then_swap_fn(env, obs=None):
    """purify_then_swap baseline wrapped to the mc_eval policy_fn(env, obs)
    signature (the repo heuristic takes only env)."""
    return strategies.purify_then_swap(env)


def make_conditional_fn(coef_path, seed, p_gen, p_swap, cutoff):
    """State-conditioned q(state) variant of `make_hybrid_fn`.

    Same purify_then_swap skeleton (purify-only -> PURIFY, swap-only ->
    SWAP), but in the both-legal branch the coin is Bernoulli(q_i(state))
    with q_i read off the exported logistic in `coef_path`
    (experiments/q_heuristic/fit_q_conditional.py's contract):

        q = sigmoid(coef . ((x - mu) / sigma) + intercept)

    `x` is the 34-feature row (`experiments.policy_probes.purify_map`'s
    35-column schema minus `can_swap`, which is constant 1.0 in the
    both-legal subset) built by `purify_map.node_row` at decision time and
    reordered to the JSON's `columns` BY NAME.

    Requires torch: `purify_map` imports `experiments.policy_probes._collect`,
    which pulls torch (the greedy-agent rollout it was built for). That
    import is therefore lazy, INSIDE this factory, so importing this module
    (`hybrid_policy.py`) stays numpy-only, matching `make_hybrid_fn`'s
    contract; the eval roster that calls `make_conditional_fn` already needs
    torch for the trained-agent policies anyway.

    RNG discipline mirrors `make_hybrid_fn`: `rng` is a fresh
    `np.random.default_rng(seed)`, independent of `env.rng`, and exactly one
    `rng.random()` draw is consumed per both-legal MICRO-STEP (never zero,
    never two) so the rng stream lines up 1:1 with `make_hybrid_fn`'s when
    q_i happens to be constant, that is what the bit-identity test exploits.
    """
    import json
    from experiments.policy_probes import purify_map  # lazy: pulls torch

    with open(coef_path) as f:
        data = json.load(f)
    columns = data["columns"]
    mu = np.asarray(data["mu"], dtype=np.float64)
    sigma = np.asarray(data["sigma"], dtype=np.float64)
    coef = np.asarray(data["coef"], dtype=np.float64)
    intercept = float(data["intercept"])
    # map the JSON's 34 columns -> positions in purify_map.COLUMNS (35), by
    # name, never by position (purify_map.node_row's row is COLUMNS-ordered).
    col_idx = np.asarray([purify_map.COLUMNS.index(c) for c in columns],
                         dtype=np.int64)

    rng = np.random.default_rng(seed)
    ctx = dict(p_gen=p_gen, p_swap=p_swap, cutoff=cutoff)

    def policy(env, obs=None):
        if obs is None:
            obs = env.get_observation()
        i = env.active_node
        mask = env.get_action_mask()
        if mask[i, PURIFY] and mask[i, SWAP]:          # both-legal
            dummy_acts = np.zeros(env.N, dtype=int)   # only feeds node_row's label, unused
            r = purify_map.node_row(env, obs, mask, dummy_acts, i, ctx)
            if r is None:                              # defensive: no candidate partner
                return SWAP
            row = np.asarray(r[0], dtype=np.float64)
            x = row[col_idx]
            z = float(np.dot(coef, (x - mu) / sigma) + intercept)
            q_i = 1.0 / (1.0 + np.exp(-z))
            return PURIFY if rng.random() < q_i else SWAP
        elif mask[i, PURIFY]:                          # purify-only
            return PURIFY
        elif mask[i, SWAP]:                            # swap-only
            return SWAP
        return NOOP

    return policy
