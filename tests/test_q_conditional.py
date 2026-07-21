"""
--------------------------------------------------------------------------------
Wiring guards for the q-conditional purify policy (make_conditional_fn).

Proves make_conditional_fn is plumbed correctly by exploiting its RNG
discipline against make_hybrid_fn's: EXACTLY one rng.random() draw per
both-legal node per step means a constant-q logistic (coef=0) reproduces the
constant-q hybrid bit-for-bit, and a saturated logistic (q~1) reproduces the
deterministic purify_then_swap heuristic bit-for-bit.
--------------------------------------------------------------------------------
"""
import json
import math
import os

import numpy as np
import pytest

from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY
from experiments.q_heuristic.hybrid_policy import (
    make_hybrid_fn, make_conditional_fn, purify_then_swap_fn)

ENV_KW = dict(n_repeaters=6, n_ch=4, p_gen=0.4, p_swap=0.8, cutoff=30,
              F0=1.0, channel_loss=0.0, topology="chain")
H = 200


def _write_coef_json(tmp_path, columns, coef, mu, sigma, intercept, name="q.json"):
    path = tmp_path / name
    data = dict(columns=list(columns), coef=[float(c) for c in coef],
                mu=[float(m) for m in mu], sigma=[float(s) for s in sigma],
                intercept=float(intercept))
    path.write_text(json.dumps(data))
    return str(path)


def _synthetic_columns():
    """The JSON's 34 columns: purify_map.COLUMNS (35) minus `can_swap`,
    which is constant 1.0 in the both-legal subset (see fit_q_conditional.py's
    docstring)."""
    from experiments.policy_probes import purify_map
    return [c for c in purify_map.COLUMNS if c != "can_swap"]


def _rollout(policy_fn, seed):
    """Roll one episode, returning the per-step action arrays."""
    env = QRNEnv(max_steps=H, rng=np.random.default_rng(seed), **ENV_KW)
    obs = env.reset()
    log = []
    for _ in range(H):
        a = policy_fn(env, obs)
        log.append(np.asarray(a).copy())
        obs, _, done, _ = env.step(a)
        if done:
            break
    return log


def test_bit_identity_degenerate_case_matches_constant_hybrid(tmp_path):
    """coef=0, constant q -> the q-conditional policy IS the constant-q hybrid,
    on identical env seeds, over 3 seeded episodes."""
    columns = _synthetic_columns()
    q = 0.369
    intercept = math.log(q / (1.0 - q))
    coef_path = _write_coef_json(tmp_path, columns, np.zeros(34), np.zeros(34),
                                 np.ones(34), intercept)

    cond_fn = make_conditional_fn(coef_path, seed=7, p_gen=0.4, p_swap=0.8, cutoff=30)
    hyb_fn = make_hybrid_fn(q=q, seed=7)

    for ep_seed in (1, 2, 3):
        a_cond = _rollout(cond_fn, ep_seed)
        a_hyb = _rollout(hyb_fn, ep_seed)
        assert len(a_cond) == len(a_hyb)
        for step_i, (x, y) in enumerate(zip(a_cond, a_hyb)):
            np.testing.assert_array_equal(
                x, y, err_msg=f"seed={ep_seed} step={step_i}")


def test_saturated_case_matches_purify_then_swap(tmp_path):
    """intercept=+50 -> q~1 in every both-legal state -> reproduces
    purify_then_swap_fn action-for-action on one seeded episode."""
    columns = _synthetic_columns()
    coef_path = _write_coef_json(tmp_path, columns, np.zeros(34), np.zeros(34),
                                 np.ones(34), 50.0)
    cond_fn = make_conditional_fn(coef_path, seed=3, p_gen=0.4, p_swap=0.8, cutoff=30)

    a_cond = _rollout(cond_fn, seed=11)
    a_pts = _rollout(purify_then_swap_fn, seed=11)
    assert len(a_cond) == len(a_pts)
    for step_i, (x, y) in enumerate(zip(a_cond, a_pts)):
        np.testing.assert_array_equal(x, y, err_msg=f"step={step_i}")


def test_real_artifact_smoke():
    """If the fitted s3 artifact is present, roll one episode: every action
    is a valid discrete action, and if any both-legal state occurred, at
    least one PURIFY was drawn at the fitted q-model rate."""
    coef_path = "experiments/q_heuristic/q_conditional_s3.json"
    if not os.path.exists(coef_path):
        pytest.skip("experiments/q_heuristic/q_conditional_s3.json not present")

    cond_fn = make_conditional_fn(coef_path, seed=42, p_gen=0.4, p_swap=0.8, cutoff=30)
    env = QRNEnv(max_steps=H, rng=np.random.default_rng(123), **ENV_KW)
    obs = env.reset()

    saw_both_legal = False
    saw_purify = False
    valid_actions = {NOOP, SWAP, PURIFY}
    for _ in range(H):
        mask = env.get_action_mask()
        if bool(np.any(mask[:, PURIFY] & mask[:, SWAP])):
            saw_both_legal = True
        a = cond_fn(env, obs)
        assert set(np.unique(a).tolist()).issubset(valid_actions)
        if bool(np.any(a == PURIFY)):
            saw_purify = True
        obs, _, done, _ = env.step(a)
        if done:
            break

    if saw_both_legal:
        assert saw_purify
