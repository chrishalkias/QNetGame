#!/usr/bin/env python3
"""Test suite for the rewritten RL stack.

Verifies: env semantics, action masking, physical correctness,
swap_asap optimality, buffer integrity, and size-agnosticism.
"""

from __future__ import annotations
import sys, time
import numpy as np

sys.path.insert(0, "/home/claude")
from rl_stack.env_wrapper import (
    QRNEnv, N_ACTIONS, NOOP, SWAP, PURIFY, ACTION_NAMES)
from rl_stack.buffer import ReplayBuffer
from rl_stack import strategies

SEP = "=" * 72
OK = FAIL = 0
def _ok(msg=""): global OK; OK += 1; print(f"    \u2713 {msg}") if msg else None
def _fail(msg): global FAIL; FAIL += 1; print(f"    \u2717 FAIL: {msg}")


# ══════════════════════════════════════════════════════════════════
# 1. ENV BASICS
# ══════════════════════════════════════════════════════════════════

def test_env_creation():
    print(f"\n{SEP}\nTEST 1: Env creation and reset\n{SEP}")
    for N in range(3, 12):
        env = QRNEnv(n_repeaters=N, n_ch=4, dt_seconds=0.0,
                     max_steps=50, rng=np.random.default_rng(0))
        obs = env.reset()
        assert obs["x"].shape == (N, 8), f"N={N}: shape {obs['x'].shape}"
        assert obs["edge_index"].shape[0] == 2
        assert env.source != env.dest
        assert env.steps == 0 and not env.done
    _ok("N=3..11, obs shapes, target selection")


def test_action_space():
    print(f"\n{SEP}\nTEST 2: Action space is noop/swap/purify\n{SEP}")
    assert N_ACTIONS == 3
    assert NOOP == 0 and SWAP == 1 and PURIFY == 2
    assert ACTION_NAMES == ["noop", "swap", "purify"]
    _ok("3-action space correct")


def test_action_mask_shapes():
    print(f"\n{SEP}\nTEST 3: Action mask shapes and source/dest blocking\n{SEP}")
    rng = np.random.default_rng(42)
    for N in range(3, 11):
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, rng=rng, channel_loss=0.0)
        env.reset()
        mask = env.get_action_mask()
        assert mask.shape == (N, 3), f"N={N}: mask shape {mask.shape}"
        assert mask.dtype == bool
        assert mask[:, NOOP].all(), "NOOP always valid"
        assert not mask[env.source, SWAP], "Source cannot swap"
        assert not mask[env.dest, SWAP], "Dest cannot swap"
        assert not mask[env.source, PURIFY], "Source cannot purify"
        assert not mask[env.dest, PURIFY], "Dest cannot purify"
    _ok("N=3..10, shapes, source/dest blocked for swap AND purify")


def test_step_basic():
    print(f"\n{SEP}\nTEST 4: Basic step execution\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50, rng=np.random.default_rng(0))
    env.reset()
    actions = np.full(5, NOOP, dtype=np.int32)
    obs, reward, done, info = env.step(actions)
    assert obs["x"].shape == (5, 8)
    assert reward == QRNEnv.STEP_COST
    assert env.steps == 1
    assert info["noops"] == 5
    _ok("noop step")


def test_observation_features():
    print(f"\n{SEP}\nTEST 5: Observation feature values\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, rng=np.random.default_rng(0),
                 channel_loss=0.0, ee=True)
    env.reset()
    obs = env.get_observation()
    x = obs["x"]
    assert x[0, 2] == 1.0, "Node 0 should be source"
    assert x[4, 3] == 1.0, "Node 4 should be dest"
    assert x[1, 2] == 0.0 and x[1, 3] == 0.0, "Interior is neither"
    assert x[0, 5] == 0.0, "Source can_swap must be 0"
    assert x[0, 6] == 0.0, "Source can_purify must be 0"
    assert x[4, 5] == 0.0, "Dest can_swap must be 0"
    assert x[4, 6] == 0.0, "Dest can_purify must be 0"
    # Time remaining at step 0 should be 1.0
    assert abs(x[0, 7] - 1.0) < 1e-6, f"time_remaining should be 1.0, got {x[0,7]}"
    _ok("features correct and normalised")


def test_source_dest_clamped():
    print(f"\n{SEP}\nTEST 6: Swap/purify at source/dest clamped to NOOP\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50, rng=np.random.default_rng(0),
                 channel_loss=0.0, ee=True)
    env.reset()
    actions = np.array([SWAP, SWAP, SWAP, SWAP, SWAP], dtype=np.int32)
    _, _, _, info = env.step(actions)
    assert info["actions"][0] == NOOP, "Source swap clamped"
    assert info["actions"][4] == NOOP, "Dest swap clamped"
    _ok("swap at source/dest → noop")


def test_max_steps_termination():
    print(f"\n{SEP}\nTEST 7: Max steps termination\n{SEP}")
    env = QRNEnv(n_repeaters=10, n_ch=2, p_gen=0.1, p_swap=0.1,
                 dt_seconds=0.0, max_steps=5, rng=np.random.default_rng(0))
    env.reset()
    for _ in range(10):
        _, _, done, _ = env.step(np.full(env.N, NOOP, dtype=np.int32))
        if done:
            break
    assert done and env.steps <= 5
    _ok("terminates at max_steps=5")


def test_reward_structure():
    print(f"\n{SEP}\nTEST 8: Reward is positive-sum on success\n{SEP}")
    env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=1.0,
                 spacing=0.0, dt_seconds=0.0, max_steps=50,
                 channel_loss=0.0, F0=1.0, cutoff=999,
                 rng=np.random.default_rng(0), ee=True)
    env.reset()
    total = 0.0
    for _ in range(50):
        acts = strategies.swap_asap(env)
        _, r, done, info = env.step(acts)
        total += r
        if done:
            break
    assert done and info["fidelity"] > 0, "should succeed"
    assert total > 0, f"total return should be positive, got {total:.3f}"
    _ok(f"total return = {total:.3f} > 0 on success")


# ══════════════════════════════════════════════════════════════════
# 2. PHYSICAL CORRECTNESS
# ══════════════════════════════════════════════════════════════════

def test_no_intranode_entanglement():
    print(f"\n{SEP}\nTEST 9: No intra-node entanglement ever occurs\n{SEP}")
    rng = np.random.default_rng(2025)
    for _ in range(50):
        N = rng.integers(3, 10)
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=20,
                     rng=np.random.default_rng(rng.integers(0, 2**31)),
                     channel_loss=0.0)
        env.reset()
        for _ in range(20):
            acts = strategies.random_policy(env, env.rng)
            env.step(acts)
            # Check: no qubit points to its own repeater
            for rep in env.net.repeaters:
                for qi in rep.occupied_indices():
                    pr = int(rep.partner_repeater[qi])
                    assert pr != rep.rid, \
                        f"R{rep.rid}:q{qi} points to itself!"
            if env.done:
                break
    _ok("50 random episodes, no self-links ever")


def test_can_swap_requires_distinct_partners():
    print(f"\n{SEP}\nTEST 10: can_swap needs qubits to distinct partners\n{SEP}")
    env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50, rng=np.random.default_rng(0),
                 channel_loss=0.0, ee=True)
    env.reset()
    # After auto_entangle: R1 has links to R0 and R2 → can swap
    mask = env.get_action_mask()
    assert mask[1, SWAP], "R1 should be able to swap (links to R0 and R2)"
    assert mask[2, SWAP], "R2 should be able to swap (links to R1 and R3)"
    _ok("distinct-partner swap check correct")


# ══════════════════════════════════════════════════════════════════
# 3. STRATEGIES
# ══════════════════════════════════════════════════════════════════

def test_strategies_valid():
    print(f"\n{SEP}\nTEST 11: All strategies produce valid masked actions\n{SEP}")
    rng = np.random.default_rng(42)
    for N in [4, 6, 8]:
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=30, rng=rng,
                     channel_loss=0.0)
        env.reset()
        mask = env.get_action_mask()
        for name, fn in strategies.STRATEGY_MAP.items():
            actions = fn(env)
            assert actions.shape == (N,)
            for i in range(N):
                assert mask[i, actions[i]], \
                    f"{name}: invalid action {actions[i]} at node {i}"
        acts_rand = strategies.random_policy(env, rng)
        for i in range(N):
            assert mask[i, acts_rand[i]]
    _ok("all strategies respect masks")


def test_strategies_never_touch_targets():
    print(f"\n{SEP}\nTEST 12: Strategies never swap/purify at source/dest\n{SEP}")
    rng = np.random.default_rng(0)
    for N in [4, 6, 8]:
        for trial in range(20):
            env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                         dt_seconds=0.0, max_steps=30,
                         rng=np.random.default_rng(rng.integers(0, 2**31)),
                         channel_loss=0.0)
            env.reset()
            for step in range(10):
                for name, fn in strategies.STRATEGY_MAP.items():
                    acts = fn(env)
                    assert acts[env.source] == NOOP, \
                        f"{name}: non-noop at source"
                    assert acts[env.dest] == NOOP, \
                        f"{name}: non-noop at dest"
                env.step(strategies.swap_asap(env))
                if env.done:
                    break
    _ok("N=4,6,8 × 20 trials × 10 steps: targets always NOOP")


def test_swap_asap_3node():
    print(f"\n{SEP}\nTEST 13: SwapASAP N=3 perfect → e2e in 1 step\n{SEP}")
    for trial in range(50):
        env = QRNEnv(n_repeaters=3, n_ch=4, spacing=0.0,
                     p_gen=1.0, p_swap=1.0, cutoff=999,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                     max_steps=10, rng=np.random.default_rng(trial),
                     ee=True)
        env.reset()
        acts = strategies.swap_asap(env)
        assert acts[0] == NOOP and acts[2] == NOOP
        assert acts[1] == SWAP, f"trial {trial}: R1 should swap, got {acts[1]}"
        _, reward, done, info = env.step(acts)
        assert done and info["fidelity"] > 0, f"trial {trial}: not done"
        assert env.steps == 1
    _ok("50/50 trials: e2e in exactly 1 step")


def test_swap_asap_4node():
    print(f"\n{SEP}\nTEST 14: SwapASAP N=4 perfect → e2e in ≤2 steps\n{SEP}")
    for trial in range(50):
        env = QRNEnv(n_repeaters=4, n_ch=4, spacing=0.0,
                     p_gen=1.0, p_swap=1.0, cutoff=999,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                     max_steps=10, rng=np.random.default_rng(trial),
                     ee=True)
        env.reset()
        for _ in range(10):
            _, _, done, info = env.step(strategies.swap_asap(env))
            if done:
                break
        assert done and info["fidelity"] > 0, f"trial {trial}: failed"
        assert env.steps <= 2, f"trial {trial}: took {env.steps} steps"
    _ok("50/50 trials: e2e in ≤2 steps")


def test_swap_asap_5node():
    print(f"\n{SEP}\nTEST 15: SwapASAP N=5 perfect → e2e in ≤3 steps\n{SEP}")
    for trial in range(50):
        env = QRNEnv(n_repeaters=5, n_ch=4, spacing=0.0,
                     p_gen=1.0, p_swap=1.0, cutoff=999,
                     F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                     max_steps=10, rng=np.random.default_rng(trial),
                     ee=True)
        env.reset()
        for _ in range(10):
            _, _, done, info = env.step(strategies.swap_asap(env))
            if done:
                break
        assert done and info["fidelity"] > 0, f"trial {trial}: failed"
        assert env.steps <= 3, f"trial {trial}: took {env.steps} steps"
    _ok("50/50 trials: e2e in ≤3 steps")


def test_swap_asap_solves_stochastic():
    print(f"\n{SEP}\nTEST 16: SwapASAP solves stochastic chains\n{SEP}")
    for N in [4, 6, 8]:
        solved = 0
        for trial in range(30):
            env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=0.8, p_swap=0.7,
                         dt_seconds=0.0, max_steps=50,
                         rng=np.random.default_rng(trial),
                         channel_loss=0.02, ee=True)
            env.reset()
            for _ in range(50):
                _, _, done, info = env.step(strategies.swap_asap(env))
                if done and info["fidelity"] > 0:
                    solved += 1
                    break
        assert solved >= 10, f"N={N}: only {solved}/30 solved"
        _ok(f"N={N}: {solved}/30 solved")


# ══════════════════════════════════════════════════════════════════
# 4. BUFFER
# ══════════════════════════════════════════════════════════════════

def test_buffer():
    print(f"\n{SEP}\nTEST 17: Replay buffer with next_mask\n{SEP}")
    buf = ReplayBuffer(max_size=100)
    assert buf.size() == 0
    for i in range(150):
        buf.add({"x": np.zeros(3)}, np.array([0, 1]), float(i),
                {"x": np.ones(3)}, i == 149,
                np.ones((2, 3), dtype=bool))       # next_mask
    assert buf.size() == 100
    sample = buf.sample(10)
    assert len(sample) == 10
    assert "m_" in sample[0], "buffer must store next_mask"
    buf.clear()
    assert buf.size() == 0
    _ok("add, cap, sample (with mask), clear")


# ══════════════════════════════════════════════════════════════════
# 5. SIZE AGNOSTICISM
# ══════════════════════════════════════════════════════════════════

def test_env_sizes():
    print(f"\n{SEP}\nTEST 18: Env works for N=3..15\n{SEP}")
    rng = np.random.default_rng(0)
    for N in range(3, 16):
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=0.8, p_swap=0.7,
                     dt_seconds=1e-4, max_steps=20, rng=rng)
        obs = env.reset()
        assert obs["x"].shape == (N, 8)
        for _ in range(5):
            obs, r, d, _ = env.step(strategies.swap_asap(env))
            if d:
                break
    _ok("N=3..15 all work")


def test_stress():
    print(f"\n{SEP}\nTEST 19: Stress — 100 random episodes, no crashes\n{SEP}")
    rng = np.random.default_rng(2025)
    for ep in range(100):
        N = rng.integers(3, 12)
        env = QRNEnv(n_repeaters=N, n_ch=rng.integers(2, 8),
                     p_gen=rng.uniform(0.3, 1.0),
                     p_swap=rng.uniform(0.3, 1.0),
                     cutoff=rng.integers(5, 30),
                     dt_seconds=1e-4, max_steps=20,
                     rng=np.random.default_rng(rng.integers(0, 2**31)),
                     heterogeneous=True, channel_loss=0.02)
        env.reset()
        for _ in range(20):
            mask = env.get_action_mask()
            acts = strategies.random_policy(env, env.rng)
            for i in range(N):
                assert mask[i, acts[i]]
            _, _, d, _ = env.step(acts)
            if d:
                break
    _ok("100 random episodes without error")


# ══════════════════════════════════════════════════════════════════
# 6. SAME-PARTNER SWAP GUARD
# ══════════════════════════════════════════════════════════════════

def test_same_partner_swap_guard():
    print(f"\n{SEP}\nTEST 20: Swap with same-partner qubits does not crash\n{SEP}")
    # Force a situation where a node has multiple qubits to the same partner
    from quantum_repeater_sim.repeater import SwapPolicy
    env = QRNEnv(n_repeaters=3, n_ch=4, spacing=0.0,
                 p_gen=1.0, p_swap=1.0, cutoff=999,
                 F0=1.0, channel_loss=0.0, dt_seconds=0.0,
                 max_steps=20, rng=np.random.default_rng(42),
                 ee=True)
    env.reset()
    # Manually add extra link to make R1 have 2 qubits to R0
    env.net.entangle(0, 1)  # second link
    # R1 now may have 2 qubits to R0 and 1 to R2
    # Attempt swap — should not crash even if pair selection picks same-partner
    try:
        env.net.swap(1)
        _ok("same-partner guard works (no crash)")
    except ValueError as e:
        _fail(f"same-partner swap crashed: {e}")


# ══════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    T0 = time.perf_counter()

    test_env_creation()
    test_action_space()
    test_action_mask_shapes()
    test_step_basic()
    test_observation_features()
    test_source_dest_clamped()
    test_max_steps_termination()
    test_reward_structure()
    test_no_intranode_entanglement()
    test_can_swap_requires_distinct_partners()
    test_strategies_valid()
    test_strategies_never_touch_targets()
    test_swap_asap_3node()
    test_swap_asap_4node()
    test_swap_asap_5node()
    test_swap_asap_solves_stochastic()
    test_buffer()
    test_env_sizes()
    test_stress()
    test_same_partner_swap_guard()

    elapsed = time.perf_counter() - T0
    print(f"\n{SEP}")
    if FAIL == 0:
        print(f"ALL {OK} CHECKS PASSED    ({elapsed:.2f}s)")
    else:
        print(f"{OK} passed, {FAIL} FAILED   ({elapsed:.2f}s)")
    print(SEP)
    sys.exit(1 if FAIL else 0)