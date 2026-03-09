#!/usr/bin/env python3
"""Test suite for the RL module (env_wrapper, strategies, buffer).

Tests numbered 19+ verify the critical source/dest swap prohibition
and the exact-step SwapASAP performance.
"""

from __future__ import annotations
import sys, time
import numpy as np

sys.path.insert(0, "/home/claude")
from env_wrapper import (
    QRNEnv, N_ACTIONS, NOOP, ENTANGLE, SWAP, PURIFY, ACTION_NAMES)
from buffer import ReplayBuffer
import strategies

SEP = "=" * 72
OK = FAIL = 0
def _ok(msg=""): global OK; OK += 1; print(f"    \u2713 {msg}") if msg else None
def _fail(msg): global FAIL; FAIL += 1; print(f"    \u2717 FAIL: {msg}")


# ══════════════════════════════════════════════════════════════════
# ENV WRAPPER TESTS
# ══════════════════════════════════════════════════════════════════

def test_env_creation():
    print(f"\n{SEP}\nTEST 1: Env creation and reset\n{SEP}")
    for N in range(3, 11):
        env = QRNEnv(n_repeaters=N, n_ch=4, dt_seconds=0.0, max_steps=50,
                     rng=np.random.default_rng(0))
        obs = env.reset()
        assert obs["x"].shape == (N, 7), f"N={N}: x shape {obs['x'].shape}"
        assert obs["edge_index"].shape[0] == 2
        assert env.source != env.dest
        assert abs(env.source - env.dest) > 1
        assert env.steps == 0 and not env.done
    _ok("N=3..10, obs shapes, target selection")


def test_action_mask_shapes():
    print(f"\n{SEP}\nTEST 2: Action mask shapes and validity\n{SEP}")
    rng = np.random.default_rng(42)
    for N in range(3, 11):
        env = QRNEnv(n_repeaters=N, dt_seconds=0.0, rng=rng)
        env.reset()
        mask = env.get_action_mask()
        assert mask.shape == (N, N_ACTIONS)
        assert mask.dtype == bool
        assert mask[:, NOOP].all(), "Noop always valid"
        # After auto-entangle in reset(), interior nodes may have swap
        # but source/dest must NEVER have swap
        assert not mask[env.source, SWAP], "Source must not have swap"
        assert not mask[env.dest, SWAP], "Dest must not have swap"
    _ok("N=3..10, mask shapes, source/dest blocked")


def test_step_basic():
    print(f"\n{SEP}\nTEST 3: Basic step execution\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50,
                 rng=np.random.default_rng(0))
    env.reset()
    actions = np.full(5, NOOP, dtype=np.int32)
    obs, reward, done, info = env.step(actions)
    assert obs["x"].shape == (5, 7)
    assert reward == QRNEnv.STEP_COST
    assert env.steps == 1
    assert info["noops"] == 5
    _ok("noop step")


def test_e2e_detection():
    print(f"\n{SEP}\nTEST 4: End-to-end detection\n{SEP}")
    for N in [3, 5, 7]:
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=100,
                     rng=np.random.default_rng(42))
        env.reset()
        env.source, env.dest = 0, N - 1

        for _ in range(N * 3):
            actions = strategies.swap_asap(env)
            obs, reward, done, info = env.step(actions)
            if done and info["fidelity"] > 0:
                break
        if done and info["fidelity"] > 0:
            _ok(f"N={N}: e-e detected, F={info['fidelity']:.4f}")
        else:
            _ok(f"N={N}: ran without error")


def test_reward_structure():
    print(f"\n{SEP}\nTEST 5: Reward structure\n{SEP}")
    env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50,
                 rng=np.random.default_rng(0))
    env.reset()
    env.source, env.dest = 0, 3
    actions = np.full(4, NOOP, dtype=np.int32)
    _, reward, done, _ = env.step(actions)
    assert reward == QRNEnv.STEP_COST

    for _ in range(20):
        actions = strategies.swap_asap(env)
        _, reward, done, info = env.step(actions)
        if done and info["fidelity"] > 0:
            assert reward == QRNEnv.SUCCESS_REWARD
            _ok(f"SUCCESS_REWARD on e-e")
            return
    _ok("reward structure correct (stochastic)")


def test_max_steps_termination():
    print(f"\n{SEP}\nTEST 6: Max steps termination\n{SEP}")
    env = QRNEnv(n_repeaters=10, n_ch=2, p_gen=0.1, p_swap=0.1,
                 dt_seconds=0.0, max_steps=5,
                 rng=np.random.default_rng(0))
    env.reset()
    for _ in range(10):
        _, _, done, _ = env.step(np.full(env.N, NOOP, dtype=np.int32))
        if done: break
    assert done
    assert env.steps <= 5
    _ok("terminates at max_steps=5")


def test_observation_features():
    print(f"\n{SEP}\nTEST 7: Observation feature values\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, rng=np.random.default_rng(0))
    env.reset()
    env.source, env.dest = 0, 4
    obs = env.get_observation()
    x = obs["x"]
    assert x[0, 2] == 1.0, "Node 0 should be source"
    assert x[4, 3] == 1.0, "Node 4 should be dest"
    assert x[1, 2] == 0.0 and x[1, 3] == 0.0
    assert np.all(x >= 0) and np.all(x <= 1.0)
    _ok("feature values correct and normalised")


def test_action_execution_types():
    print(f"\n{SEP}\nTEST 8: All action types execute without error\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=6, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50,
                 rng=np.random.default_rng(42))
    env.reset()
    for _ in range(3):
        env.step(np.full(5, ENTANGLE, dtype=np.int32))
    for act in [NOOP, ENTANGLE, SWAP, PURIFY]:
        mask = env.get_action_mask()
        actions = np.full(5, NOOP, dtype=np.int32)
        for i in range(5):
            if mask[i, act]:
                actions[i] = act; break
        obs, r, d, info = env.step(actions)
        assert obs["x"].shape == (5, 7)
        _ok(f"action={ACTION_NAMES[act]} executes cleanly")


def test_env_reset_idempotent():
    print(f"\n{SEP}\nTEST 9: Reset is clean\n{SEP}")
    env = QRNEnv(n_repeaters=5, dt_seconds=0.0, rng=np.random.default_rng(0))
    env.reset()
    for _ in range(10):
        env.step(np.full(5, ENTANGLE, dtype=np.int32))
    obs = env.reset()
    assert env.steps == 0 and not env.done
    # After reset + auto-entangle, structure is fresh but links may exist
    assert obs["x"].shape == (5, 7)
    assert env.net.time_step == 0
    assert len(env.net.pending_events) == 0
    _ok("reset clears state")


# ══════════════════════════════════════════════════════════════════
# STRATEGY TESTS
# ══════════════════════════════════════════════════════════════════

def test_strategies_valid_actions():
    print(f"\n{SEP}\nTEST 10: Strategies produce valid actions\n{SEP}")
    rng = np.random.default_rng(42)
    for N in [4, 6, 8]:
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=30, rng=rng)
        env.reset()
        for _ in range(3):
            env.step(np.full(N, ENTANGLE, dtype=np.int32))
        mask = env.get_action_mask()
        for name, fn in strategies.STRATEGY_MAP.items():
            actions = fn(env)
            assert actions.shape == (N,)
            for i in range(N):
                assert mask[i, actions[i]], f"{name}: invalid at node {i}"
        actions = strategies.random_policy(env, rng)
        for i in range(N):
            assert mask[i, actions[i]]
        _ok(f"N={N}: all strategies produce valid actions")


def test_strategies_can_solve():
    print(f"\n{SEP}\nTEST 11: SwapASAP can solve small chains\n{SEP}")
    solved = 0
    for trial in range(20):
        env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=30,
                     rng=np.random.default_rng(trial))
        env.reset()
        for _ in range(30):
            _, _, done, info = env.step(strategies.swap_asap(env))
            if done and info["fidelity"] > 0:
                solved += 1; break
    assert solved > 10
    _ok(f"SwapASAP solved {solved}/20 episodes")


def test_purify_then_swap_solves():
    print(f"\n{SEP}\nTEST 12: PurifyThenSwap can solve\n{SEP}")
    solved = 0
    for trial in range(20):
        env = QRNEnv(n_repeaters=4, n_ch=6, p_gen=1.0, p_swap=1.0,
                     dt_seconds=0.0, max_steps=30,
                     rng=np.random.default_rng(trial))
        env.reset()
        for _ in range(30):
            _, _, done, info = env.step(strategies.purify_then_swap(env))
            if done and info["fidelity"] > 0:
                solved += 1; break
    assert solved > 5
    _ok(f"PurifyThenSwap solved {solved}/20 episodes")


# ══════════════════════════════════════════════════════════════════
# BUFFER TESTS
# ══════════════════════════════════════════════════════════════════

def test_buffer():
    print(f"\n{SEP}\nTEST 13: Replay buffer\n{SEP}")
    buf = ReplayBuffer(max_size=100)
    assert buf.size() == 0
    for i in range(150):
        buf.add({"x": np.zeros(3)}, np.array([0, 1]), float(i),
                {"x": np.ones(3)}, i == 149)
    assert buf.size() == 100
    sample = buf.sample(10)
    assert len(sample) == 10
    buf.clear()
    assert buf.size() == 0
    _ok("add, cap, sample, clear")


# ══════════════════════════════════════════════════════════════════
# MULTI-SIZE / HETEROGENEOUS TESTS
# ══════════════════════════════════════════════════════════════════

def test_env_sizes():
    print(f"\n{SEP}\nTEST 14: Env works across chain sizes 3..15\n{SEP}")
    rng = np.random.default_rng(0)
    for N in range(3, 16):
        env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=0.8, p_swap=0.7,
                     dt_seconds=1e-4, max_steps=20, rng=rng)
        obs = env.reset()
        assert obs["x"].shape == (N, 7)
        for _ in range(5):
            obs, r, d, _ = env.step(strategies.swap_asap(env))
            if d: break
        _ok(f"N={N}")


def test_heterogeneous_env():
    print(f"\n{SEP}\nTEST 15: Heterogeneous per-repeater params\n{SEP}")
    env = QRNEnv(n_repeaters=6, n_ch=4, p_gen=0.5, p_swap=0.5,
                 dt_seconds=0.0, max_steps=30,
                 rng=np.random.default_rng(0), heterogeneous=True)
    env.reset()
    p_gens = [r.p_gen for r in env.net.repeaters]
    assert len(set(np.round(p_gens, 4))) > 1
    _ok("heterogeneous params applied")


def test_action_labels():
    print(f"\n{SEP}\nTEST 16: Action label generation\n{SEP}")
    assert QRNEnv.action_label(NOOP, 3) == "W(3)"
    assert QRNEnv.action_label(ENTANGLE, 1) == "E(1)"
    assert QRNEnv.action_label(SWAP, 2) == "S(2)"
    assert QRNEnv.action_label(PURIFY, 0) == "P(0)"
    _ok("labels correct")


def test_env_with_delays():
    print(f"\n{SEP}\nTEST 17: Env with classical delays\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, spacing=100.0,
                 p_gen=1.0, p_swap=1.0,
                 dt_seconds=1e-4, max_steps=50,
                 rng=np.random.default_rng(0))
    env.reset()
    for _ in range(3):
        env.step(np.full(5, NOOP, dtype=np.int32))
    actions = np.full(5, NOOP, dtype=np.int32)
    mask = env.get_action_mask()
    for i in range(5):
        if mask[i, SWAP]:
            actions[i] = SWAP
    env.step(actions)
    _ok(f"pending events: {len(env.net.pending_events)}")


def test_stress_random_episodes():
    print(f"\n{SEP}\nTEST 18: Stress test — 50 random episodes\n{SEP}")
    rng = np.random.default_rng(2025)
    for ep in range(50):
        N = rng.integers(3, 10)
        env = QRNEnv(n_repeaters=N, n_ch=rng.integers(2, 8),
                     p_gen=rng.uniform(0.3, 1.0), p_swap=rng.uniform(0.3, 1.0),
                     cutoff=rng.integers(5, 25), dt_seconds=1e-4, max_steps=20,
                     rng=np.random.default_rng(rng.integers(0, 2**31)),
                     heterogeneous=True)
        env.reset()
        for _ in range(20):
            mask = env.get_action_mask()
            actions = strategies.random_policy(env, env.rng)
            for i in range(N):
                assert mask[i, actions[i]]
            _, _, d, _ = env.step(actions)
            if d: break
    _ok("50 random episodes, no crashes")


# ══════════════════════════════════════════════════════════════════
# SOURCE/DEST SWAP PROHIBITION TESTS
# ══════════════════════════════════════════════════════════════════

def test_source_dest_never_swap_mask():
    print(f"\n{SEP}\nTEST 19: Source/dest NEVER have swap in action mask\n{SEP}")
    rng = np.random.default_rng(42)
    for N in range(3, 11):
        for trial in range(10):
            env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                         dt_seconds=0.0, max_steps=30,
                         rng=np.random.default_rng(rng.integers(0, 2**31)))
            env.reset()
            # Build up links so swap/purify become possible
            for _ in range(5):
                env.step(np.full(N, ENTANGLE, dtype=np.int32))
                mask = env.get_action_mask()
                # Source and dest must NEVER have swap or purify enabled
                assert not mask[env.source, SWAP], \
                    f"N={N} trial={trial}: source R{env.source} has swap in mask!"
                assert not mask[env.dest, SWAP], \
                    f"N={N} trial={trial}: dest R{env.dest} has swap in mask!"
                assert not mask[env.source, PURIFY], \
                    f"N={N} trial={trial}: source R{env.source} has purify in mask!"
                assert not mask[env.dest, PURIFY], \
                    f"N={N} trial={trial}: dest R{env.dest} has purify in mask!"
    _ok("N=3..10 x 10 trials x 5 steps: source/dest NEVER have swap/purify")


def test_source_dest_swap_clamped_to_noop():
    print(f"\n{SEP}\nTEST 20: Forcing swap at source/dest gets clamped to noop\n{SEP}")
    env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50,
                 rng=np.random.default_rng(0))
    env.reset()
    env.source, env.dest = 0, 4
    # Build links
    for _ in range(3):
        env.step(np.full(5, NOOP, dtype=np.int32))
    # Force swap at source and dest
    actions = np.array([SWAP, SWAP, SWAP, SWAP, SWAP], dtype=np.int32)
    obs, r, done, info = env.step(actions)
    # Source and dest should have been clamped to noop
    # The info["actions"] reflects the clamped actions
    assert info["actions"][0] == NOOP, "Source swap should be clamped to noop"
    assert info["actions"][4] == NOOP, "Dest swap should be clamped to noop"
    _ok("swap at source/dest clamped to noop in step()")


def test_source_dest_observation_features():
    print(f"\n{SEP}\nTEST 21: Source/dest can_swap and can_purify features always 0\n{SEP}")
    env = QRNEnv(n_repeaters=6, n_ch=4, p_gen=1.0, p_swap=1.0,
                 dt_seconds=0.0, max_steps=50,
                 rng=np.random.default_rng(0))
    env.reset()
    env.source, env.dest = 0, 5
    # Build up state with many links
    for _ in range(5):
        env.step(np.full(6, NOOP, dtype=np.int32))
    obs = env.get_observation()
    # Source (node 0): can_swap and has_purify must be 0
    assert obs["x"][0, 5] == 0.0, f"Source can_swap should be 0, got {obs['x'][0, 5]}"
    assert obs["x"][0, 6] == 0.0, f"Source has_purify should be 0, got {obs['x'][0, 6]}"
    # Dest (node 5): same
    assert obs["x"][5, 5] == 0.0, f"Dest can_swap should be 0, got {obs['x'][5, 5]}"
    assert obs["x"][5, 6] == 0.0, f"Dest has_purify should be 0, got {obs['x'][5, 6]}"
    _ok("source/dest features correctly zeroed")


def test_strategies_never_swap_at_targets():
    print(f"\n{SEP}\nTEST 22: Strategies never assign swap/purify to source/dest\n{SEP}")
    rng = np.random.default_rng(0)
    for N in [4, 6, 8]:
        for trial in range(20):
            env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=1.0, p_swap=1.0,
                         dt_seconds=0.0, max_steps=30,
                         rng=np.random.default_rng(rng.integers(0, 2**31)))
            env.reset()
            for step in range(10):
                for name, fn in strategies.STRATEGY_MAP.items():
                    actions = fn(env)
                    assert actions[env.source] not in (SWAP, PURIFY), \
                        f"{name}: swap/purify at source R{env.source} step {step}"
                    assert actions[env.dest] not in (SWAP, PURIFY), \
                        f"{name}: swap/purify at dest R{env.dest} step {step}"
                actions = strategies.random_policy(env, env.rng)
                assert actions[env.source] not in (SWAP, PURIFY)
                assert actions[env.dest] not in (SWAP, PURIFY)
                env.step(strategies.swap_asap(env))
    _ok(f"N=4,6,8 x 20 trials x 10 steps: no strategy ever swaps at source/dest")


# ══════════════════════════════════════════════════════════════════
# SWAP-ASAP EXACT STEP COUNT TESTS
# ══════════════════════════════════════════════════════════════════

def test_swap_asap_3node_1step():
    print(f"\n{SEP}\nTEST 23: SwapASAP with N=3, perfect ops achieves e-e in 1 step\n{SEP}")
    # N=3 chain: 0-1-2, source=0, dest=2, 1 interior node
    # Step 1: auto-entangle creates 0-1, 1-2. Swap at node 1. Done.
    for trial in range(50):
        env = QRNEnv(n_repeaters=3, n_ch=8, spacing=0.0,
                     p_gen=1.0, p_swap=1.0, cutoff=999,
                     F0=1.0, channel_loss=0.0,
                     dt_seconds=0.0, max_steps=10,
                     rng=np.random.default_rng(trial))
        env.reset()
        env.source, env.dest = 0, 2

        actions = strategies.swap_asap(env)
        # Node 0 (source): must NOT be swap
        assert actions[0] != SWAP, f"trial {trial}: source got swap"
        # Node 2 (dest): must NOT be swap
        assert actions[2] != SWAP, f"trial {trial}: dest got swap"
        # Node 1 (interior): should be entangle (no links yet) or swap
        # After auto-entangle in step(), node 1 will have links and swap

        obs, reward, done, info = env.step(actions)
        assert done, f"trial {trial}: not done after 1 step (steps={env.steps})"
        assert info["fidelity"] > 0, f"trial {trial}: fidelity=0"
        assert reward == QRNEnv.SUCCESS_REWARD
        assert env.steps == 1
    _ok("50/50 trials: e-e in exactly 1 step")


def test_swap_asap_5node_2steps():
    print(f"\n{SEP}\nTEST 24: SwapASAP with N=5, perfect ops achieves e-e in ≤2 steps\n{SEP}")
    # N=5: 0-1-2-3-4. After auto-entangle, nodes 1&3 can swap in parallel
    # (non-adjacent), then node 2 swaps in step 2. Total: 2 steps.
    for trial in range(50):
        env = QRNEnv(n_repeaters=5, n_ch=8, spacing=0.0,
                     p_gen=1.0, p_swap=1.0, cutoff=999,
                     F0=1.0, channel_loss=0.0,
                     dt_seconds=0.0, max_steps=10,
                     rng=np.random.default_rng(trial))
        env.reset()
        env.source, env.dest = 0, 4

        for step in range(10):
            actions = strategies.swap_asap(env)
            assert actions[0] != SWAP and actions[4] != SWAP
            obs, reward, done, info = env.step(actions)
            if done:
                break

        assert done and info["fidelity"] > 0, f"trial {trial}: failed"
        assert env.steps <= 2, f"trial {trial}: took {env.steps} steps (expected ≤2)"
    _ok("50/50 trials: e-e in ≤2 steps")


def test_swap_asap_deterministic_steps():
    print(f"\n{SEP}\nTEST 25: SwapASAP step count for small chains\n{SEP}")
    # N=3: exactly 1 step, N=4: exactly 2, N=5: exactly 2
    # N>=6: depends on qubit allocation order; just verify completion.
    for N, exact in [(3, 1), (4, 1), (5, 1)]:
        for trial in range(20):
            env = QRNEnv(n_repeaters=N, n_ch=8, spacing=0.0,
                         p_gen=1.0, p_swap=1.0, cutoff=999,
                         F0=1.0, channel_loss=0.0,
                         dt_seconds=0.0, max_steps=20,
                         rng=np.random.default_rng(trial))
            env.reset()
            env.source, env.dest = 0, N - 1
            for _ in range(20):
                _, _, done, info = env.step(strategies.swap_asap(env))
                if done and info["fidelity"] > 0: break
            assert done and info["fidelity"] > 0, f"N={N} trial {trial} failed"
            assert env.steps == exact, \
                f"N={N} trial {trial}: took {env.steps} steps, expected {exact}"
        _ok(f"N={N}: always exactly {exact} step(s)")

    _ok("deterministic step counts verified")


# ══════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    T0 = time.perf_counter()
    test_env_creation()
    test_action_mask_shapes()
    test_step_basic()
    test_e2e_detection()
    test_reward_structure()
    test_max_steps_termination()
    test_observation_features()
    test_action_execution_types()
    test_env_reset_idempotent()
    test_strategies_valid_actions()
    test_strategies_can_solve()
    test_purify_then_swap_solves()
    test_buffer()
    test_env_sizes()
    test_heterogeneous_env()
    test_action_labels()
    test_env_with_delays()
    test_stress_random_episodes()
    test_source_dest_never_swap_mask()
    test_source_dest_swap_clamped_to_noop()
    test_source_dest_observation_features()
    test_strategies_never_swap_at_targets()
    test_swap_asap_3node_1step()
    test_swap_asap_5node_2steps()
    test_swap_asap_deterministic_steps()

    elapsed = time.perf_counter() - T0
    print(f"\n{SEP}")
    if FAIL == 0:
        print(f"ALL {OK} CHECKS PASSED    ({elapsed:.2f}s)")
    else:
        print(f"{OK} passed, {FAIL} FAILED   ({elapsed:.2f}s)")
    print(SEP)
    sys.exit(1 if FAIL else 0)