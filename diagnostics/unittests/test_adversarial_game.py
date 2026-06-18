import numpy as np
import pytest
import torch
import torch.nn as nn

from game.adversarial_game.adversary import (
    DESTROY,
    NOOP,
    AdversaryAgent,
    AdversaryFlavor,
    AdversaryQNetwork,
    QUBIT_FEATURES,
    SabotageTarget,
    build_adversary_observation,
    decode_target,
    greedy_action_tensor,
    obs_to_data,
    target_mask,
    target_pairs,
    targets_per_node,
)
from game.adversarial_game.environment import AdversarialQRNEnv
from quantum_repeater_sim.repeater import (
    NO_PARTNER,
    QUBIT_OCCUPIED,
    SwapPolicy,
    werner_to_fidelity,
)
from rl_stack.env_wrapper import QRNEnv, SWAP


def make_env(*, n_repeaters=3, n_ch=4, p_gen=0.0):
    return QRNEnv(
        n_repeaters=n_repeaters,
        n_ch=n_ch,
        spacing=50.0,
        p_gen=p_gen,
        p_swap=1.0,
        cutoff=20,
        F0=1.0,
        channel_loss=0.0,
        dt_seconds=1e-4,
        max_steps=50,
        rng=np.random.default_rng(0),
        topology="chain",
        backend="legacy",
    )


def make_adversarial_env(
    flavor=AdversaryFlavor.PHOTON_EATER,
    *,
    n_repeaters=2,
    n_ch=2,
    p_gen=0.0,
    max_steps=50,
):
    return AdversarialQRNEnv(
        flavor,
        n_repeaters=n_repeaters,
        n_ch=n_ch,
        spacing=50.0,
        p_gen=p_gen,
        p_swap=1.0,
        cutoff=20,
        F0=1.0,
        channel_loss=0.0,
        dt_seconds=0.0,
        max_steps=max_steps,
        rng=np.random.default_rng(0),
        topology="chain",
        backend="legacy",
    )


def noop_actions(env):
    return np.zeros(env.N, dtype=np.int64)


def entangle_instance_state(env):
    return (
        "entangle" in env.backend.__dict__,
        env.backend.__dict__.get("entangle"),
    )


def assert_entangle_instance_state(env, expected):
    present, value = expected
    assert ("entangle" in env.backend.__dict__) is present
    if present:
        assert env.backend.__dict__["entangle"] is value


def swap_instance_state(env):
    return (
        "swap" in env.backend.__dict__,
        env.backend.__dict__.get("swap"),
    )


def assert_swap_instance_state(env, expected):
    present, value = expected
    assert ("swap" in env.backend.__dict__) is present
    if present:
        assert env.backend.__dict__["swap"] is value


def test_adversarial_env_rejects_nonlegacy_backend_and_cosmic_ray():
    with pytest.raises(ValueError, match="backend.*legacy"):
        AdversarialQRNEnv(AdversaryFlavor.PHOTON_EATER, backend="netsquid")
    with pytest.raises(NotImplementedError, match="CosmicRay"):
        AdversarialQRNEnv(AdversaryFlavor.COSMIC_RAY, backend="legacy")


def test_adversarial_env_validates_positional_backend_before_construction():
    legacy_args = (
        2,
        2,
        50.0,
        0.0,
        1.0,
        0.0,
        0.0,
        20,
        1.0,
        0.0,
        0.0,
        50,
        np.random.default_rng(0),
        "chain",
        0.99,
        "legacy",
    )

    env = AdversarialQRNEnv(AdversaryFlavor.PHOTON_EATER, *legacy_args)
    assert env.backend.__class__.__name__ == "LegacyBackend"

    netsquid_args = legacy_args[:-1] + ("netsquid",)
    with pytest.raises(ValueError, match="backend.*legacy"):
        AdversarialQRNEnv(AdversaryFlavor.PHOTON_EATER, *netsquid_args)


def test_adversarial_env_validates_targets_defensively():
    photon_env = make_adversarial_env()
    gate_env = make_adversarial_env(AdversaryFlavor.GATE_DAEMON)

    invalid_cases = (
        (photon_env, (object(),), (TypeError, "SabotageTarget")),
        (
            photon_env,
            (SabotageTarget(node=-1, slot=0, qubits=(0,)),),
            (ValueError, "node"),
        ),
        (
            photon_env,
            (SabotageTarget(node=photon_env.N, slot=0, qubits=(0,)),),
            (ValueError, "node"),
        ),
        (
            photon_env,
            (SabotageTarget(node=0, slot=0, qubits=(2,)),),
            (ValueError, "qubit"),
        ),
        (
            photon_env,
            (SabotageTarget(node=0, slot=0, qubits=(0, 1)),),
            (ValueError, "exactly one"),
        ),
        (
            gate_env,
            (SabotageTarget(node=0, slot=0, qubits=(0,)),),
            (ValueError, "exactly two"),
        ),
    )
    for env, targets, (error, message) in invalid_cases:
        with pytest.raises(error, match=message):
            env._validate_targets(targets)

    duplicate = SabotageTarget(node=0, slot=0, qubits=(0,))
    with pytest.raises(ValueError, match="duplicate"):
        photon_env._validate_targets((duplicate, duplicate))

    numpy_duplicate = SabotageTarget(
        node=np.int64(0),
        slot=np.int64(0),
        qubits=(np.int64(0),),
    )
    with pytest.raises(ValueError, match="duplicate"):
        photon_env._validate_targets((numpy_duplicate, duplicate))

    distinct = SabotageTarget(node=0, slot=1, qubits=(1,))
    with pytest.raises(ValueError, match="K=1"):
        photon_env._validate_targets((duplicate, distinct))


@pytest.mark.parametrize(
    "target",
    [
        SabotageTarget(node=0, slot=0, qubits=(1,)),
        SabotageTarget(node=0, slot=1, qubits=(0,)),
    ],
)
def test_adversarial_env_rejects_noncanonical_slot_qubits(target):
    env = make_adversarial_env()

    with pytest.raises(ValueError, match="canonical"):
        env._validate_targets((target,))


def test_adversarial_env_rejects_out_of_range_target_slot():
    env = make_adversarial_env()
    target = SabotageTarget(node=0, slot=2, qubits=(0,))

    with pytest.raises((IndexError, ValueError), match="slot.*range"):
        env._validate_targets((target,))


@pytest.mark.parametrize("slot", [-1, np.int64(-1), 0.5, "0", True])
def test_adversarial_env_rejects_invalid_target_slot(slot):
    env = make_adversarial_env()
    target = SabotageTarget(node=0, slot=slot, qubits=(0,))

    with pytest.raises((TypeError, ValueError), match="slot"):
        env._validate_targets((target,))


def test_adversarial_step_normalizes_numpy_target_and_generator_qubits():
    env = make_adversarial_env()
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
    target = SabotageTarget(
        node=np.int64(0),
        slot=np.int64(0),
        qubits=(qubit for qubit in [0]),
    )

    _, _, _, info = env.step_adversarial(noop_actions(env), (target,))

    normalized = SabotageTarget(node=0, slot=0, qubits=(0,))
    assert info["sabotage_targets"] == (normalized,)
    assert type(info["sabotage_targets"][0].node) is int
    assert type(info["sabotage_targets"][0].slot) is int
    assert type(info["sabotage_targets"][0].qubits[0]) is int
    assert info["sabotage_triggered"] is True


def test_photon_eater_blocks_exact_allocation_for_one_turn():
    env = make_adversarial_env()
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
    target = SabotageTarget(node=0, slot=0, qubits=(0,))
    prior_entangle_state = entangle_instance_state(env)

    _, _, done, info = env.step_adversarial(noop_actions(env), (target,))

    assert not done
    assert info["sabotage_targets"] == (target,)
    assert info["sabotage_triggered"] is True
    assert info["sabotage_result"]["reason"] == "photon_eater"
    assert not env.backend.node_state(0).occupied.any()
    assert not env.backend.node_state(1).occupied.any()
    assert_entangle_instance_state(env, prior_entangle_state)
    assert env._sabotage_triggered is False

    _, _, done, info = env.step_adversarial(noop_actions(env))

    assert not done
    assert info["sabotage_targets"] == ()
    assert info["sabotage_triggered"] is False
    assert env.backend.node_state(0).occupied.tolist() == [True, False]
    assert env.backend.node_state(1).occupied.tolist() == [True, False]
    assert_entangle_instance_state(env, prior_entangle_state)
    assert env._sabotage_triggered is False


def test_photon_eater_nonmatching_target_delegates_to_allocator():
    env = make_adversarial_env()
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
    target = SabotageTarget(node=0, slot=1, qubits=(1,))

    _, _, _, info = env.step_adversarial(noop_actions(env), (target,))

    assert info["sabotage_triggered"] is False
    assert env.backend.node_state(0).occupied.tolist() == [True, False]
    assert env.backend.node_state(1).occupied.tolist() == [True, False]


def test_photon_eater_does_not_mutate_existing_link_when_blocking_next_slot():
    env = make_adversarial_env(n_repeaters=3, n_ch=3)
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
        repeater.cutoff = 10**9
    assert env.backend.entangle(0, 1)["success"]
    before_left = env.backend.node_state(0)
    before_right = env.backend.node_state(1)
    target = SabotageTarget(node=0, slot=1, qubits=(1,))

    _, _, _, info = env.step_adversarial(noop_actions(env), (target,))

    after_left = env.backend.node_state(0)
    after_right = env.backend.node_state(1)
    assert info["sabotage_triggered"] is True
    assert info["sabotage_result"]["reason"] == "photon_eater"
    assert after_left.partner_node[0] == before_left.partner_node[0]
    assert after_left.partner_qubit[0] == before_left.partner_qubit[0]
    assert after_left.fidelity[0] == pytest.approx(before_left.fidelity[0])
    assert after_right.partner_node[0] == before_right.partner_node[0]
    assert after_right.partner_qubit[0] == before_right.partner_qubit[0]
    assert after_right.fidelity[0] == pytest.approx(before_right.fidelity[0])


def test_adversarial_step_restores_interceptor_after_exception(monkeypatch):
    env = make_adversarial_env()
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
    raw_entangle = env.backend.entangle
    env.backend.entangle = raw_entangle
    prior_entangle_state = entangle_instance_state(env)
    target = SabotageTarget(node=0, slot=0, qubits=(0,))

    def trigger_then_raise():
        result = env.backend.entangle(0, 1)
        assert result["reason"] == "photon_eater"
        assert env._sabotage_triggered is True
        raise RuntimeError("boom")

    monkeypatch.setattr(env, "_auto_entangle", trigger_then_raise)

    with pytest.raises(RuntimeError, match="boom"):
        env.step_adversarial(noop_actions(env), (target,))

    assert_entangle_instance_state(env, prior_entangle_state)
    assert env.active_targets == ()
    assert env._sabotage_triggered is False


def test_photon_eater_triggers_only_once_per_transition(monkeypatch):
    env = make_adversarial_env(n_repeaters=3, n_ch=2, p_gen=0.0)
    env.reset()
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 1.0
    outcomes = []

    def two_attempts():
        outcomes.append(env.backend.entangle(0, 1))
        outcomes.append(env.backend.entangle(1, 2))

    monkeypatch.setattr(env, "_auto_entangle", two_attempts)
    target = SabotageTarget(node=1, slot=0, qubits=(0,))

    _, _, _, info = env.step_adversarial(noop_actions(env), (target,))

    assert outcomes[0]["reason"] == "photon_eater"
    assert outcomes[1]["success"] is True
    assert info["sabotage_triggered"] is True


def test_photon_eater_preserves_generation_rng_advancement(monkeypatch):
    sabotaged = make_adversarial_env(p_gen=1.0)
    control = make_adversarial_env(p_gen=1.0)
    sabotaged.reset()
    control.reset()
    shared_state = np.random.default_rng(91).bit_generator.state
    sabotaged.backend.net.rng.bit_generator.state = shared_state
    control.backend.net.rng.bit_generator.state = shared_state
    monkeypatch.setattr(
        sabotaged,
        "_auto_entangle",
        lambda: sabotaged.backend.entangle(0, 1),
    )
    monkeypatch.setattr(
        control,
        "_auto_entangle",
        lambda: control.backend.entangle(0, 1),
    )

    sabotaged.step_adversarial(
        noop_actions(sabotaged),
        (SabotageTarget(node=0, slot=0, qubits=(0,)),),
    )
    control.step(noop_actions(control))

    assert sabotaged.backend.net.rng.random() == control.backend.net.rng.random()


def test_adversarial_step_restores_interceptor_on_terminal_return():
    env = make_adversarial_env(max_steps=1)
    env.reset()
    prior_entangle_state = entangle_instance_state(env)
    target = SabotageTarget(node=0, slot=0, qubits=(0,))

    _, _, done, info = env.step_adversarial(noop_actions(env), (target,))

    assert done
    assert info["sabotage_targets"] == (target,)
    assert info["sabotage_triggered"] is False
    assert_entangle_instance_state(env, prior_entangle_state)
    assert env.active_targets == ()
    assert env._sabotage_triggered is False


def test_gate_daemon_forces_targeted_swap_to_fail_and_consume_links():
    env = make_adversarial_env(
        AdversaryFlavor.GATE_DAEMON,
        n_repeaters=3,
        n_ch=2,
        p_gen=1.0,
    )
    env.reset()
    middle = env.backend.net.repeaters[1]
    middle.p_gen = 0.0
    env.backend.net.repeaters[0].p_gen = 0.0
    env.backend.net.repeaters[2].p_gen = 0.0
    prior_p_swap = middle.p_swap
    prior_swap_state = swap_instance_state(env)
    target = SabotageTarget(node=1, slot=0, qubits=(0, 1))
    actions = noop_actions(env)
    actions[1] = SWAP

    _, _, done, info = env.step_adversarial(actions, (target,))

    assert not done
    assert info["sabotage_triggered"] is True
    assert info["sabotage_result"]["reason"] == "swap_failed"
    assert info["failed_actions"] == 1
    assert not env.backend.node_state(1).occupied.any()
    assert middle.p_swap == prior_p_swap
    assert_swap_instance_state(env, prior_swap_state)


def test_gate_daemon_nonmatching_pair_uses_normal_swap_probability():
    env = make_adversarial_env(
        AdversaryFlavor.GATE_DAEMON,
        n_repeaters=3,
        n_ch=3,
        p_gen=1.0,
    )
    env.reset()
    target = SabotageTarget(node=1, slot=1, qubits=(0, 2))
    actions = noop_actions(env)
    actions[1] = SWAP

    _, _, done, info = env.step_adversarial(actions, (target,))

    assert done
    assert info["sabotage_triggered"] is False
    assert info["failed_actions"] == 0


def test_gate_daemon_restores_swap_and_probability_after_exception():
    env = make_adversarial_env(
        AdversaryFlavor.GATE_DAEMON,
        n_repeaters=3,
        n_ch=2,
        p_gen=1.0,
    )
    env.reset()
    middle = env.backend.net.repeaters[1]
    prior_p_swap = middle.p_swap

    def exploding_swap(node):
        assert node == 1
        assert middle.p_swap == 0.0
        raise RuntimeError("boom")

    env.backend.swap = exploding_swap
    prior_swap_state = swap_instance_state(env)
    target = SabotageTarget(node=1, slot=0, qubits=(0, 1))
    actions = noop_actions(env)
    actions[1] = SWAP

    with pytest.raises(RuntimeError, match="boom"):
        env.step_adversarial(actions, (target,))

    assert middle.p_swap == prior_p_swap
    assert_swap_instance_state(env, prior_swap_state)
    assert env.active_targets == ()
    assert env._sabotage_triggered is False


def test_gate_daemon_random_preview_preserves_selected_pair():
    env = make_adversarial_env(
        AdversaryFlavor.GATE_DAEMON,
        n_repeaters=3,
        n_ch=4,
        p_gen=1.0,
    )
    env.reset()
    assert env.backend.entangle(0, 1)["success"]
    middle = env.backend.net.repeaters[1]
    middle.swap_policy = SwapPolicy.RANDOM
    for repeater in env.backend.net.repeaters:
        repeater.p_gen = 0.0

    expected = None
    for seed in range(100):
        state = np.random.default_rng(seed).bit_generator.state
        env.backend.net.rng.bit_generator.state = state
        candidate = middle.select_swap_pair(
            env.backend.net._positions,
            rng=env.backend.net.rng,
        )
        env.backend.net.rng.bit_generator.state = state
        if (
            middle.partner_repeater[candidate[0]]
            != middle.partner_repeater[candidate[1]]
        ):
            expected = tuple(sorted(candidate))
            break
    assert expected is not None
    slot = target_pairs(middle.n_ch).index(expected)
    target = SabotageTarget(node=1, slot=slot, qubits=expected)
    actions = noop_actions(env)
    actions[1] = SWAP

    _, _, _, info = env.step_adversarial(actions, (target,))

    assert info["sabotage_triggered"] is True
    occupied = env.backend.node_state(1).occupied
    assert all(not occupied[qubit] for qubit in expected)


def test_gate_daemon_fails_when_probability_draw_is_exactly_zero():
    class ZeroRandom:
        def __init__(self, rng):
            self._rng = rng

        def __getattr__(self, name):
            return getattr(self._rng, name)

        def random(self, *args, **kwargs):
            self._rng.random(*args, **kwargs)
            return 0.0

    env = make_adversarial_env(
        AdversaryFlavor.GATE_DAEMON,
        n_repeaters=3,
        n_ch=2,
        p_gen=1.0,
    )
    env.reset()
    proxy = ZeroRandom(env.backend.net.rng)
    env.backend.net.rng = proxy
    actions = noop_actions(env)
    actions[1] = SWAP

    _, _, _, info = env.step_adversarial(
        actions,
        (SabotageTarget(node=1, slot=0, qubits=(0, 1)),),
    )

    assert info["sabotage_result"]["reason"] == "swap_failed"
    assert info["failed_actions"] == 1
    assert env.backend.net.rng is proxy


def test_adversarial_game_public_exports():
    from game.adversarial_game import (
        AdversarialQRNEnv as PublicEnvironment,
        AdversaryAgent as PublicAgent,
        AdversaryFlavor as PublicFlavor,
        SabotageTarget as PublicTarget,
    )

    assert PublicEnvironment is AdversarialQRNEnv
    assert PublicAgent is AdversaryAgent
    assert PublicFlavor is AdversaryFlavor
    assert PublicTarget is SabotageTarget


def test_target_pairs_are_lexicographic():
    assert target_pairs(4) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_target_indexing_and_validation():
    assert targets_per_node(AdversaryFlavor.PHOTON_EATER, 4) == 4
    assert targets_per_node(AdversaryFlavor.GATE_DAEMON, 4) == 6
    assert decode_target(AdversaryFlavor.PHOTON_EATER, 2, 3, 4).qubits == (3,)
    assert decode_target(AdversaryFlavor.GATE_DAEMON, 2, 4, 4).qubits == (1, 3)

    target = decode_target(
        AdversaryFlavor.GATE_DAEMON,
        np.int64(2),
        np.int32(4),
        np.int64(4),
    )
    assert type(target.node) is int
    assert type(target.slot) is int
    assert all(type(qubit) is int for qubit in target.qubits)

    with pytest.raises(ValueError, match="n_ch must be >= 2"):
        target_pairs(1)
    with pytest.raises(ValueError, match="n_ch must be >= 2"):
        targets_per_node(AdversaryFlavor.GATE_DAEMON, 1)
    with pytest.raises(IndexError):
        decode_target(AdversaryFlavor.PHOTON_EATER, 0, 4, 4)
    with pytest.raises(NotImplementedError, match="CosmicRay"):
        targets_per_node(AdversaryFlavor.COSMIC_RAY, 4)


@pytest.mark.parametrize("n_ch", [0, 1])
def test_photon_eater_rejects_too_few_channels(n_ch):
    with pytest.raises(ValueError, match="n_ch must be >= 2"):
        targets_per_node(AdversaryFlavor.PHOTON_EATER, n_ch)


@pytest.mark.parametrize("n_ch", [True, False, 2.5, "4"])
def test_channel_count_must_be_a_non_boolean_integer(n_ch):
    with pytest.raises(TypeError, match="n_ch must be an integer"):
        targets_per_node(AdversaryFlavor.PHOTON_EATER, n_ch)


@pytest.mark.parametrize("node", [-1, -5])
def test_decode_target_rejects_negative_node(node):
    with pytest.raises(ValueError, match="node must be >= 0"):
        decode_target(AdversaryFlavor.PHOTON_EATER, node, 0, 4)


@pytest.mark.parametrize("node", [True, 1.5, "1"])
def test_decode_target_rejects_non_integral_node(node):
    with pytest.raises(TypeError, match="node must be an integer"):
        decode_target(AdversaryFlavor.PHOTON_EATER, node, 0, 4)


@pytest.mark.parametrize("slot", [True, 1.5, "1"])
def test_decode_target_rejects_non_integral_slot(slot):
    with pytest.raises(TypeError, match="slot must be an integer"):
        decode_target(AdversaryFlavor.PHOTON_EATER, 0, slot, 4)


def test_adversary_observation_has_fixed_qubit_features_and_copied_edges():
    env = make_env(n_ch=4, p_gen=0.0)
    base_obs = env.reset()
    repeater = env.backend.net.repeaters[0]
    qubit = 2
    repeater.status[qubit] = QUBIT_OCCUPIED
    repeater.locked[qubit] = True
    repeater.partner_repeater[qubit] = 2
    repeater.partner_qubit[qubit] = 0
    repeater.werner_param[qubit] = np.float32(0.8)
    repeater.age[qubit] = 5

    obs = build_adversary_observation(env, base_obs, n_ch=4)

    assert QUBIT_FEATURES == 5
    assert obs["x"].shape == (3, 30)
    assert obs["x"].dtype == np.float32
    np.testing.assert_array_equal(obs["x"][:, :10], base_obs["x"])
    start = 10 + qubit * QUBIT_FEATURES
    expected_qubit = np.array(
        [
            1.0,
            1.0,
            1.0,
            werner_to_fidelity(np.float32(0.8)),
            0.25,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        obs["x"][0, start:start + QUBIT_FEATURES],
        expected_qubit,
    )
    assert obs["edge_index"].dtype == np.int64
    assert np.array_equal(obs["edge_index"], base_obs["edge_index"])
    assert not np.shares_memory(obs["edge_index"], base_obs["edge_index"])
    assert np.isfinite(obs["x"]).all()


def test_observation_and_mask_reject_mixed_channel_widths():
    env = make_env(n_ch=4)
    base_obs = env.reset()
    env.backend.net.repeaters[1].n_ch = 3

    with pytest.raises(ValueError, match="n_ch"):
        build_adversary_observation(env, base_obs, n_ch=4)
    with pytest.raises(ValueError, match="n_ch"):
        target_mask(env, AdversaryFlavor.PHOTON_EATER, n_ch=4)


def test_photon_eater_mask_marks_free_unlocked_qubits():
    env = make_env(n_ch=4, p_gen=0.0)
    env.reset()

    mask = target_mask(env, AdversaryFlavor.PHOTON_EATER, n_ch=4)

    assert mask.shape == (3, 4)
    assert mask.dtype == np.bool_
    assert mask.all()

    repeater = env.backend.net.repeaters[1]
    repeater.status[0] = QUBIT_OCCUPIED
    repeater.locked[2] = True
    mask = target_mask(env, AdversaryFlavor.PHOTON_EATER, n_ch=4)
    assert not mask[1, 0]
    assert not mask[1, 2]


def test_gate_daemon_mask_requires_distinct_present_partners():
    env = make_env(n_ch=2, p_gen=1.0)
    env.reset()

    mask = target_mask(env, AdversaryFlavor.GATE_DAEMON, n_ch=2)

    assert mask.shape == (3, 1)
    assert mask[:, 0].tolist() == [False, True, False]

    repeater = env.backend.net.repeaters[1]
    original_partners = repeater.partner_repeater.copy()

    repeater.locked[0] = True
    assert not target_mask(env, AdversaryFlavor.GATE_DAEMON, n_ch=2)[1, 0]
    repeater.locked[0] = False

    repeater.partner_repeater[0] = NO_PARTNER
    assert not target_mask(env, AdversaryFlavor.GATE_DAEMON, n_ch=2)[1, 0]
    repeater.partner_repeater[:] = original_partners

    repeater.partner_repeater[1] = repeater.partner_repeater[0]
    assert not target_mask(env, AdversaryFlavor.GATE_DAEMON, n_ch=2)[1, 0]
    repeater.partner_repeater[:] = original_partners


def test_adversary_model_outputs_action_values_per_target():
    env = make_env(n_ch=4, p_gen=0.0)
    obs = build_adversary_observation(env, env.reset(), n_ch=4)
    data = obs_to_data(obs, device="cpu")
    model = AdversaryQNetwork(
        node_dim=30,
        hidden=16,
        targets_per_node=6,
    )

    output = model(data)

    assert output.shape == (3, 6, 2)
    assert isinstance(data.x, torch.Tensor)
    assert not hasattr(model, "conv2")


def test_greedy_action_tensor_selects_best_positive_advantage_globally():
    q_values = torch.tensor(
        [
            [[2.0, 1.0], [0.0, 3.0]],
            [[1.0, 2.0], [4.0, 3.0]],
        ]
    )
    destroy_mask = torch.tensor([[True, True], [True, False]])

    actions = greedy_action_tensor(q_values, destroy_mask, k=1)

    assert actions.dtype == torch.long
    assert actions.tolist() == [[NOOP, DESTROY], [NOOP, NOOP]]


def test_greedy_action_tensor_keeps_nonpositive_advantages_as_noop():
    q_values = torch.tensor([[[1.0, 1.0], [3.0, 2.0]]])
    destroy_mask = torch.ones((1, 2), dtype=torch.bool)

    actions = greedy_action_tensor(q_values, destroy_mask, k=2)

    assert torch.equal(actions, torch.zeros_like(destroy_mask, dtype=torch.long))


def test_greedy_action_tensor_selects_up_to_k_positive_targets():
    q_values = torch.tensor(
        [[[0.0, 4.0], [1.0, 3.0], [2.0, 1.0]]]
    )
    destroy_mask = torch.ones((1, 3), dtype=torch.bool)

    actions = greedy_action_tensor(q_values, destroy_mask, k=2)

    assert actions.tolist() == [[DESTROY, DESTROY, NOOP]]


def test_greedy_action_tensor_never_selects_invalid_high_advantage_target():
    q_values = torch.tensor([[[0.0, 1.0], [0.0, 100.0]]])
    destroy_mask = torch.tensor([[True, False]])

    actions = greedy_action_tensor(q_values, destroy_mask, k=2)

    assert actions.tolist() == [[DESTROY, NOOP]]


@pytest.mark.parametrize("k", [True, 0, -1, 1.5, "1"])
def test_greedy_action_tensor_rejects_invalid_k(k):
    q_values = torch.zeros((1, 1, 2))
    destroy_mask = torch.ones((1, 1), dtype=torch.bool)

    with pytest.raises((TypeError, ValueError), match="k"):
        greedy_action_tensor(q_values, destroy_mask, k=k)


@pytest.mark.parametrize(
    ("q_values", "destroy_mask"),
    [
        (torch.zeros((2,)), torch.ones((), dtype=torch.bool)),
        (torch.zeros((3, 2)), torch.ones((3,), dtype=torch.bool)),
        (torch.zeros((1, 2, 3, 2)), torch.ones((1, 2, 3), dtype=torch.bool)),
    ],
)
def test_greedy_action_tensor_rejects_q_values_with_invalid_rank(
    q_values,
    destroy_mask,
):
    with pytest.raises(ValueError, match="rank-3"):
        greedy_action_tensor(q_values, destroy_mask, k=1)


def test_greedy_action_tensor_rejects_mask_with_invalid_rank():
    q_values = torch.zeros((2, 3, 2))
    destroy_mask = torch.ones((1, 2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="rank-2"):
        greedy_action_tensor(q_values, destroy_mask, k=1)


@pytest.mark.parametrize(
    ("q_values", "destroy_mask"),
    [
        (torch.zeros((2, 3, 2)), torch.ones((2, 2), dtype=torch.bool)),
    ],
)
def test_greedy_action_tensor_rejects_incompatible_shapes(q_values, destroy_mask):
    with pytest.raises(ValueError, match="shape"):
        greedy_action_tensor(q_values, destroy_mask, k=1)


class FixedAdversaryQ(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", q_values)

    def forward(self, data):
        return self.q_values[: data.x.shape[0]]


def test_select_actions_returns_only_the_positive_greedy_target():
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    base_obs = env.reset()
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        epsilon=0.0,
        k=1,
        device="cpu",
    )
    q_values = torch.zeros((3, 4, 2))
    q_values[..., DESTROY] = -1.0
    q_values[2, 3, DESTROY] = 1.0
    agent.policy_net = FixedAdversaryQ(q_values)

    actions, selected = agent.select_actions(env, base_obs, training=True)

    expected = np.zeros((3, 4), dtype=np.int64)
    expected[2, 3] = DESTROY
    np.testing.assert_array_equal(actions, expected)
    assert actions.dtype == np.int64
    assert selected == [SabotageTarget(node=2, slot=3, qubits=(3,))]


def test_exploration_includes_global_noop_and_respects_mask_and_k():
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    base_obs = env.reset()
    env.backend.net.repeaters[0].status[0] = QUBIT_OCCUPIED
    env.backend.net.repeaters[1].locked[1] = True
    valid = target_mask(env, AdversaryFlavor.PHOTON_EATER, n_ch=4)
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        epsilon=1.0,
        k=2,
        rng=np.random.default_rng(17),
        device="cpu",
    )
    saw_noop = False
    saw_destroy = False

    for _ in range(100):
        actions, selected = agent.select_actions(env, base_obs, training=True)
        assert np.count_nonzero(actions == DESTROY) <= 2
        assert not np.any(actions[~valid] == DESTROY)
        assert len(selected) == np.count_nonzero(actions == DESTROY)
        saw_noop |= not np.any(actions == DESTROY)
        saw_destroy |= np.any(actions == DESTROY)

    assert saw_noop
    assert saw_destroy


def _transition(env, agent, reward=2.0, done=False):
    base_obs = env.reset()
    state = agent.observe(env, base_obs)
    mask = agent.get_target_mask(env)
    actions = np.zeros(mask.shape, dtype=np.int64)
    return state, actions, reward, state, done, mask


def test_train_step_returns_none_before_batch_is_available():
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        batch_size=2,
        device="cpu",
    )

    assert agent.train_step() is None


def test_train_step_updates_policy_parameters():
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=1,
        device="cpu",
    )
    agent.memory.add(*_transition(env, agent, reward=10.0, done=True))
    before = [parameter.detach().clone() for parameter in agent.policy_net.parameters()]

    loss = agent.train_step()

    assert isinstance(loss, float)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy_net.parameters())
    )


def _agent_with_invalid_transition(*, action=None, reward=2.0, done=False, mask=None):
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=1,
        k=1,
        device="cpu",
    )
    state, default_action, _, next_state, _, default_mask = _transition(env, agent)
    agent.memory.add(
        state,
        default_action if action is None else action,
        reward,
        next_state,
        done,
        default_mask if mask is None else mask,
    )
    return agent


def test_train_step_rejects_action_with_wrong_target_width():
    agent = _agent_with_invalid_transition(
        action=np.zeros((3, 1), dtype=np.int64)
    )

    with pytest.raises(ValueError, match="action.*shape"):
        agent.train_step()


def test_train_step_rejects_unknown_action_value():
    action = np.zeros((3, 4), dtype=np.int64)
    action[0, 0] = 2
    agent = _agent_with_invalid_transition(action=action)

    with pytest.raises(ValueError, match="action.*NOOP.*DESTROY"):
        agent.train_step()


def test_train_step_rejects_action_exceeding_destroy_budget():
    action = np.zeros((3, 4), dtype=np.int64)
    action[0, :2] = DESTROY
    agent = _agent_with_invalid_transition(action=action)

    with pytest.raises(ValueError, match="action.*k=1"):
        agent.train_step()


def test_train_step_rejects_nonboolean_successor_mask():
    agent = _agent_with_invalid_transition(
        mask=np.ones((3, 4), dtype=np.int64)
    )

    with pytest.raises(TypeError, match="successor mask.*boolean"):
        agent.train_step()


def test_train_step_rejects_nan_reward():
    agent = _agent_with_invalid_transition(reward=np.nan)

    with pytest.raises(ValueError, match="reward.*finite"):
        agent.train_step()


def test_train_step_rejects_nonboolean_done():
    agent = _agent_with_invalid_transition(done=1)

    with pytest.raises(TypeError, match="done.*bool"):
        agent.train_step()


def test_train_step_applies_successor_top_k_per_graph(monkeypatch):
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=2,
        gamma=1.0,
        k=1,
        device="cpu",
    )
    env_three = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    env_four = make_env(n_repeaters=4, n_ch=4, p_gen=0.0)
    for env in (env_three, env_four):
        state, actions, _, next_state, _, next_mask = _transition(env, agent)
        agent.memory.add(state, actions, 0.0, next_state, False, next_mask)
    monkeypatch.setattr(
        agent.memory,
        "sample",
        lambda batch_size: agent.memory.buffer[:batch_size],
    )

    current_q = torch.zeros((7, 4, 2))
    successor_policy_q = torch.zeros((7, 4, 2))
    successor_policy_q[..., DESTROY] = -1.0
    successor_policy_q[0, 0, DESTROY] = 3.0
    successor_policy_q[0, 1, DESTROY] = 2.0
    successor_policy_q[4, 0, DESTROY] = 4.0
    successor_policy_q[4, 1, DESTROY] = 1.0
    successor_target_q = torch.zeros((7, 4, 2))
    successor_target_q[0, 0, DESTROY] = 10.0
    successor_target_q[0, 1, DESTROY] = 30.0
    successor_target_q[4, 0, DESTROY] = 20.0
    successor_target_q[4, 1, DESTROY] = 40.0
    policy_calls = 0
    parameter_anchor = next(agent.policy_net.parameters())

    def policy_forward(data):
        nonlocal policy_calls
        values = current_q if policy_calls == 0 else successor_policy_q
        policy_calls += 1
        return values.to(data.x.device) + parameter_anchor.sum() * 0.0

    def target_forward(data):
        return successor_target_q.to(data.x.device)

    monkeypatch.setattr(agent.policy_net, "forward", policy_forward)
    monkeypatch.setattr(agent.target_net, "forward", target_forward)

    loss = agent.train_step()

    constrained_targets = torch.zeros((7, 4))
    constrained_targets[0, 0] = 10.0
    constrained_targets[4, 0] = 20.0
    expected_loss = torch.nn.functional.smooth_l1_loss(
        torch.zeros_like(constrained_targets),
        constrained_targets,
    ).item()
    batch_global_targets = torch.zeros_like(constrained_targets)
    batch_global_targets[4, 0] = 20.0
    batch_global_loss = torch.nn.functional.smooth_l1_loss(
        torch.zeros_like(batch_global_targets),
        batch_global_targets,
    ).item()
    all_positive_targets = constrained_targets.clone()
    all_positive_targets[0, 1] = 30.0
    all_positive_targets[4, 1] = 40.0
    all_positive_loss = torch.nn.functional.smooth_l1_loss(
        torch.zeros_like(all_positive_targets),
        all_positive_targets,
    ).item()

    assert policy_calls == 2
    assert loss == pytest.approx(expected_loss)
    assert loss != pytest.approx(batch_global_loss)
    assert loss != pytest.approx(all_positive_loss)


def test_train_step_polyak_updates_every_target_parameter():
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    tau = 0.25
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=1,
        tau=tau,
        device="cpu",
    )
    agent.memory.add(*_transition(env, agent, reward=10.0, done=True))
    old_policy = [
        parameter.detach().clone() for parameter in agent.policy_net.parameters()
    ]
    old_target = [
        parameter.detach().clone() for parameter in agent.target_net.parameters()
    ]

    agent.train_step()

    new_policy = list(agent.policy_net.parameters())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(old_policy, new_policy)
    )
    for policy_parameter, old_target_parameter, new_target_parameter in zip(
        new_policy,
        old_target,
        agent.target_net.parameters(),
    ):
        expected = (
            tau * policy_parameter.detach()
            + (1.0 - tau) * old_target_parameter
        )
        torch.testing.assert_close(
            new_target_parameter,
            expected,
            rtol=1e-6,
            atol=1e-7,
        )


def test_train_step_batches_variable_size_graphs():
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=2,
        device="cpu",
    )
    env_three = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    env_four = make_env(n_repeaters=4, n_ch=4, p_gen=0.0)
    agent.memory.add(*_transition(env_three, agent))
    agent.memory.add(*_transition(env_four, agent))

    loss = agent.train_step()

    assert isinstance(loss, float)


def test_cosmic_ray_agent_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="CosmicRay"):
        AdversaryAgent(AdversaryFlavor.COSMIC_RAY, device="cpu")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"k": True}, TypeError),
        ({"k": 0}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"buffer_size": -1}, ValueError),
        ({"epsilon": 1.1}, ValueError),
        ({"tau": 0.0}, ValueError),
        ({"gamma": -0.1}, ValueError),
    ],
)
def test_adversary_agent_validates_training_configuration(kwargs, error):
    with pytest.raises(error):
        AdversaryAgent(AdversaryFlavor.PHOTON_EATER, device="cpu", **kwargs)


def test_adversary_agent_builds_matching_policy_and_target_networks():
    agent = AdversaryAgent(
        AdversaryFlavor.GATE_DAEMON,
        n_ch=4,
        hidden=12,
        device="cpu",
    )

    assert agent.target_count == 6
    assert agent.node_dim == 10 + 4 * QUBIT_FEATURES
    assert not agent.target_net.training
    for policy_parameter, target_parameter in zip(
        agent.policy_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.equal(policy_parameter, target_parameter)


def test_load_defender_initializes_policy_and_target(tmp_path):
    from game.adversarial_game.train import load_defender
    from rl_stack.agent import QRNAgent

    source = QRNAgent(hidden=64, rng=np.random.default_rng(8))
    checkpoint = tmp_path / "policy.pth"
    torch.save(source.policy_net.state_dict(), checkpoint)

    loaded = load_defender(
        str(checkpoint),
        lr=1e-4,
        rng=np.random.default_rng(9),
    )

    assert not loaded.target_net.training
    for expected, policy, target in zip(
        source.policy_net.parameters(),
        loaded.policy_net.parameters(),
        loaded.target_net.parameters(),
    ):
        assert torch.equal(expected, policy)
        assert torch.equal(policy, target)


def test_one_game_step_stores_exact_opposite_rewards(tmp_path):
    from game.adversarial_game.train import (
        StageIIIConfig,
        build_training_state,
        play_step,
    )

    config = StageIIIConfig(
        defender_checkpoint="checkpoints/inhomo_001/policy.pth",
        flavor="photon_eater",
        episodes=1,
        max_steps=2,
        n_range=(3,),
        n_ch=4,
        batch_size=64,
        output_dir=str(tmp_path),
        seed=10,
        plot=False,
    )
    state = build_training_state(config)
    observation = state.env.reset()

    _, _, _, _, record = play_step(state, observation, training=True)

    defender_entry = state.defender.memory.buffer[-1]
    adversary_entry = state.adversary.memory.buffer[-1]
    assert adversary_entry["r"] == pytest.approx(-defender_entry["r"])
    assert record["adversary_reward"] == pytest.approx(
        -record["defender_reward"]
    )
    assert defender_entry["m_"].shape == (state.env.N, 3)
    assert adversary_entry["m_"].shape == (state.env.N, 4)


def test_one_game_step_updates_both_agents_when_replay_is_ready(tmp_path):
    from game.adversarial_game.train import StageIIIConfig, build_training_state, play_step

    config = StageIIIConfig(
        defender_checkpoint="checkpoints/inhomo_001/policy.pth",
        episodes=1,
        max_steps=2,
        n_range=(3,),
        batch_size=1,
        buffer_size=4,
        output_dir=str(tmp_path),
        seed=13,
        plot=False,
    )
    state = build_training_state(config)
    defender_before = [
        parameter.detach().clone() for parameter in state.defender.policy_net.parameters()
    ]
    adversary_before = [
        parameter.detach().clone() for parameter in state.adversary.policy_net.parameters()
    ]

    play_step(state, state.env.reset(), training=True)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(defender_before, state.defender.policy_net.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(adversary_before, state.adversary.policy_net.parameters())
    )


def test_stage3_config_rejects_invalid_fixed_dimensions_and_flavor():
    from game.adversarial_game.train import StageIIIConfig

    with pytest.raises(NotImplementedError, match="CosmicRay"):
        StageIIIConfig(flavor="cosmic_ray").validate()
    with pytest.raises(ValueError, match="n_ch"):
        StageIIIConfig(n_ch=1).validate()
    with pytest.raises(ValueError, match="[Kk]"):
        StageIIIConfig(k=0).validate()
    with pytest.raises(ValueError, match="K=1"):
        StageIIIConfig(k=2).validate()
    with pytest.raises(ValueError, match="n_range"):
        StageIIIConfig(n_range=()).validate()


def test_incompatible_defender_checkpoint_fails_strict_load(tmp_path):
    from game.adversarial_game.train import load_defender

    checkpoint = tmp_path / "bad.pth"
    torch.save({"not_a_model": torch.zeros(1)}, checkpoint)

    with pytest.raises(RuntimeError):
        load_defender(
            str(checkpoint),
            lr=1e-4,
            rng=np.random.default_rng(9),
        )


def test_stage3_training_smoke_saves_loadable_outputs(tmp_path):
    import json

    from game.adversarial_game.train import StageIIIConfig, train

    config = StageIIIConfig(
        defender_checkpoint="checkpoints/inhomo_001/policy.pth",
        flavor="gate_daemon",
        episodes=2,
        max_steps=2,
        n_range=(3,),
        n_ch=4,
        batch_size=1,
        buffer_size=16,
        output_dir=str(tmp_path),
        seed=11,
        plot=False,
    )

    metrics = train(config)

    assert len(metrics["defender_return"]) == 2
    for name in ("defender_final.pth", "adversary_final.pth"):
        torch.load(tmp_path / name, map_location="cpu", weights_only=True)
    assert not (tmp_path / "defender_best.pth").exists()
    assert not (tmp_path / "adversary_best.pth").exists()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["config"]["flavor"] == "gate_daemon"


def test_stage3_training_is_reproducible_for_a_fixed_seed(tmp_path):
    from game.adversarial_game.train import StageIIIConfig, train

    base = dict(
        defender_checkpoint="checkpoints/inhomo_001/policy.pth",
        flavor="photon_eater",
        episodes=2,
        max_steps=2,
        n_range=(3,),
        n_ch=4,
        batch_size=1,
        buffer_size=16,
        seed=21,
        plot=False,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_metrics = train(StageIIIConfig(output_dir=str(first_dir), **base))
    second_metrics = train(StageIIIConfig(output_dir=str(second_dir), **base))

    assert first_metrics["defender_return"] == second_metrics["defender_return"]
    assert first_metrics["adversary_return"] == second_metrics["adversary_return"]
    assert first_metrics["selected_targets"] == second_metrics["selected_targets"]
    for name in ("defender_final.pth", "adversary_final.pth"):
        first_state = torch.load(first_dir / name, map_location="cpu", weights_only=True)
        second_state = torch.load(second_dir / name, map_location="cpu", weights_only=True)
        assert first_state.keys() == second_state.keys()
        for key in first_state:
            assert torch.equal(first_state[key], second_state[key])


def test_stage3_evaluation_uses_final_checkpoints_only(tmp_path, monkeypatch):
    from game.adversarial_game.adversary import AdversaryAgent
    from game.adversarial_game.evaluate import evaluate
    from rl_stack.agent import QRNAgent

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    defender = QRNAgent(hidden=64, rng=np.random.default_rng(31))
    adversary = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        hidden=64,
        rng=np.random.default_rng(32),
        device="cpu",
    )
    torch.save(defender.policy_net.state_dict(), run_dir / "defender_final.pth")
    torch.save(adversary.policy_net.state_dict(), run_dir / "adversary_final.pth")
    monkeypatch.setattr(
        "game.adversarial_game.evaluate._plot_summary",
        lambda result, output: None,
    )
    monkeypatch.setattr(
        "game.adversarial_game.evaluate._plot_targets",
        lambda result, output: None,
    )

    result = evaluate(
        run_dir,
        pretrained_checkpoint="checkpoints/inhomo_001/policy.pth",
        episodes=1,
        n_range=(3,),
        max_steps=1,
        seed=33,
    )

    assert result["config"]["defender_checkpoint"].endswith(
        "defender_final.pth"
    )
    assert result["config"]["adversary_checkpoint"].endswith(
        "adversary_final.pth"
    )
    assert "checkpoint_kind" not in result["config"]
    assert (run_dir / "evaluation_metrics.json").is_file()
    assert not (run_dir / "evaluation_metrics_final.json").exists()
