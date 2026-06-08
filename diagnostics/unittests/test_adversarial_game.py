import numpy as np
import pytest
import torch
import torch.nn as nn

import game.adversarial_game.adversary as adversary_module
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
from quantum_repeater_sim.repeater import (
    NO_PARTNER,
    QUBIT_OCCUPIED,
    werner_to_fidelity,
)
from rl_stack.env_wrapper import QRNEnv


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
        (torch.zeros((2, 3)), torch.ones((2, 3), dtype=torch.bool)),
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


def test_train_step_reconstructs_successor_actions_with_global_top_k(monkeypatch):
    env = make_env(n_repeaters=3, n_ch=4, p_gen=0.0)
    agent = AdversaryAgent(
        AdversaryFlavor.PHOTON_EATER,
        n_ch=4,
        batch_size=1,
        k=1,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        output_bias = agent.policy_net.head[-1].bias.reshape(agent.target_count, 2)
        output_bias[:, DESTROY] = 1.0
    agent.memory.add(*_transition(env, agent))
    calls = []
    real_helper = adversary_module.greedy_action_tensor

    def recording_helper(q_values, destroy_mask, k):
        actions = real_helper(q_values, destroy_mask, k)
        calls.append((q_values.clone(), destroy_mask.shape, k, actions.clone()))
        return actions

    monkeypatch.setattr(adversary_module, "greedy_action_tensor", recording_helper)

    agent.train_step()

    assert len(calls) == 1
    successor_q, mask_shape, k, actions = calls[0]
    assert successor_q.shape == (3, 4, 2)
    assert mask_shape == (3, 4)
    assert k == 1
    assert torch.count_nonzero(
        successor_q[..., DESTROY] > successor_q[..., NOOP]
    ) > k
    assert torch.count_nonzero(actions == DESTROY) <= 1


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
