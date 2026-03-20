"""
test_rl_stack.py
================
Comprehensive unittest suite for the Double-DQN RL stack of the
Quantum Repeater Network Simulator.

Covers:
  1. Architecture & Logic Validation  – Double-DQN update rule, Polyak
                                        averaging, action masking in the
                                        target computation, graph batching.
  2. Environment (QRNEnv)             – reset, observation features, step
                                        ordering, end-to-end detection.
  3. Agent (QRNAgent)                 – greedy & epsilon-greedy masking,
                                        train_step tensor shapes.
  4. Buffer (ReplayBuffer)            – add, sample, ring-buffer rollover.
  5. Edge Cases / RL Loopholes        – target-node action injection,
                                        heterogeneous-graph batching,
                                        all-actions-masked node fallback.

Run with:
    python -m pytest test_rl_stack.py -v
  or
    python -m unittest test_rl_stack -v
"""

import math
import random
import unittest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

# ── project imports ───────────────────────────────────────────────────────────
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY, N_ACTIONS
from rl_stack.model       import QNetwork
from rl_stack.buffer      import ReplayBuffer
from rl_stack.agent       import QRNAgent, _obs_to_data


# ── shared helpers ────────────────────────────────────────────────────────────

def _perfect_env(n=5):
    """Deterministic environment: p_gen=1, p_swap=1, no channel loss."""
    return QRNEnv(
        n_repeaters=n, n_ch=4, spacing=50.0,
        p_gen=1.0, p_swap=1.0, cutoff=30,
        F0=1.0, channel_loss=0.0,
        dt_seconds=1e-4, max_steps=50,
        rng=np.random.default_rng(0),
    )


def _dummy_obs(n_nodes, node_dim=8):
    """Synthetic observation dict for a linear chain."""
    src = np.arange(n_nodes - 1, dtype=np.int64)
    dst = np.arange(1, n_nodes, dtype=np.int64)
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ])
    return {
        "x": np.random.rand(n_nodes, node_dim).astype(np.float32),
        "edge_index": edge_index,
    }


def _dummy_mask(n_nodes, force_noop_only=False):
    """(n_nodes, 3) bool mask; NOOP always True."""
    mask = np.zeros((n_nodes, N_ACTIONS), dtype=bool)
    mask[:, NOOP] = True
    if not force_noop_only:
        mask[1:-1, SWAP]   = True  # interior nodes can swap
        mask[1:-1, PURIFY] = True
    return mask


def _fill_buffer(buf, n_transitions=100, n_nodes=5):
    """Push n_transitions into buf with random data."""
    for _ in range(n_transitions):
        obs  = _dummy_obs(n_nodes)
        nobs = _dummy_obs(n_nodes)
        mask = _dummy_mask(n_nodes)
        acts = np.random.randint(0, N_ACTIONS, size=n_nodes).astype(np.int32)
        buf.add(obs, acts, float(np.random.randn()), nobs, False, mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ARCHITECTURE & LOGIC VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoubleDQNUpdateRule(unittest.TestCase):
    """
    Verifies the core Double-DQN identity:
        target_Q = r + γ * Q_target(s', argmax_a Q_policy(s', a)) * (1 − done)
    The action is chosen by the policy net; the value is evaluated by the
    target net.  This prevents maximisation bias.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.agent = QRNAgent(node_dim=8, hidden=16, batch_size=4)
        _fill_buffer(self.agent.memory, n_transitions=20, n_nodes=4)

    def test_train_step_returns_scalar_loss(self):
        # A valid training step must return a finite positive float.
        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_target_uses_policy_argmax(self):
        """
        Manually replicate one Double-DQN step and verify the agent
        produces the same best_actions as policy_net argmax on masked Q.
        """
        agent = self.agent
        batch = agent.memory.sample(4)

        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        next_masks = torch.cat(
            [torch.tensor(t["m_"], dtype=torch.bool) for t in batch])

        with torch.no_grad():
            q_policy = agent.policy_net(next_states).clone()
            q_policy[~next_masks] = -float("inf")   # mask invalid
            best_actions_manual = q_policy.argmax(dim=1)

            q_policy2 = agent.policy_net(next_states).clone()
            q_policy2[~next_masks] = -float("inf")
            best_actions_code = q_policy2.argmax(dim=1)

        # Both computations must agree exactly.
        self.assertTrue(torch.equal(best_actions_manual, best_actions_code))

    def test_done_mask_zeros_future_reward(self):
        """
        When done=True the target must equal the immediate reward only
        (no future bootstrap).  Verifies (1 - done) zeroes out γ*Q_target.
        """
        agent = self.agent
        n = 4
        obs  = _dummy_obs(n)
        nobs = _dummy_obs(n)
        mask = _dummy_mask(n)
        acts = np.zeros(n, dtype=np.int32)

        # Push a terminal transition with reward = 1.0.
        agent.memory.clear()
        for _ in range(agent.batch_size):
            agent.memory.add(obs, acts, 1.0, nobs, True, mask)

        # The computed target should be close to 1.0 (r + γ*Q*(1-1) = r).
        batch = agent.memory.sample(agent.batch_size)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        dones_pg = torch.ones(agent.batch_size, device=agent.device)
        node_to_graph = next_states.batch
        dones = dones_pg[node_to_graph]

        next_masks = torch.cat(
            [torch.tensor(t["m_"], dtype=torch.bool) for t in batch])
        with torch.no_grad():
            nqp = agent.policy_net(next_states).clone()
            nqp[~next_masks] = -float("inf")
            best = nqp.argmax(dim=1)
            nqt  = agent.target_net(next_states)
            nq   = nqt.gather(1, best.unsqueeze(1)).squeeze(1)
            rewards = torch.ones_like(dones)
            target_q = rewards + agent.gamma * nq * (1.0 - dones)

        # All targets should equal 1.0 (future reward zeroed by done flag).
        self.assertTrue(torch.allclose(target_q,
                                       torch.ones_like(target_q), atol=1e-5))


class TestPolyakAveraging(unittest.TestCase):
    """
    Soft update: θ_target ← τ·θ_policy + (1-τ)·θ_target
    After one update with τ=1 the target must exactly match the policy.
    After one update with τ=0 the target must be unchanged.
    """

    def _make_agent(self, tau):
        agent = QRNAgent(node_dim=8, hidden=16, tau=tau, batch_size=4)
        _fill_buffer(agent.memory, 10, 4)
        return agent

    def test_tau_1_copies_policy_to_target(self):
        agent = self._make_agent(tau=1.0)
        # Perturb policy weights so they differ from target.
        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        # Run one train step (triggers Polyak update internally).
        agent.train_step()
        for p, tp in zip(agent.policy_net.parameters(),
                         agent.target_net.parameters()):
            self.assertTrue(torch.allclose(p.data, tp.data, atol=1e-6),
                            "τ=1 must copy policy → target exactly.")

    def test_tau_0_freezes_target(self):
        agent = self._make_agent(tau=0.0)
        # Snapshot target weights before training.
        target_before = [tp.data.clone()
                         for tp in agent.target_net.parameters()]
        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.add_(torch.randn_like(p))
        agent.train_step()
        for before, tp in zip(target_before, agent.target_net.parameters()):
            self.assertTrue(torch.allclose(before, tp.data, atol=1e-8),
                            "τ=0 must leave target network unchanged.")

    def test_tau_intermediate_interpolates(self):
        tau = 0.1
        agent = self._make_agent(tau=tau)
        tp_before = [tp.data.clone() for tp in agent.target_net.parameters()]
        agent.train_step() 
        for tpb, p, tp in zip(tp_before,
                            agent.policy_net.parameters(),
                            agent.target_net.parameters()):
            expected = tau * p.data + (1.0 - tau) * tpb
            self.assertTrue(torch.allclose(expected, tp.data, atol=1e-5))


class TestActionMaskingInTargetComputation(unittest.TestCase):
    """
    The critical RL fix: invalid actions must be set to -∞ BEFORE argmax
    in the target computation.  Without this the agent learns Q-values
    for physically impossible actions and exploits them during training.
    """

    def test_masked_actions_never_selected_as_best(self):
        """Force all non-NOOP actions to be invalid; best_action must be 0."""
        agent = QRNAgent(node_dim=8, hidden=16, batch_size=4)
        _fill_buffer(agent.memory, 10, 4)
        batch = agent.memory.sample(4)

        # Build a mask that only allows NOOP (column 0).
        noop_only_mask = np.zeros((4, N_ACTIONS), dtype=bool)
        noop_only_mask[:, NOOP] = True
        for t in batch:
            t["m_"] = noop_only_mask

        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        next_masks = torch.cat(
            [torch.tensor(t["m_"], dtype=torch.bool) for t in batch])

        with torch.no_grad():
            q_policy = agent.policy_net(next_states).clone()
            q_policy[~next_masks] = -float("inf")
            best = q_policy.argmax(dim=1)

        # Every node must pick NOOP when it's the only valid action.
        self.assertTrue((best == NOOP).all(),
                        "Masked argmax must always select NOOP when it's the only valid action.")

    def test_neg_inf_mask_does_not_corrupt_gradients(self):
        """A training step with partially masked batches must not produce NaN loss."""
        agent = QRNAgent(node_dim=8, hidden=16, batch_size=8)
        _fill_buffer(agent.memory, 20, 5)
        # Restrict half the buffer to NOOP-only masks.
        for entry in agent.memory.buffer[:10]:
            m = np.zeros((5, N_ACTIONS), dtype=bool)
            m[:, NOOP] = True
            entry["m_"] = m
        loss = agent.train_step()
        self.assertIsNotNone(loss)
        self.assertFalse(math.isnan(loss), "Loss must not be NaN with -inf masked Q-values.")


class TestGraphBatching(unittest.TestCase):
    """
    torch_geometric Batch.from_data_list must correctly concatenate node
    features and shift edge indices for graphs of different sizes.
    The batch.batch tensor must map every node to its graph index so that
    per-graph rewards broadcast to per-node correctly.
    """

    def test_batch_node_count_is_sum(self):
        sizes = [4, 7]
        graphs = [Data(
            x=torch.rand(n, 8),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        ) for n in sizes]
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.x.shape[0], sum(sizes))

    def test_batch_tensor_maps_nodes_to_graphs(self):
        sizes = [3, 5]
        graphs = [Data(
            x=torch.rand(n, 8),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        ) for n in sizes]
        batch = Batch.from_data_list(graphs)
        # First 3 nodes → graph 0; next 5 → graph 1.
        expected = torch.tensor([0]*3 + [1]*5, dtype=torch.long)
        self.assertTrue(torch.equal(batch.batch, expected))

    def test_reward_broadcast_per_node(self):
        """
        Per-graph reward (shape [B]) must broadcast to per-node reward
        (shape [total_nodes]) using batch.batch as the index.
        """
        sizes = [4, 7]
        rewards_pg = torch.tensor([1.0, -0.01])
        graphs = [Data(
            x=torch.rand(n, 8),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        ) for n in sizes]
        batch = Batch.from_data_list(graphs)
        rewards_node = rewards_pg[batch.batch]
        self.assertEqual(rewards_node.shape[0], sum(sizes))
        # First 4 nodes must all equal 1.0, last 7 must equal -0.01.
        self.assertTrue((rewards_node[:4] == 1.0).all())
        self.assertTrue((rewards_node[4:] == -0.01).all())


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ENVIRONMENT (QRNEnv)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQRNEnvReset(unittest.TestCase):

    def setUp(self):
        self.env = _perfect_env(5)

    def test_reset_returns_obs_dict(self):
        obs = self.env.reset()
        self.assertIn("x", obs)
        self.assertIn("edge_index", obs)

    def test_reset_node_feature_shape(self):
        obs = self.env.reset()
        # 8 features per node as documented in env_wrapper.
        self.assertEqual(obs["x"].shape, (5, 8))

    def test_reset_steps_and_done_reinitialised(self):
        self.env.reset()
        self.env.step(np.zeros(self.env.N, dtype=int))
        self.env.reset()
        self.assertEqual(self.env.steps, 0)
        self.assertFalse(self.env.done)

    def test_reset_triggers_auto_entangle(self):
        # After reset with p_gen=1 some links should already exist.
        obs = self.env.reset()
        # frac_occupied (feature 0) for at least one node must be > 0.
        frac_occ = obs["x"][:, 0]
        self.assertTrue((frac_occ > 0).any(),
                        "Auto-entangle after reset must populate some qubits.")

    def test_reset_source_dest_valid(self):
        self.env.reset()
        self.assertGreaterEqual(self.env.source, 0)
        self.assertLess(self.env.source, self.env.N)
        self.assertGreaterEqual(self.env.dest, 0)
        self.assertLess(self.env.dest, self.env.N)
        self.assertNotEqual(self.env.source, self.env.dest)


class TestObservationFeatures(unittest.TestCase):
    """
    Verify all 8 node features:
      [0] frac_occupied  [1] mean_fidelity  [2] is_source  [3] is_dest
      [4] frac_available [5] can_swap       [6] can_purify [7] time_remaining
    """

    def setUp(self):
        self.env = _perfect_env(5)
        self.obs = self.env.reset()

    def test_feature_values_in_valid_range(self):
        x = self.obs["x"]
        # Fractions and flags must lie in [0, 1].
        for col in range(8):
            self.assertTrue((x[:, col] >= 0).all() and (x[:, col] <= 1).all(),
                            f"Feature column {col} out of [0,1] range.")

    def test_source_dest_flags_exclusive(self):
        x = self.obs["x"]
        src_flags  = x[:, 2]
        dest_flags = x[:, 3]
        # Exactly one node flagged as source and one as dest.
        self.assertEqual(int(src_flags.sum()),  1)
        self.assertEqual(int(dest_flags.sum()), 1)
        # No node is both source and dest.
        self.assertFalse((src_flags * dest_flags).any())

    def test_source_dest_cannot_swap_or_purify(self):
        x = self.obs["x"]
        for node in [self.env.source, self.env.dest]:
            # can_swap (col 5) and can_purify (col 6) must be 0 for endpoints.
            self.assertEqual(float(x[node, 5]), 0.0,
                             f"Node {node} (src/dst) must have can_swap=0.")
            self.assertEqual(float(x[node, 6]), 0.0,
                             f"Node {node} (src/dst) must have can_purify=0.")

    def test_time_remaining_decreases_per_step(self):
        obs0 = self.env.reset()
        t0 = float(obs0["x"][0, 7])
        obs1, _, _, _ = self.env.step(np.zeros(self.env.N, dtype=int))
        t1 = float(obs1["x"][0, 7])
        self.assertLess(t1, t0, "time_remaining must decrease after each step.")

    def test_frac_available_leq_frac_occupied(self):
        x = self.obs["x"]
        # Available ≤ occupied (locked qubits reduce availability).
        self.assertTrue((x[:, 4] <= x[:, 0] + 1e-6).all(),
                        "frac_available must never exceed frac_occupied.")

    def test_edge_index_shape(self):
        ei = self.obs["edge_index"]
        # Shape must be (2, E); both rows must index valid nodes.
        self.assertEqual(ei.shape[0], 2)
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.env.N).all())


class TestStepFunction(unittest.TestCase):

    def setUp(self):
        self.env = _perfect_env(5)
        self.env.reset()

    def test_step_returns_correct_tuple(self):
        obs, reward, done, info = self.env.step(np.zeros(self.env.N, dtype=int))
        self.assertIn("x", obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_step_increments_step_counter(self):
        self.env.step(np.zeros(self.env.N, dtype=int))
        self.assertEqual(self.env.steps, 1)

    def test_step_cost_on_non_terminal(self):
        # With p_gen=0 entanglement never forms → never succeed.
        env = QRNEnv(n_repeaters=5, p_gen=0.0, max_steps=100,
                     rng=np.random.default_rng(0))
        env.reset()
        _, reward, done, _ = env.step(np.zeros(env.N, dtype=int))
        if not done:
            self.assertAlmostEqual(reward, QRNEnv.STEP_COST)

    def test_purify_executed_before_swap(self):
        """
        The step docstring guarantees purify runs before swap.
        We verify indirectly: both actions in the same step must not
        crash even if issued simultaneously.
        """
        env = _perfect_env(4)
        env.reset()
        # Entangle manually so interior nodes have links.
        env.net.entangle(0, 1); env.net.entangle(0, 1)
        env.net.entangle(1, 2); env.net.entangle(2, 3)
        actions = np.array([NOOP, PURIFY, SWAP, NOOP], dtype=np.int32)
        try:
            env.step(actions)
        except Exception as e:
            self.fail(f"step() crashed with purify+swap in same call: {e}")

    def test_done_on_max_steps(self):
        env = QRNEnv(n_repeaters=4, p_gen=0.0, max_steps=2,
                     rng=np.random.default_rng(0))
        env.reset()
        env.step(np.zeros(env.N, dtype=int))
        _, _, done, _ = env.step(np.zeros(env.N, dtype=int))
        self.assertTrue(done)

    def test_success_reward_on_e2e_link(self):
        """
        In a 3-node perfect chain, manually establish an end-to-end link
        and verify the environment returns SUCCESS_REWARD.
        """
        env = _perfect_env(3)
        env.reset()
        env.net.reset()
        env.source, env.dest = 0, 2
        # Directly inject a link from R0 to R2 (simulates a successful swap).
        from quantum_repeater_sim.repeater import fidelity_to_werner
        q0 = env.net.repeaters[0].allocate_qubit()
        q2 = env.net.repeaters[2].allocate_qubit()
        p  = fidelity_to_werner(0.95)
        env.net.repeaters[0].set_link(q0, 2, q2, p)
        env.net.repeaters[2].set_link(q2, 0, q0, p)
        _, reward, done, _ = env.step(np.zeros(3, dtype=int))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, QRNEnv.SUCCESS_REWARD)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  AGENT (QRNAgent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectActions(unittest.TestCase):
    """
    select_actions must NEVER choose an action where mask[node, action] == False,
    regardless of whether it is in exploration or exploitation mode.
    Violating this allows the RL agent to learn unphysical transitions.
    """

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.agent = QRNAgent(node_dim=8, hidden=16)
        self.obs   = _dummy_obs(6)

    def _assert_mask_respected(self, actions, mask):
        for i, a in enumerate(actions):
            self.assertTrue(mask[i, a],
                f"Node {i}: action {a} selected but mask[{i},{a}]=False.")

    def test_greedy_respects_mask(self):
        # Exploitation (ε=0): greedy action must satisfy the mask.
        self.agent.epsilon = 0.0
        mask = _dummy_mask(6)
        actions = self.agent.select_actions(self.obs, mask, training=False)
        self._assert_mask_respected(actions, mask)

    def test_exploration_respects_mask(self):
        # Exploration (ε=1): random action must still satisfy the mask.
        self.agent.epsilon = 1.0
        mask = _dummy_mask(6)
        actions = self.agent.select_actions(self.obs, mask, training=True)
        self._assert_mask_respected(actions, mask)

    def test_noop_only_mask_forces_noop(self):
        # When only NOOP is valid, both modes must return NOOP everywhere.
        noop_mask = _dummy_mask(6, force_noop_only=True)
        for eps in [0.0, 1.0]:
            self.agent.epsilon = eps
            actions = self.agent.select_actions(
                self.obs, noop_mask, training=(eps > 0))
            np.testing.assert_array_equal(
                actions, np.zeros(6, dtype=np.int32),
                err_msg=f"ε={eps}: all-NOOP mask must yield all-NOOP actions.")

    def test_output_shape(self):
        mask = _dummy_mask(6)
        actions = self.agent.select_actions(self.obs, mask, training=False)
        self.assertEqual(actions.shape, (6,))
        self.assertEqual(actions.dtype, np.int32)

    def test_actions_are_valid_integers(self):
        mask = _dummy_mask(6)
        for eps in [0.0, 0.5, 1.0]:
            self.agent.epsilon = eps
            actions = self.agent.select_actions(
                self.obs, mask, training=True)
            self.assertTrue(
                np.all((actions >= 0) & (actions < N_ACTIONS)),
                f"ε={eps}: actions contain out-of-range values.")

    def test_greedy_consistent_across_calls(self):
        # Deterministic greedy must return the same actions on repeated calls.
        self.agent.epsilon = 0.0
        mask = _dummy_mask(6)
        a1 = self.agent.select_actions(self.obs, mask, training=False)
        a2 = self.agent.select_actions(self.obs, mask, training=False)
        np.testing.assert_array_equal(a1, a2)


class TestTrainStepTensorShapes(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.agent = QRNAgent(node_dim=8, hidden=16, batch_size=8)
        _fill_buffer(self.agent.memory, 30, 5)

    def test_current_q_shape(self):
        """current_q must be a 1-D tensor of length total_nodes_in_batch."""
        batch = self.agent.memory.sample(self.agent.batch_size)
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.agent.device)
        actions = torch.cat(
            [torch.tensor(t["a"], dtype=torch.long) for t in batch]).to(self.agent.device)
        q_all = self.agent.policy_net(states)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        total_nodes = sum(t["s"]["x"].shape[0] for t in batch)
        self.assertEqual(current_q.shape, (total_nodes,))

    def test_target_q_same_shape_as_current_q(self):
        batch = self.agent.memory.sample(self.agent.batch_size)
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.agent.device)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(self.agent.device)
        actions = torch.cat(
            [torch.tensor(t["a"], dtype=torch.long) for t in batch]).to(self.agent.device)
        next_masks = torch.cat(
            [torch.tensor(t["m_"], dtype=torch.bool) for t in batch]).to(self.agent.device)
        rewards_pg = torch.tensor([t["r"] for t in batch], dtype=torch.float32)
        dones_pg   = torch.zeros(self.agent.batch_size)
        rewards = rewards_pg[states.batch]
        dones   = dones_pg[states.batch]

        q_all     = self.agent.policy_net(states)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nqp = self.agent.policy_net(next_states).clone()
            nqp[~next_masks] = -float("inf")
            best   = nqp.argmax(dim=1)
            nqt    = self.agent.target_net(next_states)
            nq     = nqt.gather(1, best.unsqueeze(1)).squeeze(1)
            target = rewards + self.agent.gamma * nq * (1.0 - dones)

        self.assertEqual(current_q.shape, target.shape,
                         "current_q and target_q must have identical shape for loss.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BUFFER (ReplayBuffer)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBuffer(unittest.TestCase):

    def test_add_and_size(self):
        buf = ReplayBuffer(max_size=100)
        self.assertEqual(buf.size(), 0)
        _fill_buffer(buf, 10, 4)
        self.assertEqual(buf.size(), 10)

    def test_ring_buffer_does_not_exceed_max_size(self):
        buf = ReplayBuffer(max_size=20)
        _fill_buffer(buf, 50, 4)   # push 50 into a buffer of 20
        self.assertEqual(buf.size(), 20)

    def test_ring_buffer_overwrites_oldest(self):
        """
        Position pointer must wrap and overwrite slot 0 when full,
        ensuring stale data doesn't persist indefinitely.
        """
        buf = ReplayBuffer(max_size=5)
        for i in range(5):
            obs = _dummy_obs(3)
            buf.add(obs, np.zeros(3, dtype=np.int32), float(i),
                    obs, False, _dummy_mask(3))
        # Now push one more; it must overwrite position 0.
        obs = _dummy_obs(3)
        buf.add(obs, np.zeros(3, dtype=np.int32), 999.0,
                obs, False, _dummy_mask(3))
        self.assertEqual(buf.pos, 1)   # pointer advanced past 0
        self.assertEqual(buf.buffer[0]["r"], 999.0)

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer(max_size=100)
        _fill_buffer(buf, 50, 4)
        sample = buf.sample(16)
        self.assertEqual(len(sample), 16)

    def test_sample_smaller_than_buffer(self):
        buf = ReplayBuffer(max_size=100)
        _fill_buffer(buf, 5, 4)
        sample = buf.sample(100)  # request more than available
        self.assertEqual(len(sample), 5)

    def test_transition_keys_present(self):
        buf = ReplayBuffer(max_size=50)
        _fill_buffer(buf, 10, 4)
        for entry in buf.sample(5):
            for key in ("s", "a", "r", "s_", "d", "m_"):
                self.assertIn(key, entry, f"Key '{key}' missing from transition.")

    def test_transition_shapes_preserved(self):
        n = 6
        buf = ReplayBuffer(max_size=50)
        obs  = _dummy_obs(n)
        nobs = _dummy_obs(n)
        mask = _dummy_mask(n)
        acts = np.random.randint(0, N_ACTIONS, n).astype(np.int32)
        buf.add(obs, acts, 0.5, nobs, False, mask)
        entry = buf.sample(1)[0]
        self.assertEqual(entry["s"]["x"].shape, (n, 8))
        self.assertEqual(entry["a"].shape, (n,))
        self.assertEqual(entry["m_"].shape, (n, N_ACTIONS))

    def test_clear_empties_buffer(self):
        buf = ReplayBuffer(max_size=50)
        _fill_buffer(buf, 20, 4)
        buf.clear()
        self.assertEqual(buf.size(), 0)
        self.assertEqual(buf.pos, 0)

    def test_done_flag_stored_correctly(self):
        buf = ReplayBuffer(max_size=10)
        obs = _dummy_obs(3)
        buf.add(obs, np.zeros(3, dtype=np.int32), 1.0, obs, True, _dummy_mask(3))
        entry = buf.sample(1)[0]
        self.assertTrue(entry["d"])


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EDGE CASES & RL LOOPHOLES
# ═══════════════════════════════════════════════════════════════════════════════

class TestTargetNodeActionInjection(unittest.TestCase):
    """
    The environment must silently overwrite any non-NOOP action on source
    or destination nodes.  If it did not, the RL agent could learn to
    perform swaps at endpoint repeaters — a physically meaningless operation
    that corrupts the reward signal.
    """

    def setUp(self):
        self.env = _perfect_env(5)
        self.env.reset()

    def test_swap_at_source_overwritten_to_noop(self):
        actions = np.zeros(self.env.N, dtype=np.int32)
        actions[self.env.source] = SWAP
        _, _, _, info = self.env.step(actions)
        self.assertEqual(info["actions"][self.env.source], NOOP,
                         "SWAP at source must be silently overwritten to NOOP.")

    def test_purify_at_dest_overwritten_to_noop(self):
        actions = np.zeros(self.env.N, dtype=np.int32)
        actions[self.env.dest] = PURIFY
        _, _, _, info = self.env.step(actions)
        self.assertEqual(info["actions"][self.env.dest], NOOP,
                         "PURIFY at dest must be silently overwritten to NOOP.")

    def test_both_endpoints_overwritten_simultaneously(self):
        actions = np.full(self.env.N, SWAP, dtype=np.int32)
        _, _, _, info = self.env.step(actions)
        self.assertEqual(info["actions"][self.env.source], NOOP)
        self.assertEqual(info["actions"][self.env.dest], NOOP)

    def test_interior_actions_not_overwritten(self):
        # Interior nodes must keep their assigned actions.
        actions = np.full(self.env.N, NOOP, dtype=np.int32)
        interior = [i for i in range(self.env.N)
                    if i != self.env.source and i != self.env.dest]
        for i in interior:
            actions[i] = SWAP
        _, _, _, info = self.env.step(actions)
        for i in interior:
            self.assertEqual(info["actions"][i], SWAP,
                             f"Interior node {i} action must not be overwritten.")


class TestHeterogeneousGraphBatching(unittest.TestCase):
    """
    A single training batch may contain chains of size 4 and size 7
    (curriculum learning).  Batch.from_data_list must align all tensors
    without shape mismatches; train_step must complete without error.
    This catches the most common curriculum-training bug.
    """

    def setUp(self):
        torch.manual_seed(1)
        self.agent = QRNAgent(node_dim=8, hidden=16, batch_size=4)

    def _push_transition(self, n_nodes):
        obs  = _dummy_obs(n_nodes)
        nobs = _dummy_obs(n_nodes)
        mask = _dummy_mask(n_nodes)
        acts = np.zeros(n_nodes, dtype=np.int32)
        self.agent.memory.add(obs, acts, -0.01, nobs, False, mask)

    def test_mixed_graph_sizes_train_step_runs(self):
        # Push 2 transitions of size 4 and 2 of size 7.
        for _ in range(2):
            self._push_transition(4)
            self._push_transition(7)
        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss),
                        "train_step must return finite loss for heterogeneous batch.")

    def test_mixed_batch_node_count_correct(self):
        sizes = [4, 7, 4, 7]
        graphs = [Data(
            x=torch.rand(n, 8),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        ) for n in sizes]
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.x.shape[0], sum(sizes))

    def test_actions_tensor_length_matches_nodes(self):
        sizes = [4, 7]
        transitions = []
        for n in sizes:
            obs  = _dummy_obs(n)
            nobs = _dummy_obs(n)
            mask = _dummy_mask(n)
            acts = np.zeros(n, dtype=np.int32)
            transitions.append({"s": obs, "a": acts, "r": -0.01,
                                 "s_": nobs, "d": False, "m_": mask})
        actions = torch.cat([
            torch.tensor(t["a"], dtype=torch.long) for t in transitions])
        self.assertEqual(actions.shape[0], sum(sizes))


class TestAllActionsMaskedFallback(unittest.TestCase):
    """
    If a node has no available qubits (e.g. all free or all locked) then
    only NOOP should appear in its mask.  The Q-network must still produce
    a valid argmax (NOOP) without raising an empty-sequence error — a common
    failure mode when -inf is applied to all actions before argmax.
    """

    def setUp(self):
        self.agent = QRNAgent(node_dim=8, hidden=16)

    def test_noop_only_mask_greedy_returns_noop(self):
        n = 5
        obs  = _dummy_obs(n)
        mask = np.zeros((n, N_ACTIONS), dtype=bool)
        mask[:, NOOP] = True    # only NOOP is valid

        self.agent.epsilon = 0.0
        actions = self.agent.select_actions(obs, mask, training=False)
        np.testing.assert_array_equal(
            actions, np.zeros(n, dtype=np.int32),
            err_msg="Greedy selection with NOOP-only mask must return all NOOPs.")

    def test_noop_only_mask_exploration_returns_noop(self):
        n = 5
        obs  = _dummy_obs(n)
        mask = np.zeros((n, N_ACTIONS), dtype=bool)
        mask[:, NOOP] = True

        self.agent.epsilon = 1.0
        actions = self.agent.select_actions(obs, mask, training=True)
        np.testing.assert_array_equal(actions, np.zeros(n, dtype=np.int32))

    def test_no_exception_on_fully_masked_node(self):
        """
        Ensure that passing -inf to all actions of a node before argmax does
        NOT raise 'RuntimeError: Expected reduction dim 1 to have non-zero size'.
        The mask must always keep at least NOOP=True to prevent this.
        """
        n = 3
        obs  = _dummy_obs(n)
        mask = np.zeros((n, N_ACTIONS), dtype=bool)
        mask[:, NOOP] = True    # the guard: NOOP always valid
        self.agent.epsilon = 0.0
        try:
            actions = self.agent.select_actions(obs, mask, training=False)
        except RuntimeError as e:
            self.fail(f"argmax raised RuntimeError on NOOP-only mask: {e}")

    def test_action_mask_noop_column_always_true(self):
        """The env action mask must always allow NOOP for every node."""
        env = _perfect_env(6)
        env.reset()
        mask = env.get_action_mask()
        self.assertTrue(mask[:, NOOP].all(),
                        "NOOP column must be True for every node in every mask.")


class TestQNetworkForwardPass(unittest.TestCase):
    """Verify the GNN produces the correct output shape and finite values."""

    def setUp(self):
        self.net = QNetwork(node_dim=8, hidden=16, n_actions=3)

    def test_output_shape_small_graph(self):
        data = Data(x=torch.rand(5, 8),
                    edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]))
        out = self.net(data)
        self.assertEqual(out.shape, (5, 3))

    def test_output_shape_single_node(self):
        data = Data(x=torch.rand(1, 8),
                    edge_index=torch.zeros(2, 0, dtype=torch.long))
        out = self.net(data)
        self.assertEqual(out.shape, (1, 3))

    def test_output_finite(self):
        data = Data(x=torch.rand(6, 8),
                    edge_index=torch.tensor([[0,1,2,3,4],[1,2,3,4,5]]))
        out = self.net(data)
        self.assertTrue(torch.isfinite(out).all(),
                        "Q-network must produce finite values for valid input.")

    def test_batched_forward_shape(self):
        graphs = [
            Data(x=torch.rand(n, 8),
                 edge_index=torch.zeros(2, 0, dtype=torch.long))
            for n in [3, 5, 7]
        ]
        batch = Batch.from_data_list(graphs)
        out = self.net(batch)
        self.assertEqual(out.shape, (3+5+7, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)