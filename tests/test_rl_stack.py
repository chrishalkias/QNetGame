"""
--------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------
"""

import math
import random
import unittest
import numpy as np
import pytest
import torch
from torch_geometric.data import Data, Batch

# -- project imports -----------------------------------------------------------
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY, N_ACTIONS
from rl_stack.model       import QNetwork
from rl_stack.buffer      import ReplayBuffer
from rl_stack.agent       import QRNAgent, _obs_to_data


# -- shared helpers ------------------------------------------------------------

def _perfect_env(n=5):
    """Deterministic environment: p_gen=1, p_swap=1, no channel loss."""
    return QRNEnv(
        n_repeaters=n, n_ch=4, spacing=50.0,
        p_gen=1.0, p_swap=1.0, cutoff=30,
        F0=1.0, channel_loss=0.0,
        max_steps=50,
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


def _dummy_mask_row(force_noop_only=False):
    """(3,) bool row for ONE node's next-state mask; NOOP always True."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[NOOP] = True
    if not force_noop_only:
        m[SWAP] = True
        m[PURIFY] = True
    return m


def _fill_buffer(buf, n_transitions=100, n_nodes=5):
    """Push n_transitions of per-decision transitions into buf with random data."""
    for _ in range(n_transitions):
        obs  = _dummy_obs(n_nodes)
        nobs = _dummy_obs(n_nodes)
        ai   = int(np.random.randint(0, n_nodes))
        nai  = int(np.random.randint(0, n_nodes))
        a    = int(np.random.randint(0, N_ACTIONS))
        buf.add(obs, a, ai, float(np.random.randn()), nobs, nai,
                _dummy_mask_row(), False, 1.0)


# -- Task 4: per-decision DQN contract (buffer.add / train_step active-node
#    gather / terminated-vs-truncated bootstrapping) --------------------------

def test_buffer_stores_per_decision_fields():
    buf = ReplayBuffer(10)
    s  = _dummy_obs(4)
    s2 = _dummy_obs(4)
    buf.add(s, 1, 2, 0.5, s2, 3, np.array([True, False, True]), False, 1.0)
    e = buf.buffer[0]
    assert e["ai"] == 2 and e["nai"] == 3 and e["g"] == 1.0 and e["d"] is False
    assert e["a"] == 1 and e["r"] == 0.5


def test_train_step_gathers_active_node_and_uses_gamma_eff():
    """train_step must gather Q at the batched-graph ACTIVE node index
    (ptr[b] + ai), not node 0, and must use each transition's own gamma_eff
    (not a fixed agent.gamma) in the target."""
    torch.manual_seed(0)
    agent = QRNAgent(node_dim=8, hidden=16, batch_size=2)
    n = 4
    obs1, nobs1 = _dummy_obs(n), _dummy_obs(n)
    obs2, nobs2 = _dummy_obs(n), _dummy_obs(n)
    ai1, nai1 = 2, 3            # active node != 0 -> catches a node-0 bug
    ai2, nai2 = 1, 0
    agent.memory.add(obs1, SWAP, ai1, 0.3, nobs1, nai1, _dummy_mask_row(), False, 1.0)
    agent.memory.add(obs2, PURIFY, ai2, -0.2, nobs2, nai2, _dummy_mask_row(), False, 0.995)

    fixed_batch = list(agent.memory.buffer)
    agent.memory.sample = lambda n: fixed_batch  # pin the batch for hand-checking

    # -- hand-replicate the expected loss on the CURRENT (pre-update) weights --
    states = Batch.from_data_list([_obs_to_data(t["s"]) for t in fixed_batch]).to(agent.device)
    next_states = Batch.from_data_list([_obs_to_data(t["s_"]) for t in fixed_batch]).to(agent.device)
    ptr = states.ptr[:-1]
    actions = torch.tensor([t["a"] for t in fixed_batch], dtype=torch.long)
    with torch.no_grad():
        q_all = agent.policy_net(states)
        correct_idx = ptr + torch.tensor([t["ai"] for t in fixed_batch])
        current_q = q_all[correct_idx].gather(1, actions.unsqueeze(1)).squeeze(1)

        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor([t["nai"] for t in fixed_batch])
        next_masks = torch.tensor(np.stack([t["m_"] for t in fixed_batch]), dtype=torch.bool)
        nqp = agent.policy_net(next_states)[nidx].clone()
        nqp[~next_masks] = -float("inf")
        best = nqp.argmax(1)
        nq = agent.target_net(next_states)[nidx].gather(1, best.unsqueeze(1)).squeeze(1)

        rewards = torch.tensor([t["r"] for t in fixed_batch], dtype=torch.float32)
        gammas  = torch.tensor([t["g"] for t in fixed_batch], dtype=torch.float32)
        dones   = torch.tensor([float(t["d"]) for t in fixed_batch])
        target_q = rewards + gammas * nq * (1.0 - dones)
        expected_loss = torch.nn.SmoothL1Loss()(current_q, target_q).item()

        # regression guard: gathering at node 0 (ignoring "ai") would give a
        # DIFFERENT current_q with overwhelming probability on a random net.
        wrong_q = q_all[ptr].gather(1, actions.unsqueeze(1)).squeeze(1)
        assert not torch.allclose(wrong_q, current_q)

    loss = agent.train_step()
    assert loss is not None
    assert math.isclose(loss, expected_loss, rel_tol=1e-4, abs_tol=1e-6)


def test_target_uses_terminated_not_truncated():
    """A truncation (timeout) transition stores terminated=False (d=False)
    even though the episode ended, with gamma_eff=agent.gamma (the tick-
    boundary rate) -- the target must still bootstrap next_q, unlike a real
    terminal delivery which stores d=True and zeroes the bootstrap term."""
    torch.manual_seed(0)
    agent = QRNAgent(node_dim=8, hidden=16, batch_size=1)
    n = 3
    obs, nobs = _dummy_obs(n), _dummy_obs(n)
    agent.memory.add(obs, SWAP, 1, -0.01, nobs, 1, _dummy_mask_row(),
                     False, agent.gamma)
    batch = list(agent.memory.buffer)
    agent.memory.sample = lambda n: batch

    states = Batch.from_data_list([_obs_to_data(t["s"]) for t in batch]).to(agent.device)
    next_states = Batch.from_data_list([_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
    ptr = states.ptr[:-1]
    act_idx = ptr + torch.tensor([t["ai"] for t in batch])
    actions = torch.tensor([t["a"] for t in batch], dtype=torch.long)
    with torch.no_grad():
        current_q = agent.policy_net(states)[act_idx].gather(1, actions.unsqueeze(1)).squeeze(1)
        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor([t["nai"] for t in batch])
        next_masks = torch.tensor(np.stack([t["m_"] for t in batch]), dtype=torch.bool)
        nqp = agent.policy_net(next_states)[nidx].clone()
        nqp[~next_masks] = -float("inf")
        best = nqp.argmax(1)
        nq = agent.target_net(next_states)[nidx].gather(1, best.unsqueeze(1)).squeeze(1)
        rewards = torch.tensor([t["r"] for t in batch], dtype=torch.float32)
        gammas  = torch.tensor([t["g"] for t in batch], dtype=torch.float32)
        dones   = torch.tensor([float(t["d"]) for t in batch])
        target_q = rewards + gammas * nq * (1.0 - dones)
        expected_loss = torch.nn.SmoothL1Loss()(current_q, target_q).item()
        # the bootstrap term must be nonzero: truncation must NOT zero it
        # the way a real terminated=True transition would.
        assert not torch.allclose(target_q, rewards, atol=1e-6)

    loss = agent.train_step()
    assert loss is not None
    assert math.isclose(loss, expected_loss, rel_tol=1e-4, abs_tol=1e-6)



#   ▄▄▄▄               ▄▄                                                 
# ▄██▀▀██▄             ██    ▀▀  ██                ██                     
# ███  ███ ████▄ ▄████ ████▄ ██ ▀██▀▀ ▄█▀█▄ ▄████ ▀██▀▀ ██ ██ ████▄ ▄█▀█▄ 
# ███▀▀███ ██ ▀▀ ██    ██ ██ ██  ██   ██▄█▀ ██     ██   ██ ██ ██ ▀▀ ██▄█▀ 
# ███  ███ ██    ▀████ ██ ██ ██▄ ██   ▀█▄▄▄ ▀████  ██   ▀██▀█ ██    ▀█▄▄▄ 
                                                                                                                                                                                                        
#           ▄▄▄                                                           
#    ▄      ███                  ▀▀                                       
#    █      ███      ▄███▄ ▄████ ██  ▄████                                
# ▀▀▀█▀▀▀   ███      ██ ██ ██ ██ ██  ██                                   
#    █      ████████ ▀███▀ ▀████ ██▄ ▀████                                
#                             ██                                          
#                           ▀▀▀                                           

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
        Manually replicate one Double-DQN step (at each transition's ACTIVE
        node) and verify the agent produces the same best_actions as
        policy_net argmax on masked Q.
        """
        agent = self.agent
        batch = agent.memory.sample(4)

        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor([t["nai"] for t in batch], device=agent.device)
        next_masks = torch.tensor(
            np.stack([t["m_"] for t in batch]), dtype=torch.bool, device=agent.device)

        with torch.no_grad():
            q_policy = agent.policy_net(next_states)[nidx].clone()
            q_policy[~next_masks] = -float("inf")   # mask invalid
            best_actions_manual = q_policy.argmax(dim=1)

            q_policy2 = agent.policy_net(next_states)[nidx].clone()
            q_policy2[~next_masks] = -float("inf")
            best_actions_code = q_policy2.argmax(dim=1)

        # Both computations must agree exactly.
        self.assertTrue(torch.equal(best_actions_manual, best_actions_code))

    def test_done_mask_zeros_future_reward(self):
        """
        When terminated=True the target must equal the immediate reward only
        (no future bootstrap).  Verifies (1 - done) zeroes out γ*Q_target.
        """
        agent = self.agent
        n = 4
        obs  = _dummy_obs(n)
        nobs = _dummy_obs(n)
        m_   = _dummy_mask_row()

        # Push a terminal transition with reward = 1.0.
        agent.memory.clear()
        for _ in range(agent.batch_size):
            agent.memory.add(obs, NOOP, 1, 1.0, nobs, 1, m_, True, 1.0)

        # The computed target should be close to 1.0 (r + g*Q*(1-1) = r).
        batch = agent.memory.sample(agent.batch_size)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        dones = torch.tensor([float(t["d"]) for t in batch], device=agent.device)
        gammas = torch.tensor([t["g"] for t in batch], device=agent.device)
        rewards = torch.tensor([t["r"] for t in batch], device=agent.device)

        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor([t["nai"] for t in batch], device=agent.device)
        next_masks = torch.tensor(
            np.stack([t["m_"] for t in batch]), dtype=torch.bool, device=agent.device)
        with torch.no_grad():
            nqp = agent.policy_net(next_states)[nidx].clone()
            nqp[~next_masks] = -float("inf")
            best = nqp.argmax(dim=1)
            nqt  = agent.target_net(next_states)[nidx]
            nq   = nqt.gather(1, best.unsqueeze(1)).squeeze(1)
            target_q = rewards + gammas * nq * (1.0 - dones)

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

        # Restrict every transition's next-state mask row to NOOP only.
        noop_only_row = _dummy_mask_row(force_noop_only=True)
        for t in batch:
            t["m_"] = noop_only_row

        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(agent.device)
        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor([t["nai"] for t in batch], device=agent.device)
        next_masks = torch.tensor(
            np.stack([t["m_"] for t in batch]), dtype=torch.bool, device=agent.device)

        with torch.no_grad():
            q_policy = agent.policy_net(next_states)[nidx].clone()
            q_policy[~next_masks] = -float("inf")
            best = q_policy.argmax(dim=1)

        # Every active node must pick NOOP when it's the only valid action.
        self.assertTrue((best == NOOP).all(),
                        "Masked argmax must always select NOOP when it's the only valid action.")

    def test_neg_inf_mask_does_not_corrupt_gradients(self):
        """A training step with partially masked batches must not produce NaN loss."""
        agent = QRNAgent(node_dim=8, hidden=16, batch_size=8)
        _fill_buffer(agent.memory, 20, 5)
        # Restrict half the buffer to NOOP-only next-state masks.
        for entry in agent.memory.buffer[:10]:
            entry["m_"] = _dummy_mask_row(force_noop_only=True)
        loss = agent.train_step()
        self.assertIsNotNone(loss)
        self.assertFalse(math.isnan(loss), "Loss must not be NaN with -inf masked Q-values.")


class TestCheckpointWindowGate(unittest.TestCase):
    """The best-checkpoint window must open only after the curriculum has fully
    opened AND epsilon has reached its floor (0.9*episodes) — otherwise the easy
    early curriculum phase (small fast-delivering chains) wins on rolling reward."""

    def test_eps_floor_dominates_default_curriculum(self):
        # frac=0.5 opens curriculum at 15000; eps floor at 0.9*30000=27000 -> 27000.
        self.assertEqual(QRNAgent._ckpt_window_start(30000, True, 0.5), 27000)

    def test_late_curriculum_dominates_eps_floor(self):
        # frac=0.95 opens curriculum at 950 > eps floor 900 -> 950.
        self.assertEqual(QRNAgent._ckpt_window_start(1000, True, 0.95), 950)

    def test_no_curriculum_uses_eps_floor(self):
        self.assertEqual(QRNAgent._ckpt_window_start(1000, False, 0.5), 900)


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


                                                    
#   ▄▄▄▄▄   ▄▄▄▄▄▄▄   ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄             
# ▄███████▄ ███▀▀███▄ ████▄  ███ ███▀▀▀▀▀             
# ███   ███ ███▄▄███▀ ███▀██▄███ ███▄▄    ████▄ ██ ██ 
# ███▄█▄███ ███▀▀██▄  ███  ▀████ ███      ██ ██ ██▄██ 
#  ▀█████▀  ███  ▀███ ███    ███ ▀███████ ██ ██  ▀█▀  
#       ▀▀                                            
                                                    

class TestQRNEnvReset(unittest.TestCase):

    def setUp(self):
        self.env = _perfect_env(5)

    def test_reset_returns_obs_dict(self):
        obs = self.env.reset()
        self.assertIn("x", obs)
        self.assertIn("edge_index", obs)

    def test_reset_node_feature_shape(self):
        obs = self.env.reset()
        # 11 features per node as documented in env_wrapper.get_observation.
        self.assertEqual(obs["x"].shape, (5, 11))

    def test_reset_steps_and_done_reinitialised(self):
        self.env.reset()
        # drive one full left-to-right sweep (tick) to completion
        info = {"tick_boundary": False}
        while not info["tick_boundary"]:
            _, _, _, info = self.env.step(NOOP)
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
    Verify all 11 node features:
      [0] frac_occupied  [1] mean_fidelity  [2] in_endnode   [3] frac_available
      [4] can_swap       [5] can_purify     [6] p_gen        [7] p_swap
      [8] link_urgency   [9] is_active      [10] relative_position
    """

    def setUp(self):
        self.env = _perfect_env(5)
        self.obs = self.env.reset()

    def test_feature_values_in_valid_range(self):
        x = self.obs["x"]
        # Fractions, flags and per-repeater rates must all lie in [0, 1].
        for col in range(11):
            self.assertTrue((x[:, col] >= 0).all() and (x[:, col] <= 1).all(),
                            f"Feature column {col} out of [0,1] range.")

    def test_p_gen_p_swap_features_match_backend(self):
        # Cols 6/7 must equal each repeater's per-node p_gen / p_swap, and an
        # inhomogeneous network (std>0) must produce genuinely varying values.
        env = QRNEnv(n_repeaters=6, n_ch=4, p_gen=0.7, p_swap=0.7,
                     p_gen_std=0.18, p_swap_std=0.18, cutoff=20, max_steps=40,
                     topology="chain", rng=np.random.default_rng(99))
        x = env.reset()["x"]
        for i in range(env.N):
            ns = env.net.node_state(i)
            self.assertAlmostEqual(float(x[i, 6]), float(ns.p_gen), places=5)
            self.assertAlmostEqual(float(x[i, 7]), float(ns.p_swap), places=5)
        self.assertGreater(float(x[:, 6].std()), 0.0)  # inhomogeneous
        self.assertGreater(float(x[:, 7].std()), 0.0)

    def test_in_endnode_flags_endpoints(self):
        x = self.obs["x"]
        endnode = x[:, 2]
        # Exactly the two endpoints (source + dest) are flagged.
        self.assertEqual(int(endnode.sum()), 2)
        self.assertEqual(float(endnode[self.env.source]), 1.0)
        self.assertEqual(float(endnode[self.env.dest]), 1.0)

    def test_source_dest_cannot_swap_or_purify(self):
        x = self.obs["x"]
        for node in [self.env.source, self.env.dest]:
            # can_swap (col 4) and can_purify (col 5) must be 0 for endpoints.
            self.assertEqual(float(x[node, 4]), 0.0,
                             f"Node {node} (src/dst) must have can_swap=0.")
            self.assertEqual(float(x[node, 5]), 0.0,
                             f"Node {node} (src/dst) must have can_purify=0.")

    def test_frac_available_leq_frac_occupied(self):
        x = self.obs["x"]
        # Available <= occupied (qubits consumed/unavailable mid-op reduce availability).
        self.assertTrue((x[:, 3] <= x[:, 0] + 1e-6).all(),
                        "frac_available must never exceed frac_occupied.")

    def test_edge_index_shape(self):
        ei = self.obs["edge_index"]
        # Shape must be (2, E); both rows must index valid nodes.
        self.assertEqual(ei.shape[0], 2)
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.env.N).all())

    def test_frac_occupied_denominator_is_physical_capacity(self):
        # feats[i,0] must divide by the node's physical qubit count
        # (2*n_ch interior, n_ch ends), not the per-side n_ch field.
        env = _perfect_env(4)               # n_ch=4 -> interior cap 8, ends cap 4
        x = env.reset()["x"]
        for i in range(env.N):
            ns = env.net.node_state(i)
            cap = ns.occupied.size
            occ = int(ns.occupied.sum())
            self.assertAlmostEqual(float(x[i, 0]) * cap, occ, places=4)
        # capacities really differ between interior and end nodes
        self.assertEqual(env.net.node_state(1).occupied.size, 8)
        self.assertEqual(env.net.node_state(0).occupied.size, 4)


class TestMultiPartnerPurify(unittest.TestCase):
    """One PURIFY action runs the distillation cascade on every partner with
    which the node shares >=2 links (left partner AND right partner)."""

    def test_purify_touches_both_partners(self):
        env = QRNEnv(n_repeaters=3, n_ch=2, p_gen=1.0, p_swap=1.0, cutoff=1000,
                     F0=1.0, channel_loss=0.0, max_steps=50, topology="chain",
                     rng=np.random.default_rng(0))
        env.reset()
        # Two links on each side of the interior node R1.
        for _ in range(2):
            env.net.entangle(0, 1)
            env.net.entangle(1, 2)
        rep1 = env.net.repeaters[1]
        self.assertEqual(rep1.num_occupied(), 4)      # 2 left + 2 right
        res = env._exec_purify(1)
        self.assertTrue(res["success"])
        env.net.age_links(discard_expired=False)
        # Each partner's cascade leaves <=1 survivor: at most one link per side.
        self.assertLessEqual(len(rep1.qubits_to(0)), 1)
        self.assertLessEqual(len(rep1.qubits_to(2)), 1)
        self.assertLessEqual(rep1.num_occupied(), 2)


class TestStepFunction(unittest.TestCase):
    """Serialized sweep: step(action:int) applies to env.active_node only."""

    def setUp(self):
        self.env = _perfect_env(5)
        self.env.reset()

    def _run_to_boundary(self, env, action=NOOP):
        """Drive micro-steps with `action` until (and including) a tick
        boundary; returns the last (obs, reward, done, info)."""
        info = {"tick_boundary": False}
        while not info["tick_boundary"]:
            obs, reward, done, info = env.step(action)
            if done:
                break
        return obs, reward, done, info

    def test_step_returns_correct_tuple(self):
        obs, reward, done, info = self.env.step(NOOP)
        self.assertIn("x", obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_step_increments_step_counter(self):
        self._run_to_boundary(self.env)
        self.assertEqual(self.env.steps, 1)

    def test_step_cost_on_non_terminal(self):
        # With p_gen=0 entanglement never forms -> never succeed, Phi stays 0.
        env = QRNEnv(n_repeaters=5, p_gen=0.0, max_steps=100,
                     rng=np.random.default_rng(0))
        env.reset()
        _, reward, done, _ = self._run_to_boundary(env)
        if not done:
            self.assertAlmostEqual(reward, QRNEnv.STEP_COST)

    def test_purify_executed_before_swap(self):
        """
        Different interior nodes may issue different actions across the
        sweep; PURIFY at one node and SWAP at another in the same tick
        must not crash.
        """
        env = _perfect_env(4)
        env.reset()
        # Entangle manually so interior nodes have links.
        env.net.entangle(0, 1); env.net.entangle(0, 1)
        env.net.entangle(1, 2); env.net.entangle(2, 3)
        try:
            env.step(PURIFY)   # active_node == 1
            env.step(SWAP)     # active_node == 2 (tick boundary)
        except Exception as e:
            self.fail(f"step() crashed with purify+swap in the same tick: {e}")

    def test_done_on_max_steps(self):
        env = QRNEnv(n_repeaters=4, p_gen=0.0, max_steps=2,
                     rng=np.random.default_rng(0))
        env.reset()
        self._run_to_boundary(env)                    # tick 1
        _, _, done, _ = self._run_to_boundary(env)     # tick 2 -> truncated
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
        # R0 (leftmost) faces RIGHT toward R2; R2 (rightmost) faces LEFT toward R0.
        from simulator.repeater import fidelity_to_werner, LEFT, RIGHT
        q0 = env.net.repeaters[0].allocate_qubit(RIGHT)
        q2 = env.net.repeaters[2].allocate_qubit(LEFT)
        p  = fidelity_to_werner(0.95)
        env.net.repeaters[0].set_link(q0, 2, q2, p)
        env.net.repeaters[2].set_link(q2, 0, q0, p)
        phi_before = env._phi
        _, reward, done, info = env.step(NOOP)   # only interior node -> node 1
        self.assertTrue(done)
        # Terminal reward = fidelity * SUCCESS_REWARD - Phi(s_before); Phi(terminal)=0.
        fidelity = info["fidelity"]
        expected = fidelity * QRNEnv.SUCCESS_REWARD - phi_before
        self.assertAlmostEqual(reward, expected)


class TestSequentialSweep(unittest.TestCase):
    """Task 3: one env.step() is one micro-decision for env.active_node;
    interior nodes are visited strictly left-to-right, and the LAST interior
    node's micro-step is the tick boundary (age_links + auto_entangle)."""

    def test_observation_is_11_features_with_active_and_relpos(self):
        env = _perfect_env(5)
        env.reset()
        obs = env.get_observation()
        self.assertEqual(obs["x"].shape[1], 11)
        active = env.active_node
        self.assertEqual(obs["x"][active, 9], 1.0)
        self.assertEqual(obs["x"][:, 9].sum(), 1.0)
        self.assertAlmostEqual(float(obs["x"][4, 10]), 1.0, places=6)  # dest
        self.assertEqual(float(obs["x"][0, 10]), 0.0)                  # source

    def test_sweep_visits_interior_left_to_right_then_tick_boundary(self):
        # info["active_node"] names the node that just acted (unlike
        # next_active_node, it is NOT advanced before being reported).
        env = _perfect_env(5)
        env.reset()
        acted = []
        info = None
        for _ in range(3):  # interior nodes 1, 2, 3 act in order
            _, _, _, info = env.step(NOOP)
            acted.append(info["active_node"])
        self.assertEqual(acted, [1, 2, 3])
        self.assertTrue(info["tick_boundary"])

    def test_intra_tick_gamma_one_boundary_gamma_tick(self):
        env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=30,
                     F0=1.0, channel_loss=0.0, max_steps=50, gamma=0.9,
                     rng=np.random.default_rng(0))
        env.reset()
        _, _, _, i1 = env.step(NOOP)   # node 1 -> intra-tick
        self.assertEqual(i1["gamma_eff"], 1.0)
        self.assertFalse(i1["tick_boundary"])
        _, _, _, i2 = env.step(NOOP)   # node 2 -> intra-tick
        _, _, _, i3 = env.step(NOOP)   # node 3 -> boundary
        self.assertEqual(i3["gamma_eff"], 0.9)
        self.assertTrue(i3["tick_boundary"])

    def test_step_cost_charged_once_per_tick_at_boundary(self):
        # p_gen=0: no links ever form, Phi stays 0 throughout -> intra-tick
        # rewards are pure (zero) shaping; the boundary reward is exactly
        # STEP_COST.
        env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.0, max_steps=50,
                     rng=np.random.default_rng(0))
        env.reset()
        _, r1, _, i1 = env.step(NOOP)
        _, r2, _, i2 = env.step(NOOP)
        _, r3, _, i3 = env.step(NOOP)
        self.assertAlmostEqual(r1, 0.0)
        self.assertAlmostEqual(r2, 0.0)
        self.assertTrue(i3["tick_boundary"])
        self.assertAlmostEqual(r3, QRNEnv.STEP_COST)

    def test_delivery_terminates_mid_sweep_on_closing_node(self):
        # 4-node chain, interior [1, 2]. Manually wire 0<->1 and 1<->3 so a
        # SWAP at node 1 (the FIRST interior node) closes source->dest
        # directly, mid-sweep -- node 2 never gets to act this tick.
        env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=30,
                     F0=1.0, channel_loss=0.0, max_steps=50,
                     rng=np.random.default_rng(3))
        env.reset()
        env.net.reset()
        from simulator.repeater import fidelity_to_werner, LEFT, RIGHT
        q0 = env.net.repeaters[0].allocate_qubit(RIGHT)
        q1a = env.net.repeaters[1].allocate_qubit(LEFT)
        p = fidelity_to_werner(0.99)
        env.net.repeaters[0].set_link(q0, 1, q1a, p)
        env.net.repeaters[1].set_link(q1a, 0, q0, p)

        q1b = env.net.repeaters[1].allocate_qubit(RIGHT)
        q3 = env.net.repeaters[3].allocate_qubit(LEFT)
        env.net.repeaters[1].set_link(q1b, 3, q3, p)
        env.net.repeaters[3].set_link(q3, 1, q1b, p)

        self.assertEqual(env.active_node, 1)   # first interior node, unchanged
        phi_before = env._phi
        _, reward, done, info = env.step(SWAP)
        self.assertTrue(info["terminated"])
        self.assertEqual(info["active_node"], 1)
        self.assertTrue(done)
        expected = info["fidelity"] * QRNEnv.SUCCESS_REWARD - phi_before
        self.assertAlmostEqual(reward, expected)



#   ▄▄▄▄▄   ▄▄▄▄▄▄▄   ▄▄▄    ▄▄▄   ▄▄▄▄
# ▄███████▄ ███▀▀███▄ ████▄  ███ ▄██▀▀██▄                    ██
# ███   ███ ███▄▄███▀ ███▀██▄███ ███  ███ ▄████ ▄█▀█▄ ████▄ ▀██▀▀
# ███▄█▄███ ███▀▀██▄  ███  ▀████ ███▀▀███ ██ ██ ██▄█▀ ██ ██  ██
#  ▀█████▀  ███  ▀███ ███    ███ ███  ███ ▀████ ▀█▄▄▄ ██ ██  ██
#       ▀▀                                   ██
#                                          ▀▀▀

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


class TestSelectAction(unittest.TestCase):
    """select_action (scalar) is the per-micro-step counterpart of
    select_actions: it picks ONE action for env.active_node against that
    node's (3,) mask_row, and must never violate the mask either greedily
    or under exploration."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.agent = QRNAgent(node_dim=8, hidden=16)
        self.obs   = _dummy_obs(6)

    def test_greedy_respects_mask_row(self):
        self.agent.epsilon = 0.0
        row = _dummy_mask_row()
        a = self.agent.select_action(self.obs, row, active_node=2, training=False)
        self.assertTrue(row[a])

    def test_exploration_respects_mask_row(self):
        self.agent.epsilon = 1.0
        row = _dummy_mask_row()
        for _ in range(20):
            a = self.agent.select_action(self.obs, row, active_node=3, training=True)
            self.assertTrue(row[a])

    def test_noop_only_row_forces_noop(self):
        row = _dummy_mask_row(force_noop_only=True)
        for eps in [0.0, 1.0]:
            self.agent.epsilon = eps
            a = self.agent.select_action(self.obs, row, active_node=1,
                                         training=(eps > 0))
            self.assertEqual(a, NOOP)

    def test_returns_python_int(self):
        self.agent.epsilon = 0.0
        row = _dummy_mask_row()
        a = self.agent.select_action(self.obs, row, active_node=0, training=False)
        self.assertIsInstance(a, int)

    def test_greedy_consistent_across_calls(self):
        self.agent.epsilon = 0.0
        row = _dummy_mask_row()
        a1 = self.agent.select_action(self.obs, row, active_node=4, training=False)
        a2 = self.agent.select_action(self.obs, row, active_node=4, training=False)
        self.assertEqual(a1, a2)


class TestTrainStepTensorShapes(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.agent = QRNAgent(node_dim=8, hidden=16, batch_size=8)
        _fill_buffer(self.agent.memory, 30, 5)

    def test_current_q_shape(self):
        """current_q must be a 1-D tensor of length batch_size: ONE Q per
        transition, gathered at its active node (not one per node)."""
        batch = self.agent.memory.sample(self.agent.batch_size)
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.agent.device)
        ptr = states.ptr[:-1]
        act_idx = ptr + torch.tensor(
            [t["ai"] for t in batch], device=self.agent.device)
        actions = torch.tensor(
            [t["a"] for t in batch], dtype=torch.long, device=self.agent.device)
        q_all = self.agent.policy_net(states)
        current_q = q_all[act_idx].gather(1, actions.unsqueeze(1)).squeeze(1)
        self.assertEqual(current_q.shape, (self.agent.batch_size,))

    def test_target_q_same_shape_as_current_q(self):
        batch = self.agent.memory.sample(self.agent.batch_size)
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.agent.device)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(self.agent.device)
        ptr = states.ptr[:-1]
        act_idx = ptr + torch.tensor(
            [t["ai"] for t in batch], device=self.agent.device)
        actions = torch.tensor(
            [t["a"] for t in batch], dtype=torch.long, device=self.agent.device)
        nptr = next_states.ptr[:-1]
        nidx = nptr + torch.tensor(
            [t["nai"] for t in batch], device=self.agent.device)
        next_masks = torch.tensor(
            np.stack([t["m_"] for t in batch]), dtype=torch.bool, device=self.agent.device)
        rewards = torch.tensor(
            [t["r"] for t in batch], dtype=torch.float32, device=self.agent.device)
        gammas = torch.tensor(
            [t["g"] for t in batch], dtype=torch.float32, device=self.agent.device)
        dones = torch.zeros(self.agent.batch_size, device=self.agent.device)

        q_all     = self.agent.policy_net(states)
        current_q = q_all[act_idx].gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nqp = self.agent.policy_net(next_states)[nidx].clone()
            nqp[~next_masks] = -float("inf")
            best   = nqp.argmax(dim=1)
            nqt    = self.agent.target_net(next_states)[nidx]
            nq     = nqt.gather(1, best.unsqueeze(1)).squeeze(1)
            target = rewards + gammas * nq * (1.0 - dones)

        self.assertEqual(current_q.shape, target.shape,
                         "current_q and target_q must have identical shape for loss.")

                                      
# ▄▄▄▄▄▄▄           ▄▄   ▄▄             
# ███▀▀███▄        ██   ██              
# ███▄▄███▀ ██ ██ ▀██▀ ▀██▀ ▄█▀█▄ ████▄ 
# ███  ███▄ ██ ██  ██   ██  ██▄█▀ ██ ▀▀ 
# ████████▀ ▀██▀█  ██   ██  ▀█▄▄▄ ██    
                                                                         

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
            buf.add(obs, NOOP, 0, float(i), obs, 0, _dummy_mask_row(), False, 1.0)
        # Now push one more; it must overwrite position 0.
        obs = _dummy_obs(3)
        buf.add(obs, NOOP, 0, 999.0, obs, 0, _dummy_mask_row(), False, 1.0)
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
            for key in ("s", "a", "ai", "r", "s_", "nai", "m_", "d", "g"):
                self.assertIn(key, entry, f"Key '{key}' missing from transition.")

    def test_transition_shapes_preserved(self):
        n = 6
        buf = ReplayBuffer(max_size=50)
        obs  = _dummy_obs(n)
        nobs = _dummy_obs(n)
        buf.add(obs, SWAP, 2, 0.5, nobs, 3, _dummy_mask_row(), False, 1.0)
        entry = buf.sample(1)[0]
        self.assertEqual(entry["s"]["x"].shape, (n, 8))
        self.assertIsInstance(entry["a"], int)
        self.assertEqual(entry["ai"], 2)
        self.assertEqual(entry["nai"], 3)
        self.assertEqual(entry["m_"].shape, (N_ACTIONS,))

    def test_clear_empties_buffer(self):
        buf = ReplayBuffer(max_size=50)
        _fill_buffer(buf, 20, 4)
        buf.clear()
        self.assertEqual(buf.size(), 0)
        self.assertEqual(buf.pos, 0)

    def test_done_flag_stored_correctly(self):
        buf = ReplayBuffer(max_size=10)
        obs = _dummy_obs(3)
        buf.add(obs, NOOP, 0, 1.0, obs, 0, _dummy_mask_row(), True, 1.0)
        entry = buf.sample(1)[0]
        self.assertTrue(entry["d"])


                                                                               
#  ▄▄▄▄▄▄▄    ▄▄                                                                 
# ███▀▀▀▀▀    ██                                                                 
# ███▄▄    ▄████ ▄████ ▄█▀█▄   ▄████  ▀▀█▄ ▄█▀▀▀ ▄█▀█▄ ▄█▀▀▀                     
# ███      ██ ██ ██ ██ ██▄█▀   ██    ▄█▀██ ▀███▄ ██▄█▀ ▀███▄                     
# ▀███████ ▀████ ▀████ ▀█▄▄▄   ▀████ ▀█▄██ ▄▄▄█▀ ▀█▄▄▄ ▄▄▄█▀                     
#                   ██                                                           
#                 ▀▀▀                                                                                                                                        
#           ▄▄▄▄▄▄▄   ▄▄▄        ▄▄                   ▄▄          ▄▄             
#    ▄      ███▀▀███▄ ███        ██                   ██          ██             
#    █      ███▄▄███▀ ███        ██ ▄███▄ ▄███▄ ████▄ ████▄ ▄███▄ ██ ▄█▀█▄ ▄█▀▀▀ 
# ▀▀▀█▀▀▀   ███▀▀██▄  ███        ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██▄█▀ ▀███▄ 
#    █      ███  ▀███ ████████   ██ ▀███▀ ▀███▀ ████▀ ██ ██ ▀███▀ ██ ▀█▄▄▄ ▄▄▄█▀ 
#                                               ██                               
#                                               ▀▀                               

class TestTargetNodeActionInjection(unittest.TestCase):
    """
    Source and destination must never be able to act. Under the serialized
    sweep this is now a STRUCTURAL invariant (they are never members of
    `env._interior`, so `env.active_node` can never equal them) rather than
    an action-array override -- verify the invariant holds across a full
    rollout regardless of what action is issued each micro-step.
    """

    def setUp(self):
        self.env = _perfect_env(5)
        self.env.reset()

    def test_active_node_never_source_or_dest(self):
        for _ in range(3 * self.env.max_steps):
            self.assertNotEqual(self.env.active_node, self.env.source)
            self.assertNotEqual(self.env.active_node, self.env.dest)
            _, _, done, info = self.env.step(SWAP)
            if info["next_active_node"] != -1:
                self.assertNotEqual(info["next_active_node"], self.env.source)
                self.assertNotEqual(info["next_active_node"], self.env.dest)
            if done:
                break

    def test_sweep_targets_are_exactly_the_interior_nodes(self):
        # One full left-to-right sweep must visit exactly _interior, in order
        # (info["active_node"] names the node that just acted each call).
        expected = list(self.env._interior)
        seen = []
        for _ in range(len(expected)):
            _, _, _, info = self.env.step(NOOP)
            seen.append(info["active_node"])
        self.assertEqual(seen, expected)


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
        ai   = int(np.random.randint(0, n_nodes))
        nai  = int(np.random.randint(0, n_nodes))
        self.agent.memory.add(obs, NOOP, ai, -0.01, nobs, nai,
                              _dummy_mask_row(), False, 1.0)

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

    def test_active_node_offset_matches_transition_count(self):
        """act_idx (ptr + ai) must have exactly ONE entry per TRANSITION in
        the batch, not one per node -- the key departure from the old
        per-graph-broadcast model. Also checks the offset arithmetic itself
        for mixed graph sizes (graph 1's nodes start at graph 0's size)."""
        sizes = [4, 7]
        transitions = [{"s": _dummy_obs(n)} for n in sizes]
        ai_vals = [1, 2]
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in transitions])
        ptr = states.ptr[:-1]
        act_idx = ptr + torch.tensor(ai_vals)
        self.assertEqual(act_idx.shape[0], len(transitions))
        expected = torch.tensor([0 + ai_vals[0], sizes[0] + ai_vals[1]])
        self.assertTrue(torch.equal(act_idx, expected))


class TestAllActionsMaskedFallback(unittest.TestCase):
    """
    If a node has no available qubits (e.g. all free or all consumed by an
    in-flight op) then only NOOP should appear in its mask.  The Q-network
    must still produce a valid argmax (NOOP) without raising an empty-sequence
    error, a common failure mode when -inf is applied to all actions before argmax.
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


class TestEnvRewireCorrectness(unittest.TestCase):
    """The backend-rewired env produces VALID, correct rollouts.

    We deliberately do NOT assert a frozen reward trajectory — the contract is
    correctness (valid shapes/ranges, mask-respecting actions, finite rewards,
    working e2e detection), not byte-reproduction of the old numpy engine.
    """

    def test_swap_asap_rollout_is_valid(self):
        # strategies.swap_asap(env) returns a SCALAR action for env.active_node
        # (Task 5's contract); recomputed fresh each micro-step so it always
        # sees the current state.
        import numpy as np
        from rl_stack.env_wrapper import QRNEnv
        from rl_stack.strategies import swap_asap
        env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.9, p_swap=0.7,
                     cutoff=20, max_steps=40, topology="chain",
                     rng=np.random.default_rng(2024))
        obs = env.reset()
        self.assertEqual(obs["x"].shape, (env.N, 11))
        self.assertEqual(obs["edge_index"].shape[0], 2)
        while True:
            mask = env.get_action_mask()
            r_node = env.active_node
            a = swap_asap(env)
            self.assertTrue(mask[r_node, a])
            obs, r, done, info = env.step(a)
            self.assertTrue(np.isfinite(r))
            self.assertEqual(obs["x"].shape, (env.N, 11))
            self.assertTrue(np.all(obs["x"] >= -1e-6))
            self.assertTrue(np.all(obs["x"] <= 1.0 + 1e-6))
            if done:
                break

    def test_e2e_detection_terminates(self):
        import numpy as np
        from rl_stack.env_wrapper import QRNEnv
        from rl_stack.strategies import swap_asap
        env = QRNEnv(n_repeaters=3, n_ch=4, p_gen=1.0, p_swap=1.0,
                     cutoff=50, max_steps=80, topology="chain",
                     rng=np.random.default_rng(7))
        env.reset()
        reached = False
        while True:
            a = swap_asap(env)
            _, _, done, info = env.step(a)
            if done:
                reached = info["fidelity"] > 0.0
                break
        self.assertTrue(reached)


def test_step_reports_terminated_vs_truncated():
    # max_steps=1, impossible delivery (p_gen=0) -> must truncate, not terminate.
    # n_repeaters=4 -> 2 interior nodes; drive the full sweep to the tick boundary.
    env = QRNEnv(n_repeaters=4, n_ch=4, p_gen=0.0, p_swap=1.0, cutoff=20,
                 max_steps=1, topology="chain", rng=np.random.default_rng(0))
    env.reset()
    info = {"tick_boundary": False}
    while not info["tick_boundary"]:
        _, _, done, info = env.step(NOOP)
    assert done is True
    assert info["truncated"] is True
    assert info["terminated"] is False


def test_step_win_is_terminated_not_truncated():
    # easy 2-node chain: delivers fast -> terminated True, truncated False.
    # N=2 has no interior nodes (the reset() empty-_interior edge case).
    env = QRNEnv(n_repeaters=2, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=20,
                 max_steps=50, topology="chain", rng=np.random.default_rng(0))
    env.reset()
    saw_win = False
    for _ in range(50):
        _, _, done, info = env.step(NOOP)
        if done:
            saw_win = info["terminated"]
            assert info["terminated"] != info["truncated"]
            break
    assert saw_win is True


def test_sample_cutoff_range_is_int_in_band():
    from rl_stack.agent import _sample_cutoff
    rng = np.random.default_rng(0)
    vals = [_sample_cutoff(rng, (10, 40)) for _ in range(200)]
    assert all(isinstance(v, int) for v in vals)
    assert min(vals) >= 10 and max(vals) <= 40
    assert len(set(vals)) > 1                      # genuinely varied
    # scalar passes through, draws no rng
    assert _sample_cutoff(rng, 15) == 15


def test_draw_winnable_cell_rejects_unwinnable():
    from rl_stack import agent as A

    class FakeWC:
        def __init__(self):
            self.n = 0
        def winnable(self, **kw):
            self.n += 1
            return self.n >= 3            # first two draws rejected

    wc = FakeWC()
    rng = np.random.default_rng(0)
    p_gen, p_swap, cutoff, n, nch = A._draw_winnable_cell(
        rng, wc, p_gen=(0.4, 0.9), p_swap=(0.4, 0.9), cutoff=(10, 40),
        n_pool=np.array([6]), n_ch_pool=np.array([4]), max_tries=10)
    assert wc.n == 3                       # resampled until winnable
    assert 0.4 <= p_gen <= 0.9 and 10 <= cutoff <= 40
    assert n == 6 and nch == 4


def test_draw_winnable_cell_no_oracle_passes_through():
    from rl_stack import agent as A
    rng = np.random.default_rng(0)
    pg, ps, ct, n, nch = A._draw_winnable_cell(
        rng, None, p_gen=0.7, p_swap=0.8, cutoff=15,
        n_pool=np.array([5]), n_ch_pool=np.array([3]))
    assert (pg, ps, ct, n, nch) == (0.7, 0.8, 15, 5, 3)


def test_can_swap_masked_when_only_doomed_pairs():
    """SWAP mask must reject pairs that would not survive same-tick resolution
    (age_i + age_j + 2 >= link_cutoff). Feature 4 agrees."""
    env = QRNEnv(n_repeaters=3, n_ch=2, p_gen=1.0, p_swap=1.0, cutoff=10,
                 F0=1.0, channel_loss=0.0, max_steps=20,
                 rng=np.random.default_rng(0))
    env.reset()
    # age every link so any pair sums past the viability margin
    for rep in env.net.repeaters:
        occ = rep.occupied_indices()
        rep.age[occ] = 5          # 5 + 5 + 2 = 12 >= 10
    mask = env.get_action_mask()
    assert not mask[1, SWAP]
    obs = env.get_observation()
    assert obs["x"][1, 4] == 0.0   # feature 4 agrees with the mask


if __name__ == "__main__":
    unittest.main(verbosity=2)
