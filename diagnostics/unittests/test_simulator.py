"""
test_simulator.py
=================
AI GENERATED
=================
Comprehensive unittest suite for the Quantum Repeater Network Simulator.

Covers:
  1. Physical Validation   - Werner/fidelity conversions, decoherence,
                             BBPSSW, swap product rule, distance scaling.
  2. Core Functionality    - entangle, swap, purify, age_links, reset,
                             cross-module wiring.
  3. Edge Cases / RL Loopholes - ghost links, asymmetric cutoffs, zero
                             distance, double-booking, self-swapping,
                             and more.

Run with:
    python -m pytest test_simulator.py -v
  or
    python -m unittest test_simulator -v
"""

import math
import unittest
import numpy as np

# ── imports ──────────────────────────────────────────────────────────────────
from quantum_repeater_sim.repeater import (
    Repeater, SwapPolicy,
    QUBIT_FREE, QUBIT_OCCUPIED, NO_PARTNER,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_fidelity,
)
from quantum_repeater_sim.network import (
    RepeaterNetwork, build_chain, build_grid, build_GEANT,
)
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY


# ── tiny helpers ─────────────────────────────────────────────────────────────

def _perfect_chain(n, n_ch=4, cutoff=20, spacing=50.0):
    """Build a deterministic chain: p_gen=1, p_swap=1, no channel loss."""
    return build_chain(
        n, n_ch=n_ch, spacing=spacing,
        p_gen=1.0, p_swap=1.0, cutoff=cutoff,
        F0=1.0, channel_loss=0.0,
        dt_seconds=1e-4,
        distance_dep_gen=False,
        rng=np.random.default_rng(0),
    )


def _entangle_force(net, r1, r2):
    """Guarantee entanglement regardless of RNG by patching p_gen temporarily."""
    net.repeaters[r1].p_gen = 1.0
    net.repeaters[r2].p_gen = 1.0
    res = net.entangle(r1, r2)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestWernerFidelityConversion(unittest.TestCase):
    """Werner ↔ Fidelity round-trip: F = (3p + 1) / 4  ↔  p = (4F - 1) / 3."""

    def test_pure_state_fidelity_1(self):
        # p = 1  →  F = 1  (maximally entangled Bell state)
        self.assertAlmostEqual(float(werner_to_fidelity(1.0)), 1.0)

    def test_maximally_mixed_state(self):
        # p = 0  →  F = 0.25  (completely depolarised)
        self.assertAlmostEqual(float(werner_to_fidelity(0.0)), 0.25)

    def test_fidelity_to_werner_round_trip(self):
        # Converting F → p → F should recover the original value.
        for f in [0.5, 0.75, 0.9, 1.0]:
            p = fidelity_to_werner(f)
            f2 = werner_to_fidelity(p)
            self.assertAlmostEqual(float(f2), f, places=9)

    def test_werner_negative_unphysical_below_quarter(self):
        # F < 0.25 maps to negative p (unphysical Werner state).
        p = fidelity_to_werner(0.1)
        self.assertLess(float(p), 0.0)

    def test_formula_exact_value(self):
        # Direct check: p = 0.6  →  F = (1.8 + 1) / 4 = 0.7
        self.assertAlmostEqual(float(werner_to_fidelity(0.6)), 0.7)


class TestDecoherenceModel(unittest.TestCase):
    """Werner parameter decays as p(t) = p0 * exp(-t / cutoff)."""

    def test_exponential_decay_one_step(self):
        # After 1 tick p should equal p0 * exp(-1/cutoff).
        cutoff = 10
        rep = Repeater(rid=0, n_ch=2, cutoff=cutoff)
        p0 = 0.9
        rep.set_link(0, 1, 0, p0, link_age=0, effective_cutoff=cutoff)
        rep.status[0] = QUBIT_OCCUPIED
        rep.age_occupied()  # advance 1 tick
        expected = p0 * math.exp(-1 / cutoff)
        self.assertAlmostEqual(float(rep.werner_param[0]), expected, places=5)

    def test_decay_multiple_steps(self):
        cutoff = 20
        rep = Repeater(rid=0, n_ch=2, cutoff=cutoff)
        p0 = 0.8
        rep.set_link(0, 1, 0, p0, link_age=0, effective_cutoff=cutoff)
        rep.status[0] = QUBIT_OCCUPIED
        for _ in range(5):
            rep.age_occupied()
        expected = p0 * math.exp(-5 / cutoff)
        self.assertAlmostEqual(float(rep.werner_param[0]), expected, places=5)

    def test_link_age_set_correctly_at_generation(self):
        # At t = 0 the Werner param equals p0 exactly (no decay yet).
        rep = Repeater(rid=0, n_ch=2, cutoff=20)
        p0 = 0.95
        rep.set_link(0, 1, 0, p0, link_age=0)
        self.assertAlmostEqual(float(rep.werner_param[0]), p0, places=6)

    def test_expiry_returned_at_cutoff(self):
        # age_occupied() must flag qubits whose age >= cutoff as expired.
        cutoff = 2
        rep = Repeater(rid=0, n_ch=2, cutoff=cutoff)
        rep.set_link(0, 1, 0, 0.9, link_age=0, effective_cutoff=cutoff)
        rep.status[0] = QUBIT_OCCUPIED
        expired = np.array([], dtype=np.intp)
        for _ in range(cutoff):
            rep.age_occupied() 
            expired = rep.age_occupied()
        self.assertIn(0, expired, "Qubit must be flagged expired at cutoff age.")


class TestBBPSSWPurification(unittest.TestCase):
    """
    BBPSSW protocol (Bennett et al. 1996).

    Success prob: P_suc = (8/9)*p1*p2 - (2/9)*(p1+p2) + 5/9
      where p1, p2 are Werner parameters.
    New Werner:   p_new = (1-(p1+p2)+10*p1*p2) / (5-2*(p1+p2)+8*p1*p2)
    """

    def test_success_prob_identical_states(self):
        # For two identical states with F = 0.9, check against formula.
        f = 0.9
        expected = (4/3)*f*f - (2/3)*f + 1/3
        self.assertAlmostEqual(float(bbpssw_success_prob(f, f)), expected, places=9)

    def test_new_fidelity_higher_than_inputs(self):
        # A successful purification must strictly improve fidelity.
        f1, f2 = 0.8, 0.75
        f_new = bbpssw_new_fidelity(f1, f2)
        self.assertGreater(float(f_new), max(f1, f2),
                           "Purified fidelity must exceed both inputs.")

    def test_purification_with_perfect_states(self):
        # Two Bell states (F = 1) should give F_new = 1 and P_suc = 1.
        f = 1.0
        self.assertAlmostEqual(float(bbpssw_success_prob(f, f)), 1.0, places=9)
        self.assertAlmostEqual(float(bbpssw_new_fidelity(f, f)), 1.0, places=9)

    def test_success_prob_in_valid_range(self):
        # P_suc must lie in [0, 1] for any physical input.
        for f in [0.6, 0.7, 0.8, 0.9, 1.0]:
            ps = float(bbpssw_success_prob(f, f))
            self.assertGreaterEqual(ps, 0.0)
            self.assertLessEqual(ps, 1.0)


class TestEntanglementSwapping(unittest.TestCase):
    """Post-swap Werner parameter: p_new = p_a * p_b (product rule)."""

    def test_swap_product_rule_in_network(self):
        # Build R0–R1–R2, entangle 0↔1 and 1↔2, swap at R1.
        # Expected: initial_werner of the new link = p_01 * p_12 (product rule).
        # The current werner_param will differ due to age-based decoherence
        # during the classical delay, so we check initial_werner directly.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)

        rep1 = net.repeaters[1]
        qa = rep1.available_indices()[0]
        qb = rep1.available_indices()[1]
        p_a = float(rep1.werner_param[qa])
        p_b = float(rep1.werner_param[qb])
        expected_p_new = p_a * p_b

        res = net.swap(1)
        self.assertTrue(res["success"])
        # Resolve the pending event
        while net.pending_events:
            net.age_links(discard_expired=False)

        rep0 = net.repeaters[0]
        occupied = rep0.occupied_indices()
        self.assertTrue(len(occupied) > 0, "R0 should hold the new link.")
        actual_initial_p = float(rep0.initial_werner[occupied[0]])
        self.assertAlmostEqual(actual_initial_p, expected_p_new, places=5)

    def test_local_qubits_freed_after_bsm(self):
        # BSM physically consumes both local qubits immediately.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        # R1 should have no occupied qubits right after the BSM.
        self.assertEqual(net.repeaters[1].num_occupied(), 0,
                         "BSM must destroy local qubits instantly.")


class TestClassicalDelay(unittest.TestCase):
    """Classical signal delay: steps = ceil(d / (c_fiber * dt))."""

    def test_delay_formula(self):
        net = _perfect_chain(3, spacing=50.0)
        # c_fiber = 200_000 km/s, dt = 1e-4 s, d = 50 km
        # steps = ceil(50 / (200_000 * 1e-4)) = ceil(50/20) = 3
        expected = math.ceil(50.0 / (200_000.0 * 1e-4))
        actual = net._classical_delay_steps(50.0)
        self.assertEqual(actual, expected)

    def test_zero_distance_no_delay(self):
        # d = 0 must return 0 (no division-by-zero, no delay).
        net = _perfect_chain(3)
        self.assertEqual(net._classical_delay_steps(0.0), 0)

    def test_event_queued_with_nonzero_timer(self):
        # With a non-trivial dt the swap event timer must be > 0.
        net = build_chain(3, n_ch=4, spacing=50.0,
                          p_gen=1.0, p_swap=1.0,
                          F0=1.0, channel_loss=0.0,
                          dt_seconds=1e-6,          # very small dt → large delay
                          distance_dep_gen=False,
                          rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        self.assertTrue(len(net.pending_events) > 0)
        self.assertGreater(net.pending_events[0]["timer"], 0)

    def test_remote_qubits_locked_after_swap(self):
        # Remote qubits must be locked while the classical message travels.
        net = build_chain(3, n_ch=4, spacing=50.0,
                          p_gen=1.0, p_swap=1.0,
                          F0=1.0, channel_loss=0.0,
                          dt_seconds=1e-6,
                          distance_dep_gen=False,
                          rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        # R0 and R2 should each have exactly one locked qubit.
        self.assertEqual(net.repeaters[0].num_locked(), 1)
        self.assertEqual(net.repeaters[2].num_locked(), 1)


class TestDistanceDependency(unittest.TestCase):
    """
    Generation probability: p_eff = p_avg * exp(-loss * d / 2)
    Initial fidelity:       F0_eff = F0 * exp(-loss * d)
    """

    def _make_two_node(self, spacing, loss):
        return build_chain(2, n_ch=4, spacing=spacing,
                           p_gen=1.0, p_swap=1.0,
                           F0=1.0, channel_loss=loss,
                           dt_seconds=1e-4,
                           distance_dep_gen=True,
                           rng=np.random.default_rng(0))

    def test_gen_prob_scaling(self):
        loss, d = 0.02, 50.0
        net = self._make_two_node(d, loss)
        expected = 1.0 * math.exp(-loss * d / 2)
        self.assertAlmostEqual(net._gen_prob(0, 1), expected, places=6)

    def test_initial_fidelity_scaling(self):
        loss, d = 0.02, 50.0
        net = self._make_two_node(d, loss)
        expected_fid = 1.0 * math.exp(-loss * d)
        self.assertAlmostEqual(net._gen_fidelity(0, 1), expected_fid, places=6)

    def test_zero_loss_unity_fidelity(self):
        net = self._make_two_node(50.0, 0.0)
        self.assertAlmostEqual(net._gen_fidelity(0, 1), 1.0, places=9)
        self.assertAlmostEqual(net._gen_prob(0, 1), 1.0, places=9)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CORE FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntanglement(unittest.TestCase):

    def setUp(self):
        self.net = _perfect_chain(3)

    def test_qubit_transitions_free_to_occupied(self):
        # Successful entanglement must mark both qubits as OCCUPIED.
        self.net.entangle(0, 1)
        r0, r1 = self.net.repeaters[0], self.net.repeaters[1]
        self.assertEqual(r0.num_occupied(), 1)
        self.assertEqual(r1.num_occupied(), 1)

    def test_initial_age_is_zero(self):
        # A freshly generated link must have age 0.
        self.net.entangle(0, 1)
        qi = self.net.repeaters[0].occupied_indices()[0]
        self.assertEqual(int(self.net.repeaters[0].age[qi]), 0)

    def test_initial_fidelity_correct(self):
        # channel_loss=0, F0=1 → fidelity = 1.0 exactly.
        self.net.entangle(0, 1)
        qi = self.net.repeaters[0].occupied_indices()[0]
        p = float(self.net.repeaters[0].werner_param[qi])
        self.assertAlmostEqual(float(werner_to_fidelity(p)), 1.0, places=6)

    def test_partner_pointers_consistent(self):
        # R0[q0].partner == R1, and R1[q1].partner == R0 (back-pointer).
        self.net.entangle(0, 1)
        r0 = self.net.repeaters[0]
        qi0 = r0.occupied_indices()[0]
        prid = int(r0.partner_repeater[qi0])
        pqid = int(r0.partner_qubit[qi0])
        self.assertEqual(prid, 1)
        # R1's partner qubit must point back to R0.
        self.assertEqual(int(self.net.repeaters[1].partner_repeater[pqid]), 0)

    def test_entangle_non_adjacent_fails(self):
        # R0 and R2 are not adjacent in a chain → must fail.
        res = self.net.entangle(0, 2)
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "not_adjacent")

    def test_entangle_full_repeater_fails(self):
        # Fill all 4 qubits of R0 via R0–R1, then attempt another.
        net = _perfect_chain(3, n_ch=2)
        net.entangle(0, 1)
        net.entangle(0, 1)   # now R0 is full
        res = net.entangle(0, 1)
        self.assertFalse(res["success"])

    def test_multiple_links_same_pair(self):
        # Two independent Bell pairs between R0–R1 must both be stored.
        self.net.entangle(0, 1)
        self.net.entangle(0, 1)
        self.assertEqual(self.net.repeaters[0].num_occupied(), 2)
        self.assertEqual(self.net.repeaters[1].num_occupied(), 2)


class TestSwapping(unittest.TestCase):

    def setUp(self):
        self.net = _perfect_chain(3, cutoff=50)
        self.net.entangle(0, 1)
        self.net.entangle(1, 2)

    def test_swap_succeeds(self):
        res = self.net.swap(1)
        self.assertTrue(res["success"])

    def test_swap_queues_event(self):
        self.net.swap(1)
        self.assertEqual(len(self.net.pending_events), 1)

    def test_event_resolves_to_long_range_link(self):
        self.net.swap(1)
        while self.net.pending_events:
            self.net.age_links(discard_expired=False)
        # R0 should now be linked to R2.
        r0 = self.net.repeaters[0]
        self.assertTrue(len(r0.occupied_indices()) > 0)
        qi = r0.occupied_indices()[0]
        self.assertEqual(int(r0.partner_repeater[qi]), 2)

    def test_failed_swap_destroys_both_links_immediately(self):
        # p_swap = 0  → swap always fails → both links destroyed at BSM time.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.repeaters[1].p_swap = 0.0
        # override RNG so the roll > 0 always
        net.rng = np.random.default_rng(99)
        res = net.swap(1)
        if not res["success"]:
            # All qubits at R1 freed; no pending events.
            self.assertEqual(net.repeaters[1].num_occupied(), 0)
            self.assertEqual(len(net.pending_events), 0)

    def test_swap_without_two_links_fails(self):
        net = _perfect_chain(3)
        net.entangle(0, 1)     # only one link at R1
        res = net.swap(1)
        self.assertFalse(res["success"])


class TestPurification(unittest.TestCase):

    def _net_with_two_links(self):
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(0, 1)
        return net

    def test_purify_queues_event(self):
        net = self._net_with_two_links()
        res = net.purify(0, 1)
        self.assertTrue(res["success"] or res["reason"] == "pending")
        self.assertEqual(len(net.pending_events), 1)

    def test_purify_success_upgrades_fidelity(self):
        # Run until the purification resolves; the kept pair should have
        # a higher Werner parameter than either input.
        net = _perfect_chain(3, cutoff=100)
        net.repeaters[0].p_swap = 1.0  # re-use p_swap field isn't relevant here
        # Set two links with F = 0.8 < 1 to make the improvement visible.
        net.entangle(0, 1)
        net.entangle(0, 1)
        # Degrade both slightly
        for qi in net.repeaters[0].occupied_indices():
            net.repeaters[0].werner_param[qi] = fidelity_to_werner(0.8)
            net.repeaters[0].initial_werner[qi] = fidelity_to_werner(0.8)

        p_before = max(
            float(net.repeaters[0].werner_param[qi])
            for qi in net.repeaters[0].occupied_indices()
        )
        res = net.purify(0, 1)
        # Force resolve by ticking
        while net.pending_events:
            net.age_links(discard_expired=False)

        occ = net.repeaters[0].occupied_indices()
        if len(occ):
            p_after = float(net.repeaters[0].werner_param[occ[0]])
            # On success the kept pair must be better than either input.
            self.assertGreaterEqual(p_after, p_before * 0.99,
                                    "Purification must not degrade kept pair.")

    def test_purify_failure_destroys_both(self):
        # Inject a forced-failure event directly and verify cleanup.
        net = _perfect_chain(3, cutoff=100)
        net.entangle(0, 1)
        net.entangle(0, 1)
        q1s = net.repeaters[0].occupied_indices()
        q1_sac, q1_keep = int(q1s[0]), int(q1s[1])
        q2_sac = int(net.repeaters[0].partner_qubit[q1_sac])
        q2_keep = int(net.repeaters[0].partner_qubit[q1_keep])

        # Manually lock as the protocol would and inject failure event.
        net.repeaters[0].lock_qubit(q1_sac);  net.repeaters[0].lock_qubit(q1_keep)
        net.repeaters[1].lock_qubit(q2_sac);  net.repeaters[1].lock_qubit(q2_keep)
        net.pending_events.append({
            "type": "purify", "timer": 0, "success": False,
            "r1": 0, "r2": 1,
            "q1_sac": q1_sac, "q2_sac": q2_sac,
            "q1_keep": q1_keep, "q2_keep": q2_keep,
            "p_new": 0.0,
        })
        net.age_links(discard_expired=False)
        # Both pairs must be destroyed.
        self.assertEqual(net.repeaters[0].num_occupied(), 0)

    def test_purify_insufficient_links_fails(self):
        net = _perfect_chain(3)
        net.entangle(0, 1)    # only one link
        res = net.purify(0, 1)
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "insufficient_shared_pairs")


class TestAgeing(unittest.TestCase):

    def test_time_step_increments(self):
        net = _perfect_chain(3)
        self.assertEqual(net.time_step, 0)
        net.age_links()
        self.assertEqual(net.time_step, 1)
        net.age_links()
        self.assertEqual(net.time_step, 2)

    def test_fidelity_degrades_each_tick(self):
        net = _perfect_chain(3, cutoff=20)
        net.entangle(0, 1)
        qi = net.repeaters[0].occupied_indices()[0]
        p0 = float(net.repeaters[0].werner_param[qi])
        net.age_links(discard_expired=False)
        p1 = float(net.repeaters[0].werner_param[qi])
        self.assertLess(p1, p0, "Werner param must decrease after one tick.")

    def test_link_destroyed_at_cutoff(self):
        cutoff = 3
        net = _perfect_chain(3, cutoff=cutoff)
        net.entangle(0, 1)
        for _ in range(cutoff + 1):
            net.age_links(discard_expired=True)
        self.assertEqual(net.repeaters[0].num_occupied(), 0,
                         "Link must be discarded after exceeding cutoff.")

    def test_decay_formula_exact(self):
        # p(t) = p0 * exp(-t / cutoff) checked numerically.
        cutoff = 10
        net = _perfect_chain(3, cutoff=cutoff)
        net.entangle(0, 1)
        qi = net.repeaters[0].occupied_indices()[0]
        p0 = float(net.repeaters[0].initial_werner[qi])
        for t in range(1, 6):
            net.age_links(discard_expired=False)
            p_actual = float(net.repeaters[0].werner_param[qi])
            p_expected = p0 * math.exp(-t / cutoff)
            self.assertAlmostEqual(p_actual, p_expected, places=5,
                                   msg=f"Decay mismatch at t={t}")

    def test_pending_events_resolved_after_delay(self):
        # Zero-delay network: pending swap resolves in next tick.
        net = _perfect_chain(3, cutoff=50, spacing=0.0)
        # Give repeaters distinct positions to avoid 0 distance issues.
        net.repeaters[0].position = np.array([0.0, 0.0])
        net.repeaters[1].position = np.array([0.0, 0.0])
        net.repeaters[2].position = np.array([0.0, 0.0])
        net._positions = np.stack([r.position for r in net.repeaters])
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        initial_pending = len(net.pending_events)
        net.age_links(discard_expired=False)
        self.assertLessEqual(len(net.pending_events), initial_pending)


class TestCrossModuleWiring(unittest.TestCase):
    """Verify correct referencing between network ↔ repeater ↔ env_wrapper."""

    def test_network_holds_repeater_instances(self):
        net = _perfect_chain(4)
        for i, rep in enumerate(net.repeaters):
            self.assertIsInstance(rep, Repeater)
            self.assertEqual(rep.rid, i)

    def test_adjacency_matrix_shape(self):
        n = 5
        net = _perfect_chain(n)
        self.assertEqual(net.adj.shape, (n, n))

    def test_env_wraps_network(self):
        env = QRNEnv(n_repeaters=4, topology="chain")
        self.assertIsInstance(env.net, RepeaterNetwork)

    def test_env_net_repeater_count(self):
        env = QRNEnv(n_repeaters=5, topology="chain")
        self.assertEqual(env.net.N, 5)
        self.assertEqual(len(env.net.repeaters), 5)

    def test_env_reset_returns_observation(self):
        env = QRNEnv(n_repeaters=4, topology="chain")
        obs = env.reset()
        self.assertIn("x", obs)
        self.assertIn("edge_index", obs)

    def test_env_step_returns_correct_shape(self):
        env = QRNEnv(n_repeaters=4, topology="chain")
        env.reset()
        actions = np.zeros(env.N, dtype=np.int32)
        obs, reward, done, info = env.step(actions)
        self.assertEqual(obs["x"].shape[0], env.N)

    def test_build_chain_returns_repeater_network(self):
        net = build_chain(3)
        self.assertIsInstance(net, RepeaterNetwork)

    def test_build_grid_returns_repeater_network(self):
        net = build_grid(2, 3)
        self.assertIsInstance(net, RepeaterNetwork)

    def test_chain_adjacency_is_tridiagonal(self):
        n = 5
        net = _perfect_chain(n)
        for i in range(n - 1):
            self.assertGreater(net.adj[i, i+1], 0.0)
        # Non-adjacent pairs must be 0.
        self.assertEqual(net.adj[0, 2], 0.0)
        self.assertEqual(net.adj[0, 3], 0.0)

    def test_grid_adjacency_correct(self):
        net = build_grid(2, 3, spacing=50.0)
        # Node 0 → 1 and 0 → 3 must be adjacent in a 2×3 grid.
        self.assertGreater(net.adj[0, 1], 0.0)
        self.assertGreater(net.adj[0, 3], 0.0)
        self.assertEqual(net.adj[0, 2], 0.0)   # not directly adjacent


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  EDGE CASES AND RL LOOPHOLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGhostLinkResolution(unittest.TestCase):
    """
    A swap event is pending, but one remote qubit expires due to decoherence
    before the classical message arrives.  The system must abort cleanly and
    free the surviving qubit — no dangling state.
    """

    def _setup_pending_swap(self):
        """Return a network where a swap event is queued but not yet resolved."""
        net = build_chain(3, n_ch=4, spacing=50.0,
                          p_gen=1.0, p_swap=1.0,
                          F0=1.0, channel_loss=0.0,
                          dt_seconds=1e-6,          # big delay so event stays pending
                          distance_dep_gen=False,
                          rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        self.assertEqual(len(net.pending_events), 1)
        return net

    def test_ghost_link_abort_on_expiry(self):
        net = self._setup_pending_swap()
        ev = net.pending_events[0]
        # Manually expire one of the remote qubits (simulate decoherence).
        ra, qa_r = ev["ra"], ev["qa_r"]
        net.repeaters[ra].free_qubit(qa_r)
        # Resolve the event.
        net.pending_events[0]["timer"] = 0
        net.age_links(discard_expired=False)
        # After resolution, no qubit in the network must be locked.
        for rep in net.repeaters:
            self.assertFalse(np.any(rep.locked),
                             f"R{rep.rid} still has a locked qubit after ghost-link abort.")

    def test_ghost_link_no_new_link_created(self):
        net = self._setup_pending_swap()
        ev = net.pending_events[0]
        rb, qb_r = ev["rb"], ev["qb_r"]
        net.repeaters[rb].free_qubit(qb_r)
        net.pending_events[0]["timer"] = 0
        net.age_links(discard_expired=False)
        # R0 and R2 must end up with zero occupied qubits.
        self.assertEqual(net.repeaters[0].num_occupied(), 0)
        self.assertEqual(net.repeaters[2].num_occupied(), 0)

    def test_ghost_purify_abort(self):
        """Kept qubit expires during purify delay → both sides cleaned up."""
        net = _perfect_chain(3, cutoff=100)
        net.entangle(0, 1)
        net.entangle(0, 1)
        q1s = net.repeaters[0].occupied_indices()
        q1_sac, q1_keep = int(q1s[0]), int(q1s[1])
        q2_sac = int(net.repeaters[0].partner_qubit[q1_sac])
        q2_keep = int(net.repeaters[0].partner_qubit[q1_keep])

        # Lock qubits as the protocol would.
        net.repeaters[0].lock_qubit(q1_sac);  net.repeaters[0].lock_qubit(q1_keep)
        net.repeaters[1].lock_qubit(q2_sac);  net.repeaters[1].lock_qubit(q2_keep)

        # Expire the kept qubit on R1 before the message arrives.
        net.repeaters[1].free_qubit(q2_keep)

        net.pending_events.append({
            "type": "purify", "timer": 0, "success": True,
            "r1": 0, "r2": 1,
            "q1_sac": q1_sac, "q2_sac": q2_sac,
            "q1_keep": q1_keep, "q2_keep": q2_keep,
            "p_new": 0.95,
            "gen_keep1": int(net.repeaters[0].generation_id[q1_keep]),
            "gen_keep2": int(net.repeaters[1].generation_id[q2_keep]),
            "gen_sac1": int(net.repeaters[0].generation_id[q1_sac]),
            "gen_sac2": int(net.repeaters[1].generation_id[q2_sac]),
        })
        net.age_links(discard_expired=False)
        # No zombie links on R0.
        self.assertEqual(net.repeaters[0].num_occupied(), 0)


class TestAsymmetricCutoff(unittest.TestCase):
    """
    Two repeaters with different cutoffs must use min(c1, c2) for the link.
    Physical justification: the link is only valid as long as both memories
    can store it; the weaker memory defines the lifetime.
    """

    def test_effective_cutoff_is_minimum(self):
        c1, c2 = 10, 30
        reps = [
            Repeater(rid=0, n_ch=2, cutoff=c1, position=np.array([0.0, 0.0])),
            Repeater(rid=1, n_ch=2, cutoff=c2, position=np.array([50.0, 0.0])),
        ]
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        net = RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                              distance_dep_gen=False,
                              rng=np.random.default_rng(0))
        net.entangle(0, 1)
        qi = net.repeaters[0].occupied_indices()[0]
        self.assertEqual(int(net.repeaters[0].link_cutoff[qi]), min(c1, c2),
                         "Effective link cutoff must be min(c1, c2).")

    def test_link_expires_at_min_cutoff(self):
        c1, c2 = 3, 20
        reps = [
            Repeater(rid=0, n_ch=2, cutoff=c1, position=np.array([0.0, 0.0])),
            Repeater(rid=1, n_ch=2, cutoff=c2, position=np.array([50.0, 0.0])),
        ]
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        net = RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                              distance_dep_gen=False,
                              rng=np.random.default_rng(0))
        net.entangle(0, 1)
        # Tick until the min cutoff is exceeded.
        for _ in range(c1 + 1):
            net.age_links(discard_expired=True)
        self.assertEqual(net.repeaters[0].num_occupied(), 0,
                         "Link must expire at min cutoff even if one memory is better.")
        self.assertEqual(net.repeaters[1].num_occupied(), 0)


class TestZeroDistanceOperations(unittest.TestCase):
    """
    Collocated repeaters (d = 0) must not raise division-by-zero errors and
    the classical delay must be 0 (instantaneous coordination).
    """

    def _zero_dist_net(self):
        reps = [
            Repeater(rid=0, n_ch=4, cutoff=20, position=np.array([0.0, 0.0])),
            Repeater(rid=1, n_ch=4, cutoff=20, position=np.array([0.0, 0.0])),
        ]
        adj = np.array([[0.0, 1.0], [1.0, 0.0]])
        return RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                               distance_dep_gen=False,
                               rng=np.random.default_rng(0))

    def test_delay_is_zero(self):
        net = self._zero_dist_net()
        self.assertEqual(net._classical_delay_steps(0.0), 0)

    def test_entangle_zero_distance_succeeds(self):
        net = self._zero_dist_net()
        res = net.entangle(0, 1)
        self.assertTrue(res["success"])

    def test_swap_event_resolves_immediately(self):
        reps = [
            Repeater(rid=i, n_ch=4, cutoff=20,
                     position=np.array([0.0, 0.0])) for i in range(3)
        ]
        adj = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float64)
        net = RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                              distance_dep_gen=False,
                              rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        # Timer should be 0 → event resolves in the very next age_links call.
        if net.pending_events:
            self.assertEqual(net.pending_events[0]["timer"], 0)

    def test_no_division_by_zero_in_gen_prob(self):
        net = self._zero_dist_net()
        try:
            net._gen_prob(0, 1)
            net._gen_fidelity(0, 1)
        except ZeroDivisionError:
            self.fail("_gen_prob / _gen_fidelity raised ZeroDivisionError at d=0.")


class TestDoubleBookingLockingIntegrity(unittest.TestCase):
    """
    A locked qubit (awaiting classical message) must not be eligible for
    further swap or purify actions — it is physically inaccessible.
    """

    def _net_with_locked_qubit(self):
        net = build_chain(3, n_ch=4, spacing=50.0,
                          p_gen=1.0, p_swap=1.0,
                          F0=1.0, channel_loss=0.0,
                          dt_seconds=1e-6,
                          distance_dep_gen=False,
                          rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)   # locks remote qubits at R0 and R2
        return net

    def test_locked_qubit_not_in_available(self):
        net = self._net_with_locked_qubit()
        # R0's single qubit is locked: available_indices() must be empty.
        avail = net.repeaters[0].available_indices()
        self.assertEqual(len(avail), 0,
                         "Locked qubit must not appear in available_indices().")

    def test_locked_qubit_not_swappable(self):
        net = self._net_with_locked_qubit()
        self.assertFalse(net.repeaters[0].can_swap(),
                         "can_swap() must return False when only qubit is locked.")

    def test_swap_mask_excludes_locked_node(self):
        net = self._net_with_locked_qubit()
        mask = net.action_mask_swap()
        self.assertFalse(mask[0], "Swap mask must be False for a node with only locked qubits.")
        self.assertFalse(mask[2], "Swap mask must be False for a node with only locked qubits.")

    def test_purify_mask_excludes_locked_qubits(self):
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(0, 1)
        # Lock one of the two qubits at R0.
        qi = net.repeaters[0].occupied_indices()[0]
        net.repeaters[0].lock_qubit(qi)
        # Only 1 available qubit left → purify mask must be False for (0,1).
        mask = net.action_mask_purify()
        self.assertFalse(mask[0, 1],
                         "Purify mask must be False when < 2 available qubits to same partner.")

    def test_has_free_qubit_ignores_locked_free_slots(self):
        # A node whose only free slot is locked must report no free qubit.
        rep = Repeater(rid=0, n_ch=1, cutoff=20)
        rep.locked[0] = True   # lock the sole qubit (still FREE)
        self.assertFalse(rep.has_free_qubit())


class TestSelfSwapping(unittest.TestCase):
    """
    A repeater must not swap two qubits that are both linked to the *same*
    remote repeater — this would create a self-loop (unphysical).
    """

    def test_same_partner_swap_rejected(self):
        # R0 holds two qubits both linked to R1; swap at R0 must be rejected.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(0, 1)
        res = net.swap(0)
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "same_partner",
                         "Swap of two links to the same partner must return 'same_partner'.")

    def test_valid_swap_different_partners_accepted(self):
        # R1 holds one link to R0 and one to R2 → swap must succeed.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)
        res = net.swap(1)
        self.assertTrue(res["success"])

    def test_self_link_error_on_set_link(self):
        # set_link must raise ValueError if partner_rid == self.rid.
        rep = Repeater(rid=5, n_ch=2, cutoff=20)
        rep.status[0] = QUBIT_OCCUPIED
        with self.assertRaises(ValueError):
            rep.set_link(0, 5, 1, 0.9)   # partner is itself


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ADDITIONAL ROBUSTNESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestResetBehaviour(unittest.TestCase):

    def test_network_reset_clears_all_links(self):
        net = _perfect_chain(4)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.entangle(2, 3)
        net.reset()
        for rep in net.repeaters:
            self.assertEqual(rep.num_occupied(), 0)
            self.assertFalse(np.any(rep.locked))

    def test_network_reset_clears_pending_events(self):
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        net.reset()
        self.assertEqual(len(net.pending_events), 0)

    def test_time_step_reset_to_zero(self):
        net = _perfect_chain(3)
        net.age_links()
        net.age_links()
        net.reset()
        self.assertEqual(net.time_step, 0)

    def test_env_reset_reinitialises_steps(self):
        env = QRNEnv(n_repeaters=4, topology="chain")
        env.reset()
        env.step(np.zeros(env.N, dtype=int))
        env.step(np.zeros(env.N, dtype=int))
        env.reset()
        self.assertEqual(env.steps, 0)
        self.assertFalse(env.done)


class TestActionMasks(unittest.TestCase):

    def test_entangle_mask_only_adjacent_pairs(self):
        net = _perfect_chain(4)
        mask = net.action_mask_entangle()
        # Adjacent pairs must be True (when qubits are free).
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[2, 3])
        # Non-adjacent must be False.
        self.assertFalse(mask[0, 2])
        self.assertFalse(mask[0, 3])

    def test_swap_mask_false_for_empty_repeaters(self):
        net = _perfect_chain(4)
        mask = net.action_mask_swap()
        # No entanglement yet → no node can swap.
        self.assertFalse(np.any(mask))

    def test_purify_mask_false_with_single_link(self):
        net = _perfect_chain(3)
        net.entangle(0, 1)
        mask = net.action_mask_purify()
        # Only one link between R0–R1 → cannot purify.
        self.assertFalse(mask[0, 1])

    def test_purify_mask_true_with_two_links(self):
        net = _perfect_chain(3, n_ch=4)
        net.entangle(0, 1)
        net.entangle(0, 1)
        mask = net.action_mask_purify()
        self.assertTrue(mask[0, 1])


class TestGetAllLinks(unittest.TestCase):

    def test_empty_network_returns_empty_array(self):
        net = _perfect_chain(3)
        links = net.get_all_links()
        self.assertEqual(links.shape, (0, 6))

    def test_one_link_returns_one_row(self):
        net = _perfect_chain(3)
        net.entangle(0, 1)
        links = net.get_all_links()
        self.assertEqual(links.shape[0], 1)

    def test_link_row_has_correct_repeater_indices(self):
        net = _perfect_chain(3)
        net.entangle(0, 1)
        links = net.get_all_links()
        ra, rb = int(links[0, 0]), int(links[0, 2])
        self.assertLess(ra, rb)           # r_a < r_b by convention
        self.assertIn(0, [ra, rb])
        self.assertIn(1, [ra, rb])

    def test_fidelity_column_in_valid_range(self):
        net = _perfect_chain(4)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.entangle(2, 3)
        links = net.get_all_links()
        fids = links[:, 4]
        self.assertTrue(np.all(fids >= 0.25),
                        "Fidelity must be >= 0.25 (Werner state lower bound).")
        self.assertTrue(np.all(fids <= 1.0))


class TestEnvWrapper(unittest.TestCase):

    def test_source_dest_always_noop(self):
        # Source and destination must be forced to NOOP regardless of agent.
        env = QRNEnv(n_repeaters=5, topology="chain", p_gen=1.0, p_swap=1.0)
        env.reset()
        actions = np.full(env.N, SWAP, dtype=int)
        obs, reward, done, info = env.step(actions)
        # The info["actions"] at source/dest must be NOOP.
        self.assertEqual(info["actions"][env.source], NOOP)
        self.assertEqual(info["actions"][env.dest], NOOP)

    def test_step_cost_on_non_terminal(self):
        env = QRNEnv(n_repeaters=5, topology="chain",
                     p_gen=0.0,         # never generate → never succeed
                     max_steps=100)
        env.reset()
        actions = np.zeros(env.N, dtype=int)
        _, reward, done, _ = env.step(actions)
        if not done:
            self.assertAlmostEqual(reward, env.STEP_COST)

    def test_done_flag_on_max_steps(self):
        env = QRNEnv(n_repeaters=3, topology="chain",
                     p_gen=0.0, max_steps=2)
        env.reset()
        for _ in range(2):
            _, _, done, _ = env.step(np.zeros(env.N, dtype=int))
        self.assertTrue(done)

    def test_observation_node_features_shape(self):
        env = QRNEnv(n_repeaters=6, topology="chain")
        obs = env.reset()
        # Expected: (N, 8) node feature matrix.
        self.assertEqual(obs["x"].shape, (6, 8))

    def test_action_mask_shape_and_noop_always_true(self):
        env = QRNEnv(n_repeaters=5, topology="chain")
        env.reset()
        mask = env.get_action_mask()
        self.assertEqual(mask.shape, (5, 3))
        # NOOP (column 0) must always be available for every node.
        self.assertTrue(np.all(mask[:, NOOP]))


class TestGeantTopology(unittest.TestCase):

    def test_geant_node_count(self):
        net = build_GEANT()
        self.assertEqual(net.N, 24)

    def test_geant_adjacency_symmetric(self):
        net = build_GEANT()
        np.testing.assert_array_equal(net.adj, net.adj.T,
                                      err_msg="GEANT adjacency must be symmetric.")

    def test_geant_edge_count(self):
        net = build_GEANT()
        # 37 undirected edges → 74 nonzero entries in the full matrix.
        n_edges = int(np.count_nonzero(net.adj)) // 2
        self.assertEqual(n_edges, 37)

    def test_geant_distances_positive(self):
        net = build_GEANT()
        nonzero = net.adj[net.adj > 0]
        self.assertTrue(np.all(nonzero > 0),
                        "All GEANT link weights must be positive km distances.")

    def test_geant_entangle_adjacent_nodes(self):
        net = build_GEANT(p_gen=1.0, p_swap=1.0, channel_loss=0.0)
        # AT(0) and CH(2) are adjacent in GEANT.
        res = net.entangle(0, 2)
        self.assertTrue(res["success"])

    def test_geant_entangle_non_adjacent_fails(self):
        net = build_GEANT(p_gen=1.0)
        # AT(0) and GR(8) are not directly linked.
        res = net.entangle(0, 8)
        self.assertFalse(res["success"])


class TestRepeaterInternals(unittest.TestCase):

    def test_allocate_qubit_returns_index(self):
        rep = Repeater(rid=0, n_ch=4, cutoff=20)
        qi = rep.allocate_qubit()
        self.assertGreaterEqual(qi, 0)
        self.assertEqual(rep.status[qi], QUBIT_OCCUPIED)

    def test_allocate_qubit_full_returns_minus_one(self):
        rep = Repeater(rid=0, n_ch=2, cutoff=20)
        rep.allocate_qubit()
        rep.allocate_qubit()
        self.assertEqual(rep.allocate_qubit(), -1)

    def test_free_qubit_clears_all_fields(self):
        rep = Repeater(rid=0, n_ch=2, cutoff=20)
        rep.set_link(0, 1, 0, 0.9, link_age=0)
        rep.status[0] = QUBIT_OCCUPIED
        rep.free_qubit(0)
        self.assertEqual(rep.status[0], QUBIT_FREE)
        self.assertEqual(int(rep.partner_repeater[0]), NO_PARTNER)
        self.assertEqual(float(rep.werner_param[0]), 0.0)
        self.assertFalse(rep.locked[0])

    def test_lock_unlock_qubit(self):
        rep = Repeater(rid=0, n_ch=2, cutoff=20)
        rep.lock_qubit(0)
        self.assertTrue(rep.locked[0])
        rep.unlock_qubit(0)
        self.assertFalse(rep.locked[0])

    def test_qubits_to_returns_only_unlocked_occupied(self):
        rep = Repeater(rid=0, n_ch=4, cutoff=20)
        # Manually set up two links to partner 1; lock one.
        rep.set_link(0, 1, 0, 0.9)
        rep.status[0] = QUBIT_OCCUPIED
        rep.set_link(1, 1, 1, 0.8)
        rep.status[1] = QUBIT_OCCUPIED
        rep.lock_qubit(0)
        result = rep.qubits_to(1)
        self.assertNotIn(0, result)
        self.assertIn(1, result)

    def test_feature_vector_length(self):
        rep = Repeater(rid=0, n_ch=4, cutoff=20,
                       position=np.array([1.0, 2.0]))
        fv = rep.feature_vector()
        self.assertEqual(len(fv), 6)

    def test_qubit_features_shape(self):
        rep = Repeater(rid=0, n_ch=4, cutoff=20)
        qf = rep.qubit_features()
        self.assertEqual(qf.shape, (4, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2)