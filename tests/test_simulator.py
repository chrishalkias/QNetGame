"""
--------------------------------------------------------------------------------
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
    PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_simulator.py -v
--------------------------------------------------------------------------------
"""

import math
import unittest
from unittest.mock import patch
import numpy as np
import pytest

# -- imports ------------------------------------------------------------------
from simulator.repeater import (
    Repeater, SwapPolicy,
    QUBIT_FREE, QUBIT_OCCUPIED, NO_PARTNER, LEFT, RIGHT,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_fidelity,
)
from simulator.network import (
    RepeaterNetwork, build_chain, build_network,
)
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY


# -- tiny helpers -------------------------------------------------------------

def _perfect_chain(n, n_ch=4, cutoff=20, spacing=50.0):
    """Build a deterministic chain: p_gen=1, p_swap=1, no channel loss."""
    return build_chain(
        n, n_ch=n_ch, spacing=spacing,
        p_gen=1.0, p_swap=1.0, cutoff=cutoff,
        F0=1.0, channel_loss=0.0,
        distance_dep_gen=False,
        rng=np.random.default_rng(0),
    )


def _entangle_force(net, r1, r2):
    """Guarantee entanglement regardless of RNG by patching p_gen temporarily."""
    net.repeaters[r1].p_gen = 1.0
    net.repeaters[r2].p_gen = 1.0
    res = net.entangle(r1, r2)
    return res


_REP_MUT_ARRAYS = ("status", "partner_repeater", "partner_qubit", "werner_param",
                   "initial_werner", "age", "link_cutoff", "locked",
                   "position")


def _chain_with_links(n_ch=4, cutoff=50):
    """3-node chain: B (rid=1) holds one LEFT link to A (rid=0) and one RIGHT
    link to C (rid=2)."""
    net = _perfect_chain(3, n_ch=n_ch, cutoff=cutoff)
    net.entangle(0, 1)
    net.entangle(1, 2)
    return net


def test_swap_applies_immediately_no_pending():
    # two adjacent links A-B, B-C; swap at B creates A-C now, before age_links
    net = _chain_with_links()
    rep0 = net.repeaters[0]
    qA = int(rep0.occupied_indices()[0])
    res = net.swap(1)
    assert res["success"]
    # A now points at C immediately, no age_links call
    assert int(rep0.partner_repeater[qA]) == 2
    assert not hasattr(net, "pending_events")


def test_swap_value_is_product_and_sum_ages():
    # value == w_A*w_B at creation; inherited age == age_A+age_B; single decoherence
    tau = 50
    net = _chain_with_links(cutoff=tau)
    for _ in range(3):
        net.age_links(discard_expired=False)   # age both links to 3 each
    rep0, rep2 = net.repeaters[0], net.repeaters[2]
    qA = int(rep0.occupied_indices()[0])
    qC = int(rep2.occupied_indices()[0])
    wA = float(rep0.werner_param[qA])
    wC = float(rep2.werner_param[qC])
    res = net.swap(1)
    assert res["success"]
    assert abs(float(rep0.werner_param[qA]) - wA * wC) < 1e-6
    assert int(rep0.age[qA]) == 6
    # after ONE age_links, decays once from the summed-age baseline
    base = float(rep0.initial_werner[qA])
    net.age_links(discard_expired=False)
    expected = base * math.exp(-7 / tau)
    assert abs(float(rep0.werner_param[qA]) - expected) < 1e-6


def test_purify_applies_immediately():
    # survivor upgraded / sacrificed destroyed immediately; no pending_events
    net = build_chain(3, n_ch=4, p_gen=1.0, p_swap=1.0, cutoff=1000,
                      F0=1.0, channel_loss=0.0, distance_dep_gen=False,
                      rng=np.random.default_rng(0))
    rep0, rep1 = net.repeaters[0], net.repeaters[1]
    for f in (0.8, 0.8):
        q0 = rep0.allocate_qubit(RIGHT)
        q1 = rep1.allocate_qubit(LEFT)
        p = float(fidelity_to_werner(f))
        rep0.set_link(q0, 1, q1, p, link_age=0, effective_cutoff=1000)
        rep1.set_link(q1, 0, q0, p, link_age=0, effective_cutoff=1000)

    class _ZeroRNG:
        def random(self):
            return 0.0
    net.rng = _ZeroRNG()

    res = net.purify(0, 1)
    assert res["success"]
    assert rep0.num_occupied() == 1        # sacrificed destroyed immediately
    assert not hasattr(net, "pending_events")
    occ = int(rep0.occupied_indices()[0])
    assert float(werner_to_fidelity(rep0.werner_param[occ])) > 0.8   # survivor upgraded


def test_age_links_no_resolution_keys():
    net = _perfect_chain(3)
    out = net.age_links()
    assert "resolved_count" not in out and "pending_count" not in out


def test_repeater_deepcopy_isolation():
    """Repeater.__deepcopy__ (the exact-DP fast clone) must reproduce every
    mutable array and give the clone independent storage: mutating the clone must
    not touch the original. Guards the optimization behind build_kernel."""
    import copy
    net = _perfect_chain(4, n_ch=2, cutoff=20)
    _entangle_force(net, 1, 2)          # populate arrays with real link state
    rep = net.repeaters[1]
    clone = copy.deepcopy(rep)
    # faithful copy
    for name in _REP_MUT_ARRAYS:
        np.testing.assert_array_equal(getattr(clone, name), getattr(rep, name))
    assert clone.rid == rep.rid and clone.p_swap == rep.p_swap
    # independent storage: every mutable array is a distinct object
    for name in _REP_MUT_ARRAYS:
        assert getattr(clone, name) is not getattr(rep, name), name
    # mutating the clone leaves the original untouched
    before = rep.age.copy()
    clone.age += 7
    clone.status[:] = 0
    np.testing.assert_array_equal(rep.age, before)
    assert rep.status.any()             # original link still occupied


                                            
# ▄▄▄▄▄▄▄   ▄▄                                
# ███▀▀███▄ ██                ▀▀              
# ███▄▄███▀ ████▄ ██ ██ ▄█▀▀▀ ██  ▄████ ▄█▀▀▀ 
# ███▀▀▀▀   ██ ██ ██▄██ ▀███▄ ██  ██    ▀███▄ 
# ███       ██ ██  ▀██▀ ▄▄▄█▀ ██▄ ▀████ ▄▄▄█▀ 
#                   ██                        
#                 ▀▀▀                         

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

    Success prob (in fidelities): P_suc = (8*F1*F2 - 2*(F1+F2) + 5)/9
      equivalently in Werner parameters P_suc = (p1*p2 + 1)/2.
    New Werner:   p_new = (1-(p1+p2)+10*p1*p2) / (5-2*(p1+p2)+8*p1*p2)
    """

    def test_success_prob_identical_states(self):
        # Canonical BBPSSW success rate for two identical F=0.9 Werner pairs.
        # In fidelities: (8*F1*F2 - 2*(F1+F2) + 5)/9; for F1=F2=f this is
        # (8*f*f - 4*f + 5)/9 = (8*0.81 - 3.6 + 5)/9 = 7.88/9 = 0.875556.
        f = 0.9
        expected = (8*f*f - 4*f + 5)/9
        self.assertAlmostEqual(float(bbpssw_success_prob(f, f)), expected, places=9)

    def test_success_prob_canonical_anchor_half(self):
        # At F1=F2=0.5 (fully separable Werner p=1/3) the canonical BBPSSW rate
        # is 5/9. The old non-canonical (3*p1*p2+1)/4 map gives 1/3 here.
        self.assertAlmostEqual(float(bbpssw_success_prob(0.5, 0.5)), 5/9, places=12)

    def test_success_prob_equals_werner_form(self):
        # Canonical rate in Werner params: (p1*p2 + 1)/2. Cross-check the
        # fidelity-domain function against it over several inputs.
        for f1, f2 in [(0.5, 0.5), (0.7, 0.9), (0.6, 0.85), (1.0, 1.0)]:
            p1, p2 = fidelity_to_werner(f1), fidelity_to_werner(f2)
            expected = (float(p1)*float(p2) + 1)/2
            self.assertAlmostEqual(float(bbpssw_success_prob(f1, f2)),
                                   expected, places=12)

    def test_success_prob_is_ninth_of_new_fidelity_denominator(self):
        # Internal-consistency: the success rate must be exactly 1/9 of the
        # denominator of bbpssw_new_fidelity (both come from one twirled
        # BBPSSW density matrix). D = 5 - 2*(F1+F2) + 8*F1*F2.
        for f1, f2 in [(0.5, 0.5), (0.7, 0.9), (0.6, 0.85)]:
            denom = 5 - 2*(f1 + f2) + 8*f1*f2
            self.assertAlmostEqual(float(bbpssw_success_prob(f1, f2)),
                                   denom/9, places=12)

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
        # Swap applies immediately: the new link exists right away.

        rep0 = net.repeaters[0]
        occupied = rep0.occupied_indices()
        self.assertTrue(len(occupied) > 0, "R0 should hold the new link.")
        actual_initial_p = float(rep0.initial_werner[occupied[0]])
        self.assertAlmostEqual(actual_initial_p, expected_p_new, places=5)

    def test_swap_resolution_single_counts_decoherence(self):
        # HIGH-1 (swap): the resolved link's value must be exactly the product
        # of the two links' decohered Werner values at resolution time, with
        # NO extra decay factor re-applied by set_link.
        # Setup: two links p0=1 aged to exactly 5 ticks each, tau=cutoff=20.
        #   each decohered value w = e^(-5/20); product = e^(-10/20) ~ 0.6065.
        #   Resolved link age must be age_A + age_B = 10.
        # Old (buggy) code re-applied max(age)=5 on top: e^(-10/20)*e^(-5/20)
        #   = e^(-15/20) ~ 0.4724, and stored age 5.
        tau = 20
        net = _perfect_chain(3, cutoff=tau)
        net.entangle(0, 1)
        net.entangle(1, 2)
        for _ in range(5):                       # age both links to exactly 5
            net.age_links(discard_expired=False)
        res = net.swap(1)
        self.assertTrue(res["success"])
        # Swap applies immediately (no intervening age tick), so ages stay 5 + 5.
        rep0 = net.repeaters[0]
        occ = rep0.occupied_indices()
        self.assertEqual(len(occ), 1, "R0 should hold exactly the new link.")
        q = int(occ[0])
        self.assertAlmostEqual(float(rep0.werner_param[q]),
                               math.exp(-10 / tau), places=5)
        self.assertEqual(int(rep0.age[q]), 10)

    def test_swap_resolution_future_decay_from_resolution_value(self):
        # HIGH-1 (swap) invariant (c): k ticks after resolution the value is
        # (resolution value)*exp(-k/tau). resolution value = e^(-10/20); after
        # 3 more ticks -> e^(-13/20).
        tau = 20
        net = _perfect_chain(3, cutoff=tau)
        net.entangle(0, 1)
        net.entangle(1, 2)
        for _ in range(5):
            net.age_links(discard_expired=False)
        net.swap(1)
        rep0 = net.repeaters[0]
        q = int(rep0.occupied_indices()[0])
        for _ in range(3):
            rep0.age_occupied()
        self.assertAlmostEqual(float(rep0.werner_param[q]),
                               math.exp(-13 / tau), places=5)

    def test_local_qubits_freed_after_bsm(self):
        # BSM physically consumes both local qubits immediately.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)
        # R1 should have no occupied qubits right after the BSM.
        self.assertEqual(net.repeaters[1].num_occupied(), 0,
                         "BSM must destroy local qubits instantly.")


class TestPurifyResolutionDecoherence(unittest.TestCase):
    """Task 3 (arXiv 2401.13168 Eq. (4)): the resolved kept-pair value must
    equal the BBPSSW output of the decision-time fidelities (ev['p_new'])
    represented as an age on a FRESH (p0=1) baseline: m' = ceil(-tau*ln(p_new))
    plus ticks accrued since the decision. This replaces the old sum-of-
    endpoint-ages bookkeeping (age_A + age_B, which doubled the expiry clock)
    and its back-solved baseline (which could exceed 1)."""

    def test_purify_resolution_single_counts_decoherence(self):
        # Two parallel links between R0 and R1, both p0=1 aged to 5, tau=20.
        # Decision fidelities come from w=e^(-5/20); p_new = BBPSSW output in
        # Werner. Resolving directly (no intervening age_links call) means
        # zero ticks accrue since the decision, so the resolved age is just
        # m' = ceil(-tau*ln(p_new)) and the resolved werner is exp(-m'/tau)
        # on a fresh p0=1 baseline (NOT p_new itself, and NOT age_A+age_B=10).
        tau = 20
        net = build_chain(2, n_ch=4, spacing=50.0,
                          p_gen=1.0, p_swap=1.0, cutoff=tau,
                          F0=1.0, channel_loss=0.0,                          distance_dep_gen=False, rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(0, 1)
        for _ in range(5):
            net.age_links(discard_expired=False)

        # Force the BBPSSW coin to succeed deterministically (rng.random()==0).
        class _ZeroRNG:
            def random(self):
                return 0.0
        net.rng = _ZeroRNG()

        # Both links are identical (same decohered w), so the decision-time
        # p_new is exactly the BBPSSW output of that fidelity with itself.
        rep0 = net.repeaters[0]
        w_before = float(rep0.werner_param[rep0.occupied_indices()[0]])
        f_before = float(werner_to_fidelity(w_before))
        f_new = float(bbpssw_new_fidelity(f_before, f_before))
        p_new = float(fidelity_to_werner(f_new))

        # Purify applies immediately (two equal F<1 links -> beneficial,
        # forced success -> the survivor is purified, not merely kept).
        res = net.purify(0, 1)
        self.assertTrue(res["success"])
        occ = rep0.occupied_indices()
        self.assertEqual(len(occ), 1, "R0 should hold exactly the kept pair.")
        q = int(occ[0])
        m_equiv = int(math.ceil(-tau * math.log(p_new)))
        self.assertEqual(int(rep0.age[q]), m_equiv)
        self.assertAlmostEqual(float(rep0.initial_werner[q]), 1.0, places=9)
        self.assertAlmostEqual(float(rep0.werner_param[q]),
                               math.exp(-m_equiv / tau), places=5)
        # invariant (c): 3 more ticks -> exp(-(m_equiv+3)/tau).
        for _ in range(3):
            rep0.age_occupied()
        self.assertAlmostEqual(float(rep0.werner_param[q]),
                               math.exp(-(m_equiv + 3) / tau), places=5)


class TestDistanceDependency(unittest.TestCase):
    """
    Generation probability: p_eff = p_avg * exp(-loss * d / 2)
    Initial fidelity:       F0_eff = F0 * exp(-loss * d)
    """

    def _make_two_node(self, spacing, loss):
        return build_chain(2, n_ch=4, spacing=spacing,
                           p_gen=1.0, p_swap=1.0,
                           F0=1.0, channel_loss=loss,
                           distance_dep_gen=True,
                           rng=np.random.default_rng(0))

    def test_gen_prob_scaling(self):
        loss, d = 0.02, 50.0
        net = self._make_two_node(d, loss)
        expected = 1.0 * math.exp(-loss * d / 2)
        self.assertAlmostEqual(net._gen_prob(0, 1), expected, places=6)

    def test_initial_fidelity_scaling(self):
        # Depolarizing loss damps the Werner parameter: p0 = w(F0)*exp(-loss*d),
        # reported as F = werner_to_fidelity(p0). At F0=1 the baseline w(1)=1, so
        # p0 = exp(-loss*d) and F = (3*exp(-loss*d) + 1)/4.
        # d=50, loss=0.02: exp(-1)=0.3678794; F=(3*0.3678794+1)/4=0.5259096.
        loss, d = 0.02, 50.0
        net = self._make_two_node(d, loss)
        expected_fid = (3 * math.exp(-loss * d) + 1) / 4
        self.assertAlmostEqual(net._gen_fidelity(0, 1), expected_fid, places=6)

    def test_zero_loss_unity_fidelity(self):
        net = self._make_two_node(50.0, 0.0)
        self.assertAlmostEqual(net._gen_fidelity(0, 1), 1.0, places=9)
        self.assertAlmostEqual(net._gen_prob(0, 1), 1.0, places=9)

    def test_generated_fidelity_never_below_quarter(self):
        # HIGH-2: fiber loss is a depolarizing channel — it must damp the Werner
        # parameter p (F -> 1/4 as d -> inf), not the fidelity itself. The old
        # F0*exp(-loss*d) fell below the 1/4 mixed-state floor past ~69 km.
        loss, F0 = 0.02, 1.0
        for d in [50.0, 69.0, 100.0, 300.0, 1000.0]:
            net = self._make_two_node(d, loss)
            f = float(net._gen_fidelity(0, 1))
            self.assertGreaterEqual(f, 0.25 - 1e-12,
                                    f"generated F={f} below 1/4 floor at d={d}")

    def test_depolarizing_loss_damps_werner(self):
        # Explicit depolarizing form: p0 = w(F0)*exp(-loss*d), so the reported
        # fidelity is werner_to_fidelity(p0). At F0=1, d=100, loss=0.02:
        #   p0 = exp(-2) ~ 0.13534, F = (3*0.13534 + 1)/4 ~ 0.35150.
        loss, d, F0 = 0.02, 100.0, 1.0
        net = self._make_two_node(d, loss)
        p0 = fidelity_to_werner(F0) * math.exp(-loss * d)
        expected_f = (3 * p0 + 1) / 4
        self.assertAlmostEqual(float(net._gen_fidelity(0, 1)), expected_f, places=6)


                           
#  ▄▄▄▄▄▄▄                   
# ███▀▀▀▀▀                   
# ███      ▄███▄ ████▄ ▄█▀█▄ 
# ███      ██ ██ ██ ▀▀ ██▄█▀ 
# ▀███████ ▀███▀ ██    ▀█▄▄▄ 
                           
                           

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

    def test_swap_creates_long_range_link_immediately(self):
        res = self.net.swap(1)
        self.assertTrue(res["success"])
        # R0 should now be linked to R2 immediately, before any age_links call.
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
            # All qubits at R1 freed immediately.
            self.assertEqual(net.repeaters[1].num_occupied(), 0)

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

    def test_purify_applies_immediately(self):
        net = self._net_with_two_links()
        res = net.purify(0, 1)
        self.assertTrue(res["success"])
        # Guard-skip cascade (both F=1.0) resolves instantly: one survivor.
        self.assertEqual(net.repeaters[0].num_occupied(), 1)

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
        # Applies immediately: no age_links call needed.
        occ = net.repeaters[0].occupied_indices()
        if len(occ):
            p_after = float(net.repeaters[0].werner_param[occ[0]])
            # On success the kept pair must be better than either input.
            self.assertGreaterEqual(p_after, p_before * 0.99,
                                    "Purification must not degrade kept pair.")

    def test_purify_failure_destroys_both(self):
        # Force BBPSSW to fail (real purify -> age_links path) and verify both
        # pairs are destroyed. Links must be F<1 so the beneficial-purify guard
        # actually attempts BBPSSW (two F=1 links are kept, not purified).
        import simulator.network as netmod
        net = _perfect_chain(3, cutoff=100)
        net.entangle(0, 1)
        net.entangle(0, 1)
        for qi in net.repeaters[0].occupied_indices():
            net.repeaters[0].werner_param[qi] = fidelity_to_werner(0.8)
            net.repeaters[0].initial_werner[qi] = fidelity_to_werner(0.8)
        with patch.object(netmod, "bbpssw_success_prob", return_value=0.0):
            res = net.purify(0, 1)
        self.assertFalse(res["success"])
        # Applied immediately: both pairs must already be destroyed.
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

    def test_chain_adjacency_is_tridiagonal(self):
        n = 5
        net = _perfect_chain(n)
        for i in range(n - 1):
            self.assertGreater(net.adj[i, i+1], 0.0)
        # Non-adjacent pairs must be 0.
        self.assertEqual(net.adj[0, 2], 0.0)
        self.assertEqual(net.adj[0, 3], 0.0)


                                                           
#  ▄▄▄▄▄▄▄    ▄▄                                             
# ███▀▀▀▀▀    ██                                             
# ███▄▄    ▄████ ▄████ ▄█▀█▄   ▄████  ▀▀█▄ ▄█▀▀▀ ▄█▀█▄ ▄█▀▀▀ 
# ███      ██ ██ ██ ██ ██▄█▀   ██    ▄█▀██ ▀███▄ ██▄█▀ ▀███▄ 
# ▀███████ ▀████ ▀████ ▀█▄▄▄   ▀████ ▀█▄██ ▄▄▄█▀ ▀█▄▄▄ ▄▄▄█▀ 
#                   ██                                       
#                 ▀▀▀                                        
                                                           
#           ▄▄▄▄▄▄▄   ▄▄▄                                    
#    ▄      ███▀▀███▄ ███                                    
#    █      ███▄▄███▀ ███                                    
# ▀▀▀█▀▀▀   ███▀▀██▄  ███                                    
#    █      ███  ▀███ ████████                               
                                                           
                                                           

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
        res = net.swap(1)
        # Swap applies immediately: the fused 0-2 link exists right after the
        # call, before any age_links.
        if res["success"]:
            self.assertIn(2, net.repeaters[0].partner_repeater[
                net.repeaters[0].occupied_indices()])

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
                          distance_dep_gen=False,
                          rng=np.random.default_rng(0))
        net.entangle(0, 1)
        net.entangle(1, 2)
        net.swap(1)   # fuses R0<->R2 immediately; remote qubits are NOT locked
        return net

    def test_remote_qubit_available_immediately_after_swap(self):
        # Swap applies immediately (no deferral window), so R0's fused qubit
        # is occupied and available right away, not locked.
        net = self._net_with_locked_qubit()
        avail = net.repeaters[0].available_indices()
        self.assertEqual(len(avail), 1,
                         "Fused remote qubit must be immediately available.")

    def test_locked_qubit_not_swappable(self):
        # R0/R2 are chain endpoints (one port width 0), so they structurally
        # can never satisfy the one-LEFT-plus-one-RIGHT swap requirement,
        # independent of any locking.
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
        # A single-port node whose only free slot is locked must report no free qubit.
        rep = Repeater(rid=0, n_ch=1, cutoff=20, n_left=1, n_right=0)
        rep.locked[0] = True   # lock the sole qubit (still FREE)
        self.assertFalse(rep.has_free_qubit())


class TestSelfSwapping(unittest.TestCase):
    """
    A repeater must not swap two qubits that are both linked to the *same*
    remote repeater — this would create a self-loop (unphysical). With
    left/right ports this is structurally impossible: both links to one partner
    live on the same port, and a swap needs one LEFT + one RIGHT link.
    """

    def test_same_partner_swap_rejected(self):
        # R0 (an end node) holds two RIGHT-port links both to R1; with no LEFT
        # link a swap at R0 cannot even form a pair and is rejected.
        net = _perfect_chain(3, cutoff=50)
        net.entangle(0, 1)
        net.entangle(0, 1)
        res = net.swap(0)
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "insufficient_qubits",
                         "Same-partner links share a port, so no left x right pair exists.")

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


                                                                                  
# ▄▄▄▄▄▄▄                                                                           
# ███▀▀███▄                          ██                                ▄       ▄    
# ███▄▄███▀ ▄███▄ ████▄ ██ ██ ▄█▀▀▀ ▀██▀▀ ████▄ ▄█▀█▄ ▄█▀▀▀ ▄█▀▀▀      █       █    
# ███▀▀██▄  ██ ██ ██ ██ ██ ██ ▀███▄  ██   ██ ██ ██▄█▀ ▀███▄ ▀███▄   ▀▀▀█▀▀▀ ▀▀▀█▀▀▀ 
# ███  ▀███ ▀███▀ ██ ██ ▀██▀█ ▄▄▄█▀  ██   ██ ██ ▀█▄▄▄ ▄▄▄█▀ ▄▄▄█▀      █       █    
                                                                                  
                                                                                  

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
        # Expected: (N, 9) node feature matrix.
        self.assertEqual(obs["x"].shape, (6, 9))

    def test_action_mask_shape_and_noop_always_true(self):
        env = QRNEnv(n_repeaters=5, topology="chain")
        env.reset()
        mask = env.get_action_mask()
        self.assertEqual(mask.shape, (5, 3))
        # NOOP (column 0) must always be available for every node.
        self.assertTrue(np.all(mask[:, NOOP]))


class TestRepeaterInternals(unittest.TestCase):

    def test_allocate_qubit_returns_index(self):
        rep = Repeater(rid=0, n_ch=4, cutoff=20)
        qi = rep.allocate_qubit(RIGHT)
        self.assertGreaterEqual(qi, 0)
        self.assertEqual(rep.status[qi], QUBIT_OCCUPIED)

    def test_allocate_qubit_full_returns_minus_one(self):
        # n_ch qubits per side; fill the RIGHT port then the next RIGHT alloc fails.
        rep = Repeater(rid=0, n_ch=2, cutoff=20)
        rep.allocate_qubit(RIGHT)
        rep.allocate_qubit(RIGHT)
        self.assertEqual(rep.allocate_qubit(RIGHT), -1)

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


class TestSwapDecisionGate(unittest.TestCase):
    """Paper (arXiv 2401.13168) step 2: swapping is attempted only if the
    fused link would still be inside its cutoff at resolution
    (age_a + age_b + 2 < ec for dt=0 same-tick resolution)."""

    def _chain3(self, cutoff=10):
        net = build_network(topology='chain', n_repeaters=3, n_ch=2,
                            p_gen=1.0, p_swap=1.0, cutoff=cutoff,
                            F0=1.0, channel_loss=0.0,                            rng=np.random.default_rng(0))
        assert net.entangle(0, 1)["success"]
        assert net.entangle(1, 2)["success"]
        return net

    def test_overage_pair_is_refused(self):
        net = self._chain3(cutoff=10)
        rep1 = net.repeaters[1]
        # age both of R1's links so summed age + 2 >= cutoff: 4 + 4 + 2 = 10 >= 10
        for rep in net.repeaters:
            for q in rep.occupied_indices():
                rep.age[q] = 4
        res = net.swap(1)
        self.assertFalse(res["success"])
        self.assertEqual(res["reason"], "no_valid_pair")
        # nothing was consumed or locked
        self.assertEqual(rep1.num_occupied(), 2)
        self.assertEqual(net.repeaters[0].num_locked(), 0)
        self.assertEqual(net.repeaters[2].num_locked(), 0)

    def test_viable_pair_still_swaps(self):
        net = self._chain3(cutoff=10)
        # 3 + 3 + 2 = 8 < 10: viable
        for q in range(2):
            for rep in net.repeaters:
                if rep.status[q] == QUBIT_OCCUPIED:
                    rep.age[q] = 3
        res = net.swap(1)
        self.assertTrue(res["success"])

    def test_selection_skips_doomed_picks_viable(self):
        # node 1 has TWO pairs: one doomed (old links), one viable (fresh);
        # FARTHEST would tie, so the viability filter must decide
        net = build_network(topology='chain', n_repeaters=3, n_ch=4,
                            p_gen=1.0, p_swap=1.0, cutoff=10,
                            F0=1.0, channel_loss=0.0,                            rng=np.random.default_rng(0))
        assert net.entangle(0, 1)["success"]
        assert net.entangle(1, 2)["success"]
        # age the first pair to doom any pairing that uses either old qubit
        rep0, rep1, rep2 = net.repeaters
        old_q1 = rep1.qubits_to(0)[0]
        old_q0 = int(rep1.partner_qubit[old_q1])
        rep0.age[old_q0] = 9
        rep1.age[old_q1] = 9
        # fresh second pair on both sides
        assert net.entangle(0, 1)["success"]
        assert net.entangle(1, 2)["success"]
        res = net.swap(1)
        self.assertTrue(res["success"])
        # the doomed pair must be untouched: the aged link at node 0 still
        # exists with its age intact (the swap consumed the fresh pair)
        self.assertEqual(int(rep0.age[old_q0]), 9)
        self.assertEqual(rep0.status[old_q0], QUBIT_OCCUPIED)


class TestPorts(unittest.TestCase):
    """Left/right port split: n_ch qubits per side, 2*n_ch on interior nodes,
    n_ch on end nodes; a swap fuses one LEFT link with one RIGHT link."""

    def test_interior_capacity_is_2nch_ends_nch(self):
        net = _perfect_chain(4, n_ch=2)
        caps = [(r.n_left, r.n_right, r.status.size) for r in net.repeaters]
        self.assertEqual(caps[0],  (0, 2, 2))   # source: RIGHT only
        self.assertEqual(caps[1],  (2, 2, 4))   # interior: both ports
        self.assertEqual(caps[2],  (2, 2, 4))
        self.assertEqual(caps[-1], (2, 0, 2))   # dest: LEFT only

    def test_entangle_allocates_facing_ports(self):
        net = _perfect_chain(3, n_ch=2)
        rep1 = net.repeaters[1]
        net.entangle(0, 1)                      # R1 faces LEFT toward R0
        left = rep1.available_on_side(LEFT)
        self.assertEqual(len(left), 1)
        self.assertLess(int(left[0]), rep1.n_left)          # in LEFT block
        self.assertEqual(int(rep1.partner_repeater[left[0]]), 0)
        net.entangle(1, 2)                      # R1 faces RIGHT toward R2
        right = rep1.available_on_side(RIGHT)
        self.assertEqual(len(right), 1)
        self.assertGreaterEqual(int(right[0]), rep1.n_left)  # in RIGHT block
        self.assertEqual(int(rep1.partner_repeater[right[0]]), 2)

    def test_swap_needs_one_link_per_side(self):
        net = _perfect_chain(3, n_ch=2, cutoff=50)
        rep1 = net.repeaters[1]
        net.entangle(0, 1)                      # only a LEFT link -> no swap
        self.assertFalse(rep1.can_swap())
        self.assertFalse(net.swap(1)["success"])
        net.entangle(1, 2)                      # now a RIGHT link too -> swap ok
        self.assertTrue(rep1.can_swap())
        self.assertTrue(net.swap(1)["success"])

    def test_two_same_side_links_cannot_swap(self):
        # An end node with two same-partner (same-side) links can never swap.
        net = _perfect_chain(3, n_ch=2, cutoff=50)
        net.entangle(0, 1)
        net.entangle(0, 1)
        self.assertFalse(net.repeaters[0].can_swap())


class TestPurifyCascade(unittest.TestCase):
    """Sorted-adjacent distillation cascade (arXiv 2401.13168) with the
    beneficial-purify guard F_new(F1,F2) > max(F1,F2)."""

    def _place(self, net, fids):
        """Hand-place len(fids) links R0->R1 with the given fidelities."""
        rep0, rep1 = net.repeaters[0], net.repeaters[1]
        for f in fids:
            q0 = rep0.allocate_qubit(RIGHT)
            q1 = rep1.allocate_qubit(LEFT)
            p = float(fidelity_to_werner(f))
            rep0.set_link(q0, 1, q1, p, link_age=0, effective_cutoff=1000)
            rep1.set_link(q1, 0, q0, p, link_age=0, effective_cutoff=1000)

    def _fresh(self, n_ch=4):
        return build_chain(3, n_ch=n_ch, p_gen=1.0, p_swap=1.0, cutoff=1000,
                           F0=1.0, channel_loss=0.0, distance_dep_gen=False,
                           rng=np.random.default_rng(0))

    def test_guard_skips_when_not_beneficial(self):
        # F_new(0.95, 0.55) < 0.95 -> guard skips BBPSSW: keep the 0.95 link
        # untouched, discard the weak one, and consume NO rng draw.
        self.assertLess(float(bbpssw_new_fidelity(0.95, 0.55)), 0.95)
        net = self._fresh()
        self._place(net, [0.95, 0.55])
        net.rng = np.random.default_rng(0)
        state_before = net.rng.bit_generator.state
        net.purify(0, 1)
        self.assertEqual(net.rng.bit_generator.state, state_before)  # no coin flip
        net.age_links(discard_expired=False)
        links = net.get_all_links()
        self.assertEqual(len(links), 1)
        self.assertAlmostEqual(float(links[0, 4]), 0.95, places=1)

    def test_beneficial_purify_improves(self):
        # Two equal F=0.8 links -> beneficial; forced success raises the survivor.
        net = self._fresh()
        self._place(net, [0.8, 0.8])
        class _ZeroRNG:                     # rng.random()==0 -> always succeeds
            def random(self): return 0.0
        net.rng = _ZeroRNG()
        net.purify(0, 1)
        net.age_links(discard_expired=False)
        links = net.get_all_links()
        self.assertEqual(len(links), 1)
        self.assertGreater(float(links[0, 4]), 0.8)

    def test_cascade_leaves_at_most_one_survivor(self):
        for seed in range(8):
            net = self._fresh()
            self._place(net, [0.7, 0.72, 0.78, 0.85])
            net.rng = np.random.default_rng(seed)
            net.purify(0, 1)
            net.age_links(discard_expired=False)
            self.assertLessEqual(len(net.get_all_links()), 1)

    def test_all_fail_annihilates(self):
        # Two beneficial links, BBPSSW forced to fail -> nothing survives.
        import simulator.network as netmod
        net = self._fresh()
        self._place(net, [0.8, 0.8])
        with patch.object(netmod, "bbpssw_success_prob", return_value=0.0):
            res = net.purify(0, 1)
        self.assertFalse(res["success"])
        net.age_links(discard_expired=False)
        self.assertEqual(len(net.get_all_links()), 0)

