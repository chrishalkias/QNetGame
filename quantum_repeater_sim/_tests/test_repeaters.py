#!/usr/bin/env python3
"""Exhaustive test suite for quantum_repeater_sim with classical delays."""

from __future__ import annotations
import sys, time, itertools
import numpy as np

sys.path.insert(0, ".")
from quantum_repeater_sim import (
    Repeater, SwapPolicy, RepeaterNetwork, build_chain, build_grid,
    network_to_heterodata, fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_werner,
)
from quantum_repeater_sim.repeater import NO_PARTNER, QUBIT_OCCUPIED

SEP = "=" * 72
OK = FAIL = 0

def _ok(msg=""): global OK; OK += 1; print(f"    \u2713 {msg}") if msg else None
def _fail(msg): global FAIL; FAIL += 1; print(f"    \u2717 FAIL: {msg}")

def _sym_check(net):
    """Bilateral symmetry — skip locked qubits."""
    for rep in net.repeaters:
        for qi in rep.occupied_indices():
            if rep.locked[qi]: continue
            pr, pq = int(rep.partner_repeater[qi]), int(rep.partner_qubit[qi])
            if pr == NO_PARTNER: return False
            rem = net.repeaters[pr]
            if rem.locked[pq]: continue
            if rem.status[pq] != QUBIT_OCCUPIED: return False
            if int(rem.partner_repeater[pq]) != rep.rid: return False
            if int(rem.partner_qubit[pq]) != qi: return False
            if not np.isclose(rep.werner_param[qi], rem.werner_param[pq], atol=1e-12):
                return False
            if rep.age[qi] != rem.age[pq]: return False
    return True

def _mk(N, nc, cutoff, rng, pol=SwapPolicy.FARTHEST, sp=10.0, **kw):
    reps = [Repeater(rid=i, n_ch=nc, swap_policy=pol,
                     position=np.array([i*sp, 0.0]),
                     p_gen=rng.uniform(0.01,1.0),
                     p_swap=rng.uniform(0.01,1.0), cutoff=cutoff)
            for i in range(N)]
    adj = np.zeros((N,N), dtype=np.float64)
    for i in range(N-1): adj[i,i+1]=adj[i+1,i]=1.0
    return RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                           distance_dep_gen=False, rng=rng, **kw)

# ══════════════════════════════════════════════════════════════════
# ORIGINAL TESTS (updated for 6-dim qubit features and locking)
# ══════════════════════════════════════════════════════════════════

def test_repeater_unit():
    print(f"\n{SEP}\nTEST 1: Repeater unit (n_ch 2..8)\n{SEP}")
    rng = np.random.default_rng(0)
    for nc in range(2, 9):
        r = Repeater(rid=0, n_ch=nc, p_gen=rng.uniform(0.01,1), p_swap=rng.uniform(0.01,1))
        assert r.num_occupied() == 0 and r.num_locked() == 0
        qs = [r.allocate_qubit() for _ in range(nc)]
        assert r.has_free_qubit() == False and r.allocate_qubit() == -1
        for q in qs: r.free_qubit(q)
        assert r.feature_vector().shape == (6,) and r.qubit_features().shape == (nc, 6)
        _ok(f"n_ch={nc}")

def test_entangle_invariants():
    print(f"\n{SEP}\nTEST 2: Entangle invariants (N 3..10)\n{SEP}")
    rng = np.random.default_rng(42)
    for N in range(3, 11):
        for nc in [2, 4, 8]:
            net = _mk(N, nc, 50, rng, dt_seconds=0.0)  # zero delay
            for rep in net.repeaters: rep.p_gen = 1.0
            for i in range(N-1): assert net.entangle(i, i+1)["success"]
            assert _sym_check(net)
            assert sum(r.num_occupied() for r in net.repeaters) == 2*(N-1)
        _ok(f"N={N}")

def test_swap_basic():
    print(f"\n{SEP}\nTEST 3: Swap correctness with zero delay (N 3..10)\n{SEP}")
    rng = np.random.default_rng(99)
    for N in range(3, 11):
        for pol in SwapPolicy:
            net = _mk(N, 4, 50, rng, pol=pol, dt_seconds=0.0)
            for rep in net.repeaters: rep.p_gen = rep.p_swap = 1.0
            for i in range(N-1): net.entangle(i, i+1)
            pre = net.repeaters[1].num_available()
            assert pre >= 2
            res = net.swap(1)
            assert res["success"] and res["reason"] == "pending"
            # With dt=0, timer=0, so resolve on next age_links
            net.age_links()
            assert _sym_check(net)
        _ok(f"N={N}")

def test_farthest_optimality():
    print(f"\n{SEP}\nTEST 4: FARTHEST optimality (N 4..10)\n{SEP}")
    rng = np.random.default_rng(7)

    for N in range(4, 11):
        for nc in [3, 5, 8]:
            net = _mk(N, nc, 100, rng, pol=SwapPolicy.FARTHEST, dt_seconds=0.0)
            for rep in net.repeaters: 
                rep.p_gen = rep.p_swap = 1.0
            for _ in range(5):
                for i in range(N-1): 
                    net.entangle(i, i+1)
            pos = net._positions
            for rep in net.repeaters:
                occ = rep.available_indices()

                if len(occ) < 2: 
                    continue
                pair = rep.select_swap_pair(pos)
                if not pair: 
                    continue
                qa, qb = pair
                chosen_d = np.linalg.norm(pos[rep.partner_repeater[qa]]-pos[rep.partner_repeater[qb]])
                ii, jj = np.triu_indices(len(occ), k=1)
                best_d = np.max(np.linalg.norm(
                    pos[rep.partner_repeater[occ[ii]]]-pos[rep.partner_repeater[occ[jj]]], axis=1))
                assert np.isclose(chosen_d, best_d, atol=1e-10)
        _ok(f"N={N}")

def test_strongest_optimality():
    print(f"\n{SEP}\nTEST 5: STRONGEST optimality (N 4..10)\n{SEP}")
    rng = np.random.default_rng(11)
    for N in range(4, 11):
        for nc in [3, 5, 8]:
            net = _mk(N, nc, 100, rng, pol=SwapPolicy.STRONGEST, dt_seconds=0.0)
            for rep in net.repeaters: rep.p_gen = rep.p_swap = 1.0
            for _ in range(3):
                for i in range(N-1): net.entangle(i, i+1)
                net.age_links(discard_expired=False)
            pos = net._positions
            for rep in net.repeaters:
                occ = rep.available_indices()
                if len(occ) < 2: continue
                pair = rep.select_swap_pair(pos)
                if not pair: continue
                qa, qb = pair
                ii, jj = np.triu_indices(len(occ), k=1)
                best_p = np.max(rep.werner_param[occ[ii]]*rep.werner_param[occ[jj]])
                assert np.isclose(rep.werner_param[qa]*rep.werner_param[qb], best_p, atol=1e-12)
        _ok(f"N={N}")

def test_chain_e2e():
    print(f"\n{SEP}\nTEST 6: E2E chain propagation with zero delay (N 3..10)\n{SEP}")
    rng = np.random.default_rng(2024)
    for N in range(3, 11):
        net = _mk(N, 4, 200, rng, pol=SwapPolicy.STRONGEST, dt_seconds=0.0)
        for rep in net.repeaters: rep.p_gen = rep.p_swap = 1.0
        for i in range(N-1): assert net.entangle(i, i+1)["success"]
        for k in range(1, N-1):
            assert net.swap(k)["success"]
            net.age_links()  # resolve
        lk = net.get_all_links()
        assert len(lk) == 1 and int(lk[0][0]) == 0 and int(lk[0][2]) == N-1
        _ok(f"N={N}")

def test_aging():
    print(f"\n{SEP}\nTEST 7: Aging & decoherence (cutoff 3..30)\n{SEP}")
    rng = np.random.default_rng(0)
    for c in [3, 5, 10, 15, 20, 30]:
        net = build_chain(3, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                          cutoff=c, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
        net.entangle(0, 1)
        p0 = net.repeaters[0].initial_werner[0]
        for m in range(1, c):
            net.age_links()
            assert np.isclose(net.repeaters[0].werner_param[0], p0*np.exp(-m/c), atol=1e-12)
        assert net.repeaters[0].num_occupied() == 1
        net.age_links()
        assert net.repeaters[0].num_occupied() == 0
        _ok(f"cutoff={c}")

def test_conditional_discard():
    print(f"\n{SEP}\nTEST 8: Conditional discard (cutoff 3..20)\n{SEP}")
    rng = np.random.default_rng(0)
    for c in [3, 5, 8, 12, 20]:
        net = build_chain(3, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                          cutoff=c, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
        net.entangle(0, 1)
        for _ in range(c+4): net.age_links(discard_expired=False)
        assert net.repeaters[0].num_occupied() == 1
        net.age_links(discard_expired=True)
        assert net.repeaters[0].num_occupied() == 0
        _ok(f"cutoff={c}")

def test_gen_rate():
    print(f"\n{SEP}\nTEST 9: Gen rate statistics\n{SEP}")
    rng = np.random.default_rng(777)
    for pg1 in np.linspace(0.1, 0.5, 5):
        pg2 = rng.uniform(0.1, 0.5)
        exp = (pg1+pg2)/2.0
        reps = [Repeater(rid=0, n_ch=4, position=np.zeros(2), p_gen=pg1, p_swap=1.0, cutoff=999),
                Repeater(rid=1, n_ch=4, position=np.zeros(2), p_gen=pg2, p_swap=1.0, cutoff=999)]
        net = RepeaterNetwork(reps, np.array([[0,1],[1,0]], dtype=np.float64),
                              channel_loss=0.0, F0=1.0, distance_dep_gen=False, rng=rng)
        s = sum(1 for _ in range(3000) if (net.reset() or True) and net.entangle(0,1)["success"])
        obs = s/3000
        assert abs(obs-exp) < max(3*np.sqrt(exp*(1-exp)/3000), 0.05)
        _ok(f"p_gen=({pg1:.2f},{pg2:.2f}) exp={exp:.3f} obs={obs:.3f}")

def test_swap_rate():
    print(f"\n{SEP}\nTEST 10: Swap rate statistics\n{SEP}")
    rng = np.random.default_rng(321)
    for ps in np.linspace(0.50, 0.99, 5):
        reps = [Repeater(rid=i, n_ch=4, position=np.zeros(2),
                         p_gen=1.0, p_swap=(ps if i==1 else 1.0), cutoff=999)
                for i in range(3)]
        adj = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float64)
        net = RepeaterNetwork(reps, adj, channel_loss=0.0, F0=1.0,
                              distance_dep_gen=False, rng=rng, dt_seconds=0.0)
        s = 0
        for _ in range(3000):
            net.reset(); net.entangle(0,1); net.entangle(1,2)
            if net.swap(1)["success"]: s += 1
        obs = s/3000
        assert abs(obs-ps) < max(3*np.sqrt(ps*(1-ps)/3000), 0.05)
        _ok(f"p_swap={ps:.3f} obs={obs:.3f}")

def test_graph_shapes():
    print(f"\n{SEP}\nTEST 11: Graph shapes (N 3..10, n_ch 2..8)\n{SEP}")
    rng = np.random.default_rng(55)
    for N in range(3, 11):
        for nc in range(2, 9):
            net = _mk(N, nc, 50, rng, dt_seconds=0.0)
            for rep in net.repeaters: rep.p_gen = rep.p_swap = 1.0
            nl = sum(1 for i in range(N-1) if net.entangle(i,i+1)["success"])
            d = network_to_heterodata(net)
            assert d["repeater"].x.shape == (N, 6)
            assert d["qubit"].x.shape == (N*nc, 6)  # 6-dim now
            assert d["repeater","adjacent","repeater"].edge_index.shape[1] == 2*(N-1)
        _ok(f"N={N}")

def test_bbpssw_formulas():
    print(f"\n{SEP}\nTEST 12: BBPSSW formula correctness\n{SEP}")
    pass # FIXME Add this test

def test_purify_deterministic():
    print(f"\n{SEP}\nTEST 13: Purification deterministic (zero delay)\n{SEP}")
    rng = np.random.default_rng(0)
    for N in range(3, 9):
        net = build_chain(N, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                          cutoff=999, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
        net.entangle(0,1); net.entangle(0,1)
        res = net.purify(0,1)
        assert res["success"] and res["reason"] == "pending"
        net.age_links()  # resolve
        assert net.repeaters[0].num_occupied() == 1
        assert net.repeaters[0].num_locked() == 0
        assert _sym_check(net)
        _ok(f"N={N}")

def test_purify_insufficient():
    print(f"\n{SEP}\nTEST 14: Purification insufficient pairs\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(3, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
    net.entangle(0, 1)
    assert net.purify(0, 1)["reason"] == "insufficient_shared_pairs"
    _ok("rejected with 1 shared pair")

def test_purify_rate():
    print(f"\n{SEP}\nTEST 15: Purification rate statistics\n{SEP}")
    rng = np.random.default_rng(555)
    for p_t in [0.3, 0.5, 0.7, 0.9]:
        n_succ = 0
        for _ in range(2000):
            net = build_chain(3, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                              cutoff=999, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
            net.entangle(0,1); net.entangle(0,1)
            for qi in net.repeaters[0].qubits_to(1):
                net.repeaters[0].werner_param[qi] = net.repeaters[0].initial_werner[qi] = p_t
                pq = int(net.repeaters[0].partner_qubit[qi])
                net.repeaters[1].werner_param[pq] = net.repeaters[1].initial_werner[pq] = p_t
            if net.purify(0,1)["success"]: n_succ += 1
        obs = n_succ/2000
        exp = bbpssw_success_prob(p_t, p_t)
        assert abs(obs-exp) < max(3*np.sqrt(exp*(1-exp)/2000), 0.05)
        _ok(f"p={p_t:.1f} exp={exp:.3f} obs={obs:.3f}")

# ══════════════════════════════════════════════════════════════════
# CLASSICAL DELAY TESTS
# ══════════════════════════════════════════════════════════════════

def test_locking_unit():
    print(f"\n{SEP}\nTEST 16: Qubit locking unit tests\n{SEP}")
    r = Repeater(rid=0, n_ch=4)
    q0 = r.allocate_qubit()
    r.set_link(q0, 1, 0, 0.9)
    assert q0 in r.available_indices()

    r.lock_qubit(q0)
    assert q0 not in r.available_indices()
    assert q0 in r.occupied_indices()  # raw query still sees it
    assert r.num_available() == 0
    assert r.num_locked() == 1
    assert not r.can_swap()  # only 1 occupied, and it's locked

    # Allocate second, lock it too
    q1 = r.allocate_qubit()
    r.set_link(q1, 2, 0, 0.8)
    r.lock_qubit(q1)
    assert r.num_available() == 0 and r.num_occupied() == 2

    # Unlock one
    r.unlock_qubit(q0)
    assert q0 in r.available_indices()
    assert r.num_available() == 1
    assert not r.can_swap()  # need 2 available

    r.unlock_qubit(q1)
    assert r.can_swap()

    # free_qubit clears lock
    r.lock_qubit(q0)
    r.free_qubit(q0)
    assert not r.locked[q0]

    # reset clears all locks
    r.lock_qubit(q1)
    r.reset()
    assert r.num_locked() == 0
    _ok("all locking invariants")

def test_delay_calculation():
    print(f"\n{SEP}\nTEST 17: Classical delay calculation\n{SEP}")
    rng = np.random.default_rng(0)
    # c_fiber = 200,000 km/s, dt = 1e-4 s => 1 step covers 20 km
    net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    assert net._classical_delay_steps(0.0) == 0
    assert net._classical_delay_steps(20.0) == 1  # 20/(200000*1e-4) = 1
    assert net._classical_delay_steps(100.0) == 5
    assert net._classical_delay_steps(50.0) == 3   # ceil(2.5) = 3
    # Zero dt => zero delay
    net2 = build_chain(3, spacing=100.0, rng=rng, dt_seconds=0.0)
    assert net2._classical_delay_steps(100.0) == 0
    _ok("delay formula correct")

def test_deferred_swap_zero_delay():
    print(f"\n{SEP}\nTEST 18: Deferred swap with zero delay\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(3, n_ch=4, spacing=0.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng, dt_seconds=0.0)
    net.entangle(0, 1); net.entangle(1, 2)
    res = net.swap(1)
    assert res["success"] and res["reason"] == "pending"
    # Timer=0, qubits locked now
    assert net.repeaters[1].num_locked() == 2
    assert net.repeaters[0].num_locked() == 1
    assert net.repeaters[2].num_locked() == 1
    assert len(net.pending_events) == 1

    # Resolve
    ar = net.age_links()
    assert ar["resolved_count"] == 1 and ar["pending_count"] == 0
    # All locks cleared
    for rep in net.repeaters:
        assert rep.num_locked() == 0
    # R1 freed, R0-R2 connected
    assert net.repeaters[1].num_occupied() == 0
    lk = net.get_all_links()
    assert len(lk) == 1 and {int(lk[0][0]), int(lk[0][2])} == {0, 2}
    assert _sym_check(net)
    _ok("zero-delay swap works")

def test_deferred_swap_known_delay():
    print(f"\n{SEP}\nTEST 19: Deferred swap with 5-step delay\n{SEP}")
    rng = np.random.default_rng(0)
    # spacing=100 km, dt=1e-4 => delay = ceil(100/20) = 5
    net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0, 1); net.entangle(1, 2)
    res = net.swap(1)
    assert res["success"] and res["reason"] == "pending"
    assert len(net.pending_events) == 1
    assert net.pending_events[0]["timer"] == 5

    # Steps 1-4: event still pending
    for step in range(1, 5):
        ar = net.age_links()
        assert ar["resolved_count"] == 0 and ar["pending_count"] == 1
        # Qubits still locked
        assert net.repeaters[0].num_locked() == 1
        assert net.repeaters[2].num_locked() == 1
        # Can't swap or purify the locked qubits
        assert not net.action_mask_swap()[0]  # R0 has 1 occ, 1 locked => 0 available

    # Step 5: resolved
    ar = net.age_links()
    assert ar["resolved_count"] == 1 and ar["pending_count"] == 0
    for rep in net.repeaters:
        assert rep.num_locked() == 0
    assert net.repeaters[1].num_occupied() == 0
    lk = net.get_all_links()
    assert len(lk) == 1 and {int(lk[0][0]), int(lk[0][2])} == {0, 2}
    assert _sym_check(net)
    _ok("5-step delay swap correct")

def test_locked_invisible_to_masks():
    print(f"\n{SEP}\nTEST 20: Locked qubits invisible to action masks\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(5, n_ch=2, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(1,2); net.entangle(2,3)
    # Swap at R1: locks R0:q0, R1:q0, R1:q1, R2:q(to R0)
    res = net.swap(1)
    assert res["success"]

    sm = net.action_mask_swap()
    # R2 has 2 occupied but some may be locked
    # R2 should have: 1 qubit to R1 (locked), 1 qubit to R3 (unlocked)
    assert net.repeaters[2].num_available() <= 1
    assert not sm[2]  # can't swap with < 2 available

    # Entangle mask: nodes far from the swap should still work
    em = net.action_mask_entangle()
    # R3-R4 are completely unaffected by swap at R1
    assert em[3, 4], "Unrelated nodes R3-R4 should still be entangleable"

    # R1 has n_ch=2, both qubits locked => no free qubit on R1
    assert not em[0, 1], "R1 full (locked) => R0-R1 entangle should be False"

    # Purify mask: no pair has 2 available shared links
    pm = net.action_mask_purify()
    assert not pm.any()
    _ok("masks correctly exclude locked qubits")

def test_deferred_purify_with_delay():
    print(f"\n{SEP}\nTEST 21: Deferred purify with 3-step delay\n{SEP}")
    rng = np.random.default_rng(0)
    # spacing=50 km, dt=1e-4 => delay = ceil(50/20) = 3
    net = build_chain(3, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(0,1)
    res = net.purify(0,1)
    assert res["reason"] == "pending"
    assert len(net.pending_events) == 1
    assert net.pending_events[0]["timer"] == 3

    # 4 qubits locked (2 on R0, 2 on R1)
    assert net.repeaters[0].num_locked() == 2
    assert net.repeaters[1].num_locked() == 2

    # Steps 1-2: still pending
    for _ in range(2):
        ar = net.age_links()
        assert ar["resolved_count"] == 0

    # Step 3: resolved
    ar = net.age_links()
    assert ar["resolved_count"] == 1
    assert net.repeaters[0].num_locked() == 0
    assert net.repeaters[1].num_locked() == 0

    if res["success"]:
        assert net.repeaters[0].num_occupied() == 1  # 2->1
        assert _sym_check(net)
    else:
        assert net.repeaters[0].num_occupied() == 0  # both destroyed
    _ok("deferred purify with delay")

def test_concurrent_events():
    print(f"\n{SEP}\nTEST 22: Multiple concurrent pending events\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(5, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    # Create links
    net.entangle(0,1); net.entangle(1,2)
    net.entangle(2,3); net.entangle(3,4)

    # Swap at R1 and R3 simultaneously
    res1 = net.swap(1)
    res3 = net.swap(3)
    assert res1["success"] and res3["success"]
    assert len(net.pending_events) == 2

    # Both should have timer=3 (spacing=50km, delay=3)
    assert all(ev["timer"] == 3 for ev in net.pending_events)

    # Resolve both
    for _ in range(3):
        net.age_links()
    assert len(net.pending_events) == 0
    for rep in net.repeaters:
        assert rep.num_locked() == 0
    # R0-R2 and R2-R4 should be linked
    lk = net.get_all_links()
    assert len(lk) == 2
    assert _sym_check(net)
    _ok("2 concurrent swaps resolved independently")

def test_expiry_during_delay():
    print(f"\n{SEP}\nTEST 23: Link expiry during delay window\n{SEP}")
    rng = np.random.default_rng(0)
    # cutoff=3, delay=5 => link expires before event resolves
    net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=3, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(1,2)
    res = net.swap(1)
    assert res["success"]
    delay = net.pending_events[0]["timer"]
    assert delay == 5

    # Age 5 steps — links expire at cutoff=3, event resolves at step 5
    for step in range(1, 6):
        ar = net.age_links(discard_expired=True)

    # Event resolved but qubits were already freed by expiry
    # The guard in _resolve_swap should handle this gracefully
    assert len(net.pending_events) == 0
    # No locks should remain
    for rep in net.repeaters:
        assert rep.num_locked() == 0, f"R{rep.rid} has {rep.num_locked()} locked"
    _ok("expiry during delay handled gracefully")

def test_swap_then_entangle_concurrently():
    print(f"\n{SEP}\nTEST 24: Swap in flight + concurrent entangle elsewhere\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(5, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(1,2)
    net.swap(1)
    assert len(net.pending_events) == 1

    # While swap at R1 is pending, entangle at R3-R4 (unrelated nodes)
    res = net.entangle(3, 4)
    assert res["success"], "Should be able to entangle unrelated nodes during pending swap"

    # Resolve swap
    for _ in range(3): net.age_links()
    assert len(net.pending_events) == 0
    # Should have R0-R2 link (from swap) and R3-R4 link (from concurrent entangle)
    lk = net.get_all_links()
    assert len(lk) == 2
    assert _sym_check(net)
    _ok("concurrent entangle during pending swap")

def test_purify_swap_interaction_delayed():
    print(f"\n{SEP}\nTEST 25: Purify then swap (both delayed)\n{SEP}")
    rng = np.random.default_rng(42)
    net = build_chain(4, n_ch=6, spacing=50.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(0,1); net.entangle(1,2)

    # Purify R0-R1 (delay = 3 steps)
    res_p = net.purify(0, 1)
    assert res_p["reason"] == "pending"

    # R1's qubit to R2 should still be available (not locked by purify)
    # But the purify locked 2 qubits on R1 (to R0), so R1 has 1 available (to R2)
    # R1 can't swap (needs 2 available, only has 1)
    assert not net.action_mask_swap()[1]

    # Resolve purify
    for _ in range(3): net.age_links()
    assert len(net.pending_events) == 0

    if res_p["success"]:
        # Now R1 has: 1 qubit to R0 (upgraded, unlocked) + 1 qubit to R2 (unlocked)
        assert net.repeaters[1].num_available() >= 2
        assert net.action_mask_swap()[1]
        # Swap at R1
        res_s = net.swap(1)
        assert res_s["success"]
        for _ in range(3): net.age_links()
        lk = net.get_all_links()
        assert len(lk) == 1
        assert {int(lk[0][0]), int(lk[0][2])} == {0, 2}
    assert _sym_check(net)
    _ok("purify then swap (both delayed)")

def test_failed_swap_immediate():
    print(f"\n{SEP}\nTEST 26: Failed swap resolves immediately\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(3, n_ch=4, spacing=100.0,
                      p_gen=1.0, p_swap=0.0,  # always fail
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(1,2)
    res = net.swap(1)
    assert not res["success"] and res["reason"] == "swap_failed"
    # No event queued
    assert len(net.pending_events) == 0
    # All qubits freed immediately, no locks
    assert net.repeaters[0].num_occupied() == 0
    assert net.repeaters[1].num_occupied() == 0
    assert net.repeaters[2].num_occupied() == 0
    for rep in net.repeaters:
        assert rep.num_locked() == 0
    _ok("failed swap: immediate, no event, no locks")

def test_graph_with_locked_qubits():
    print(f"\n{SEP}\nTEST 27: Graph builder with locked qubits\n{SEP}")
    rng = np.random.default_rng(0)
    net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    net.entangle(0,1); net.entangle(1,2)
    net.swap(1)  # locks 4 qubits

    d = network_to_heterodata(net)
    qf = d["qubit"].x  # (N*n_ch, 6)
    assert qf.shape == (12, 6)

    # Count locked qubits in features
    locked_col = qf[:, 5]  # last column
    n_locked = int((locked_col > 0.5).sum())
    assert n_locked == 4, f"Expected 4 locked qubits in features, got {n_locked}"

    # Network state should show 1 pending event
    ns = d["repeater"].network_state
    assert float(ns[1]) == 1.0  # pending_count

    _ok("graph correctly encodes locked qubits and pending events")

def test_rl_stress_with_delays():
    print(f"\n{SEP}\nTEST 28: RL stress with classical delays (72 configs x 30 steps)\n{SEP}")
    mrng = np.random.default_rng(2025)
    configs = list(itertools.product(range(3,11), [2,4,8], [5,10,20]))
    for N, nc, c in configs:
        rng = np.random.default_rng(mrng.integers(0, 2**31))
        net = _mk(N, nc, c, rng, dt_seconds=1e-4)
        for _ in range(30):
            net.age_links(discard_expired=True)
            d = network_to_heterodata(net)
            assert d["repeater"].x.shape[0] == N
            sm, em, pm = net.action_mask_swap(), net.action_mask_entangle(), net.action_mask_purify()
            rv = rng.random()
            if pm.any() and rv < 0.15:
                ps = np.argwhere(np.triu(pm))
                if len(ps): i=rng.integers(len(ps)); net.purify(int(ps[i,0]),int(ps[i,1]))
            elif sm.any() and rv < 0.45:
                net.swap(int(rng.choice(np.flatnonzero(sm))))
            elif em.any():
                ps = np.argwhere(np.triu(em))
                if len(ps): i=rng.integers(len(ps)); net.entangle(int(ps[i,0]),int(ps[i,1]))
        # Final: no dangling locks after draining all events
        while net.pending_events:
            net.age_links()
        for rep in net.repeaters:
            assert rep.num_locked() == 0, f"R{rep.rid} has {rep.num_locked()} dangling locks"
        assert _sym_check(net)
    _ok(f"{len(configs)} configs all passed")

def test_grid_stress_with_delays():
    print(f"\n{SEP}\nTEST 29: Grid stress with delays (2x2..4x4)\n{SEP}")
    mrng = np.random.default_rng(404)
    for rows in range(2, 5):
        for cols in range(2, 5):
            N = rows*cols
            rng = np.random.default_rng(mrng.integers(0, 2**31))
            reps = [Repeater(rid=idx, n_ch=4, swap_policy=SwapPolicy.FARTHEST,
                             position=np.array([c*10.0, r*10.0]),
                             p_gen=rng.uniform(0.01,1.0),
                             p_swap=rng.uniform(0.01,1.0),
                             cutoff=int(rng.integers(5,25)))
                    for idx in range(N) for r, c in [divmod(idx, cols)]]
            adj = np.zeros((N,N), dtype=np.float64)
            for idx in range(N):
                r, c = divmod(idx, cols)
                if c+1<cols: adj[idx,idx+1]=adj[idx+1,idx]=1.0
                if r+1<rows: adj[idx,idx+cols]=adj[idx+cols,idx]=1.0
            net = RepeaterNetwork(reps, adj, channel_loss=0.01, F0=0.99,
                                  distance_dep_gen=True, rng=rng, dt_seconds=1e-4)
            for _ in range(30):
                net.age_links(discard_expired=True)
                sm,em,pm = net.action_mask_swap(), net.action_mask_entangle(), net.action_mask_purify()
                rv = rng.random()
                if pm.any() and rv<0.2:
                    ps=np.argwhere(np.triu(pm))
                    if len(ps): i=rng.integers(len(ps)); net.purify(int(ps[i,0]),int(ps[i,1]))
                elif sm.any() and rv<0.5:
                    net.swap(int(rng.choice(np.flatnonzero(sm))))
                elif em.any():
                    ps=np.argwhere(np.triu(em))
                    if len(ps): i=rng.integers(len(ps)); net.entangle(int(ps[i,0]),int(ps[i,1]))
            while net.pending_events: net.age_links()
            for rep in net.repeaters: assert rep.num_locked()==0
            assert _sym_check(net)
            _ok(f"{rows}x{cols} grid")

def test_policy_divergence():
    print(f"\n{SEP}\nTEST 30: FARTHEST vs STRONGEST divergence\n{SEP}")
    rng = np.random.default_rng(13)
    diverged = 0
    for N in range(5, 11):
        for _ in range(10):
            seed = rng.integers(0, 2**31)
            nets = {}
            for pol in [SwapPolicy.FARTHEST, SwapPolicy.STRONGEST]:
                lr = np.random.default_rng(seed)
                n = _mk(N, 6, 200, lr, pol=pol, sp=25.0, dt_seconds=0.0)
                for rep in n.repeaters: rep.p_gen = rep.p_swap = 1.0
                ar = np.random.default_rng(seed+1)
                for _ in range(4):
                    for i in range(N-1): n.entangle(i, i+1)
                    for _ in range(ar.integers(1,5)): n.age_links(discard_expired=False)
                nets[pol] = n
            for rid in range(N):
                rf = nets[SwapPolicy.FARTHEST].repeaters[rid]
                rs = nets[SwapPolicy.STRONGEST].repeaters[rid]
                if rf.num_available() < 3: continue
                pf = rf.select_swap_pair(nets[SwapPolicy.FARTHEST]._positions)
                ps_ = rs.select_swap_pair(nets[SwapPolicy.STRONGEST]._positions)
                if pf and ps_ and set(pf) != set(ps_):
                    diverged += 1; break
    assert diverged > 0
    _ok(f"{diverged} divergent cases")

def test_asymmetric_link_swap_bug():
    print(f"\n{SEP}\nTEST 31: Asymmetric Link Corruption on Partial Swap Expiry\n{SEP}")
    rng = np.random.default_rng(42)
    
    # 3 nodes, 100km spacing. Delay = 5 steps.
    net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    
    # Force R0 cutoff to be shorter than the delay (expires at step 3)
    net.repeaters[0].cutoff = 3
    # Force R2 cutoff to be longer than the delay (survives)
    net.repeaters[2].cutoff = 10
    
    net.entangle(0, 1)
    net.entangle(1, 2)
    res = net.swap(1)
    
    if not res["success"]:
        _fail("Swap failed to initiate")
        return
        
    # Age until event resolves (5 steps)
    for _ in range(5):
        net.age_links(discard_expired=True)
        
    # Check if network symmetry is maintained. 
    # Current codebase WILL FAIL this check because R2 points to R1, but R1 is empty.
    if not _sym_check(net):
        _fail("Bilateral symmetry broken: Surviving remote link points to empty slot.")
    else:
        _ok("Bilateral symmetry maintained.")

def test_purify_partial_expiry_bug():
    print(f"\n{SEP}\nTEST 32: Purify Partial Expiry\n{SEP}")
    rng = np.random.default_rng(42)
    
    net = build_chain(2, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                      cutoff=999, F0=1.0, channel_loss=0.0, rng=rng,
                      dt_seconds=1e-4, distance_dep_gen=False)
    
    net.entangle(0, 1)
    net.entangle(0, 1)
    
    # Manually hack one link to expire early
    q_sac_0 = net.repeaters[0].occupied_indices()[0]
    q_sac_1 = net.repeaters[1].occupied_indices()[0]
    net.repeaters[0].link_cutoff[q_sac_0] = 2
    net.repeaters[1].link_cutoff[q_sac_1] = 2
    
    net.purify(0, 1)
    
    for _ in range(5):
        net.age_links(discard_expired=True)
        
    if not _sym_check(net):
        _fail("Bilateral symmetry broken after partial purify expiry.")
    else:
        _ok("Bilateral symmetry maintained.")

# ══════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    T0 = time.perf_counter()
    test_repeater_unit()
    test_entangle_invariants()
    test_swap_basic()
    test_farthest_optimality()
    test_strongest_optimality()
    test_chain_e2e()
    test_aging()
    test_conditional_discard()
    test_gen_rate()
    test_swap_rate()
    test_graph_shapes()
    test_bbpssw_formulas()
    test_purify_deterministic()
    test_purify_insufficient()
    test_purify_rate()
    test_locking_unit()
    test_delay_calculation()
    test_deferred_swap_zero_delay()
    test_deferred_swap_known_delay()
    test_locked_invisible_to_masks()
    test_deferred_purify_with_delay()
    test_concurrent_events()
    test_expiry_during_delay()
    test_swap_then_entangle_concurrently()
    test_purify_swap_interaction_delayed()
    test_failed_swap_immediate()
    test_graph_with_locked_qubits()
    test_rl_stress_with_delays()
    test_grid_stress_with_delays()
    test_policy_divergence()
    test_asymmetric_link_swap_bug()
    test_purify_partial_expiry_bug()

    elapsed = time.perf_counter() - T0
    print(f"\n{SEP}")
    if FAIL == 0: print(f"ALL {OK} CHECKS PASSED    ({elapsed:.2f}s)")
    else: print(f"{OK} passed, {FAIL} FAILED   ({elapsed:.2f}s)")
    print(SEP)
    sys.exit(1 if FAIL else 0)