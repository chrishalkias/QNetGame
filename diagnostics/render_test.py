#!/usr/bin/env python3
"""
test_render.py  -  Visual regression tests for RepeaterNetwork.render()

Produces five PNGs, each depicting a distinct network state:

    1. empty_chain        - 5-node chain, no entanglement
    2. elementary_links   - elementary Bell pairs on several edges
    3. post_swap          - long-range link after a BSM at the middle node
    4. pending_swap       - swap issued but not yet resolved (locked qubits)
    5. grid_mixed         - 2x3 grid with mixed entanglement

All scenarios use  channel_loss = 0  so that entanglement generation
is deterministic (p_gen = 1  ⇒  100 % success) and initial fidelity
F₀ = 0.98 is distance-independent.
"""

import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quantum_repeater_sim.network  import build_chain, build_grid, build_GEANT

OUT = pathlib.Path("diagnostics/renders")
OUT.mkdir(exist_ok=True)

# Common keyword arguments shared by every scenario
COMMON = dict(n_ch=4, p_gen=1.0, p_swap=1.0, F0=0.98, channel_loss=0.0)


# ==============================================================
#  Scenario 1 – Empty chain (topology only)
# ==============================================================
def scenario_empty_chain():
    net = build_chain(5, spacing=50.0, cutoff=20,
                      rng=np.random.default_rng(0), **COMMON)
    net.render(filepath=str(OUT / "1_empty_chain.png"))
    print("[1] empty_chain ✓")


# ==============================================================
#  Scenario 2 – Elementary links on several edges
# ==============================================================
def scenario_elementary_links():
    net = build_chain(5, spacing=50.0, cutoff=20,
                      rng=np.random.default_rng(1), **COMMON)
    # Two links between R0–R1, one between R2–R3, one between R3–R4
    net.entangle(0, 1)
    net.entangle(0, 1)
    net.entangle(2, 3)
    net.entangle(3, 4)

    # Age a few steps so fidelities diverge slightly
    for _ in range(3):
        net.age_links(discard_expired=False)

    net.render(filepath=str(OUT / "2_elementary_links.png"))
    print("[2] elementary_links ✓")


# ==============================================================
#  Scenario 3 – Post-swap: long-range link R0 ↔ R2
# ==============================================================
def scenario_post_swap():
    net = build_chain(5, spacing=50.0, cutoff=20, dt_seconds=1.0,
                      rng=np.random.default_rng(2), **COMMON)

    net.entangle(0, 1)          # R0:q0 ↔ R1:q0
    net.entangle(1, 2)          # R1:q1 ↔ R2:q0
    net.entangle(3, 4)          # R3:q0 ↔ R4:q0

    res = net.swap(1)           # BSM at R1 → pending R0 ↔ R2
    assert res["success"], f"swap failed: {res}"

    # Resolve pending event (delay ≈ 1 step with dt = 1 s)
    for _ in range(3):
        net.age_links()

    net.render(filepath=str(OUT / "3_post_swap.png"))
    print("[3] post_swap ✓")


# ==============================================================
#  Scenario 4 – Pending swap (locked qubits visible)
# ==============================================================
def scenario_pending_swap():
    # Small dt → large classical delay → event stays pending
    net = build_chain(5, spacing=50.0, cutoff=50, dt_seconds=1e-5,
                      rng=np.random.default_rng(3), **COMMON)

    net.entangle(0, 1)          # R0:q0 ↔ R1:q0
    net.entangle(1, 2)          # R1:q1 ↔ R2:q0
    net.entangle(3, 4)          # R3:q0 ↔ R4:q0

    res = net.swap(1)
    assert res["success"], f"swap failed: {res}"

    # One tick — event NOT yet resolved
    net.age_links()
    assert len(net.pending_events) > 0, "event resolved too soon"

    net.render(filepath=str(OUT / "4_pending_swap.png"))
    print("[4] pending_swap ✓")


# ==============================================================
#  Scenario 5 – 2 × 3 grid with mixed entanglement
# ==============================================================
def scenario_grid():
    net = build_grid(2, 3, spacing=60.0, cutoff=20,
                     rng=np.random.default_rng(4), **COMMON)
    #
    #   0 ── 1 ── 2
    #   |    |    |
    #   3 ── 4 ── 5
    #
    net.entangle(0, 1)
    net.entangle(1, 2)
    net.entangle(0, 3)
    net.entangle(3, 4)
    net.entangle(4, 5)
    net.entangle(2, 5)

    for _ in range(3):
        net.age_links(discard_expired=False)

    net.render(filepath=str(OUT / "5_grid_mixed.png"))
    print("[5] grid_mixed ✓")

# ==============================================================
#  Scenario 6 – Geant with some Entanglements
# ==============================================================
def geant_net():
    net= build_GEANT()

    net.entangle(0, 1)
    net.entangle(1, 2)
    net.entangle(0, 3)
    net.entangle(3, 4)
    net.entangle(4, 5)
    net.entangle(2, 5)

    for _ in range(3):
        net.age_links(discard_expired=False)

    net.render(filepath=str(OUT / "6_GEANT_mixed.png"))
    print("[6] geant_net ✓")


# ==============================================================

if __name__ == "__main__":
    scenario_empty_chain()
    scenario_elementary_links()
    scenario_post_swap()
    scenario_pending_swap()
    scenario_grid()
    geant_net()
    print(f"\nAll renders saved to  {OUT.resolve()}/")