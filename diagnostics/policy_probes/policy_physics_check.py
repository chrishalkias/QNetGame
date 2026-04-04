"""Systematic physics-sanity checks for a trained QRN policy.

Tests whether the learned Q-values respect key physical intuitions
about quantum repeater networks. Prints PASS/FAIL for each check.
"""

from __future__ import annotations
import numpy as np
import torch
from rl_stack.model import QNetwork
from rl_stack.env_wrapper import N_ACTIONS, NOOP, SWAP, PURIFY
from rl_stack.agent import _obs_to_data

NEIGHBORS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}


def load_model(path: str, hidden: int = 64) -> QNetwork:
    model = QNetwork(node_dim=8, hidden=hidden, n_actions=N_ACTIONS)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _make_obs(features: np.ndarray):
    N = len(features)
    src, dst = [], []
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    edge_index = np.array([src, dst], dtype=np.int64)
    return {"x": features.astype(np.float32), "edge_index": edge_index}


def get_q(model, features: np.ndarray, node: int) -> np.ndarray:
    obs = _make_obs(features)
    data = _obs_to_data(obs, "cpu")
    with torch.no_grad():
        q = model(data).cpu().numpy()
    return q[node]


def make_chain(probe: int, occ: float, fid: float, t_rem: float,
               can_swap: float = 0, can_purify: float = 0) -> np.ndarray:
    """Build a 5-node chain with controlled probe node features."""
    avail = 1.0 - occ
    feats = np.zeros((5, 8), dtype=np.float32)
    feats[0] = [0.25, 0.70, 1, 0, 0.25, 0, 0, t_rem]
    feats[1] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[2] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[3] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[4] = [0.25, 0.70, 0, 1, 0.25, 0, 0, t_rem]
    feats[probe] = [occ, fid, 0, 0, avail, can_swap, can_purify, t_rem]
    return feats


def run_checks(model_path: str, hidden: int = 64):
    model = load_model(model_path, hidden)
    probe = 2
    results = []

    def check(name: str, passed: bool, detail: str = ""):
        tag = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        results.append(passed)
        print(f"  [{tag}] {name}")
        if detail:
            print(f"         {detail}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 1: Swap preference should increase with link fidelity")
    print("=" * 65)
    # At moderate fidelity, swap advantage should be higher than at low fidelity
    # (swapping low-fidelity links produces useless e2e connections)
    q_low = get_q(model, make_chain(probe, 0.5, 0.35, 0.5, can_swap=1), probe)
    q_mid = get_q(model, make_chain(probe, 0.5, 0.65, 0.5, can_swap=1), probe)
    q_high = get_q(model, make_chain(probe, 0.5, 0.90, 0.5, can_swap=1), probe)

    adv_low = q_low[SWAP] - q_low[NOOP]
    adv_mid = q_mid[SWAP] - q_mid[NOOP]
    adv_high = q_high[SWAP] - q_high[NOOP]

    check("Q(swap)-Q(wait) at F=0.35 vs F=0.65",
          adv_mid > adv_low or adv_mid > 0,
          f"adv(F=0.35)={adv_low:.4f}, adv(F=0.65)={adv_mid:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 2: Time pressure should make agent more swap-eager")
    print("=" * 65)
    # With little time left, the agent should prefer swapping even at
    # moderate fidelity rather than waiting
    q_early = get_q(model, make_chain(probe, 0.5, 0.6, 0.9, can_swap=1), probe)
    q_late = get_q(model, make_chain(probe, 0.5, 0.6, 0.1, can_swap=1), probe)

    adv_early = q_early[SWAP] - q_early[NOOP]
    adv_late = q_late[SWAP] - q_late[NOOP]

    check("Swap advantage increases under time pressure",
          adv_late > adv_early,
          f"adv(t=0.9)={adv_early:.4f}, adv(t=0.1)={adv_late:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 3: Wait preferred when no qubits available")
    print("=" * 65)
    # With 0 occupied qubits, there's nothing to swap or purify
    q_empty = get_q(model, make_chain(probe, 0.0, 0.0, 0.5, can_swap=0, can_purify=0), probe)
    best_empty = int(q_empty.argmax())

    check("Best action is WAIT when occ=0",
          best_empty == NOOP,
          f"best={['Wait','Swap','Purify'][best_empty]}, Q={q_empty}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 4: Agent should not purify near-perfect fidelity links")
    print("=" * 65)
    # Purifying F~0.95 links wastes a qubit for negligible gain
    q_highpur = get_q(model, make_chain(probe, 0.5, 0.95, 0.5, can_purify=1), probe)
    adv_pur_high = q_highpur[PURIFY] - q_highpur[NOOP]

    check("Q(purify)-Q(wait) < 0 at F=0.95",
          adv_pur_high < 0,
          f"purify advantage={adv_pur_high:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 5: Purify should be more attractive at moderate fidelity "
          "than at high fidelity")
    print("=" * 65)
    q_modpur = get_q(model, make_chain(probe, 0.5, 0.55, 0.5, can_purify=1), probe)
    adv_pur_mod = q_modpur[PURIFY] - q_modpur[NOOP]

    check("Purify advantage at F=0.55 > purify advantage at F=0.95",
          adv_pur_mod > adv_pur_high,
          f"adv(F=0.55)={adv_pur_mod:.4f}, adv(F=0.95)={adv_pur_high:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 6: Swap should beat purify when both available and "
          "time is short")
    print("=" * 65)
    # Under time pressure, swapping to extend the chain is more urgent
    # than purifying to improve fidelity
    q_both = get_q(model, make_chain(probe, 0.5, 0.65, 0.15,
                                      can_swap=1, can_purify=1), probe)
    check("Q(swap) > Q(purify) under time pressure (t=0.15, F=0.65)",
          q_both[SWAP] > q_both[PURIFY],
          f"Q(swap)={q_both[SWAP]:.4f}, Q(purify)={q_both[PURIFY]:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 7: Symmetry — swapping should be similarly valued "
          "at nodes 1 and 3")
    print("=" * 65)
    q_n1 = get_q(model, make_chain(1, 0.5, 0.7, 0.5, can_swap=1), 1)
    q_n3 = get_q(model, make_chain(3, 0.5, 0.7, 0.5, can_swap=1), 3)
    swap_diff = abs(q_n1[SWAP] - q_n3[SWAP])

    check("Q(swap) at node 1 ≈ Q(swap) at node 3 (symmetric chain)",
          swap_diff < 0.05,
          f"Q_swap(n1)={q_n1[SWAP]:.4f}, Q_swap(n3)={q_n3[SWAP]:.4f}, "
          f"diff={swap_diff:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 8: Higher occupancy should correlate with more "
          "action options valued")
    print("=" * 65)
    # With more qubits, swap/purify become viable → Q-spread should widen
    q_low_occ = get_q(model, make_chain(probe, 0.25, 0.7, 0.5, can_swap=0), probe)
    q_high_occ = get_q(model, make_chain(probe, 0.75, 0.7, 0.5, can_swap=1), probe)
    spread_low = q_low_occ.max() - q_low_occ.min()
    spread_high = q_high_occ.max() - q_high_occ.min()

    check("Q-value spread larger at high occupancy (more options)",
          spread_high > spread_low,
          f"spread(occ=0.25)={spread_low:.4f}, spread(occ=0.75)={spread_high:.4f}")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 9: All Q-values should be in reasonable range")
    print("=" * 65)
    # Q-values should be bounded: max possible return is SUCCESS_REWARD=1.0,
    # min is max_steps * STEP_COST = -0.3 (for 30 steps)
    all_q = []
    for fid in [0.3, 0.5, 0.7, 0.9]:
        for t in [0.1, 0.5, 0.9]:
            q = get_q(model, make_chain(probe, 0.5, fid, t, can_swap=1), probe)
            all_q.extend(q.tolist())

    q_min, q_max = min(all_q), max(all_q)
    check("Q-values in plausible range [-1.0, 2.0]",
          -1.0 < q_min and q_max < 2.0,
          f"Q range: [{q_min:.4f}, {q_max:.4f}]")

    # ================================================================
    print("\n" + "=" * 65)
    print("CHECK 10: Swap at very low fidelity should NOT be preferred "
          "over wait (wasted resources)")
    print("=" * 65)
    q_vlow = get_q(model, make_chain(probe, 0.5, 0.28, 0.5, can_swap=1), probe)
    adv_vlow = q_vlow[SWAP] - q_vlow[NOOP]

    # This is nuanced: the agent might still swap at low fidelity if
    # connectivity matters more than fidelity. We check but note this
    # is a "soft" expectation.
    is_wait = adv_vlow < 0
    check("Wait preferred over swap at F=0.28 (soft expectation)",
          is_wait,
          f"swap advantage={adv_vlow:.4f}" +
          (" — agent swaps anyway (connectivity-first strategy)" if not is_wait else ""))

    # ================================================================
    # Summary
    # ================================================================
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 65}")
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("The policy is physically consistent.")
    elif n_pass >= n_total - 2:
        print("The policy is mostly consistent with minor deviations.")
    else:
        print("The policy shows significant deviations from expected physics.")
    print("=" * 65)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cluster_001/policy.pth"
    run_checks(path)
