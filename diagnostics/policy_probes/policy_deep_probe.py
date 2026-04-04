"""Deeper probes into the failed physics checks to diagnose root causes."""

from __future__ import annotations
import numpy as np
import torch
from rl_stack.model import QNetwork
from rl_stack.env_wrapper import N_ACTIONS, NOOP, SWAP, PURIFY
from rl_stack.agent import _obs_to_data

ACTION_NAMES = ['Wait', 'Swap', 'Purify']


def load_model(path, hidden=64):
    model = QNetwork(node_dim=8, hidden=hidden, n_actions=N_ACTIONS)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _make_obs(features):
    N = len(features)
    src, dst = [], []
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return {"x": features.astype(np.float32),
            "edge_index": np.array([src, dst], dtype=np.int64)}


def get_q(model, features, node):
    data = _obs_to_data(_make_obs(features), "cpu")
    with torch.no_grad():
        return model(data).cpu().numpy()[node]


def make_chain(probe, occ, fid, t_rem, can_swap=0, can_purify=0):
    feats = np.zeros((5, 8), dtype=np.float32)
    feats[0] = [0.25, 0.70, 1, 0, 0.25, 0, 0, t_rem]
    feats[1] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[2] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[3] = [0.50, 0.70, 0, 0, 0.50, 1, 0, t_rem]
    feats[4] = [0.25, 0.70, 0, 1, 0.25, 0, 0, t_rem]
    avail = 1.0 - occ
    feats[probe] = [occ, fid, 0, 0, avail, can_swap, can_purify, t_rem]
    return feats


def run_probes(model_path):
    model = load_model(model_path)
    probe = 2

    # ---- PROBE A: Does the mask (can_swap / can_purify) actually matter? ----
    print("=" * 65)
    print("PROBE A: Sensitivity to action-availability flags")
    print("=" * 65)
    print(f"{'can_swap':>10} {'can_pur':>10} | {'Q(wait)':>10} {'Q(swap)':>10} "
          f"{'Q(purify)':>10} | {'best':>8}")
    for cs in [0, 1]:
        for cp in [0, 1]:
            q = get_q(model, make_chain(probe, 0.5, 0.7, 0.5,
                                         can_swap=cs, can_purify=cp), probe)
            best = ACTION_NAMES[int(q.argmax())]
            print(f"{cs:>10} {cp:>10} | {q[0]:>10.4f} {q[1]:>10.4f} "
                  f"{q[2]:>10.4f} | {best:>8}")

    # ---- PROBE B: Q-values across fidelity sweep ----
    print(f"\n{'=' * 65}")
    print("PROBE B: Q-values vs fidelity (can_swap=1, t=0.5)")
    print("=" * 65)
    print(f"{'fid':>6} | {'Q(wait)':>10} {'Q(swap)':>10} {'Q(purify)':>10} | "
          f"{'best':>8} {'swap-wait':>12}")
    for fid in np.arange(0.25, 1.01, 0.05):
        q = get_q(model, make_chain(probe, 0.5, fid, 0.5, can_swap=1), probe)
        best = ACTION_NAMES[int(q.argmax())]
        print(f"{fid:>6.2f} | {q[0]:>10.4f} {q[1]:>10.4f} {q[2]:>10.4f} | "
              f"{best:>8} {q[1]-q[0]:>+12.4f}")

    # ---- PROBE C: Q-values across time sweep ----
    print(f"\n{'=' * 65}")
    print("PROBE C: Q-values vs time_remaining (can_swap=1, F=0.7)")
    print("=" * 65)
    print(f"{'t_rem':>6} | {'Q(wait)':>10} {'Q(swap)':>10} {'Q(purify)':>10} | "
          f"{'best':>8} {'swap-wait':>12}")
    for t in np.arange(0.05, 1.01, 0.05):
        q = get_q(model, make_chain(probe, 0.5, 0.7, t, can_swap=1), probe)
        best = ACTION_NAMES[int(q.argmax())]
        print(f"{t:>6.2f} | {q[0]:>10.4f} {q[1]:>10.4f} {q[2]:>10.4f} | "
              f"{best:>8} {q[1]-q[0]:>+12.4f}")

    # ---- PROBE D: Does the model distinguish action flags at all? ----
    print(f"\n{'=' * 65}")
    print("PROBE D: Impact of can_swap flag on Q(swap)")
    print("=" * 65)
    q_no = get_q(model, make_chain(probe, 0.5, 0.7, 0.5, can_swap=0), probe)
    q_yes = get_q(model, make_chain(probe, 0.5, 0.7, 0.5, can_swap=1), probe)
    print(f"  can_swap=0: Q(swap)={q_no[SWAP]:.4f}")
    print(f"  can_swap=1: Q(swap)={q_yes[SWAP]:.4f}")
    print(f"  Difference: {q_yes[SWAP]-q_no[SWAP]:+.4f}")

    print(f"\n{'=' * 65}")
    print("PROBE E: Impact of can_purify flag on Q(purify)")
    print("=" * 65)
    q_no = get_q(model, make_chain(probe, 0.5, 0.7, 0.5, can_purify=0), probe)
    q_yes = get_q(model, make_chain(probe, 0.5, 0.7, 0.5, can_purify=1), probe)
    print(f"  can_purify=0: Q(purify)={q_no[PURIFY]:.4f}")
    print(f"  can_purify=1: Q(purify)={q_yes[PURIFY]:.4f}")
    print(f"  Difference: {q_yes[PURIFY]-q_no[PURIFY]:+.4f}")

    # ---- PROBE F: What does the agent see at occ=0? ----
    print(f"\n{'=' * 65}")
    print("PROBE F: Q-values at zero occupancy (nothing to act on)")
    print("=" * 65)
    for cp in [0, 1]:
        q = get_q(model, make_chain(probe, 0.0, 0.0, 0.5,
                                     can_swap=0, can_purify=cp), probe)
        print(f"  occ=0, can_purify={cp}: Q={q}, best={ACTION_NAMES[int(q.argmax())]}")

    # ---- PROBE G: Q-value magnitude analysis ----
    print(f"\n{'=' * 65}")
    print("PROBE G: Q-value range and separation analysis")
    print("=" * 65)
    separations = []
    for fid in [0.3, 0.5, 0.7, 0.9]:
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            q = get_q(model, make_chain(probe, 0.5, fid, t, can_swap=1, can_purify=1), probe)
            sep = q.max() - q.min()
            separations.append(sep)

    print(f"  Mean Q-value separation:   {np.mean(separations):.4f}")
    print(f"  Median Q-value separation: {np.median(separations):.4f}")
    print(f"  Max Q-value separation:    {np.max(separations):.4f}")
    print(f"  Min Q-value separation:    {np.min(separations):.4f}")
    if np.mean(separations) < 0.02:
        print("  WARNING: Q-values are very close together — the model may not")
        print("  have learned strong action preferences.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cluster_001/policy.pth"
    run_probes(path)
