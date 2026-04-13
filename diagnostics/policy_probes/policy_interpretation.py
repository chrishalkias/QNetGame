"""
Diagnostics for a trained QRNAgent.

Probes the agent's Q-value landscape by constructing synthetic observations
with controlled parameters and reading out per-node Q-values. Produces
interpretable plots showing how the agent reasons about swap, purify,
and wait decisions.

Usage:
    # defaults: node 2 of 5-chain
    PYTHONPATH=. python diagnostics/policy_probes/policy_interpretation.py

    # node 5 of 10-chain
    PYTHONPATH=. python diagnostics/policy_probes/policy_interpretation.py \
        --n_nodes 10 --probe 5
"""

from __future__ import annotations
import argparse
import os
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rl_stack.model import QNetwork
from rl_stack.env_wrapper import NOOP, SWAP, PURIFY, N_ACTIONS

# ── action labels / colours ────────────────────────────────────────────────
_ANAMES  = {NOOP: "Wait", SWAP: "Swap", PURIFY: "Purify"}
_ACOLORS = {NOOP: "#aaaaaa", SWAP: "#cc4444", PURIFY: "#44aa44"}

T_REM = 0.5   # mid-episode time remaining used for all neutral nodes


# ── shared helpers ─────────────────────────────────────────────────────────

def _make_obs(n_nodes: int, features: np.ndarray) -> dict:
    src, dst = [], []
    for i in range(n_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    edge_index = (np.array([src, dst], dtype=np.int64)
                  if src else np.zeros((2, 0), dtype=np.int64))
    return {"x": features.astype(np.float32), "edge_index": edge_index}


def _neutral_chain(n_nodes: int, probe: int,
                   probe_feats: np.ndarray) -> np.ndarray:
    """Return (n_nodes, 8) feature matrix.

    Source = node 0, dest = node n_nodes-1.
    All non-probe interior nodes get neutral mid-range features.
    probe_feats: length-8 array for the probe node.
    """
    feats = np.zeros((n_nodes, 8), dtype=np.float32)
    for i in range(n_nodes):
        if i == 0:
            feats[i] = [0.25, 0.6, 1, 0, 0.25, 0, 0, T_REM]   # source
        elif i == n_nodes - 1:
            feats[i] = [0.25, 0.6, 0, 1, 0.25, 0, 0, T_REM]   # dest
        elif i == probe:
            feats[i] = probe_feats
        else:
            feats[i] = [0.5, 0.7, 0, 0, 0.5, 1, 0, T_REM]     # interior
    return feats


def _get_q(model, obs, device: str = "cpu") -> np.ndarray:
    from rl_stack.agent import _obs_to_data
    data = _obs_to_data(obs, device)
    with torch.no_grad():
        return model(data).cpu().numpy()


def load_model(model_path: str, node_dim: int = 8,
               hidden: int = 64, device: str = "cpu") -> QNetwork:
    model = QNetwork(node_dim, hidden, N_ACTIONS)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _tag(n_nodes: int, probe: int) -> str:
    """Filename suffix to distinguish runs."""
    return f"_n{n_nodes}_p{probe}"


# ── Diagnostic 1: swap preference f(F_left, F_right) ──────────────────────

def plot_swap_preference(model, save_dir=".", resolution=40, device="cpu",
                         n_nodes=5, probe=2):
    """Q(swap) - Q(wait) at probe node as f(left-link fidelity, right-link fidelity)."""
    f_range    = np.linspace(0.25, 1.0, resolution)
    q_swap_adv = np.zeros((resolution, resolution))
    q_swap_raw = np.zeros((resolution, resolution))

    for i, f1 in enumerate(f_range):
        for j, f2 in enumerate(f_range):
            mean_f = (f1 + f2) / 2.0
            pf = np.array([0.5, mean_f, 0, 0, 0.5, 1, 0, T_REM], dtype=np.float32)
            feats = _neutral_chain(n_nodes, probe, pf)
            # Propagate link fidelity hints to source/dest
            feats[0, 1] = f1
            feats[n_nodes - 1, 1] = f2

            obs = _make_obs(n_nodes, feats)
            q   = _get_q(model, obs, device)
            q_swap_adv[i, j] = q[probe, SWAP] - q[probe, NOOP]
            q_swap_raw[i, j] = q[probe, SWAP]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    im0 = axes[0].imshow(q_swap_adv, origin="lower", aspect="auto",
                         extent=[0.25, 1.0, 0.25, 1.0], cmap="RdBu_r")
    axes[0].set_xlabel("Right link fidelity $F_2$")
    axes[0].set_ylabel("Left link fidelity $F_1$")
    axes[0].set_title(f"$Q(\\mathrm{{swap}}) - Q(\\mathrm{{wait}})$ at node {probe}")
    axes[0].contour(f_range, f_range, q_swap_adv, levels=[0],
                    colors="black", linewidths=1.5)
    fig.colorbar(im0, ax=axes[0], label="Advantage")

    im1 = axes[1].imshow(q_swap_raw, origin="lower", aspect="auto",
                         extent=[0.25, 1.0, 0.25, 1.0], cmap="viridis")
    axes[1].set_xlabel("Right link fidelity $F_2$")
    axes[1].set_ylabel("Left link fidelity $F_1$")
    axes[1].set_title(f"$Q(\\mathrm{{swap}})$ at node {probe}")
    fig.colorbar(im1, ax=axes[1], label="Q-value")

    fig.suptitle(
        f"Swap preference at node {probe} ({n_nodes}-node chain, "
        f"src=0, dst={n_nodes-1})", fontsize=11)
    plt.tight_layout()
    fname = f"diag_swap_preference{_tag(n_nodes, probe)}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Diagnostic 2: purify preference f(F1, F2) ─────────────────────────────

def plot_purify_preference(model, save_dir=".", resolution=40, device="cpu",
                           n_nodes=5, probe=2):
    """Q(purify) - Q(wait) at probe node as f(pair-1 fidelity, pair-2 fidelity)."""
    # Purify probe: place it one node in from the source so it is plausible
    # to have 2 qubits to the same neighbour.  We keep probe as-is but set
    # can_purify=1, can_swap=0.
    f_range   = np.linspace(0.25, 1.0, resolution)
    q_pur_adv = np.zeros((resolution, resolution))

    for i, f1 in enumerate(f_range):
        for j, f2 in enumerate(f_range):
            mean_f = (f1 + f2) / 2.0
            pf = np.array([0.5, mean_f, 0, 0, 0.5, 0, 1, T_REM], dtype=np.float32)
            feats = _neutral_chain(n_nodes, probe, pf)

            obs = _make_obs(n_nodes, feats)
            q   = _get_q(model, obs, device)
            q_pur_adv[i, j] = q[probe, PURIFY] - q[probe, NOOP]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(q_pur_adv, origin="lower", aspect="auto",
                   extent=[0.25, 1.0, 0.25, 1.0], cmap="PiYG")
    ax.set_xlabel("Second pair fidelity $F_2$")
    ax.set_ylabel("First pair fidelity $F_1$")
    ax.set_title(f"$Q(\\mathrm{{purify}}) - Q(\\mathrm{{wait}})$ at node {probe}")
    ax.contour(f_range, f_range, q_pur_adv, levels=[0],
               colors="black", linewidths=1.5)
    fig.colorbar(im, ax=ax, label="Advantage")
    fig.suptitle(
        f"Purify preference at node {probe} ({n_nodes}-node chain, "
        f"2 pairs to same neighbour)", fontsize=11)
    plt.tight_layout()
    fname = f"diag_purify_preference{_tag(n_nodes, probe)}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Diagnostic 3: best action map f(occupancy, fidelity) ──────────────────

def plot_best_action_map(model, save_dir=".", resolution=30, device="cpu",
                         n_nodes=5, probe=2):
    """Preferred action at probe node as f(frac_occupied, mean_fidelity)."""
    occ_range = np.linspace(0.0, 1.0, resolution)
    fid_range = np.linspace(0.25, 1.0, resolution)
    best_map  = np.zeros((resolution, resolution), dtype=int)

    for i, occ in enumerate(occ_range):
        for j, fid in enumerate(fid_range):
            avail    = occ
            can_swap = 1.0 if avail >= 0.5 else 0.0
            can_pur  = 1.0 if avail >= 0.5 else 0.0
            pf = np.array([occ, fid, 0, 0, avail, can_swap, can_pur, T_REM],
                          dtype=np.float32)
            # neighbouring interior nodes share same occ/fid context
            feats = _neutral_chain(n_nodes, probe, pf)
            for k in range(1, n_nodes - 1):
                if k != probe:
                    feats[k] = [occ, fid, 0, 0, avail, can_swap, 0, T_REM]

            obs = _make_obs(n_nodes, feats)
            q   = _get_q(model, obs, device)
            best_map[i, j] = int(q[probe].argmax())

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_discrete = plt.matplotlib.colors.ListedColormap(
        [_ACOLORS[a] for a in range(N_ACTIONS)])
    ax.imshow(best_map, origin="lower", aspect="auto",
              extent=[0.25, 1.0, 0.0, 1.0], cmap=cmap_discrete,
              vmin=-0.5, vmax=2.5)
    ax.set_xlabel("Mean fidelity of occupied qubits")
    ax.set_ylabel("Fraction of qubits occupied")
    ax.set_title(
        f"Preferred action at node {probe} ({n_nodes}-chain, "
        f"src=0, dst={n_nodes-1})")
    patches = [mpatches.Patch(color=_ACOLORS[a], label=_ANAMES[a])
               for a in range(N_ACTIONS)]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    plt.tight_layout()
    fname = f"diag_best_action_map{_tag(n_nodes, probe)}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── Diagnostic 4: swap vs wait boundary f(avail, fidelity) ────────────────

def plot_swap_vs_wait(model, save_dir=".", resolution=40, device="cpu",
                      n_nodes=5, probe=2):
    """Q(swap) - Q(wait) at probe node as f(frac_available, mean_fidelity)."""
    avail_range = np.linspace(0.0, 1.0, resolution)
    fid_range   = np.linspace(0.25, 1.0, resolution)
    preference  = np.zeros((resolution, resolution))

    for i, avail in enumerate(avail_range):
        for j, fid in enumerate(fid_range):
            occ      = max(avail, 0.5)
            can_swap = 1.0 if avail >= 0.5 else 0.0
            pf = np.array([occ, fid, 0, 0, avail, can_swap, 0, T_REM],
                          dtype=np.float32)
            feats = _neutral_chain(n_nodes, probe, pf)
            for k in range(1, n_nodes - 1):
                if k != probe:
                    feats[k] = [occ, fid, 0, 0, avail, can_swap, 0, T_REM]

            obs = _make_obs(n_nodes, feats)
            q   = _get_q(model, obs, device)
            preference[i, j] = q[probe, SWAP] - q[probe, NOOP]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(preference, origin="lower", aspect="auto",
                   extent=[0.25, 1.0, 0.0, 1.0], cmap="RdYlBu_r")
    ax.contour(fid_range, avail_range, preference, levels=[0],
               colors="black", linewidths=2)
    ax.set_xlabel("Mean fidelity")
    ax.set_ylabel("Fraction available qubits")
    ax.set_title(
        f"$Q(\\mathrm{{swap}}) - Q(\\mathrm{{wait}})$ at node {probe} "
        f"({n_nodes}-chain)")
    fig.colorbar(im, ax=ax, label="Swap preference")
    plt.tight_layout()
    fname = f"diag_swap_vs_wait{_tag(n_nodes, probe)}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


# ── entry point ────────────────────────────────────────────────────────────

def run_all(model_path: str, save_dir: str = "diagnostics",
            device: str = "cpu", n_nodes: int = 5, probe: int = 2):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)
    print(f"Running diagnostics — node {probe} of {n_nodes}-chain "
          f"(saving to {save_dir}/)...\n")

    plot_swap_preference(model,  save_dir, device=device, n_nodes=n_nodes, probe=probe)
    plot_purify_preference(model, save_dir, device=device, n_nodes=n_nodes, probe=probe)
    plot_best_action_map(model,  save_dir, device=device, n_nodes=n_nodes, probe=probe)
    plot_swap_vs_wait(model,     save_dir, device=device, n_nodes=n_nodes, probe=probe)

    print(f"\nAll diagnostics saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QRN Agent Diagnostics")
    parser.add_argument("--model",    default="checkpoints/cluster_004/policy.pth")
    parser.add_argument("--save_dir", default="checkpoints/cluster_004/diagnostics")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--n_nodes",  type=int, default=5)
    parser.add_argument("--probe",    type=int, default=2)
    args = parser.parse_args()

    run_all(args.model, args.save_dir, args.device, args.n_nodes, args.probe)
