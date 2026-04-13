"""PCA visualisation of a trained policy's internal representations.

Sweeps synthetic observations across (fidelity, occupancy, time_remaining)
for the middle node of a chain, captures the post-conv3 embeddings
via a forward hook, then projects to 2D with PCA.

Four panels:
  1. Coloured by preferred action   (Wait / Swap / Purify)
  2. Coloured by mean fidelity
  3. Coloured by occupancy
  4. Coloured by time remaining

Usage:
    python -m diagnostics.policy_probes.PCA_viz
    # or with a custom checkpoint:
    PYTHONPATH=. python diagnostics/policy_probes/PCA_viz.py \
        --model checkpoints/cluster_008/policy.pth \
        --save_dir checkpoints/cluster_008/diagnostics
"""

from __future__ import annotations
import argparse
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from sklearn.decomposition import PCA

from rl_stack.model import QNetwork
from rl_stack.env_wrapper import N_ACTIONS, NOOP, SWAP, PURIFY
from rl_stack.agent import _obs_to_data

# ── constants ──────────────────────────────────────────────────────────────
ACTION_NAMES  = {NOOP: "Wait", SWAP: "Swap", PURIFY: "Purify"}
ACTION_COLORS = {NOOP: "#aaaaaa", SWAP: "#cc4444", PURIFY: "#44aa44"}
N_NODES       = 5
PROBE_NODE    = 2          # middle node of the 5-chain
RESOLUTION    = 25         # grid points per axis (25³ = 15 625 observations)
DEFAULT_T_REM = 0.5        # fixed time-remaining for neighbour nodes


# ── helpers ────────────────────────────────────────────────────────────────
def load_model(path: str, hidden: int = 64, device: str = "cpu") -> QNetwork:
    model = QNetwork(node_dim=8, hidden=hidden, n_actions=N_ACTIONS)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _make_obs(fid: float, occ: float, t_rem: float,
              n_nodes: int = N_NODES, probe: int = PROBE_NODE) -> dict:
    """Synthetic n_nodes chain; sweep probe node features.

    All non-probe interior nodes get neutral mid-range features.
    Source = node 0, dest = node n_nodes-1.
    """
    avail    = occ
    can_swap = 1.0 if avail >= 0.5 else 0.0
    feats    = np.zeros((n_nodes, 8), dtype=np.float32)

    for i in range(n_nodes):
        if i == 0:
            feats[i] = [0.25, 0.70, 1, 0, 0.25, 0, 0, DEFAULT_T_REM]
        elif i == n_nodes - 1:
            feats[i] = [0.25, 0.70, 0, 1, 0.25, 0, 0, DEFAULT_T_REM]
        elif i == probe:
            feats[i] = [occ, fid, 0, 0, avail, can_swap, 0, t_rem]
        else:
            feats[i] = [0.50, 0.70, 0, 0, 0.50, 1, 0, DEFAULT_T_REM]

    src, dst = [], []
    for i in range(n_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return {"x": feats,
            "edge_index": np.array([src, dst], dtype=np.int64)}


def collect_embeddings(
    model: QNetwork,
    device: str = "cpu",
    resolution: int = RESOLUTION,
    n_nodes: int = N_NODES,
    probe: int = PROBE_NODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    embeddings : (M, 64)  — post-conv3 node embedding for the probe node
    features   : (M, 3)   — [fidelity, occupancy, time_remaining]
    best_action: (M,)     — argmax Q-value index
    """
    fid_vals   = np.linspace(0.25, 1.0, resolution)
    occ_vals   = np.linspace(0.0,  1.0, resolution)
    t_rem_vals = np.linspace(0.05, 1.0, resolution)

    embeddings   = []
    features_out = []
    best_actions = []

    # Register hook on conv3 to capture node embeddings after the 3rd SAGE layer
    _hook_store: list[torch.Tensor] = []

    def _hook(module, input, output):  # noqa: ARG001
        _hook_store.append(output.detach().cpu())

    handle = model.conv3.register_forward_hook(_hook)

    with torch.no_grad():
        for fid in fid_vals:
            for occ in occ_vals:
                for t_rem in t_rem_vals:
                    obs  = _make_obs(float(fid), float(occ), float(t_rem),
                                     n_nodes, probe)
                    data = _obs_to_data(obs, device)

                    _hook_store.clear()
                    q = model(data)          # triggers hook

                    conv3_out  = _hook_store[0]          # (n_nodes, 64)
                    probe_emb  = conv3_out[probe].numpy()
                    best_a     = int(q[probe].argmax())

                    embeddings.append(probe_emb)
                    features_out.append([fid, occ, t_rem])
                    best_actions.append(best_a)

    handle.remove()

    return (np.array(embeddings, dtype=np.float32),
            np.array(features_out, dtype=np.float32),
            np.array(best_actions, dtype=np.int32))


# ── plotting ───────────────────────────────────────────────────────────────
def plot_pca(
    model_path: str,
    save_dir: str = ".",
    resolution: int = RESOLUTION,
    device: str = "cpu",
    n_nodes: int = N_NODES,
    probe: int = PROBE_NODE,
):
    os.makedirs(save_dir, exist_ok=True)

    # Derive run name from model path (e.g. "checkpoints/cluster_008/policy.pth" → "cluster_008")
    run_name = os.path.basename(os.path.dirname(os.path.abspath(model_path)))

    print(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)

    print(f"Collecting embeddings ({resolution}³ = {resolution**3} observations) "
          f"— node {probe} of {n_nodes}-chain…")
    embeddings, feats, actions = collect_embeddings(
        model, device, resolution, n_nodes, probe)

    print("Running PCA…")
    pca   = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)   # (M, 2)
    var1, var2 = pca.explained_variance_ratio_ * 100

    print(f"  PC1 {var1:.1f}%   PC2 {var2:.1f}%")

    fid_vals   = feats[:, 0]
    occ_vals   = feats[:, 1]
    t_rem_vals = feats[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"PCA of conv3 embeddings — probe node {probe} of {n_nodes}-chain\n"
        f"({run_name}, PC1={var1:.1f}%, PC2={var2:.1f}%)",
        fontsize=11)

    dot = dict(s=4, alpha=0.5, linewidths=0)

    # ── Panel 1: preferred action ──────────────────────────────────────────
    ax = axes[0, 0]
    for a in [NOOP, SWAP, PURIFY]:
        mask = actions == a
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=ACTION_COLORS[a], label=ACTION_NAMES[a], **dot)
    ax.set_title("Preferred action")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")

    # ── Panel 2: mean fidelity ─────────────────────────────────────────────
    ax = axes[0, 1]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=fid_vals, cmap="plasma", **dot)
    fig.colorbar(sc, ax=ax, label="Mean fidelity")
    ax.set_title("Mean fidelity")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")

    # ── Panel 3: occupancy ────────────────────────────────────────────────
    ax = axes[1, 0]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=occ_vals, cmap="viridis", **dot)
    fig.colorbar(sc, ax=ax, label="Frac qubits occupied")
    ax.set_title("Occupancy")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")

    # ── Panel 4: time remaining ────────────────────────────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=t_rem_vals, cmap="coolwarm", **dot)
    fig.colorbar(sc, ax=ax, label="Time remaining")
    ax.set_title("Time remaining")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"diag_pca_embeddings_n{n_nodes}_p{probe}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # ── Variance explained bar chart ───────────────────────────────────────
    pca_full = PCA(n_components=min(64, len(embeddings)))
    pca_full.fit(embeddings)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(range(1, len(cumvar) + 1),
            pca_full.explained_variance_ratio_ * 100,
            color="steelblue", alpha=0.7, label="Per-PC variance")
    ax2.plot(range(1, len(cumvar) + 1), cumvar,
             color="darkorange", lw=2, marker=".", label="Cumulative")
    ax2.axhline(90, color="grey", linestyle="--", lw=1, label="90% threshold")
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Variance explained (%)")
    ax2.set_title(f"PCA variance explained — conv3 embeddings "
                  f"({run_name}, node {probe} of {n_nodes}-chain)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.5, min(20, len(cumvar)) + 0.5)
    plt.tight_layout()
    var_path = os.path.join(save_dir, f"diag_pca_variance_n{n_nodes}_p{probe}.png")
    plt.savefig(var_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {var_path}")


# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="checkpoints/cluster_004/policy.pth")
    parser.add_argument("--save_dir",   default="checkpoints/cluster_004/diagnostics")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_nodes",    type=int, default=N_NODES)
    parser.add_argument("--probe",      type=int, default=PROBE_NODE)
    args = parser.parse_args()

    plot_pca(args.model, args.save_dir, args.resolution,
             args.device, args.n_nodes, args.probe)
