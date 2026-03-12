"""
Diagnostics for a trained QRNAgent.

Probes the agent's Q-value landscape by constructing synthetic observations
with controlled parameters and reading out per-node Q-values. Produces
interpretable plots showing how the agent reasons about swap, purify,
entangle, and wait decisions.

Usage:
    python -m quantum_repeater_sim.rl.diagnostics --model checkpoints/policy.pth
"""

from __future__ import annotations
import argparse, os
from typing import Optional
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

from rl_stack.model import QNetwork
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY, N_ACTIONS
from rl_stack import strategies

# ── action names for labels ───────────────────────────────────────
_ANAMES = {NOOP: "Wait", SWAP: "Swap", PURIFY: "Purify"}
_ACOLORS = {NOOP: "#aaaaaa", SWAP: "#cc4444", PURIFY: "#44aa44"}


def _make_obs(N: int, features: np.ndarray, spacing: float = 50.0):
    """Build a synthetic observation dict for a chain of N nodes.

    Args:
        features: (N, 8) float32 array of node features.
        spacing: only needed to construct edge_index (chain adjacency).
    """
    src, dst = [], []
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return {"x": features.astype(np.float32), "edge_index": edge_index}


def _get_q(model, obs, device="cpu"):
    """Run the model on an obs dict and return (N, 4) numpy Q-values."""
    from rl_stack.agent import _obs_to_data
    data = _obs_to_data(obs, device)
    with torch.no_grad():
        q = model(data).cpu().numpy()
    return q


def load_model(model_path: str, node_dim: int = 8, hidden: int = 64,
               device: str = "cpu") -> QNetwork:
    model = QNetwork(node_dim, hidden, N_ACTIONS)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Swap preference as f(f1, f2)
# ══════════════════════════════════════════════════════════════════

def plot_swap_preference(model, save_dir=".", resolution=40, device="cpu"):
    """Heatmap of Q(swap) - Q(wait) at a middle node as a function of
    the fidelities of its two links (one left, one right).

    Setup: 5-node chain, source=0, dest=4, probe node=2.
    Node 2 has can_swap=1, frac_occupied swept, mean_fidelity swept.
    """
    N = 5
    f_range = np.linspace(0.25, 1.0, resolution)
    q_swap_adv = np.zeros((resolution, resolution))  # Q(swap) - Q(wait)
    q_swap_raw = np.zeros((resolution, resolution))

    for i, f1 in enumerate(f_range):
        for j, f2 in enumerate(f_range):
            features = np.zeros((N, 8), dtype=np.float32)
            # Source (node 0): has one link
            features[0, :7] = [0.25, f1, 1, 0, 0.25, 0, 0]
            # Node 1: intermediate, has links
            features[1, :7] = [0.5, (f1 + 0.5) / 2, 0, 0, 0.25, 1, 0]
            # Node 2 (probe): has two links with f1 and f2
            mean_f = (f1 + f2) / 2.0
            features[2, :7] = [0.5, mean_f, 0, 0, 0.0, 1, 0]
            # Node 3: intermediate
            features[3, :7] = [0.5, (f2 + 0.5) / 2, 0, 0, 0.25, 1, 0]
            # Dest (node 4)
            features[4, :7] = [0.25, f2, 0, 1, 0.25, 0, 0]

            obs = _make_obs(N, features)
            q = _get_q(model, obs, device)  # (5, 4)
            q_swap_adv[i, j] = q[2, SWAP] - q[2, NOOP]
            q_swap_raw[i, j] = q[2, SWAP]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Advantage: Q(swap) - Q(wait)
    im0 = axes[0].imshow(q_swap_adv, origin="lower", aspect="auto",
                          extent=[0.25, 1.0, 0.25, 1.0], cmap="RdBu_r")
    axes[0].set_xlabel("Right link fidelity $F_2$")
    axes[0].set_ylabel("Left link fidelity $F_1$")
    axes[0].set_title("$Q(\\mathrm{swap}) - Q(\\mathrm{wait})$ at node 2")
    axes[0].contour(f_range, f_range, q_swap_adv, levels=[0],
                     colors="black", linewidths=1.5)
    fig.colorbar(im0, ax=axes[0], label="Advantage")

    # Raw Q(swap)
    im1 = axes[1].imshow(q_swap_raw, origin="lower", aspect="auto",
                          extent=[0.25, 1.0, 0.25, 1.0], cmap="viridis")
    axes[1].set_xlabel("Right link fidelity $F_2$")
    axes[1].set_ylabel("Left link fidelity $F_1$")
    axes[1].set_title("$Q(\\mathrm{swap})$ at node 2")
    fig.colorbar(im1, ax=axes[1], label="Q-value")

    fig.suptitle("Swap preference at a middle node (5-node chain, src=0, dst=4)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_swap_preference.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved diag_swap_preference.png")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Purify preference as f(f1, f2)
# ══════════════════════════════════════════════════════════════════

def plot_purify_preference(model, save_dir=".", resolution=40, device="cpu"):
    """Heatmap of Q(purify) - Q(wait) at a node with two links to the
    same neighbour (purification scenario)."""
    N = 5
    f_range = np.linspace(0.25, 1.0, resolution)
    q_pur_adv = np.zeros((resolution, resolution))

    for i, f1 in enumerate(f_range):
        for j, f2 in enumerate(f_range):
            features = np.zeros((N, 8), dtype=np.float32)
            mean_f = (f1 + f2) / 2.0
            # Node 1 (probe): has 2 links to same neighbor → can purify
            features[0, :7] = [0.25, 0.5, 1, 0, 0.0, 0, 0]  # source
            features[1, :7] = [0.5, mean_f, 0, 0, 0.0, 0, 1]  # probe: can purify
            features[2, :7] = [0.5, mean_f, 0, 0, 0.0, 0, 0]
            features[3, :7] = [0.25, 0.5, 0, 0, 0.25, 0, 0]
            features[4, :7] = [0.0, 0.0, 0, 1, 0.0, 0, 0]  # dest

            obs = _make_obs(N, features)
            q = _get_q(model, obs, device)
            q_pur_adv[i, j] = q[1, PURIFY] - q[1, NOOP]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(q_pur_adv, origin="lower", aspect="auto",
                    extent=[0.25, 1.0, 0.25, 1.0], cmap="PiYG")
    ax.set_xlabel("Second pair fidelity $F_2$")
    ax.set_ylabel("First pair fidelity $F_1$")
    ax.set_title("$Q(\\mathrm{purify}) - Q(\\mathrm{wait})$ at node 1")
    ax.contour(f_range, f_range, q_pur_adv, levels=[0],
                colors="black", linewidths=1.5)
    fig.colorbar(im, ax=ax, label="Advantage")
    fig.suptitle("Purify preference (2 pairs to same neighbour)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_purify_preference.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved diag_purify_preference.png")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Preferred action vs node position
# ══════════════════════════════════════════════════════════════════

def plot_action_vs_position(model, save_dir=".", device="cpu"):
    """For chains of N=5..12, show the preferred action at each node
    when all interior nodes have 2 links (can_swap=1) and moderate
    fidelity. Source=0, dest=N-1."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    axes = axes.flatten()

    for idx, N in enumerate(range(5, 13)):
        features = np.zeros((N, 8), dtype=np.float32)
        for i in range(N):
            is_src = 1.0 if i == 0 else 0.0
            is_dst = 1.0 if i == N - 1 else 0.0
            is_end = (i == 0 or i == N - 1)
            frac_occ = 0.25 if is_end else 0.5
            can_sw = 0.0 if is_end else 1.0
            features[i, :7] = [frac_occ, 0.7, is_src, is_dst, 0.25, can_sw, 0]

        obs = _make_obs(N, features)
        q = _get_q(model, obs, device)  # (N, 4)

        ax = axes[idx]
        x = np.arange(N)
        width = 0.18
        for a in range(N_ACTIONS):
            ax.bar(x + a * width - 0.27, q[:, a], width,
                   label=_ANAMES[a], color=_ACOLORS[a], alpha=0.8)

        best = q.argmax(axis=1)
        for i in range(N):
            ax.scatter(i, q[i, best[i]] + 0.5, marker="v", color="black", s=15, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i}" for i in range(N)], fontsize=7)
        ax.set_title(f"N={N}", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        if idx == 0:
            ax.set_ylabel("Q-value")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Q-values per action at each node (all interior nodes can swap, F≈0.7)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_action_vs_position.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved diag_action_vs_position.png")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Best action map (action type heatmap)
# ══════════════════════════════════════════════════════════════════

def plot_best_action_map(model, save_dir=".", resolution=30, device="cpu"):
    """For the middle node of a 5-node chain, show which action is
    preferred as a function of (frac_occupied, mean_fidelity)."""
    N = 5
    occ_range = np.linspace(0.0, 1.0, resolution)
    fid_range = np.linspace(0.25, 1.0, resolution)
    best_map = np.zeros((resolution, resolution), dtype=int)

    for i, occ in enumerate(occ_range):
        for j, fid in enumerate(fid_range):
            features = np.zeros((N, 8), dtype=np.float32)
            features[0, :7] = [0.25, 0.6, 1, 0, 0.25, 0, 0]
            features[1, :7] = [occ, fid, 0, 0, max(0, occ - 0.25), occ >= 0.5, occ >= 0.5]
            features[2, :7] = [occ, fid, 0, 0, max(0, occ - 0.25), occ >= 0.5, 0]
            features[3, :7] = [occ, fid, 0, 0, max(0, occ - 0.25), occ >= 0.5, 0]
            features[4, :7] = [0.25, 0.6, 0, 1, 0.25, 0, 0]

            obs = _make_obs(N, features)
            q = _get_q(model, obs, device)
            best_map[i, j] = int(q[2].argmax())

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_discrete = plt.cm.colors.ListedColormap(
        [_ACOLORS[a] for a in range(N_ACTIONS)])
    im = ax.imshow(best_map, origin="lower", aspect="auto",
                    extent=[0.25, 1.0, 0.0, 1.0], cmap=cmap_discrete,
                    vmin=-0.5, vmax=3.5)
    ax.set_xlabel("Mean fidelity of occupied qubits")
    ax.set_ylabel("Fraction of qubits occupied")
    ax.set_title("Preferred action at middle node (5-chain, src=0, dst=4)")

    patches = [mpatches.Patch(color=_ACOLORS[a], label=_ANAMES[a])
               for a in range(N_ACTIONS)]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_best_action_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved diag_best_action_map.png")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Live episode Q-value trace
# ══════════════════════════════════════════════════════════════════

def plot_live_q_trace(model, save_dir=".", device="cpu",
                      N=6, p_gen=0.8, p_swap=0.7, cutoff=15,
                      max_steps=40, seed=42):
    """Run a live episode with the greedy policy and plot Q-values at
    each step for every node, coloured by chosen action."""
    from rl_stack.agent import QRNAgent, _obs_to_data

    env = QRNEnv(n_repeaters=N, n_ch=4, spacing=50.0,
                 p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                 dt_seconds=0.0, max_steps=max_steps,
                 rng=np.random.default_rng(seed))
    obs = env.reset()

    # Traces: q_trace[node][step] = (q_values, chosen_action)
    q_trace = {i: [] for i in range(N)}
    step = 0

    while step < max_steps:
        q = _get_q(model, obs, device)
        mask = env.get_action_mask()
        q_masked = q.copy()
        q_masked[~mask] = -float("inf")
        actions = q_masked.argmax(axis=1)

        for i in range(N):
            q_trace[i].append((q[i].copy(), int(actions[i])))

        obs, _, done, info = env.step(actions)
        step += 1
        if done:
            break

    # Plot: one subplot per node
    n_cols = min(N, 4)
    n_rows = (N + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True, squeeze=False)
    steps_arr = np.arange(len(q_trace[0]))

    for i in range(N):
        ax = axes[i // n_cols][i % n_cols]
        qs = np.array([t[0] for t in q_trace[i]])  # (steps, 4)
        chosen = [t[1] for t in q_trace[i]]

        for a in range(N_ACTIONS):
            ax.plot(steps_arr, qs[:, a], color=_ACOLORS[a],
                    alpha=0.5, linewidth=1, label=_ANAMES[a] if i == 0 else None)

        # Mark chosen action
        for t in range(len(chosen)):
            ax.scatter(t, qs[t, chosen[t]], color=_ACOLORS[chosen[t]],
                       s=12, zorder=5, edgecolors="black", linewidths=0.3)

        src_tag = " (SRC)" if i == env.source else (" (DST)" if i == env.dest else "")
        ax.set_title(f"R{i}{src_tag}", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    # Remove unused axes
    for i in range(N, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Q-value traces during greedy episode (N={N}, steps={step},"
                 f" F={info.get('fidelity', 0):.3f})", fontsize=11)
    fig.supxlabel("Step")
    fig.supylabel("Q-value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_q_trace.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved diag_q_trace.png (episode ran {step} steps, "
          f"F={info.get('fidelity', 0):.3f})")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 6: Swap vs Entangle boundary
# ══════════════════════════════════════════════════════════════════

def plot_swap_vs_wait(model, save_dir=".", resolution=40, device="cpu"):
    """When should a node swap vs try to wait more? Show the
    decision boundary as a function of (frac_available, mean_fidelity)."""
    N = 5
    avail_range = np.linspace(0.0, 1.0, resolution)
    fid_range = np.linspace(0.25, 1.0, resolution)
    preference = np.zeros((resolution, resolution))

    for i, avail in enumerate(avail_range):
        for j, fid in enumerate(fid_range):
            occ = max(avail, 0.5)  # at least 2 qubits occupied for swap
            features = np.zeros((N, 8), dtype=np.float32)
            features[0, :7] = [0.25, 0.6, 1, 0, 0.25, 0, 0]
            features[1, :7] = [occ, fid, 0, 0, avail, 1, 0]
            features[2, :7] = [occ, fid, 0, 0, avail, 1, 0]
            features[3, :7] = [occ, fid, 0, 0, avail, 1, 0]
            features[4, :7] = [0.25, 0.6, 0, 1, 0.25, 0, 0]

            obs = _make_obs(N, features)
            q = _get_q(model, obs, device)
            # Preference: positive means swap preferred over wait
            preference[i, j] = q[2, SWAP] - q[2, NOOP]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(preference, origin="lower", aspect="auto",
                    extent=[0.25, 1.0, 0.0, 1.0], cmap="RdYlBu_r")
    ax.contour(fid_range, avail_range, preference, levels=[0],
                colors="black", linewidths=2)
    ax.set_xlabel("Mean fidelity")
    ax.set_ylabel("Fraction available qubits")
    ax.set_title("$Q(\\mathrm{swap}) - Q(\\mathrm{wait})$ at interior node")
    fig.colorbar(im, ax=ax, label="Swap preference")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diag_swap_vs_wait.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  Saved diag_swap_vs_wait.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run_all(model_path: str, save_dir: str = "diagnostics", device: str = "cpu"):
    """Run all diagnostics and save plots."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)
    print(f"Running diagnostics (saving to {save_dir}/)...\n")

    plot_swap_preference(model, save_dir, device=device)
    plot_purify_preference(model, save_dir, device=device)
    plot_action_vs_position(model, save_dir, device=device)
    plot_best_action_map(model, save_dir, device=device)
    plot_swap_vs_wait(model, save_dir, device=device)
    plot_live_q_trace(model, save_dir, device=device)

    print(f"\nAll diagnostics saved to {save_dir}/")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="QRN Agent Diagnostics")
    # parser.add_argument("--model", required=True, help="Path to policy.pth")
    # parser.add_argument("--save-dir", default="Save directory")
    # parser.add_argument("--device", default="cpu")
    # args = parser.parse_args()
    # run_all(args.model, args.save_dir, args.device)

    run_all(model_path='checkpoints/policy.pth', save_dir='diagnostics/figures', device='cpu')