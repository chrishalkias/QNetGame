"""Double-DQN agent for quantum repeater network routing.

The agent learns a per-node policy on small networks and generalises
zero-shot to larger, differently-parameterised ones.
"""

from __future__ import annotations
import math, os, time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

from .model import QNetwork
from .buffer import ReplayBuffer
from .env_wrapper import QRNEnv, N_ACTIONS, NOOP, ENTANGLE, SWAP, PURIFY, ACTION_NAMES
from . import strategies

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _obs_to_data(obs: Dict[str, np.ndarray], device="cpu") -> Data:
    """Convert env observation dict to a PyG Data object."""
    return Data(
        x=torch.tensor(obs["x"], dtype=torch.float32, device=device),
        edge_index=torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
    )


def _running_avg(vals, window=20):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(np.mean(vals[lo:i+1]))
    return out


# ── repeater colour palette ──────────────────────────────────────

def _repeater_colors(N: int):
    """Return N distinct RGBA colours for repeaters."""
    cmap = plt.cm.tab10 if N <= 10 else plt.cm.tab20
    return [to_rgba(cmap(i / max(N - 1, 1))) for i in range(N)]


# Action type → hatch pattern
_ACTION_HATCH = {NOOP: "",  ENTANGLE: "",  SWAP: "///",  PURIFY: "..."}
_ACTION_LABEL = {NOOP: "Wait", ENTANGLE: "Entangle", SWAP: "Swap", PURIFY: "Purify"}


# ══════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════

class QRNAgent:
    """Double-DQN agent with per-node Q-values on a GNN backbone.

    The agent selects one of {noop, entangle, swap, purify} for every
    repeater node simultaneously. Training uses shared global reward
    broadcast to each node.
    """

    def __init__(self, 
                 node_dim: int = 7, 
                 hidden: int = 64,
                 lr: float = 5e-4, 
                 gamma: float = 0.99,
                 buffer_size: int = 50_000, 
                 batch_size: int = 64,
                 tau: float = 0.005, 
                 epsilon: float = 1.0
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon

        self.policy_net = QNetwork(node_dim, hidden, N_ACTIONS).to(self.device)
        self.target_net = QNetwork(node_dim, hidden, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(max_size=buffer_size)

    # ── action selection ──────────────────────────────────────────

    def select_actions(self, 
                       obs: Dict[str, np.ndarray],
                       mask: np.ndarray, 
                       training: bool = True
                       ) -> np.ndarray:
        """Select one action per node via epsilon-greedy over masked Q-values.

        Args:
            obs: env observation dict with 'x' and 'edge_index'.
            mask: (N, 4) boolean action mask.
            training: if True, apply epsilon-greedy exploration.

        Returns:
            (N,) int32 action array.
        """
        N = mask.shape[0]
        data = _obs_to_data(obs, self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q = self.policy_net(data)  # (N, 4)
            q[~mask_t] = -float("inf")

        if training and np.random.random() < self.epsilon:
            # Epsilon-greedy: random valid action per node
            actions = np.zeros(N, dtype=np.int32)
            for i in range(N):
                valid = np.flatnonzero(mask[i])
                actions[i] = np.random.choice(valid) if len(valid) > 0 else NOOP
        else:
            actions = q.argmax(dim=1).cpu().numpy().astype(np.int32)

        return actions

    # ── training step ─────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """Sample batch, compute double-DQN loss, backprop. Returns loss or None."""
        if self.memory.size() < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        # Build PyG batches
        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.device)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(self.device)

        # Rewards and dones: broadcast to every node in each graph
        rewards_per_graph = torch.tensor(
            [t["r"] for t in batch], dtype=torch.float32, device=self.device)
        dones_per_graph = torch.tensor(
            [float(t["d"]) for t in batch], dtype=torch.float32, device=self.device)

        # Map each node to its graph index
        node_to_graph = states.batch  # (total_nodes,)
        rewards = rewards_per_graph[node_to_graph]  # broadcast to nodes
        dones = dones_per_graph[node_to_graph]

        # Actions: concatenate per-node actions across batch
        actions = torch.cat(
            [torch.tensor(t["a"], dtype=torch.long, device=self.device)
             for t in batch])  # (total_nodes,)

        # ── Current Q ──
        q_all = self.policy_net(states)  # (total_nodes, 4)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Target Q (double DQN) ──
        with torch.no_grad():
            # Policy net picks best action
            next_q_policy = self.policy_net(next_states)  # (total_nodes, 4)
            best_actions = next_q_policy.argmax(dim=1)  # (total_nodes,)

            # Target net evaluates that action
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak update
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return loss.item()

    # ── training loop ─────────────────────────────────────────────

    def train(self, 
              episodes: int = 2000, 
              max_steps: int = 50,
              n_range: List[int] = [4, 5, 6], 
              jitter: int = 1,
              p_gen: float = 0.8, 
              p_swap: float = 0.7, 
              cutoff: int = 50,
              F0: float = 0.95, 
              channel_loss: float = 0.02,
              dt_seconds: float = 1e-3,
              heterogeneous: bool = True,
              curriculum: bool = True,
              save_path: Optional[str] = None,
              plot: bool = True) -> Dict[str, list]:
        """Train the agent.

        Args:
            episodes: number of training episodes.
            n_range: range of chain sizes to sample from.
            jitter: re-sample N every `jitter` episodes.
            heterogeneous: randomise per-repeater params each episode.
            curriculum: progressive difficulty (small → large N).
            save_path: directory to save model checkpoints.

        Returns:
            Dict of training metrics lists.
        """
        metrics = {"reward": [], "loss": [], "steps": [], "success": []}
        n_nodes = np.random.choice(n_range)
        eps_init, eps_fin = 1.0, 0.05

        try:
            for ep in range(episodes):
                # Curriculum / jitter
                if jitter and ep % jitter == 0:
                    if curriculum:
                        prog = ep / max(episodes, 1)
                        if prog < 0.15:
                            pool = [r for r in n_range if r <= min(n_range) + 1]
                        elif prog < 0.6:
                            mid = (min(n_range) + max(n_range)) // 2
                            pool = [r for r in n_range if r <= mid + 1]
                        else:
                            pool = n_range
                        n_nodes = np.random.choice(pool)
                    else:
                        n_nodes = np.random.choice(n_range)

                env = QRNEnv(n_repeaters=n_nodes, 
                             n_ch=4, 
                             spacing=50.0,
                             p_gen=p_gen, 
                             p_swap=p_swap, 
                             cutoff=cutoff,
                             F0=F0, 
                             channel_loss=channel_loss,
                             dt_seconds=dt_seconds, 
                             max_steps=max_steps,
                             rng=np.random.default_rng(),
                             heterogeneous=heterogeneous
                             )
                obs = env.reset()
                score = 0.0
                ep_loss = []

                for _ in range(max_steps):
                    mask = env.get_action_mask()
                    actions = self.select_actions(obs, mask, training=True)
                    next_obs, reward, done, info = env.step(actions)

                    self.memory.add(obs, actions, reward, next_obs, done)
                    loss = self.train_step()
                    if loss is not None:
                        ep_loss.append(loss)

                    obs = next_obs
                    score += reward

                    if done:
                        break

                # Epsilon decay (cosine annealing)
                self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (
                    1 + math.cos(math.pi * ep / max(episodes, 1)))

                metrics["reward"].append(score)
                metrics["loss"].append(np.mean(ep_loss) if ep_loss else 0.0)
                metrics["steps"].append(env.steps)
                metrics["success"].append(1.0 if info.get("fidelity", 0) > 0 else 0.0)

                if ep % 100 == 0 or ep == episodes - 1:
                    avg_r = np.mean(metrics["reward"][-100:])
                    avg_s = np.mean(metrics["success"][-100:])
                    print(f"Ep {ep:>5d} | R={avg_r:>7.2f} | "
                          f"succ={avg_s:.2f} | eps={self.epsilon:.3f} | N={n_nodes}")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.policy_net.state_dict(),
                       os.path.join(save_path, "policy.pth"))

        if plot:
            self._plot_training(metrics)

        return metrics

    # ── validation ────────────────────────────────────────────────

    def validate(self, model_path=None,
                    n_episodes=100, max_steps=50,
                    n_repeaters=8, p_gen=0.8, p_swap=0.7,
                    cutoff=15, F0=0.95, channel_loss=0.02,
                    dt_seconds=1e-3, plot_actions=True, save_dir=".", ee=True):
        """
        Validate agent vs baselines. Timeline records agent-chosen actions only.
        """
        if model_path is not None:
            self.policy_net.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True))

        old_eps = self.epsilon
        self.epsilon = 0.0

        strat_fns = {
            "Agent": None,
            "SwapASAP": strategies.swap_asap,
            "PurifySwap": strategies.purify_then_swap,
            "Random": None,
        }
        results = {k: {"steps": [], "fidelities": []} for k in strat_fns}
        # Timeline: {strategy: list of (N,) action arrays per timestep}
        timelines = {k: [] for k in strat_fns}

        for name, fn in strat_fns.items():
            for ep in range(n_episodes):
                env = QRNEnv(n_repeaters=n_repeaters, n_ch=4, spacing=50.0,
                                p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                                F0=F0, channel_loss=channel_loss,
                                dt_seconds=dt_seconds, max_steps=max_steps,
                                rng=np.random.default_rng(ep),ee=True)
                obs = env.reset()
                done, fid = False, 0.0

                for step in range(max_steps):
                    mask = env.get_action_mask()
                    if name == "Agent":
                        actions = self.select_actions(obs, mask, training=False)
                    elif name == "Random":
                        actions = strategies.random_policy(env, env.rng)
                    else:
                        actions = fn(env)

                    # Record ONLY agent-chosen actions (first episode)
                    if ep == 0 and plot_actions:
                        timelines[name].append(actions.copy())

                    obs, reward, done, info = env.step(actions)
                    fid = info.get("fidelity", 0.0)
                    if done:
                        break

                results[name]["steps"].append(step + 1 if done and fid > 0 else max_steps)
                if fid > 0:
                    results[name]["fidelities"].append(fid)

        self.epsilon = old_eps
        self._print_results_table(results, n_repeaters, p_gen, p_swap, cutoff)

        if plot_actions:
            self._plot_timeline_grid(timelines, n_repeaters, p_gen, p_swap,
                                        cutoff, save_dir)
        return results

    # ── plotting ──────────────────────────────────────────────────

    def _plot_training(self, metrics):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(metrics["reward"], alpha=0.3, color="blue")
        axes[0].plot(_running_avg(metrics["reward"]), color="blue")
        axes[0].set_ylabel("Reward"); axes[0].set_title("Training Metrics")
        axes[1].plot(metrics["loss"], alpha=0.3, color="red")
        axes[1].plot(_running_avg(metrics["loss"]), color="red")
        axes[1].set_ylabel("Loss"); axes[1].set_yscale("log")
        axes[2].plot(_running_avg(metrics["success"], 50), color="green")
        axes[2].set_ylabel("Success Rate"); axes[2].set_xlabel("Episode")
        plt.tight_layout()
        plt.savefig("assets/training_metrics.png", dpi=250)
        plt.close()

    @staticmethod
    def _print_results_table(results, N, pg, ps, c):
        print(f"\n{'='*70}")
        print(f"Validation: N={N}, p_gen={pg}, p_swap={ps}, cutoff={c}")
        print(f"{'='*70}")
        agent_avg = np.mean(results["Agent"]["steps"])
        pm = "\u00B1"
        print(f"{'Strategy':<14} | {'Avg Steps':>12} | {'Avg Fidelity':>14} | "
                f"{'S%':>5} | {'Succ':>5}")
        print("-" * 70)
        for name, data in results.items():
            avg_s, std_s = np.mean(data["steps"]), np.std(data["steps"])
            ns = len(data["fidelities"])
            avg_f = np.mean(data["fidelities"]) if ns > 0 else 0.0
            std_f = np.std(data["fidelities"]) if ns > 0 else 0.0
            spct = (avg_s / agent_avg * 100) if agent_avg > 0 else 0
            succ = ns / max(len(data["steps"]), 1) * 100
            print(f"{name:<14} | {avg_s:>5.1f}{pm}{std_s:<5.1f} | "
                    f"{avg_f:>6.4f}{pm}{std_f:<6.4f} | {spct:>4.0f}% | {succ:>4.0f}%")

    @staticmethod
    def _plot_timeline_grid(timelines, N, pg, ps, c, save_dir="."):
        """Plot action timeline: colour = repeater, pattern = action type.

        Each cell in the grid represents one node at one timestep.
        The grid has shape (n_strategies, max_steps), each cell is a
        vertical stack of N thin bars.
        """
        strats = list(timelines.keys())
        n_strats = len(strats)
        max_steps = max((len(tl) for tl in timelines.values()), default=1)
        rep_colors = _repeater_colors(N)

        # Cell height per strategy, bar height per repeater within that
        fig_w = min(max_steps * 0.35 + 3, 22)
        fig_h = n_strats * 1.4 + 1.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        row_height = 1.0
        bar_height = row_height / N

        for si, sname in enumerate(strats):
            tl = timelines[sname]
            y_base = (n_strats - 1 - si) * (row_height + 0.3)

            for t, actions in enumerate(tl):
                for node in range(N):
                    a = int(actions[node]) if node < len(actions) else NOOP
                    y = y_base + node * bar_height
                    color = rep_colors[node]
                    hatch = _ACTION_HATCH.get(a, "")

                    rect = mpatches.FancyBboxPatch(
                        (t - 0.45, y), 0.9, bar_height * 0.9,
                        boxstyle="square,pad=0",
                        facecolor=color, edgecolor="none", linewidth=0)
                    ax.add_patch(rect)

                    if hatch:
                        h_rect = mpatches.FancyBboxPatch(
                            (t - 0.45, y), 0.9, bar_height * 0.9,
                            boxstyle="square,pad=0",
                            facecolor="none", edgecolor="black",
                            hatch=hatch, linewidth=0, alpha=0.6)
                        ax.add_patch(h_rect)

        # Axes
        y_positions = [(n_strats - 1 - i) * (row_height + 0.3) + row_height / 2
                        for i in range(n_strats)]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(strats)
        ax.set_xlim(-0.5, max_steps + 0.5)
        ax.set_ylim(-0.3, n_strats * (row_height + 0.3))
        ax.set_xlabel("Time Step")
        ax.set_title(f"Policy Actions (N={N}, pg={pg}, ps={ps}, c={c})")
        ax.grid(False)

        # Legend: repeater colours + action patterns
        handles = []
        for i in range(N):
            handles.append(mpatches.Patch(facecolor=rep_colors[i],
                                            label=f"R{i}", edgecolor="grey",
                                            linewidth=0.5))
        handles.append(mpatches.Patch(facecolor="white", edgecolor="grey",
                                        label="Wait / Entangle"))
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                        hatch="///", label="Swap"))
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                        hatch="...", label="Purify"))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5),
                    title="Legend", fontsize=7)
        plt.savefig(os.path.join(save_dir, "validation_actions.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()