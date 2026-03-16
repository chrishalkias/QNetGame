"""Double-DQN agent for quantum repeater network routing.

The agent learns a per-node policy on small chains and generalises
zero-shot to larger, differently-parameterised ones.

Key fixes over the original:
  - Successor action mask stored in buffer and used in target Q
    computation (prevents learning Q-values for impossible actions).
  - Action space reduced to {noop, swap, purify}; entanglement is
    background-only and not an agent decision.
  - Reward scale fixed: SUCCESS >> cumulative step penalty.
  - 3-layer GNN for 3-hop receptive field.
"""

from __future__ import annotations
import math, os
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

from rl_stack.model import QNetwork
from rl_stack.buffer import ReplayBuffer
from rl_stack.env_wrapper import QRNEnv, N_ACTIONS, NOOP, SWAP, PURIFY, ACTION_NAMES
from rl_stack import strategies

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba



                                           
                # ▄▄▄   ▄▄▄       ▄▄                         
                # ███   ███       ██                         
                # █████████ ▄█▀█▄ ██ ████▄ ▄█▀█▄ ████▄ ▄█▀▀▀ 
                # ███▀▀▀███ ██▄█▀ ██ ██ ██ ██▄█▀ ██ ▀▀ ▀███▄ 
                # ███   ███ ▀█▄▄▄ ██ ████▀ ▀█▄▄▄ ██    ▄▄▄█▀ 
                #                    ██                      
                #                    ▀▀       
                              
NODE_DIM = 8   # must match env_wrapper feature count

def _obs_to_data(obs: Dict[str, np.ndarray], device="cpu") -> Data:
    return Data(
        x=torch.tensor(obs["x"], dtype=torch.float32, device=device),
        edge_index=torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
    )


def _running_avg(vals, window=30):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(np.mean(vals[lo:i+1]))
    return out


def _repeater_colors(N: int):
    cmap = plt.cm.tab10 if N <= 10 else plt.cm.tab20
    return [to_rgba(cmap(i / max(N - 1, 1))) for i in range(N)]


_ACTION_HATCH = {NOOP: "", SWAP: "///", PURIFY: "..."}



class QRNAgent:
    """
                                                     
                  ▄▄▄▄    ▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄▄    ▄▄▄ ▄▄▄▄▄▄▄▄▄ 
                ▄██▀▀██▄ ███▀▀▀▀▀  ███▀▀▀▀▀ ████▄  ███ ▀▀▀███▀▀▀ 
                ███  ███ ███       ███▄▄    ███▀██▄███    ███    
                ███▀▀███ ███  ███▀ ███      ███  ▀████    ███    
                ███  ███ ▀██████▀  ▀███████ ███    ███    ███    
                                                                             
    Double-DQN agent with per-node Q-values on a GNN backbone.

    The agent selects one of {noop, swap, purify} for every node.
    Training uses shared global reward broadcast to each node.
    The successor action mask is stored in the buffer and applied
    during target Q-value computation to ensure physical validity.
    """

    def __init__(self, node_dim = NODE_DIM, hidden = 64,
                 lr = 3e-4, gamma = 0.99,
                 buffer_size = 80_000, batch_size = 64,
                 tau = 0.005, epsilon = 1.0):
        
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

            #  ▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄ ▄▄▄       ▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄ 
            # █████▀▀▀ ███▀▀▀▀▀ ███      ███▀▀▀▀▀ ███▀▀▀▀▀ ▀▀▀███▀▀▀ 
            #  ▀████▄  ███▄▄    ███      ███▄▄    ███         ███    
            #    ▀████ ███      ███      ███      ███         ███    
            # ███████▀ ▀███████ ████████ ▀███████ ▀███████    ███   

    def select_actions(self, obs: Dict[str, np.ndarray],
                       mask: np.ndarray, training: bool = True
                       ) -> np.ndarray:
        """ε-greedy over masked Q-values.  (N,) int32 actions."""
        N = mask.shape[0]

        if training and np.random.random() < self.epsilon:
            actions = np.zeros(N, dtype=np.int32)
            for i in range(N):
                valid = np.flatnonzero(mask[i])
                actions[i] = np.random.choice(valid) if len(valid) else NOOP
            return actions

        data = _obs_to_data(obs, self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q = self.policy_net(data)
            q[~mask_t] = -float("inf")

        return q.argmax(dim=1).cpu().numpy().astype(np.int32)
    

                                                  
        #  ▄▄▄▄▄▄▄                      ▄▄▄▄▄▄▄             
        # █████▀▀▀  ██                 ███▀▀▀▀▀             
        #  ▀████▄  ▀██▀▀ ▄█▀█▄ ████▄   ███▄▄    ████▄ ██ ██ 
        #    ▀████  ██   ██▄█▀ ██ ██   ███      ██ ██ ██▄██ 
        # ███████▀  ██   ▀█▄▄▄ ████▀   ▀███████ ██ ██  ▀█▀  
        #                      ██                           
        #                      ▀▀                           


    def train_step(self) -> Optional[float]:
        """Sample batch, compute masked Double-DQN loss, backprop."""
        if self.memory.size() < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = Batch.from_data_list(
            [_obs_to_data(t["s"]) for t in batch]).to(self.device)
        next_states = Batch.from_data_list(
            [_obs_to_data(t["s_"]) for t in batch]).to(self.device)

        # Per-graph scalars → broadcast to every node
        rewards_pg = torch.tensor(
            [t["r"] for t in batch], dtype=torch.float32, device=self.device)
        dones_pg = torch.tensor(
            [float(t["d"]) for t in batch], dtype=torch.float32, device=self.device)

        node_to_graph = states.batch
        rewards = rewards_pg[node_to_graph]
        dones   = dones_pg[node_to_graph]

        # Per-node actions (concatenated across batch)
        actions = torch.cat(
            [torch.tensor(t["a"], dtype=torch.long, device=self.device)
             for t in batch])

        # Per-node next masks (concatenated across batch)
        next_masks = torch.cat(
            [torch.tensor(t["m_"], dtype=torch.bool, device=self.device)
             for t in batch])

        # ── Current Q(s, a) ──
        q_all    = self.policy_net(states)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Target Q (Double DQN with masked next actions) ──
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_policy[~next_masks] = -float("inf")   # ← the critical fix
            best_actions = next_q_policy.argmax(dim=1)

            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Polyak update
        for p, tp in zip(self.policy_net.parameters(),
                         self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return loss.item()



        # ▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄     ▄▄▄▄   ▄▄▄▄▄ ▄▄▄    ▄▄▄ 
        # ▀▀▀███▀▀▀ ███▀▀███▄ ▄██▀▀██▄  ███  ████▄  ███ 
        #    ███    ███▄▄███▀ ███  ███  ███  ███▀██▄███ 
        #    ███    ███▀▀██▄  ███▀▀███  ███  ███  ▀████ 
        #    ███    ███  ▀███ ███  ███ ▄███▄ ███    ███ 



    def train(self, 
              episodes = 3000, 
              max_steps = 50,
              n_range = [4, 5, 6, 7],
              p_gen = 0.8, 
              p_swap = 0.7,
              cutoff = 30, 
              F0 = 0.95,
              channel_loss = 0.02, 
              dt_seconds = 1e-3,
              heterogeneous = True, 
              curriculum = True,
              topology = 'chain', 
              save_path = None,
              plot = True) -> Dict[str, list]:
        """                                                                                              
        Train with curriculum over chain sizes.

        Curriculum splits training into 3 phases:
            0-20 %   small chains only  (≤ min+1)
            20-60 %  up to median
            60-100 % full range
        """
        #TODO: Add wandb logging
        metrics = {"reward": [], "loss": [], "steps": [], "success": []}
        eps_init, eps_fin = 1.0, 0.05

        try:
            for ep in range(episodes):
                # -- Curriculum: progressively widen chain size pool --
                prog = ep / max(episodes, 1)
                if curriculum:
                    if prog < 0.20:
                        pool = [r for r in n_range if r <= min(n_range) + 1]
                    elif prog < 0.60:
                        mid = (min(n_range) + max(n_range)) // 2
                        pool = [r for r in n_range if r <= mid + 1]
                    else:
                        pool = n_range
                else:
                    pool = n_range
                n_nodes = int(np.random.choice(pool))

                args = {
                    'n_repeaters': n_nodes,
                    'n_ch': 4,
                    'spacing': 50,
                    'p_gen': p_gen,
                    'p_swap': p_swap,
                    'cutoff': cutoff,
                    'F0' : F0,
                    'channel_loss' : channel_loss,
                    'dt_seconds': dt_seconds,
                    'max_steps' : max_steps,
                    'heterogeneous' : heterogeneous,
                    'topology' : topology
                    }
                
                env = QRNEnv(**args)
                obs   = env.reset()
                score = 0.0
                ep_loss = []

                for _ in range(max_steps):
                    mask    = env.get_action_mask()
                    actions = self.select_actions(obs=obs, mask=mask, training=True)

                    next_obs, reward, done, info = env.step(actions)
                    next_mask = env.get_action_mask()

                    self.memory.add(obs, actions, reward,
                                    next_obs, done, next_mask)

                    loss = self.train_step()
                    if loss is not None:
                        ep_loss.append(loss)

                    obs   = next_obs
                    score += reward
                    if done:
                        break

                # Cosine annealing ε
                if ep < 0.9* episodes:
                    self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (
                        1 + math.cos(math.pi * ep / max(episodes, 1)))
                else:
                    self.epsilon = eps_fin

                metrics["reward"].append(score)
                metrics["loss"].append(
                    np.mean(ep_loss) if ep_loss else 0.0)
                metrics["steps"].append(env.steps)
                metrics["success"].append(
                    1.0 if info.get("fidelity", 0) > 0 else 0.0)

                if ep % 200 == 0 or ep == episodes - 1 and ep>0:
                    avg_r = np.mean(metrics["reward"][-200:])
                    avg_s = np.mean(metrics["success"][-200:])
                    print(f"Ep {ep:>5d}/{episodes} | R={avg_r:>7.3f} | "
                          f"succ={avg_s:.2f} | ε={self.epsilon:.3f} | N={n_nodes}")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.policy_net.state_dict(),
                       os.path.join(save_path, "policy.pth"))

        if plot:
            self._plot_training(metrics, save_path)

        return metrics
    

        # ▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄ 
        # ▀▀▀███▀▀▀ ███▀▀▀▀▀ █████▀▀▀ ▀▀▀███▀▀▀ 
        #    ███    ███▄▄     ▀████▄     ███    
        #    ███    ███         ▀████    ███    
        #    ███    ▀███████ ███████▀    ███    
                                      

    def validate(self, 
                 model_path=None,
                 n_episodes=100, 
                 max_steps=50,
                 n_repeaters=8, 
                 n_ch = 4,
                 p_gen=0.8, 
                 p_swap=0.7,
                 cutoff=15, 
                 F0=0.95, 
                 channel_loss=0.02,
                 dt_seconds=1e-3, 
                 plot_actions=True,
                 topology = 'chain',
                 heterogeneous = False,
                 verbose = 0,
                 save_dir="."
                ):
        """
        Validate agent vs baselines; plot action timelines."""
        if model_path is not None:
            self.policy_net.load_state_dict(
                torch.load(model_path, map_location=self.device,
                           weights_only=True))

        old_eps = self.epsilon
        self.epsilon = 0.0

        strat_fns = {
            "Agent":     None,
            "SwapASAP":  strategies.swap_asap,
            "PurifySwap": strategies.purify_then_swap,
            "Random":    None,
        }
        results   = {k: {"steps": [], "fidelities": []} for k in strat_fns}
        timelines = {k: [] for k in strat_fns}

        args = {
            'n_repeaters': n_repeaters,
            'n_ch': n_ch,
            'spacing': 50,
            'p_gen': p_gen,
            'p_swap': p_swap,
            'cutoff': cutoff,
            'F0' : F0,
            'channel_loss' : channel_loss,
            'dt_seconds': dt_seconds,
            'max_steps' : max_steps,
            'heterogeneous' : heterogeneous,
            'topology' : topology
            }
        
        for name, fn in strat_fns.items():
            for ep in range(n_episodes):

                env = QRNEnv(**args)
                obs  = env.reset()
                done = False
                fid  = 0.0

                for step in range(max_steps):
                    mask = env.get_action_mask()
                    if name == "Agent":
                        acts = self.select_actions(obs, mask, training=False)
                    elif name == "Random":
                        acts = strategies.random_policy(env, env.rng)
                    else:
                        acts = fn(env)

                    if ep == 0 and plot_actions:
                        timelines[name].append(acts.copy())

                    obs, reward, done, info = env.step(acts)
                    fid = info.get("fidelity", 0.0)

                    if verbose==1 and name=="Agent": # save agent actions in geometric plots
                        savedir=f"{save_dir}visual/state_{step}.png"
                        os.makedirs(os.path.dirname(savedir), exist_ok=True)
                        env.render(filepath=savedir)
                    if done:
                        break

                steps_taken = step + 1 if done and fid > 0 else max_steps
                results[name]["steps"].append(steps_taken)
                if fid > 0:
                    results[name]["fidelities"].append(fid)

        self.epsilon = old_eps
        self._print_results_table(results, n_repeaters, p_gen, p_swap, cutoff)

        if plot_actions:
            self._plot_timeline_grid(timelines, n_repeaters,
                                     p_gen, p_swap, cutoff, save_dir)
        return results

                                       
                # ▄▄▄▄▄▄▄   ▄▄▄        ▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄ 
                # ███▀▀███▄ ███      ▄███████▄ ▀▀▀███▀▀▀ 
                # ███▄▄███▀ ███      ███   ███    ███    
                # ███▀▀▀▀   ███      ███▄▄▄███    ███    
                # ███       ████████  ▀█████▀     ███    
    
    @staticmethod
    def _plot_training(metrics, save_path='assets/'):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Training Metrics", fontsize=12, y=0.98)

        ep = range(len(metrics["reward"]))
        axes[0].fill_between(ep, metrics["reward"], alpha=0.15, color="royalblue")
        axes[0].plot(_running_avg(metrics["reward"]), color="royalblue", lw=1.2)
        axes[0].set_ylabel("Episode Return")
        axes[0].axhline(0, color="grey", ls=":", lw=0.5)

        nonzero = [v for v in metrics["loss"] if v > 0]
        if nonzero:
            axes[1].plot(metrics["loss"], alpha=0.2, color="red")
            axes[1].plot(_running_avg(metrics["loss"]), color="red", lw=1.2)
            axes[1].set_ylabel("Loss")
            axes[1].set_yscale("log")

        axes[2].plot(_running_avg(metrics["success"], 50), color="seagreen",
                     lw=1.4)
        axes[2].set_ylabel("Success Rate")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(f"{save_path}training_metrics.png", dpi=200, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _print_results_table(results, N, pg, ps, c):
        pm = "\u00B1"
        print(f"\n{'='*70}")
        print(f"Validation: N={N}, p_gen={pg}, p_swap={ps}, cutoff={c}")
        print(f"{'='*70}")
        print(f"{'Strategy':<14} | {'Avg Steps':>12} | {'Avg Fidelity':>14} | "
              f"{'Succ%':>6}")
        print("-" * 70)
        for name, data in results.items():
            avg_s = np.mean(data["steps"])
            std_s = np.std(data["steps"])
            ns    = len(data["fidelities"])
            avg_f = np.mean(data["fidelities"]) if ns else 0.0
            std_f = np.std(data["fidelities"])  if ns else 0.0
            succ  = ns / max(len(data["steps"]), 1) * 100
            print(f"{name:<14} | {avg_s:>5.1f}{pm}{std_s:<5.1f} | "
                  f"{avg_f:>6.4f}{pm}{std_f:<6.4f} | {succ:>5.0f}%")

    @staticmethod
    def _plot_timeline_grid(timelines, N, pg, ps, c, save_dir="."):
        """Plot action timeline.

        Each cell = one node at one timestep.
        - Solid colour (repeater ID) = NOOP (wait / background entangle).
        - Hatched ``///`` = SWAP.
        - Hatched ``...`` = PURIFY.
        """
        strats   = list(timelines.keys())
        n_strats = len(strats)
        max_steps = max((len(tl) for tl in timelines.values()), default=1)
        rep_colors = _repeater_colors(N)

        fig_w = min(max_steps * 0.3 + 3, 22)
        fig_h = n_strats * 1.4 + 1.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        row_h = 1.0
        bar_h = row_h / N

        for si, sname in enumerate(strats):
            tl = timelines[sname]
            y_base = (n_strats - 1 - si) * (row_h + 0.3)

            for t, actions in enumerate(tl):
                for node in range(min(N, len(actions))):
                    a = int(actions[node])
                    y = y_base + node * bar_h
                    color = rep_colors[node]
                    hatch = _ACTION_HATCH.get(a, "")

                    rect = mpatches.FancyBboxPatch(
                        (t - 0.45, y), 0.9, bar_h * 0.9,
                        boxstyle="square,pad=0",
                        facecolor=color, edgecolor="none", linewidth=0)
                    ax.add_patch(rect)

                    # Only overlay hatch for SWAP / PURIFY
                    if a in (SWAP, PURIFY):
                        h_rect = mpatches.FancyBboxPatch(
                            (t - 0.45, y), 0.9, bar_h * 0.9,
                            boxstyle="square,pad=0",
                            facecolor="none", edgecolor="black",
                            hatch=hatch, linewidth=0, alpha=0.6)
                        ax.add_patch(h_rect)
            
            # Append black patch after the end of the timeline
            t_end = len(tl)
            black_patch = mpatches.FancyBboxPatch(
                (t_end - 0.45, y_base), 0.9, row_h - (bar_h * 0.1),
                boxstyle="square,pad=0",
                facecolor="black", edgecolor="none", linewidth=0, zorder=3)
            ax.add_patch(black_patch)

        y_positions = [(n_strats - 1 - i) * (row_h + 0.3) + row_h / 2
                       for i in range(n_strats)]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(strats)
        
        # Extended xlim to ensure the appended patch is not cut off
        ax.set_xlim(-0.5, max_steps + 1.5)
        ax.set_ylim(-0.3, n_strats * (row_h + 0.3))
        ax.set_xlabel("Time Step")
        ax.set_title(f"Policy Actions (N={N}, pg={pg}, ps={ps}, c={c})")
        ax.grid(False)

        handles = []
        for i in range(N):
            handles.append(mpatches.Patch(
                facecolor=rep_colors[i], label=f"R{i}",
                edgecolor="grey", linewidth=0.5))
        handles.append(mpatches.Patch(
            facecolor="white", edgecolor="grey", label="Noop"))
        handles.append(mpatches.Patch(
            facecolor="white", edgecolor="black", hatch="///", label="Swap"))
        handles.append(mpatches.Patch(
            facecolor="white", edgecolor="black", hatch="...", label="Purify"))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1, 0.5), title="Legend", fontsize=7)

        plt.savefig(os.path.join(save_dir, "validation_actions.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
