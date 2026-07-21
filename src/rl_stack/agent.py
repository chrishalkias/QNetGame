"""
--------------------------------------------------------------------------------
Double-DQN agent for quantum repeater network routing.

The agent learns a per-node policy on small chains and generalises
zero-shot to larger, differently-parameterised ones.

Key fixes over the original:
  - Successor action mask stored in buffer and used in target Q
    computation (prevents learning Q-values for impossible actions).
  - Action space reduced to {noop, swap, purify}; entanglement is
    background-only and not an agent decision.
  - Reward scale fixed: SUCCESS >> cumulative step penalty.
  - 3-layer GNN for 3-hop receptive field.
--------------------------------------------------------------------------------
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
                              
NODE_DIM = 9   # must match env_wrapper get_observation feature count

def _obs_to_data(obs: Dict[str, np.ndarray], device="cpu") -> Data:
    x = torch.tensor(obs["x"], dtype=torch.float32, device=device)
    return Data(
        x=x,
        edge_index=torch.tensor(obs["edge_index"], dtype=torch.long, device=device),
        num_nodes=x.shape[0],   # set explicitly so PyG collation never re-infers it
    )


def _as_data(transition: Dict, key: str) -> Data:
    """Return the transition's state as a (CPU) PyG Data, converting once and
    caching it back on the transition. A replayed transition is sampled many
    times; this turns N conversions into one without touching the buffer API."""
    v = transition[key]
    if isinstance(v, Data):
        return v
    d = _obs_to_data(v)
    transition[key] = d   # cache for the next time this transition is sampled
    return d


def _sample_cutoff(rng, cutoff):
    """Scalar -> passthrough (no rng draw); (lo,hi) -> uniform int in [lo,hi]."""
    if isinstance(cutoff, (tuple, list)) and len(cutoff) == 2:
        lo, hi = int(cutoff[0]), int(cutoff[1])
        return int(rng.integers(lo, hi + 1))
    return int(cutoff)


def _draw_winnable_cell(rng, wc, *, p_gen, p_swap, cutoff,
                        n_pool, n_ch_pool, max_tries=50):
    """Draw a cell (p_gen, p_swap, cutoff, N, n_ch). If `wc` is given, resample
    until the cell is winnable (purify-then-swap can deliver), capped at `max_tries`;
    if the cap is hit, the last draw is returned anyway."""
    for _ in range(max_tries):
        pg = QRNAgent._sample_rate(rng, p_gen)
        ps = QRNAgent._sample_rate(rng, p_swap)
        ct = _sample_cutoff(rng, cutoff)
        n = int(rng.choice(n_pool))
        nch = int(rng.choice(n_ch_pool)) if len(n_ch_pool) > 1 else int(n_ch_pool[0])
        if wc is None or wc.winnable(p_gen=pg, p_swap=ps, cutoff=ct,
                                     n_repeaters=n, n_ch=nch):
            return pg, ps, ct, n, nch
    return pg, ps, ct, n, nch


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

            #   ▄▄▄▄    ▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄▄    ▄▄▄ ▄▄▄▄▄▄▄▄▄ 
            # ▄██▀▀██▄ ███▀▀▀▀▀  ███▀▀▀▀▀ ████▄  ███ ▀▀▀███▀▀▀ 
            # ███  ███ ███       ███▄▄    ███▀██▄███    ███    
            # ███▀▀███ ███  ███▀ ███      ███  ▀████    ███    
            # ███  ███ ▀██████▀  ▀███████ ███    ███    ███    
                      

class QRNAgent:
    """                                                       
    Double-DQN agent with per-node Q-values on a GNN backbone.

    The agent selects one of {noop, swap, purify} for every node.
    Training uses shared global reward broadcast to each node.
    The successor action mask is stored in the buffer and applied
    during target Q-value computation to ensure physical validity.
    """

    def __init__(self, node_dim = NODE_DIM, hidden = 64,
                 lr = 3e-4, gamma = 0.99,
                 buffer_size = 80_000, batch_size = 64,
                 tau = 0.005, epsilon = 1.0,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):

        self.rng = rng if rng is not None else np.random.default_rng()
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
        # seed the replay sampler from the master seed (None -> nondeterministic)
        self.memory = ReplayBuffer(max_size=buffer_size, seed=seed)

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

        if training and self.rng.random() < self.epsilon:
            actions = np.zeros(N, dtype=np.int32)
            for i in range(N):
                valid = np.flatnonzero(mask[i])
                actions[i] = self.rng.choice(valid) if len(valid) else NOOP
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
            [_as_data(t, "s") for t in batch]).to(self.device)
        next_states = Batch.from_data_list(
            [_as_data(t, "s_") for t in batch]).to(self.device)

        # Per-graph scalars → broadcast to every node
        rewards_pg = torch.tensor(
            [t["r"] for t in batch], dtype=torch.float32, device=self.device)
        dones_pg = torch.tensor(
            [float(t["d"]) for t in batch], dtype=torch.float32, device=self.device)

        node_to_graph = states.batch
        rewards = rewards_pg[node_to_graph]
        dones   = dones_pg[node_to_graph]

        # Per-node actions / next masks: concatenate in NumPy, then ONE tensor
        # call each (vs a torch.tensor per transition + torch.cat).
        actions = torch.tensor(
            np.concatenate([t["a"] for t in batch]),
            dtype=torch.long, device=self.device)
        next_masks = torch.tensor(
            np.concatenate([t["m_"] for t in batch]),
            dtype=torch.bool, device=self.device)

        # -- Current Q(s, a) --
        q_all    = self.policy_net(states)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # -- Target Q (Double DQN with masked next actions) --
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_policy[~next_masks] = -float("inf")   # took alot time finding this bug...
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
                
            #  ▄▄▄▄▄▄▄                                  
            # █████▀▀▀  ██         ██   ▀▀              
            #  ▀████▄  ▀██▀▀ ▀▀█▄ ▀██▀▀ ██  ▄████ ▄█▀▀▀ 
            #    ▀████  ██  ▄█▀██  ██   ██  ██    ▀███▄ 
            # ███████▀  ██  ▀█▄██  ██   ██▄ ▀████ ▄▄▄█▀ 
                                          
                                          
    @staticmethod
    def _normalize_n_ch(n_ch):
        """Resolve n_ch (int or sequence) to a non-empty list of ints >= 2.

        Int -> single-element pool (backward compatible). List/tuple -> the
        pool the training loop samples from uniformly per episode."""
        pool = list(n_ch) if isinstance(n_ch, (list, tuple)) else [n_ch]
        if not pool:
            raise ValueError("n_ch list must be non-empty")
        for c in pool:
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
                raise ValueError(f"n_ch values must be ints, got {c!r}")
            if int(c) < 2:
                raise ValueError(f"n_ch values must be >= 2, got {c}")
        return [int(c) for c in pool]

    @staticmethod
    def _sample_rate(rng, val):
        """Resolve a rate (p_gen/p_swap): scalar -> constant; (lo, hi) tuple/list
        -> uniform per-episode sample; set/frozenset -> uniform DISCRETE choice
        from its elements (used to land exactly on precomputed-optimal grid
        points). A scalar draws NO RNG, so the stream stays identical to the
        pre-change int/scalar path (randomization only consumes RNG otherwise)."""
        if isinstance(val, (set, frozenset)):
            return float(rng.choice(sorted(val)))
        if isinstance(val, (tuple, list)):
            lo, hi = float(val[0]), float(val[1])
            return float(rng.uniform(lo, hi))
        return float(val)

    @staticmethod
    def _curriculum_pool(ep, episodes, n_range,
                         curriculum=True, curriculum_frac=0.5):
        """Chain sizes eligible at episode `ep`.

        The cap linearly widens from n_min to n_max over the first
        `curriculum_frac` of training (round-half-up), then the full range is
        held. `curriculum=False` (or a degenerate range) always returns the full
        range. Reaching the full range by mid-training (default 0.5) ensures the
        LARGEST size is trained for a substantial share of episodes — the old
        schedule (full range only at 0.8) starved n_max to ~5-10%, which left its
        policy at the swap-asap default instead of the optimum."""
        n_min, n_max = min(n_range), max(n_range)
        if not curriculum or n_min >= n_max:
            return list(n_range)
        ramp = min((ep / max(episodes, 1)) / max(curriculum_frac, 1e-9), 1.0)
        cap = n_min + int(ramp * (n_max - n_min) + 0.5)   # round-half-up
        return [r for r in n_range if r <= cap]

    @staticmethod
    def _ckpt_window_start(episodes, curriculum=True, curriculum_frac=0.5,
                           eps_floor_frac=0.9):
        """First episode eligible for rolling-reward best-checkpoint selection.

        Rolling reward is only comparable once difficulty AND exploration are
        fixed: the curriculum must have opened the FULL size range
        (`curriculum_frac*episodes`) AND epsilon must have reached its floor
        (`eps_floor_frac*episodes`, matching the cosine schedule). Before that,
        the easy early phase (small chains deliver fast, high reward) would win
        and freeze the checkpoint at ~ep 300. Returns the later of the two gates;
        `curriculum=False` drops the curriculum gate but still waits for epsilon."""
        eps_floor_ep = int(eps_floor_frac * episodes)
        curr_open_ep = int(curriculum_frac * episodes) if curriculum else 0
        return max(eps_floor_ep, curr_open_ep)
    

        # ▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄     ▄▄▄▄   ▄▄▄▄▄ ▄▄▄    ▄▄▄ 
        # ▀▀▀███▀▀▀ ███▀▀███▄ ▄██▀▀██▄  ███  ████▄  ███ 
        #    ███    ███▄▄███▀ ███  ███  ███  ███▀██▄███ 
        #    ███    ███▀▀██▄  ███▀▀███  ███  ███  ▀████ 
        #    ███    ███  ▀███ ███  ███ ▄███▄ ███    ███ 


    def train(self,
              episodes = 3000,
              max_steps = 50,
              n_range = [4, 5, 6, 7],
              n_ch = 4,
              p_gen = 0.8,
              p_swap = 0.7,
              p_gen_std = 0.0,
              p_swap_std = 0.0,
              cutoff = 30,
              F0 = 0.95,
              channel_loss = 0.02,
              dt_seconds = 1e-3,
              curriculum = True,
              curriculum_frac = 0.5,
              topology = 'chain',
              prune_unwinnable = False,
              env_seed = None,
              save_path = None,
              save_best = True,
              best_window = 200,
              eps_init = 1.0,
              eps_fin = 0.05,
              lr_decay = None,
              eval_fn = None,
              eval_every = 0,
              eval_patience = 0,
              eval_mode = 'min',
              disable_actions = (),
              compare = False,
              compare_extra = None,
              plot = True) -> Dict[str, list]:
        """
        Train with curriculum over chain sizes.

        Curriculum linearly widens the eligible chain size to the full range
        over the first `curriculum_frac` of training (see _curriculum_pool).

        Checkpointing: `policy.pth` always holds the BEST agent seen, never the
        final one (late-training degradation therefore can't clobber it); the
        final weights go to `policy_final.pth`. "Best" is judged by `eval_fn`
        when given (a held-out greedy policy probe), else by rolling-mean reward.
        The rolling-mean path only opens in the settled LATE window (curriculum
        fully open + epsilon at floor, see _ckpt_window_start): otherwise the easy
        early curriculum phase — small chains deliver fast, high reward — freezes
        the checkpoint at ~ep 300 and the harder late policy never replaces it.

        Early stopping: when `eval_fn` and `eval_every` are set, the probe runs
        every `eval_every` episodes; if it fails to improve for `eval_patience`
        consecutive probes (and patience > 0), training stops — so longer phases
        self-trim instead of running a fixed, often-too-long budget. The DQN loss
        is NOT used for stopping (it is a moving-target TD error, decoupled from
        policy quality).

        `disable_actions`: action indices masked off during BOTH selection and
        the Double-DQN target (e.g. (PURIFY,) trains a pure swap-scheduler).
        `eval_mode`: 'min' (lower probe = better, e.g. delivery time) or 'max'.

        `compare`: each episode, also roll out the GREEDY agent, swap-asap and
        random on one freshly seeded network and log per-policy return, steps
        and success to metrics['cmp_{agent,swap,rand}{,_steps,_succ}'] (+ a 3-
        panel `training_compare.png`). Read crossovers off the STEPS/SUCCESS
        panels — those are the pure task metrics; return also reflects
        fidelity-weighted success and failed-action penalties, so a policy can
        lead on return while only tying on delivery time. Diagnostic only; costs
        ~3 extra rollouts/episode; default off.

        `compare_extra`: optional dict {name: policy_fn(env, obs) -> actions} of
        extra baselines to log alongside agent/swap/rand under the SAME per-
        episode seed (e.g. {'optimal': optimal_dispatch_fn}). Keeps agent.py
        decoupled from experiment-specific baselines (the DP optimum is wired in
        by the caller). Each name gets cmp_{name}{,_steps,_succ}.
        """
        #TODO: Add wandb logging
        compare_extra = dict(compare_extra or {})
        cmp_names = ['agent', 'swap', 'rand', *compare_extra.keys()]
        metrics = {"reward": [], "loss": [], "steps": [], "success": [], "eval": []}
        for nm in cmp_names:
            metrics[f"cmp_{nm}"] = []
            metrics[f"cmp_{nm}_steps"] = []
            metrics[f"cmp_{nm}_succ"] = []
        # Record the run config so plots can be annotated (sets -> sorted lists
        # for JSON). p_gen/p_swap may be a scalar, (lo,hi) range, or a grid set.
        _cfg = lambda v: sorted(v) if isinstance(v, (set, frozenset)) else v
        metrics["config"] = {
            "N": _cfg(n_range), "n_ch": _cfg(n_ch),
            "p_gen": _cfg(p_gen), "p_swap": _cfg(p_swap),
            "cutoff": cutoff, "max_steps": max_steps, "episodes": episodes,
            "disable_actions": list(disable_actions),
        }
        # eps_init/eps_fin now come from the signature (fine-tuning wants a LOW
        # eps_init so a warm-started policy isn't scrambled). lr_decay enables an
        # exponential LR schedule (default None = constant LR, unchanged).
        sched = (optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
                 if lr_decay is not None else None)
        n_ch_pool = self._normalize_n_ch(n_ch)
        if prune_unwinnable:
            from rl_stack.winnability import WinnabilityCache
            self._wc = WinnabilityCache(
                probe_steps=max(3 * max_steps, 200),
                dt_seconds=dt_seconds, channel_loss=channel_loss, F0=F0)
        else:
            self._wc = None
        disable_actions = tuple(disable_actions)
        best_metric, best_ep, best_saved = -math.inf, -1, False
        best_eval = math.inf if eval_mode == 'min' else -math.inf
        eval_stale = 0
        # Rolling-reward best-ckpt is only comparable in the settled late window
        # (curriculum fully open + epsilon at floor); see _ckpt_window_start.
        ckpt_start = self._ckpt_window_start(episodes, curriculum, curriculum_frac)

        # -- Env-RNG seeding (bit-reproducible physics) --
        # env_seed is not None -> a dedicated SeedSequence deterministically spawns
        # one child generator per episode, so ALL in-episode physics stochasticity
        # (entangle/swap/purify coin flips, auto-entangle shuffle, inhomogeneity)
        # is seed-determined. This does NOT touch self.rng, so the per-episode
        # domain draws above keep their exact pre-change stream order. env_seed=None
        # keeps the prior behavior (QRNEnv falls back to OS entropy each episode).
        env_ss = (np.random.SeedSequence(env_seed) if env_seed is not None
                  else None)

        try:
            for ep in range(episodes):
                # -- Curriculum: linearly widen max chain size --
                pool = self._curriculum_pool(
                    ep, episodes, n_range, curriculum, curriculum_frac)
                # Joint per-episode cell draw; when prune_unwinnable, resample
                # until swap-asap can deliver (skip physically-impossible cells).
                # Scalar p_gen/p_swap/cutoff + single-element n_ch draw no RNG, so
                # the no-prune scalar path keeps the pre-change RNG stream.
                p_gen_ep, p_swap_ep, cutoff_ep, n_nodes, n_ch_ep = _draw_winnable_cell(
                    self.rng, self._wc, p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                    n_pool=pool, n_ch_pool=n_ch_pool)

                args = {
                    'n_repeaters': n_nodes,
                    'n_ch': n_ch_ep,
                    'spacing': 50,
                    'p_gen': p_gen_ep,
                    'p_swap': p_swap_ep,
                    'p_gen_std': p_gen_std,
                    'p_swap_std': p_swap_std,
                    'cutoff': cutoff_ep,
                    'gamma': self.gamma,   # env PBRS gamma == DQN discount
                    'F0' : F0,
                    'channel_loss' : channel_loss,
                    'dt_seconds': dt_seconds,
                    'max_steps' : max_steps,
                    'topology' : topology,
                    }

                env_rng = (np.random.default_rng(env_ss.spawn(1)[0])
                           if env_ss is not None else None)
                env = QRNEnv(**args, rng=env_rng)
                obs   = env.reset()
                score = 0.0
                ep_loss = []

                for _ in range(max_steps):
                    mask    = env.get_action_mask()
                    if disable_actions:
                        mask[:, disable_actions] = False
                    actions = self.select_actions(obs=obs, mask=mask, training=True)

                    next_obs, reward, done, info = env.step(actions)
                    next_mask = env.get_action_mask()
                    if disable_actions:
                        next_mask[:, disable_actions] = False

                    # store terminated (not done): timeouts (truncated) must
                    # bootstrap V(s') in the DQN target, only true wins zero it.
                    self.memory.add(obs, actions, reward,
                                    next_obs, info["terminated"], next_mask)

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
                if sched is not None and ep_loss:   # only after a real optimizer step
                    sched.step()

                metrics["reward"].append(score)
                metrics["loss"].append(
                    np.mean(ep_loss) if ep_loss else 0.0)
                metrics["steps"].append(env.steps)
                metrics["success"].append(
                    1.0 if info.get("fidelity", 0) > 0 else 0.0)

                # -- Per-episode paired comparison (--compare): run the GREEDY
                # agent, swap-asap and random on ONE freshly seeded network so
                # the returns are directly comparable. Reveals the training
                # phases where the learned policy overtakes random, then
                # swap-asap. Greedy (not the eps-exploring training rollout) so
                # the crossover reflects policy quality, not the eps schedule.
                if compare:
                    cmp_seed = int(self.rng.integers(0, 2**32))
                    policies = {'agent': 'agent', 'swap': 'swap', 'rand': 'rand',
                                **compare_extra}
                    for nm, pol in policies.items():
                        ret, st, sc = self._cmp_rollout(
                            args, cmp_seed, pol, max_steps, disable_actions)
                        metrics[f"cmp_{nm}"].append(ret)
                        metrics[f"cmp_{nm}_steps"].append(st)
                        metrics[f"cmp_{nm}_succ"].append(sc)

                # -- Best-checkpoint by rolling reward (only when no eval probe) --
                # Gated to the late window so the whole rolling window lies past
                # ckpt_start (curriculum open + epsilon floor) — never the easy
                # early curriculum phase.
                if (save_best and eval_fn is None and save_path
                        and ep - best_window + 1 >= ckpt_start
                        and (ep % 50 == 0 or ep == episodes - 1)):
                    roll = float(np.mean(metrics["reward"][-best_window:]))
                    if roll > best_metric:
                        best_metric, best_ep, best_saved = roll, ep, True
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(self.policy_net.state_dict(),
                                   os.path.join(save_path, "policy.pth"))

                # -- Policy probe: best-checkpoint + early stopping --
                if eval_fn is not None and eval_every > 0 and (ep + 1) % eval_every == 0:
                    m = float(eval_fn(self))
                    metrics["eval"].append((ep, m))
                    improved = m < best_eval if eval_mode == 'min' else m > best_eval
                    if improved:
                        best_eval, best_ep, best_saved, eval_stale = m, ep, True, 0
                        if save_path:
                            os.makedirs(save_path, exist_ok=True)
                            torch.save(self.policy_net.state_dict(),
                                       os.path.join(save_path, "policy.pth"))
                    else:
                        eval_stale += 1
                    print(f"  [probe ep {ep}] eval={m:.4f} best={best_eval:.4f} "
                          f"stale={eval_stale}/{eval_patience}", flush=True)
                    if eval_patience > 0 and eval_stale >= eval_patience:
                        print(f"[early-stop] no eval improvement for "
                              f"{eval_patience} probes; stopping at ep {ep}")
                        break

                if ep % 200 == 0 or (ep == episodes - 1 and ep > 0):
                    avg_r = np.mean(metrics["reward"][-200:])
                    avg_s = np.mean(metrics["success"][-200:])
                    print(f"Ep {ep:>5d}/{episodes} | R={avg_r:>7.3f} | "
                          f"succ={avg_s:.2f} | ε={self.epsilon:.3f} | N={n_nodes}")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            final_path = os.path.join(save_path, "policy.pth")
            if best_saved:
                # policy.pth already holds the best agent; keep the final weights
                # separately for reference.
                final_path = os.path.join(save_path, "policy_final.pth")
                crit = (f"eval {best_eval:.4f}" if eval_fn is not None
                        else f"rolling reward {best_metric:.3f}")
                print(f"[best] policy.pth = best ({crit}) @ ep {best_ep}; "
                      f"final -> policy_final.pth")
            torch.save(self.policy_net.state_dict(), final_path)
            self._save_metrics(metrics, save_path)

        if plot:
            self._plot_training(metrics, save_path)

        return metrics

    @staticmethod
    def _save_metrics(metrics, save_path):
        """Dump raw per-episode metrics to metrics.json so plots can be
        regenerated (experiments/training/replot.py) without retraining."""
        import json
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, default=float)   # default=float coerces np types

    def _cmp_rollout(self, args, seed, policy, max_steps, disable_actions=()):
        """One episode on a freshly SEEDED network (so all policies are paired on
        the same net). Returns (return, steps, success): `steps` = episode length
        (max_steps if undelivered), `success` = 1.0 if the e2e link was
        delivered. Return mixes speed/fidelity/action-economy; steps+success are
        the pure task metrics (delivery time, delivery rate).

        `policy`: 'agent' (greedy, eps-free), 'swap' (swap-asap), 'rand' (random),
        or a callable(env, obs) -> actions (an extra baseline, e.g. the optimum).
        Used only by --compare; never touches the agent's training rollout."""
        env = QRNEnv(**args, rng=np.random.default_rng(seed))
        obs = env.reset()
        rand_rng = np.random.default_rng((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
        ret = 0.0
        steps = max_steps
        success = 0.0
        for t in range(max_steps):
            mask = env.get_action_mask()
            if disable_actions:
                mask[:, disable_actions] = False
            if policy == 'agent':
                acts = self.select_actions(obs, mask, training=False)
            elif policy == 'swap':
                acts = strategies.swap_asap(env)
            elif policy == 'rand':
                acts = strategies.random_policy(env, rand_rng)
            else:                      # callable extra baseline
                acts = policy(env, obs)
            obs, r, done, info = env.step(acts)
            ret += r
            if done:
                steps = t + 1
                success = 1.0 if info.get("fidelity", 0.0) > 0 else 0.0
                break
        return float(ret), int(steps), float(success)


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
                 p_gen_std=0.0,
                 p_swap_std=0.0,
                 cutoff=15,
                 F0=0.95,
                 channel_loss=0.02,
                 dt_seconds=1e-3,
                 plot_actions=True,
                 topology = 'chain',
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
            "Agent":        None,
            "SwapASAP":     strategies.swap_asap,
            "PurifySwap":   strategies.purify_then_swap,
            "Random":       None,
        }
        results   = {k: {"steps": [], "fidelities": [], "total": 0}
                     for k in strat_fns}
        timelines = {k: [] for k in strat_fns}

        args = {
            'n_repeaters': n_repeaters,
            'n_ch': n_ch,
            'spacing': 50,
            'p_gen': p_gen,
            'p_swap': p_swap,
            'p_gen_std': p_gen_std,
            'p_swap_std': p_swap_std,
            'cutoff': cutoff,
            'F0' : F0,
            'channel_loss' : channel_loss,
            'dt_seconds': dt_seconds,
            'max_steps' : max_steps,
            'topology' : topology,
            }

        action_rng = np.random.default_rng()
        seed_rng = np.random.default_rng(42)
        ep_seeds = seed_rng.integers(0, 2**32, size=n_episodes)

        # Store all episode timelines so we can pick the median one
        all_timelines = {k: [] for k in strat_fns}

        for name, fn in strat_fns.items():
            for ep in range(n_episodes):

                env = QRNEnv(**args,
                             rng=np.random.default_rng(int(ep_seeds[ep])))
                obs  = env.reset()
                done = False
                fid  = 0.0
                ep_actions = []

                for step in range(max_steps):
                    mask = env.get_action_mask()
                    if name == "Agent":
                        acts = self.select_actions(obs, mask, training=False)
                    elif name == "Random":
                        acts = strategies.random_policy(env, action_rng)
                    else:
                        acts = fn(env)

                    if plot_actions:
                        ep_actions.append(acts.copy())

                    obs, reward, done, info = env.step(acts)
                    fid = info.get("fidelity", 0.0)

                    if verbose==1 and name=="Agent":
                        savedir=f"{save_dir}visual/state_{step}.png"
                        os.makedirs(os.path.dirname(savedir), exist_ok=True)
                        env.render(filepath=savedir)
                    if done:
                        break

                succeeded = done and fid > 0
                if succeeded:
                    results[name]["steps"].append(step + 1)
                    results[name]["fidelities"].append(fid)
                results[name]["total"] += 1

                if plot_actions:
                    all_timelines[name].append(ep_actions)

        self.epsilon = old_eps
        self._print_results_table(results, n_repeaters, p_gen, p_swap, cutoff)

        if plot_actions:
            # Pick the median-length successful episode per strategy
            for name in strat_fns:
                episodes = all_timelines[name]
                succ_steps = results[name]["steps"]
                if not succ_steps:
                    # No successes — use the first episode as fallback
                    timelines[name] = episodes[0] if episodes else []
                    continue
                median_steps = int(np.median(succ_steps))
                # Find the successful episode closest to the median
                best_idx, best_diff = 0, float("inf")
                ep_succ = 0
                for ep_idx, ep_tl in enumerate(episodes):
                    ep_len = len(ep_tl)
                    if ep_len >= max_steps:
                        continue  # skip failed episodes
                    diff = abs(ep_len - median_steps)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = ep_idx
                timelines[name] = episodes[best_idx]

            self._plot_timeline_grid(timelines, n_repeaters,
                                     p_gen, p_swap, cutoff, save_dir)
        return results

                                       
                # ▄▄▄▄▄▄▄   ▄▄▄        ▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄ 
                # ███▀▀███▄ ███      ▄███████▄ ▀▀▀███▀▀▀ 
                # ███▄▄███▀ ███      ███   ███    ███    
                # ███▀▀▀▀   ███      ███▄▄▄███    ███    
                # ███       ████████  ▀█████▀     ███    
    
    @staticmethod
    def _config_caption(cfg):
        """One-line parameter caption for the figures (N, n_ch, p_gen, p_swap,
        tau=cutoff, H). Returns '' if no config was recorded (older runs)."""
        if not cfg:
            return ""
        def f(v):
            if isinstance(v, (list, tuple)):
                u = sorted(set(v))
                return str(u[0]) if len(u) == 1 else "{" + ",".join(map(str, u)) + "}"
            return str(v)
        parts = [f"N={f(cfg.get('N'))}", f"n_ch={f(cfg.get('n_ch'))}",
                 f"p_gen={f(cfg.get('p_gen'))}", f"p_swap={f(cfg.get('p_swap'))}",
                 f"τ(cutoff)={cfg.get('cutoff')}", f"max_steps={cfg.get('max_steps')}"]
        if 2 in (cfg.get("disable_actions") or []):
            parts.append("swap-only")
        return ", ".join(parts)

    @staticmethod
    def _plot_training(metrics, save_path='assets/', window=None):
        """`window`: rolling-mean window for ALL smoothed curves. None -> the
        per-panel adaptive defaults; pass an int (e.g. via replot.py --window)
        to make the figures smoother without retraining."""
        w_metric = int(window) if window else 30
        w_steps = int(window) if window else 50
        caption = QRNAgent._config_caption(metrics.get("config"))
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Training Metrics" + (f"\n{caption}" if caption else ""),
                     fontsize=11, y=0.99)

        ep = range(len(metrics["reward"]))
        axes[0].fill_between(ep, metrics["reward"], alpha=0.15, color="royalblue")
        axes[0].plot(_running_avg(metrics["reward"], w_metric), color="royalblue", lw=1.2)
        axes[0].set_ylabel("Episode Return")
        axes[0].axhline(0, color="grey", ls=":", lw=0.5)

        nonzero = [v for v in metrics["loss"] if v > 0]
        if nonzero:
            axes[1].plot(metrics["loss"], alpha=0.2, color="red")
            axes[1].plot(_running_avg(metrics["loss"], w_metric), color="red", lw=1.2)
            axes[1].set_ylabel("Loss")
            axes[1].set_yscale("log")

        axes[2].fill_between(ep, metrics["steps"], alpha=0.15, color="seagreen")
        axes[2].plot(_running_avg(metrics["steps"], w_steps), color="seagreen",
                     lw=1.4)
        axes[2].set_ylabel("Avg Steps to Termination")
        axes[2].set_xlabel("Episode")

        plt.tight_layout()
        fname = os.path.join(save_path, "training_metrics.png") if save_path else "training_metrics.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

        # --compare: dedicated crossover plot, same net/episode, GREEDY agent vs
        # baselines. Return mixes speed/fidelity/action-economy; the steps and
        # success panels are the pure TASK metrics — read crossovers off those.
        if metrics.get("cmp_agent"):
            # Fixed colour/label per known series; only those present are drawn,
            # so the optimal line appears automatically when compare_extra
            # supplies it. Add new named baselines here to colour them.
            _known = (("rand", "grey", "Random"),
                      ("swap", "darkorange", "SwapASAP"),
                      ("optimal", "seagreen", "Optimal (swap-only)"),
                      ("agent", "royalblue", "Agent (greedy)"))
            series = tuple(s for s in _known if metrics.get(f"cmp_{s[0]}"))
            n = len(metrics["cmp_agent"])
            cep = range(n)
            # Wide smoothing: per-episode (p_gen,p_swap) randomisation injects huge
            # raw variance, so a small window can't reveal the trend. No raw fog —
            # at thousands of episodes it just buries the means. `window` (e.g. via
            # replot.py --window) overrides the adaptive default.
            win = int(window) if window else max(50, n // 25)

            # Paired GAP-to-optimal panel (the key readout): because every policy
            # runs on the SAME seeded net each episode, policy_steps - opt_steps
            # cancels the param-draw noise. Agent -> 0 means it reached optimal.
            has_opt = bool(metrics.get("cmp_optimal_steps"))
            panels = [("cmp_{}", "Episode Return"),
                      ("cmp_{}_steps", "Steps to Terminate"),
                      ("cmp_{}_succ", "Success")]
            nrows = len(panels) + (1 if has_opt else 0)
            fig2, axes2 = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)

            for i, (tmpl, ylabel) in enumerate(panels):
                for short, color, label in series:
                    axes2[i].plot(cep, _running_avg(metrics[tmpl.format(short)], win),
                                  color=color, lw=1.8, label=(label if i == 0 else None))
                axes2[i].set_ylabel(ylabel)
            axes2[0].axhline(0, color="grey", ls=":", lw=0.5)

            if has_opt:
                gax = axes2[len(panels)]
                opt = np.asarray(metrics["cmp_optimal_steps"], dtype=float)
                for short, color, label in series:
                    if short == "optimal":
                        continue
                    gap = (np.asarray(metrics[f"cmp_{short}_steps"], float) - opt).tolist()
                    gax.plot(cep, _running_avg(gap, win), color=color, lw=1.8)
                gax.axhline(0, color="seagreen", ls="--", lw=1.4)  # optimal = 0
                gax.set_ylabel("Steps above Optimal\n(paired; 0 = optimal)")

            _title = (f"Per-episode paired comparison "
                      f"(same seeded network, rolling mean w={win})")
            if caption:
                _title += f"\n{caption}"
            axes2[0].set_title(_title, fontsize=10)
            axes2[0].legend(loc="best", fontsize=9)
            axes2[-1].set_xlabel("Episode")
            plt.tight_layout()
            fname2 = (os.path.join(save_path, "training_compare.png")
                      if save_path else "training_compare.png")
            plt.savefig(fname2, dpi=200, bbox_inches="tight")
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
            ns   = len(data["steps"])   # only successful episodes
            tot  = data["total"]
            succ = ns / max(tot, 1) * 100
            avg_s = np.mean(data["steps"]) if ns else float("nan")
            std_s = np.std(data["steps"])  if ns else 0.0
            avg_f = np.mean(data["fidelities"]) if ns else 0.0
            std_f = np.std(data["fidelities"])  if ns else 0.0
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
        ax.set_title(f"Policy Actions — median episode (N={N}, pg={pg}, ps={ps}, c={c})")
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
