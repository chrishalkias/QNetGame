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
from rl_stack import policies



                                           
                # ▄▄▄   ▄▄▄       ▄▄                         
                # ███   ███       ██                         
                # █████████ ▄█▀█▄ ██ ████▄ ▄█▀█▄ ████▄ ▄█▀▀▀ 
                # ███▀▀▀███ ██▄█▀ ██ ██ ██ ██▄█▀ ██ ▀▀ ▀███▄ 
                # ███   ███ ▀█▄▄▄ ██ ████▀ ▀█▄▄▄ ██    ▄▄▄█▀ 
                #                    ██                      
                #                    ▀▀       
                              
NODE_DIM = 8  # must match env_wrapper get_observation feature count

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


            #   ▄▄▄▄    ▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄▄    ▄▄▄ ▄▄▄▄▄▄▄▄▄ 
            # ▄██▀▀██▄ ███▀▀▀▀▀  ███▀▀▀▀▀ ████▄  ███ ▀▀▀███▀▀▀ 
            # ███  ███ ███       ███▄▄    ███▀██▄███    ███    
            # ███▀▀███ ███  ███▀ ███      ███  ▀████    ███    
            # ███  ███ ▀██████▀  ▀███████ ███    ███    ███    
                      

class QRNAgent:
    """
    Double-DQN agent with per-node Q-values on a GNN backbone.

    The env runs a SERIALIZED per-node sweep (one micro-decision at
    env.active_node per env.step call), so training is per-decision: each
    replay transition is ONE node's (state, action, reward, next-state)
    at its active node, with its own gamma_eff (1.0 intra-tick, self.gamma
    at the tick boundary) and terminated flag. train_step gathers exactly
    one Q-value per transition, at the batched-graph global index of that
    transition's active node (`Batch.ptr[b] + ai_b`).
    The successor action mask (for the ACTIVE node at the next micro-step)
    is stored in the buffer and applied during target Q-value computation
    to ensure physical validity.
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

    def select_action(self, obs: Dict[str, np.ndarray], mask_row: np.ndarray,
                      active_node: int, training: bool = True) -> int:
        """ε-greedy scalar action for ONE node: env.active_node, the sole
        node deciding at this micro-step. `mask_row` is that node's (3,)
        action mask.

        This is the only selection entry point. The batched `select_actions`
        (an (N,3) mask in, an (N,) action vector out) was deleted 2026-07-26:
        under the serialized sweep exactly one node decides per micro-step, so
        every caller indexed the result at env.active_node and threw the other
        N-1 argmaxes away."""
        if training and self.rng.random() < self.epsilon:
            valid = np.flatnonzero(mask_row)
            return int(self.rng.choice(valid)) if len(valid) else NOOP

        data = _obs_to_data(obs, self.device)
        with torch.no_grad():
            q = self.policy_net(data)[active_node].clone()
            q[~torch.tensor(mask_row, dtype=torch.bool, device=self.device)] = -float("inf")
        return int(q.argmax().item())


                                                  
        #  ▄▄▄▄▄▄▄                      ▄▄▄▄▄▄▄             
        # █████▀▀▀  ██                 ███▀▀▀▀▀             
        #  ▀████▄  ▀██▀▀ ▄█▀█▄ ████▄   ███▄▄    ████▄ ██ ██ 
        #    ▀████  ██   ██▄█▀ ██ ██   ███      ██ ██ ██▄██ 
        # ███████▀  ██   ▀█▄▄▄ ████▀   ▀███████ ██ ██  ▀█▀  
        #                      ██                           
        #                      ▀▀                           


    def train_step(self) -> Optional[float]:
        """Sample batch, compute masked Double-DQN loss AT THE ACTIVE NODE of
        each transition, backprop. One Q-value per transition (not one per
        node per graph): gathered at the batched-graph global index
        `Batch.ptr[b] + ai_b`. Per-transition gamma_eff (1.0 intra-tick,
        self.gamma at the tick boundary) and terminated flag drive the
        target, not fixed per-graph scalars."""
        if self.memory.size() < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = Batch.from_data_list(
            [_as_data(t, "s") for t in batch]).to(self.device)
        next_states = Batch.from_data_list(
            [_as_data(t, "s_") for t in batch]).to(self.device)

        actions = torch.tensor(
            [t["a"] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [t["r"] for t in batch], dtype=torch.float32, device=self.device)
        gammas = torch.tensor(
            [t["g"] for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor(
            [float(t["d"]) for t in batch], dtype=torch.float32, device=self.device)
        next_mask_rows = torch.tensor(
            np.stack([t["m_"] for t in batch]), dtype=torch.bool, device=self.device)

        # Batched-graph global index of each transition's active node:
        # graph b's nodes start at ptr[b], so its active node is ptr[b] + ai_b.
        ptr = states.ptr[:-1]
        act_idx = ptr + torch.tensor(
            [t["ai"] for t in batch], dtype=torch.long, device=self.device)

        # -- Current Q(s, a) at the active node --
        q_all     = self.policy_net(states)
        current_q = q_all[act_idx].gather(1, actions.unsqueeze(1)).squeeze(1)

        # -- Target Q (Double DQN with masked next actions), per-transition γ --
        with torch.no_grad():
            nptr = next_states.ptr[:-1]
            next_idx = nptr + torch.tensor(
                [t["nai"] for t in batch], dtype=torch.long, device=self.device)

            next_q_policy = self.policy_net(next_states)[next_idx].clone()
            next_q_policy[~next_mask_rows] = -float("inf")   # took alot time finding this bug...
            best_actions = next_q_policy.argmax(dim=1)

            next_q_target = self.target_net(next_states)[next_idx]
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + gammas * next_q * (1.0 - dones)

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
        """Resolve n_ch (int or sequence) to a non-empty list of ints >= 1.

        n_ch is qubits PER SIDE (left/right ports), so n_ch=1 => 2 physical
        qubits interior (one left + one right link) and is valid for swap-only
        schedules; purify just can never fire (needs >=2 to the same partner).
        Int -> single-element pool (backward compatible). List/tuple -> the
        pool the training loop samples from uniformly per episode."""
        pool = list(n_ch) if isinstance(n_ch, (list, tuple)) else [n_ch]
        if not pool:
            raise ValueError("n_ch list must be non-empty")
        for c in pool:
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
                raise ValueError(f"n_ch values must be ints, got {c!r}")
            if int(c) < 1:
                raise ValueError(f"n_ch values must be >= 1, got {c}")
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
        (`eps_floor_frac*episodes`, matching the ε schedule). Before that,
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
              F0 = 1.0,
              channel_loss = 0.0,
              curriculum = True,
              curriculum_frac = 0.5,
              prune_unwinnable = False,
              env_seed = None,
              save_path = None,
              save_best = True,
              best_window = 200,
              eps_init = 1.0,
              eps_fin = 0.05,
              eps_schedule = 'linear',
              lr_decay = None,
              eval_fn = None,
              eval_every = 0,
              eval_patience = 0,
              eval_mode = 'min',
              disable_actions = (),
              compare = False,
              compare_every = 10,
              compare_extra = None,
              plot = True) -> Dict[str, list]:
        """
        Train with curriculum over chain sizes.

        Curriculum linearly widens the eligible chain size to the full range
        over the first `curriculum_frac` of training (see _curriculum_pool).

        `F0`/`channel_loss` default to IDEALISED physics (perfect fresh links,
        lossless fibre), matching what every SLURM recipe and the train.py CLI
        already pass explicitly. The engine keeps its full lossy surface, it is
        just no longer the default nobody asked for.

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

        `compare`: every `compare_every` episodes, also roll out the GREEDY
        agent, swap-asap and random on one freshly seeded network and log
        per-policy return, steps and success to
        metrics['cmp_{agent,swap,rand}{,_steps,_succ}'], with the sampled
        episode indices in metrics['cmp_ep'] (+ a 3-panel
        `training_compare.png`). Read crossovers off the STEPS/SUCCESS panels,
        those are the pure task metrics; return also reflects fidelity-weighted
        success, so a policy can lead on return while only tying on delivery
        time. Diagnostic only; default off.

        `compare_every`: sampling period for those extra rollouts (default 10).
        Each sample costs three extra greedy rollouts, so sampling every
        episode is the dominant cost of a --compare run; the panels are rolling
        means, so a 10x sparser sample is visually identical past a few hundred
        episodes. `compare_every=1` reproduces the old every-episode behaviour
        exactly, RNG stream included.

        `compare_extra`: optional dict {name: policy_fn(env, obs) -> actions} of
        extra baselines to log alongside agent/swap/rand under the SAME per-
        episode seed (e.g. {'optimal': optimal_dispatch_fn}). Keeps agent.py
        decoupled from experiment-specific baselines (the DP optimum is wired in
        by the caller). Each name gets cmp_{name}{,_steps,_succ}.
        """
        #TODO: Add wandb logging
        compare_extra = dict(compare_extra or {})
        cmp_names = ['agent', 'swap', 'rand', *compare_extra.keys()]
        metrics = {"reward": [], "loss": [], "steps": [], "success": [], "eval": [],
                   "opt_steps": []}   # cumulative optimizer steps at each episode end
        metrics["cmp_ep"] = []   # episode index of each comparison sample
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
        assert eps_schedule in ('linear', 'cosine'), \
            f"eps_schedule must be 'linear' or 'cosine', got {eps_schedule!r}"
        if int(compare_every) < 1:
            raise ValueError(
                f"compare_every must be >= 1, got {compare_every!r}")
        compare_every = int(compare_every)
        # eps_init/eps_fin now come from the signature (fine-tuning wants a LOW
        # eps_init so a warm-started policy isn't scrambled). lr_decay enables an
        # exponential LR schedule (default None = constant LR, unchanged).
        sched = (optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
                 if lr_decay is not None else None)
        n_ch_pool = self._normalize_n_ch(n_ch)
        if prune_unwinnable:
            from rl_stack.policies import WinnabilityCache
            self._wc = WinnabilityCache(
                probe_steps=max(3 * max_steps, 200),
                channel_loss=channel_loss, F0=F0)
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
            opt_steps_total = 0   # running count of real optimizer steps (train_step != None)
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
                    'max_steps' : max_steps,
                    }

                env_rng = (np.random.default_rng(env_ss.spawn(1)[0])
                           if env_ss is not None else None)
                env = QRNEnv(**args, rng=env_rng)
                obs   = env.reset()
                score = 0.0
                ep_loss = []

                # Serialized per-node sweep: one micro-decision at
                # env.active_node per env.step call. `max_steps` (ticks) is
                # enforced inside the env itself (truncation), so the loop
                # just runs to `done`.
                while True:
                    r_node = env.active_node
                    mask_row = env.action_mask(r_node)
                    if disable_actions:
                        mask_row = mask_row.copy()
                        for d in disable_actions:
                            mask_row[d] = False
                    a = self.select_action(obs, mask_row, r_node, training=True)

                    next_obs, reward, done, info = env.step(a)
                    nai = (info["next_active_node"]
                           if info["next_active_node"] >= 0 else r_node)
                    next_mask = env.action_mask(nai)
                    if disable_actions:
                        for d in disable_actions:
                            next_mask[d] = False

                    # store terminated (not done): timeouts (truncated) must
                    # bootstrap V(s') in the DQN target, only true wins zero it.
                    self.memory.add(obs, a, r_node, reward, next_obs, nai,
                                    next_mask, info["terminated"], info["gamma_eff"])

                    loss = self.train_step()
                    if loss is not None:
                        ep_loss.append(loss)

                    obs   = next_obs
                    score += reward
                    if done:
                        break

                # ε annealing over the first 90% of episodes, then floor. Both
                # schedules hit eps_fin at 0.9·episodes so _ckpt_window_start's
                # eps_floor_frac=0.9 gate holds for either. 'linear' is the DQN
                # standard (Mnih 2015 / SB3); 'cosine' kept for reproducibility.
                if ep >= 0.9 * episodes:
                    self.epsilon = eps_fin
                elif eps_schedule == 'cosine':
                    self.epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (
                        1 + math.cos(math.pi * ep / max(episodes, 1)))
                else:  # 'linear'
                    frac = ep / (0.9 * max(episodes, 1))
                    self.epsilon = eps_init + (eps_fin - eps_init) * frac
                if sched is not None and ep_loss:   # only after a real optimizer step
                    sched.step()

                metrics["reward"].append(score)
                metrics["loss"].append(
                    np.mean(ep_loss) if ep_loss else 0.0)
                metrics["steps"].append(info["ticks"])
                metrics["success"].append(
                    1.0 if info.get("fidelity", 0) > 0 else 0.0)
                # under the serialized sweep, optimizer steps per episode vary with
                # chain length / episode duration, so track the cumulative count as
                # the honest x-axis for learning-progress plots (len(ep_loss) counts
                # only train_step calls that actually stepped the optimizer).
                opt_steps_total += len(ep_loss)
                metrics["opt_steps"].append(opt_steps_total)

                # -- Paired comparison (--compare): run the GREEDY agent,
                # swap-asap and random on ONE freshly seeded network so the
                # returns are directly comparable. Reveals the training phases
                # where the learned policy overtakes random, then swap-asap.
                # Greedy (not the eps-exploring training rollout) so the
                # crossover reflects policy quality, not the eps schedule.
                # Sampled every `compare_every` episodes (default 10): three
                # extra rollouts per sample are the dominant cost of a
                # --compare run, and the panels are rolling means, so a sparser
                # sample is visually identical. The episode index goes to
                # cmp_ep, which is the x-axis those panels plot against.
                if compare and ep % compare_every == 0:
                    cmp_seed = int(self.rng.integers(0, 2**32))
                    metrics["cmp_ep"].append(ep)
                    cmp_policies = {'agent': 'agent', 'swap': 'swap',
                                    'rand': 'rand', **compare_extra}
                    for nm, pol in cmp_policies.items():
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
            from rl_stack.plots import plot_training
            plot_training(metrics, save_path)

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
        """One episode on a freshly SEEDED network (so all policies are paired
        on the same net), driven one micro-step at a time (the serialized
        sweep; env self-truncates at `max_steps` ticks). Returns
        (return, steps, success): `steps` = ticks elapsed (info["ticks"]),
        `success` = 1.0 if the e2e link was delivered. Return mixes
        speed/fidelity/action-economy; steps+success are the pure task
        metrics (delivery time, delivery rate).

        `policy`: 'agent' (greedy, eps-free), 'swap' (swap-asap), 'rand'
        (random), or a callable policy_fn(env, obs) -> int (an extra
        baseline, e.g. the optimum) -- the scalar action for env.active_node,
        recomputed fresh every micro-step so every policy sees the current
        state. Used only by --compare / the eval probe; never touches the
        agent's training rollout."""
        env = QRNEnv(**args, rng=np.random.default_rng(seed))
        obs = env.reset()
        rand_rng = np.random.default_rng((seed ^ 0x9E3779B9) & 0xFFFFFFFF)
        ret = 0.0
        success = 0.0
        while True:
            r_node = env.active_node
            if policy == 'agent':
                # action_mask(node) returns a fresh (3,) array, so disabling
                # actions in place here cannot leak into anyone else's mask.
                mask_row = env.action_mask(r_node)
                for d in disable_actions:
                    mask_row[d] = False
                a = self.select_action(obs, mask_row, r_node, training=False)
            elif policy == 'swap':
                a = policies.swap_asap(env)
            elif policy == 'rand':
                a = policies.random_policy(env, rand_rng)
            else:                      # callable extra baseline
                a = policy(env, obs)
            obs, r, done, info = env.step(int(a))
            ret += r
            if done:
                success = 1.0 if info.get("fidelity", 0.0) > 0 else 0.0
                break
        return float(ret), int(info["ticks"]), float(success)


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
                 F0=1.0,
                 channel_loss=0.0,
                ):
        """
        Validate the agent against the heuristic baselines on paired episode
        seeds and print the results table (avg steps, avg fidelity, success%).
        """
        if model_path is not None:
            self.policy_net.load_state_dict(
                torch.load(model_path, map_location=self.device,
                           weights_only=True))

        old_eps = self.epsilon
        self.epsilon = 0.0

        strat_fns = {
            "Agent":        None,
            "SwapASAP":     policies.swap_asap,
            "PurifySwap":   policies.purify_then_swap,
            "Random":       None,
        }
        results   = {k: {"steps": [], "fidelities": [], "total": 0}
                     for k in strat_fns}

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
            'max_steps' : max_steps,
            }

        action_rng = np.random.default_rng()
        seed_rng = np.random.default_rng(42)
        ep_seeds = seed_rng.integers(0, 2**32, size=n_episodes)

        for name, fn in strat_fns.items():
            for ep in range(n_episodes):

                env = QRNEnv(**args,
                             rng=np.random.default_rng(int(ep_seeds[ep])))
                obs  = env.reset()
                done = False
                fid  = 0.0

                while True:
                    r_node = env.active_node
                    if name == "Agent":
                        a = self.select_action(obs, env.action_mask(r_node),
                                               r_node, training=False)
                    elif name == "Random":
                        a = policies.random_policy(env, action_rng)
                    else:
                        a = fn(env)

                    obs, reward, done, info = env.step(a)
                    fid = info.get("fidelity", 0.0)
                    if done:
                        break

                succeeded = done and fid > 0
                if succeeded:
                    results[name]["steps"].append(info["ticks"])
                    results[name]["fidelities"].append(fid)
                results[name]["total"] += 1

        self.epsilon = old_eps
        from rl_stack.plots import print_results_table
        print_results_table(results, n_repeaters, p_gen, p_swap, cutoff)
        return results

