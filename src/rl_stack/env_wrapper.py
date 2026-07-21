"""
--------------------------------------------------------------------------------
RL environment wrapper for the quantum repeater network simulator.

Step flow
---------
  reset() > auto-entangle > return obs
  step(actions):
    1. Execute agent actions (purify first, then swap).
    2. Age links (resolve pending events, decohere, expire).
    3. Check end-to-end.
    4. Auto-entangle (prepare links for next observation).
    5. Return (obs, reward, done, info).

The agent always sees the POST-auto-entangle state so it can
immediately choose swap / purify if links are available.

Action space
-------------
    0 = NOOP   (wait)
    1 = SWAP   (BSM at this node)
    2 = PURIFY (BBPSSW on the best shared pair at this node)

Entanglement generation is **not** an agent action - it is handled
entirely by the automatic background generation step.

Source and destination nodes are restricted to NOOP only.
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from simulator.network import build_network
from simulator.repeater import NO_PARTNER
from rl_stack import potential

# --- action constants ----------------------------------------------------
NOOP    = 0
SWAP    = 1
PURIFY  = 2
N_ACTIONS = 3
ACTION_NAMES = ["noop", "swap", "purify"]

"""

      ▄▄▄▄▄   ▄▄▄▄▄▄▄   ▄▄▄    ▄▄▄        ▄▄▄▄▄▄▄
    ▄███████▄ ███▀▀███▄ ████▄  ███       ███▀▀▀▀▀
    ███   ███ ███▄▄███▀ ███▀██▄███       ███▄▄    ████▄ ██ ██
    ███▄█▄███ ███▀▀██▄  ███  ▀████ ▀▀▀▀▀ ███      ██ ██ ██▄██
     ▀█████▀  ███  ▀███ ███    ███       ▀███████ ██ ██  ▀█▀
          ▀▀

Gym-like wrapper around RepeaterNetwork for RL training.

The agent decides SWAP / PURIFY / NOOP at each interior node.
Source and destination nodes are always forced to NOOP.
"""


class QRNEnv:
    __slots__ = (
        "rng",        # RNG: per-episode physics + action sampling
        "max_steps",  # truncation horizon (steps before time-limit)
        "gamma",      # PBRS discount (threaded to equal the DQN gamma)
        "topology",   # topology tag (always 'chain')
        "net",        # RepeaterNetwork physics engine
        "_topo",      # frozen Topology snapshot (adjacency + positions)
        "N",          # number of repeaters
        "source",     # source node id (0)
        "dest",       # destination node id (N-1)
        "steps",      # steps elapsed in the current episode
        "done",       # episode terminated or truncated
        "_phi",       # last PBRS potential Φ(s)
        "_d_src",     # BFS hop distances from source
        "_d_dst",     # BFS hop distances from dest
        "_d_total",   # source→dest hop distance (path length)
    )

    STEP_COST       = -0.01
    SUCCESS_REWARD  =  1.0
    # ponytail: 0 = no penalty for failed ops. The mask already blocks invalid
    # actions, so this term only ever fired on STOCHASTIC swap/purify failures
    # (rng > p_swap) — punishing the agent for the env's coin-flip and training
    # it swap-shy. Failure is already costed naturally (lost links → -PBRS +
    # step cost). Restore a small negative only to penalize genuinely-invalid
    # attempts if masking is ever disabled.
    FAILED_ACTION   =  0.0

    def __init__(self,
                 n_repeaters = 5,
                 n_ch = 4,
                 spacing = 50.0,
                 p_gen = 0.8,
                 p_swap = 0.5,
                 p_gen_std = 0.0,
                 p_swap_std = 0.0,
                 cutoff = 20,
                 F0 = 0.95,
                 channel_loss = 0.02,
                 max_steps = 50,
                 rng: Optional[np.random.Generator] = None,
                 topology = 'chain',
                 gamma = 0.99):

        if topology != 'chain':
            raise ValueError(f'Topology {topology} not supported')

        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_steps = max_steps
        self.gamma = gamma
        self._phi = 0.0
        self.topology = topology

        self.net = build_network(
            topology=topology, n_repeaters=n_repeaters, n_ch=n_ch,
            spacing=spacing, p_gen=p_gen, p_swap=p_swap,
            p_gen_std=p_gen_std, p_swap_std=p_swap_std, cutoff=cutoff,
            F0=F0, channel_loss=channel_loss,
            rng=self.rng)

        self._topo = self.net.topology()
        self.N = self._topo.N
        self.source = -1
        self.dest = -1
        self.steps = 0
        self.done = False
        self._pick_targets()



# ▄▄▄▄▄▄▄▄▄
# ▀▀▀███▀▀▀                       ██
#    ███  ▀▀█▄ ████▄ ▄████ ▄█▀█▄ ▀██▀▀ ▄█▀▀▀
#    ███ ▄█▀██ ██ ▀▀ ██ ██ ██▄█▀  ██   ▀███▄
#    ███ ▀█▄██ ██    ▀████ ▀█▄▄▄  ██   ▄▄▄█▀
#                       ██
#                     ▀▀▀

    def _pick_targets(self):
        # Chain endpoints are the two ends; also cache BFS hop distances that
        # the PBRS potential (_progress) reads every step.
        self.source, self.dest = 0, self.N - 1
        adj = self._topo.adjacency
        self._d_src = potential.bfs_hops(adj, self.source)
        self._d_dst = potential.bfs_hops(adj, self.dest)
        self._d_total = float(self._d_src[self.dest])

    def _entangled_edges(self):
        """Undirected (a, b) edges of the current entanglement graph."""
        edges = set()
        for i in range(self.N):
            ns = self.net.node_state(i)
            for qi in np.flatnonzero(ns.occupied):
                p = int(ns.partner_node[qi])
                if p != NO_PARTNER and p != i:
                    edges.add((min(i, p), max(i, p)))
        return edges

    def _progress(self) -> float:
        """Topology-general PBRS potential in [0, 1]."""
        return potential.path_progress(
            self._d_src, self._d_dst, self._d_total, self._entangled_edges())

    def is_target(self, node: int) -> bool:
        return node == self.source or node == self.dest

#   ▄▄▄▄▄   ▄▄
# ▄███████▄ ██                                   ██   ▀▀
# ███   ███ ████▄ ▄█▀▀▀ ▄█▀█▄ ████▄ ██ ██  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄
# ███▄▄▄███ ██ ██ ▀███▄ ██▄█▀ ██ ▀▀ ██▄██ ▄█▀██  ██   ██  ██ ██ ██ ██
#  ▀█████▀  ████▀ ▄▄▄█▀ ▀█▄▄▄ ██     ▀█▀  ▀█▄██  ██   ██▄ ▀███▀ ██ ██



    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build size-agnostic node features + topology.

        Features per node (9):
            [0] frac_occupied     — occupied / n_ch
            [1] mean_fidelity     — avg F of available (unlocked) qubits (0 if none)
            [2] in_endnode        — 1.0 if source OR dest (endpoints are symmetric)
            [3] frac_available    — available (unlocked occupied) / n_ch
            [4] can_swap          — 1.0 if a viable swap pair exists: ≥2 available qubits to different partners whose fused link survives same-tick resolution (age_i + age_j + 2 < min cutoff)
            [5] can_purify        — 1.0 if ≥2 available qubits to same partner
            [6] p_gen             — per-repeater link-generation prob. (inhomogeneity)
            [7] p_swap            — per-repeater BSM success prob. (inhomogeneity)
            [8] link_urgency      — mean(age/link_cutoff) over occupied qubits (0 if none)

        Features [4] and [5] are forced to 0 for source / dest. Features [6]/[7]
        are constant across nodes when the network is homogeneous (std=0); they
        carry node-quality signal only under per-repeater inhomogeneity.
        Feature [8] is 0 for nodes with no occupied qubits; ~1 means links are
        about to expire.
        """
        feats = np.zeros((self.N, 9), dtype=np.float32)
        for i in range(self.N):
            ns = self.net.node_state(i)
            occ = ns.occupied
            avail = occ & (~ns.locked)
            feats[i, 0] = int(occ.sum()) / ns.n_ch
            feats[i, 1] = (float(ns.fidelity[avail].mean())
                           if bool(avail.any()) else 0.0)
            feats[i, 2] = 1.0 if self.is_target(i) else 0.0
            feats[i, 3] = int(avail.sum()) / ns.n_ch
            if self.is_target(i):
                feats[i, 4] = 0.0
                feats[i, 5] = 0.0
            else:
                feats[i, 4] = 1.0 if self._can_swap_from(ns) else 0.0
                feats[i, 5] = 1.0 if self._can_purify_from(ns) else 0.0
            feats[i, 6] = ns.p_gen
            feats[i, 7] = ns.p_swap
            if bool(occ.any()):
                lc = np.maximum(ns.link_cutoff[occ], 1)
                feats[i, 8] = float(np.clip(np.mean(ns.age[occ] / lc), 0.0, 1.0))
            else:
                feats[i, 8] = 0.0
        src, dst = np.nonzero(self._topo.adjacency)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        return {"x": feats, "edge_index": edge_index}

#   ▄▄▄▄▄
# ▄███████▄                         ▀▀
# ███   ███ ██ ██ ▄█▀█▄ ████▄ ████▄ ██  ▄█▀█▄ ▄█▀▀▀
# ███▄█▄███ ██ ██ ██▄█▀ ██ ▀▀ ██ ▀▀ ██  ██▄█▀ ▀███▄
#  ▀█████▀  ▀██▀█ ▀█▄▄▄ ██    ██    ██▄ ▀█▄▄▄ ▄▄▄█▀
#       ▀▀


    def _can_swap_from(self, ns) -> bool:
        """True if ns has a VIABLE swap pair: >=2 available qubits to distinct
        partners whose fused link survives same-tick resolution
        (age_i + age_j + 2 < min(link_cutoff_i, link_cutoff_j)). Mirrors the
        engine's decision gate; exact for homogeneous cutoffs (the only
        regime in use), and the engine stays authoritative regardless."""
        avail = np.flatnonzero(ns.occupied & (~ns.locked))
        if avail.size < 2:
            return False
        partners = ns.partner_node[avail]
        real = partners != NO_PARTNER
        avail, partners = avail[real], partners[real]
        if avail.size < 2:
            return False
        i, j = np.triu_indices(avail.size, k=1)
        ages = ns.age[avail].astype(np.int64)
        cuts = ns.link_cutoff[avail].astype(np.int64)
        viable = (ages[i] + ages[j] + 2) < np.minimum(cuts[i], cuts[j])
        return bool(np.any((partners[i] != partners[j]) & viable))

    def _can_purify_from(self, ns) -> bool:
        """True if ns has ≥2 available qubits linked to the *same* partner.
        `bincount` over partner ids beats `np.unique` (no sort) for tiny arrays."""
        avail = ns.occupied & (~ns.locked)
        if int(avail.sum()) < 2:
            return False
        partners = ns.partner_node[avail]
        counts = np.bincount(partners[partners != NO_PARTNER])
        return bool(np.any(counts >= 2))


# ▄▄▄      ▄▄▄
# ████▄  ▄████             ▄▄
# ███▀████▀███  ▀▀█▄ ▄█▀▀▀ ██ ▄█▀
# ███  ▀▀  ███ ▄█▀██ ▀███▄ ████
# ███      ███ ▀█▄██ ▄▄▄█▀ ██ ▀█▄


    def get_action_mask(self) -> np.ndarray:
        """(N, 3) bool mask.  Source/dest: only NOOP."""
        mask = np.zeros((self.N, N_ACTIONS), dtype=bool)
        mask[:, NOOP] = True

        for i in range(self.N):
            if self.is_target(i):
                continue
            ns = self.net.node_state(i)
            if self._can_swap_from(ns):
                mask[i, SWAP] = True
            if self._can_purify_from(ns):
                mask[i, PURIFY] = True
        return mask

#  ▄▄▄▄▄▄▄
# █████▀▀▀  ██
#  ▀████▄  ▀██▀▀ ▄█▀█▄ ████▄
#    ▀████  ██   ██▄█▀ ██ ██
# ███████▀  ██   ▀█▄▄▄ ████▀
#                      ██
#                      ▀▀

    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step:  actions → age → check e2e → auto-entangle."""
        assert len(actions) == self.N
        actions = actions.copy()
        info = {"fidelity": 0.0, "swaps": 0, "purifies": 0,
                "noops": 0, "failed_actions": 0, "actions": actions.copy()}

        # Safety: clamp any non-NOOP at source / dest
        for t in [self.source, self.dest]:
            if actions[t] != NOOP:
                actions[t] = NOOP
                info["actions"][t] = NOOP

        # Phase 1a: execute purifications first (order matters:
        #   purify before swap ensures swapped links are freshly improved)
        for r in np.flatnonzero(actions == PURIFY):
            result = self._exec_purify(int(r))
            info["purifies"] += 1
            if not result["success"]:
                info["failed_actions"] += 1

        # Phase 1b: execute swaps
        for r in np.flatnonzero(actions == SWAP):
            result = self._exec_swap(int(r))
            info["swaps"] += 1
            if not result["success"]:
                info["failed_actions"] += 1

        info["noops"] = int(np.sum(actions == NOOP))

        # Phase 2: age links (resolves pending events, decoheres, expires)
        self.net.age_links(discard_expired=True)

        # Phase 3: check end-to-end
        self.steps += 1
        connected, fidelity = self._check_e2e()
        info["fidelity"] = fidelity

        # Reward shaping: failed actions get penalized
        penalty = info["failed_actions"] * self.FAILED_ACTION

        if connected:
            self.done = True
            # Terminal: Φ(s_terminal) = 0 by PBRS convention
            shaping = -self._phi
            reward = fidelity * self.SUCCESS_REWARD + penalty + shaping
            info["terminated"], info["truncated"] = True, False
            return self.get_observation(), reward, True, info

        # Phase 4: auto-entangle for next step's observation
        self._auto_entangle()

        # PBRS: γΦ(s') - Φ(s)  (topology-general potential)
        phi_new = self._progress()
        shaping = self.gamma * phi_new - self._phi
        self._phi = phi_new

        # A time-limit hit is truncation, NOT a true terminal: it takes the
        # same non-terminal path above (auto-entangle + normal PBRS) so V(s')
        # stays bootstrappable. terminated=False -> the DQN target bootstraps.
        truncated = self.steps >= self.max_steps
        self.done = truncated
        info["terminated"], info["truncated"] = False, truncated
        return (self.get_observation(),
                self.STEP_COST + penalty + shaping, truncated, info)


#  ▄▄▄▄▄▄▄
# ███▀▀▀▀▀                                  ██   ▀▀
# ███▄▄    ██ ██ ▄█▀█▄ ▄████    ▀▀█▄ ▄████ ▀██▀▀ ██  ▄███▄ ████▄ ▄█▀▀▀
# ███       ███  ██▄█▀ ██      ▄█▀██ ██     ██   ██  ██ ██ ██ ██ ▀███▄
# ▀███████ ██ ██ ▀█▄▄▄ ▀████   ▀█▄██ ▀████  ██   ██▄ ▀███▀ ██ ██ ▄▄▄█▀



    def _auto_entangle(self):
        """Background entanglement: one pass over all adjacent pairs."""
        pairs = list(zip(*np.nonzero(np.triu(self._topo.adjacency, k=1))))
        self.rng.shuffle(pairs)
        for r1, r2 in pairs:
            self.net.entangle(int(r1), int(r2))

    def _exec_swap(self, r: int) -> Dict:
        return self.net.swap(r)

    def _exec_purify(self, r: int) -> Dict:
        ns = self.net.node_state(r)
        avail = ns.occupied & (~ns.locked)
        if int(avail.sum()) < 2:
            return {"success": False, "reason": "insufficient_qubits"}
        partners = ns.partner_node[avail]
        unique, counts = np.unique(partners[partners != NO_PARTNER],
                                   return_counts=True)
        valid = [(int(p), c) for p, c in zip(unique, counts) if c >= 2]
        if not valid:
            return {"success": False, "reason": "no_valid_pair"}
        best_nb = max(valid, key=lambda x: x[1])[0]
        return self.net.purify(r, best_nb)

    def _check_e2e(self) -> Tuple[bool, float]:
        """Check whether source and dest share a direct entanglement link."""
        ns = self.net.node_state(self.source)
        for qi in np.flatnonzero(ns.occupied):
            if int(ns.partner_node[qi]) == self.dest:
                return True, float(ns.fidelity[qi])
        return False, 0.0


# ▄▄▄      ▄▄▄
# ████▄  ▄████ ▀▀
# ███▀████▀███ ██  ▄█▀▀▀ ▄████
# ███  ▀▀  ███ ██  ▀███▄ ██
# ███      ███ ██▄ ▄▄▄█▀ ▀████



    def reset(self) -> Dict[str, np.ndarray]:
        """Reset, auto-entangle once, return observation."""
        self.net.reset()
        self._pick_targets()
        self.steps = 0
        self.done  = False
        self._auto_entangle()
        self._phi = self._progress()
        return self.get_observation()

    @staticmethod
    def action_label(action: int, node: int) -> str:
        return f"{['W','S','P'][action]}({node})"

    def render(self, filepath=None, figsize=None, dpi=250):
        return self.net.render(filepath=filepath, figsize=figsize,
                               dpi=dpi,
                               source_dest=(self.source, self.dest))
