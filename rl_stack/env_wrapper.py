"""RL environment wrapper for the quantum repeater network simulator.

Step flow:
  reset() → auto-entangle → return obs
  step(actions):
    1. Execute agent actions.
    2. Age links (resolve pending events, decohere, expire).
    3. Check end-to-end.
    4. Auto-entangle (prepare links for next observation).
    5. Return (obs, reward, done, info).

The agent always sees the POST-auto-entangle state, so it can
immediately choose swap if links are available.

CRITICAL: Source and destination nodes NEVER swap or purify.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np

from quantum_repeater_sim.network import RepeaterNetwork, build_chain
from quantum_repeater_sim.repeater import Repeater, SwapPolicy, werner_to_fidelity, NO_PARTNER


NOOP = 0
ENTANGLE = 1
SWAP = 2
PURIFY = 3
N_ACTIONS = 4
ACTION_NAMES = ["noop", "entangle", "swap", "purify"]


class QRNEnv:
    """Gym-like wrapper around RepeaterNetwork for RL training.

    Source and destination nodes are restricted to {noop, entangle}.
    """

    STEP_COST = -0.1
    SUCCESS_REWARD = 1.0

    def __init__(self, n_repeaters: int = 5, n_ch: int = 4,
                 spacing: float = 50.0, p_gen: float = 0.8,
                 p_swap: float = 0.5, cutoff: int = 20,
                 F0: float = 0.95, channel_loss: float = 0.02,
                 dt_seconds: float = 1e-4, max_steps: int = 50,
                 rng: Optional[np.random.Generator] = None,
                 heterogeneous: bool = False, ee=False):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_steps = max_steps

        self.net = build_chain(
            n_repeaters, n_ch=n_ch, spacing=spacing,
            p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
            F0=F0, channel_loss=channel_loss,
            dt_seconds=dt_seconds,
            distance_dep_gen=True, rng=self.rng)

        if heterogeneous:
            for rep in self.net.repeaters:
                rep.p_gen = self.rng.uniform(0.3, 1.0)
                rep.p_swap = self.rng.uniform(0.3, 1.0)

        self.N = self.net.N
        self.source: int = -1
        self.dest: int = -1
        self.steps: int = 0
        self.done: bool = False
        self.ee=ee # always picks end to end if True
        self._pick_targets()

    def _pick_targets(self):
        if self.N <= 2 or self.ee:
            self.source, self.dest = 0, self.N - 1
            return
        while True:
            s, d = sorted(self.rng.choice(self.N, size=2, replace=False))
            if abs(s - d) > 1:
                self.source, self.dest = int(s), int(d)
                return

    def is_target(self, node: int) -> bool:
        return node == self.source or node == self.dest

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset, auto-entangle once, return observation."""
        self.net.reset()
        self._pick_targets()
        self.steps = 0
        self.done = False
        # Initial generation attempt so agent sees links on first obs
        self._auto_entangle()
        return self.get_observation()

    # ── observation ───────────────────────────────────────────────

    def get_observation(self) -> Dict[str, np.ndarray]:
        """(N, 7) node features + (2, E) edge_index.

        Source/dest always have can_swap=0 and can_purify=0.
        """
        feats = np.zeros((self.N, 7), dtype=np.float32)
        for i, rep in enumerate(self.net.repeaters):
            feats[i, 0] = rep.num_occupied() / rep.n_ch
            occ = rep.occupied_indices()
            feats[i, 1] = (float(np.mean(werner_to_fidelity(rep.werner_param[occ])))
                           if len(occ) > 0 else 0.0)
            feats[i, 2] = 1.0 if i == self.source else 0.0
            feats[i, 3] = 1.0 if i == self.dest else 0.0
            feats[i, 4] = rep.num_available() / rep.n_ch
            if self.is_target(i):
                feats[i, 5] = 0.0
                feats[i, 6] = 0.0
            else:
                feats[i, 5] = 1.0 if rep.can_swap() else 0.0
                feats[i, 6] = 1.0 if self._can_purify_at(i) else 0.0

        src, dst = np.nonzero(self.net.adj)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        return {"x": feats, "edge_index": edge_index}

    def _can_purify_at(self, r: int) -> bool:
        rep = self.net.repeaters[r]
        occ = rep.available_indices()
        if len(occ) < 2:
            return False
        partners = rep.partner_repeater[occ]
        _, counts = np.unique(partners, return_counts=True)
        return bool(np.any(counts >= 2))

    # ── action mask ───────────────────────────────────────────────

    def get_action_mask(self) -> np.ndarray:
        """(N, 4) bool mask. Source/dest: only {noop, entangle}."""
        mask = np.zeros((self.N, N_ACTIONS), dtype=bool)
        mask[:, NOOP] = True

        em = self.net.action_mask_entangle()
        sm = self.net.action_mask_swap()
        pm = self.net.action_mask_purify()

        for i in range(self.N):
            if em[i].any():
                mask[i, ENTANGLE] = True
            if self.is_target(i):
                continue
            if sm[i]:
                mask[i, SWAP] = True
            if pm[i].any():
                mask[i, PURIFY] = True
        return mask

    # ── step ──────────────────────────────────────────────────────

    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step: actions → age → check e-e → auto-entangle.

        The auto-entangle at the end prepares links for the agent's
        NEXT observation, so the agent always sees post-generation state.
        """
        assert len(actions) == self.N
        info = {"fidelity": 0.0, "entangles": 0, "swaps": 0,
                "purifies": 0, "noops": 0, "actions": actions.copy()}

        # Safety: clamp any swap/purify at source/dest to noop
        for t in [self.source, self.dest]:
            if actions[t] in (SWAP, PURIFY):
                actions[t] = NOOP
                info["actions"][t] = NOOP

        # Phase 1: execute agent actions (entangle first, then purify, swap, noop)
        for action_type in [ENTANGLE, PURIFY, SWAP, NOOP]:
            nodes = np.flatnonzero(actions == action_type)
            for r in nodes:
                r = int(r)
                if action_type == ENTANGLE:
                    self._exec_entangle(r); info["entangles"] += 1
                elif action_type == SWAP:
                    self._exec_swap(r); info["swaps"] += 1
                elif action_type == PURIFY:
                    self._exec_purify(r); info["purifies"] += 1
                else:
                    info["noops"] += 1

        # Phase 2: age links
        self.net.age_links(discard_expired=True)

        # Phase 3: check end-to-end
        self.steps += 1
        connected, fidelity = self._check_e2e()
        info["fidelity"] = fidelity

        if connected:
            self.done = True
            return self.get_observation(), self.SUCCESS_REWARD, True, info

        if self.steps >= self.max_steps:
            self.done = True
            return self.get_observation(), self.STEP_COST, True, info

        # Phase 4: auto-entangle for next step's observation
        self._auto_entangle()

        return self.get_observation(), self.STEP_COST, False, info

    # ── internal ──────────────────────────────────────────────────

    def _auto_entangle(self):
        """Try entanglement on all adjacent pairs (background generation)."""
        pairs = list(zip(*np.nonzero(np.triu(self.net.adj))))
        self.rng.shuffle(pairs)
        for r1, r2 in pairs:
            self.net.entangle(int(r1), int(r2))

    def _exec_entangle(self, r: int):
        neighbors = np.flatnonzero(self.net.adj[r])
        if len(neighbors) == 0:
            return
        for nb in self.rng.permutation(neighbors):
            if self.net.entangle(r, int(nb))["success"]:
                return

    def _exec_swap(self, r: int):
        self.net.swap(r)

    def _exec_purify(self, r: int):
        rep = self.net.repeaters[r]
        occ = rep.available_indices()
        if len(occ) < 2:
            return
        partners = rep.partner_repeater[occ]
        unique, counts = np.unique(partners, return_counts=True)
        valid = [(int(p), c) for p, c in zip(unique, counts)
                 if p != NO_PARTNER and c >= 2]
        if not valid:
            return
        best_nb = max(valid, key=lambda x: x[1])[0]
        self.net.purify(r, best_nb)

    def _check_e2e(self) -> Tuple[bool, float]:
        src_rep = self.net.repeaters[self.source]
        for qi in src_rep.occupied_indices():
            if int(src_rep.partner_repeater[qi]) == self.dest:
                return True, float(werner_to_fidelity(src_rep.werner_param[qi]))
        return False, 0.0

    @staticmethod
    def action_label(action: int, node: int) -> str:
        return f"{['W','E','S','P'][action]}({node})"