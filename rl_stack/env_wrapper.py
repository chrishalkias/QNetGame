"""
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
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from quantum_repeater_sim.network import RepeaterNetwork, build_chain
from quantum_repeater_sim.repeater import (
    Repeater, SwapPolicy, werner_to_fidelity, NO_PARTNER,
    QUBIT_OCCUPIED,
)

# ── action constants ──────────────────────────────────────────────
NOOP    = 0
SWAP    = 1
PURIFY  = 2
N_ACTIONS = 3
ACTION_NAMES = ["noop", "swap", "purify"]


class QRNEnv:
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

    STEP_COST       = -0.01
    SUCCESS_REWARD  =  1.0

    def __init__(self, n_repeaters: int = 5, n_ch: int = 4,
                 spacing: float = 50.0, p_gen: float = 0.8,
                 p_swap: float = 0.5, cutoff: int = 20,
                 F0: float = 0.95, channel_loss: float = 0.02,
                 dt_seconds: float = 1e-4, max_steps: int = 50,
                 rng: Optional[np.random.Generator] = None,
                 heterogeneous: bool = False, ee: bool = False):

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
                rep.p_gen  = self.rng.uniform(0.3, 1.0)
                rep.p_swap = self.rng.uniform(0.3, 1.0)

        self.N       = self.net.N
        self.source  = -1
        self.dest    = -1
        self.steps   = 0
        self.done    = False
        self.ee      = ee
        self._pick_targets()

                                           
# ▄▄▄▄▄▄▄▄▄                                  
# ▀▀▀███▀▀▀                       ██         
#    ███  ▀▀█▄ ████▄ ▄████ ▄█▀█▄ ▀██▀▀ ▄█▀▀▀ 
#    ███ ▄█▀██ ██ ▀▀ ██ ██ ██▄█▀  ██   ▀███▄ 
#    ███ ▀█▄██ ██    ▀████ ▀█▄▄▄  ██   ▄▄▄█▀ 
#                       ██                   
#                     ▀▀▀                    

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
                                                                  
#   ▄▄▄▄▄   ▄▄                                                        
# ▄███████▄ ██                                   ██   ▀▀              
# ███   ███ ████▄ ▄█▀▀▀ ▄█▀█▄ ████▄ ██ ██  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███▄▄▄███ ██ ██ ▀███▄ ██▄█▀ ██ ▀▀ ██▄██ ▄█▀██  ██   ██  ██ ██ ██ ██ 
#  ▀█████▀  ████▀ ▄▄▄█▀ ▀█▄▄▄ ██     ▀█▀  ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 
                                                                    
                                                                    

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build size-agnostic node features + topology.

        Features per node (8):
            [0] frac_occupied     — occupied / n_ch
            [1] mean_fidelity     — avg F of occupied qubits (0 if none)
            [2] is_source         — 1.0 / 0.0
            [3] is_dest           — 1.0 / 0.0
            [4] frac_available    — available (unlocked occupied) / n_ch
            [5] can_swap          — 1.0 if ≥2 available qubits to different partners
            [6] can_purify        — 1.0 if ≥2 available qubits to same partner
            [7] time_remaining    — (max_steps - steps) / max_steps

        Features [5] and [6] are forced to 0 for source / dest.
        """
        feats = np.zeros((self.N, 8), dtype=np.float32)
        for i, rep in enumerate(self.net.repeaters):
            feats[i, 0] = rep.num_occupied() / rep.n_ch
            occ = rep.occupied_indices()
            feats[i, 1] = (float(np.mean(werner_to_fidelity(rep.werner_param[occ])))
                           if len(occ) > 0 else 0.0)
            feats[i, 2] = 1.0 if i == self.source else 0.0
            feats[i, 3] = 1.0 if i == self.dest   else 0.0
            feats[i, 4] = rep.num_available() / rep.n_ch

            if self.is_target(i):
                feats[i, 5] = 0.0
                feats[i, 6] = 0.0
            else:
                feats[i, 5] = 1.0 if self._can_swap_at(i)   else 0.0
                feats[i, 6] = 1.0 if self._can_purify_at(i) else 0.0

            feats[i, 7] = (self.max_steps - self.steps) / self.max_steps

        src, dst = np.nonzero(self.net.adj)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        return {"x": feats, "edge_index": edge_index}
                                                  
#   ▄▄▄▄▄                                           
# ▄███████▄                         ▀▀              
# ███   ███ ██ ██ ▄█▀█▄ ████▄ ████▄ ██  ▄█▀█▄ ▄█▀▀▀ 
# ███▄█▄███ ██ ██ ██▄█▀ ██ ▀▀ ██ ▀▀ ██  ██▄█▀ ▀███▄ 
#  ▀█████▀  ▀██▀█ ▀█▄▄▄ ██    ██    ██▄ ▀█▄▄▄ ▄▄▄█▀ 
#       ▀▀                                          
                                                  

    def _can_swap_at(self, r: int) -> bool:
        """True if node r has ≥2 available qubits linked to *distinct* partners."""
        rep = self.net.repeaters[r]
        avail = rep.available_indices()
        if len(avail) < 2:
            return False
        partners = rep.partner_repeater[avail]
        unique = np.unique(partners[partners != NO_PARTNER])
        return len(unique) >= 2

    def _can_purify_at(self, r: int) -> bool:
        """True if node r has ≥2 available qubits linked to the *same* partner."""
        rep = self.net.repeaters[r]
        avail = rep.available_indices()
        if len(avail) < 2:
            return False
        partners = rep.partner_repeater[avail]
        _, counts = np.unique(partners[partners != NO_PARTNER],
                              return_counts=True)
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
            if self._can_swap_at(i):
                mask[i, SWAP] = True
            if self._can_purify_at(i):
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
        info = {"fidelity": 0.0, "swaps": 0, "purifies": 0,
                "noops": 0, "actions": actions.copy()}

        # Safety: clamp any non-NOOP at source / dest
        for t in [self.source, self.dest]:
            if actions[t] != NOOP:
                actions[t] = NOOP
                info["actions"][t] = NOOP

        # Phase 1a: execute purifications first (order matters:
        #   purify before swap ensures swapped links are freshly improved)
        for r in np.flatnonzero(actions == PURIFY):
            self._exec_purify(int(r))
            info["purifies"] += 1

        # Phase 1b: execute swaps
        for r in np.flatnonzero(actions == SWAP):
            self._exec_swap(int(r))
            info["swaps"] += 1

        info["noops"] = int(np.sum(actions == NOOP))

        # Phase 2: age links (resolves pending events, decoheres, expires)
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

                                                                     
#  ▄▄▄▄▄▄▄                                                             
# ███▀▀▀▀▀                                  ██   ▀▀                    
# ███▄▄    ██ ██ ▄█▀█▄ ▄████    ▀▀█▄ ▄████ ▀██▀▀ ██  ▄███▄ ████▄ ▄█▀▀▀ 
# ███       ███  ██▄█▀ ██      ▄█▀██ ██     ██   ██  ██ ██ ██ ██ ▀███▄ 
# ▀███████ ██ ██ ▀█▄▄▄ ▀████   ▀█▄██ ▀████  ██   ██▄ ▀███▀ ██ ██ ▄▄▄█▀ 
                                                                     
                                                                     

    def _auto_entangle(self):
        """Background entanglement: one pass over all adjacent pairs."""
        pairs = list(zip(*np.nonzero(np.triu(self.net.adj, k=1))))
        self.rng.shuffle(pairs)
        for r1, r2 in pairs:
            self.net.entangle(int(r1), int(r2))

    def _exec_swap(self, r: int):
        self.net.swap(r)

    def _exec_purify(self, r: int):
        rep = self.net.repeaters[r]
        avail = rep.available_indices()
        if len(avail) < 2:
            return
        partners = rep.partner_repeater[avail]
        unique, counts = np.unique(
            partners[partners != NO_PARTNER], return_counts=True)
        valid = [(int(p), c) for p, c in zip(unique, counts) if c >= 2]
        if not valid:
            return
        best_nb = max(valid, key=lambda x: x[1])[0]
        self.net.purify(r, best_nb)

    def _check_e2e(self) -> Tuple[bool, float]:
        """Check whether source and dest share a direct entanglement link."""
        src_rep = self.net.repeaters[self.source]
        for qi in src_rep.occupied_indices():
            if int(src_rep.partner_repeater[qi]) == self.dest:
                return True, float(werner_to_fidelity(src_rep.werner_param[qi]))
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
        return self.get_observation()

    @staticmethod
    def action_label(action: int, node: int) -> str:
        return f"{['W','S','P'][action]}({node})"