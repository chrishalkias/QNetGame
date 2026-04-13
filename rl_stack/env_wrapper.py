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

from quantum_repeater_sim.network import RepeaterNetwork, build_chain, build_grid, build_GEANT
from quantum_repeater_sim.repeater import (
    Repeater, SwapPolicy, werner_to_fidelity, NO_PARTNER,
    QUBIT_OCCUPIED,
)

# --- action constants ----------------------------------------------------
NOOP    = 0
SWAP    = 1
PURIFY  = 2
N_ACTIONS = 3
ACTION_NAMES = ["noop", "swap", "purify"]


class QRNEnv(RepeaterNetwork):
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
    FAILED_ACTION   = -0.05

    def __init__(self, 
                 n_repeaters = 5, 
                 n_ch = 4,
                 spacing = 50.0, 
                 p_gen = 0.8,
                 p_swap = 0.5, 
                 cutoff = 20,
                 F0 = 0.95, 
                 channel_loss = 0.02,
                 dt_seconds = 1e-4, 
                 max_steps = 50,
                 rng: Optional[np.random.Generator] = None,
                 heterogeneous = False,
                 topology = 'chain',
                 gamma = 0.99):
        
        if topology not in ['chain', 'grid', 'geant']:
            raise ValueError(f'Topology {topology} not supported')
        
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_steps = max_steps
        self.gamma = gamma
        self._phi = 0.0

        if topology == 'chain':
            self.net = build_chain(
                n_repeaters, n_ch=n_ch, spacing=spacing,
                p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                F0=F0, channel_loss=channel_loss,
                dt_seconds=dt_seconds,
                distance_dep_gen=True, rng=self.rng)
            
        elif topology == 'grid':
            self.net = build_grid(
                    rows=n_repeaters, cols=n_repeaters,
                    n_ch=n_ch, spacing=spacing,
                    swap_policy=SwapPolicy.FARTHEST,
                    p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                    rng=self.rng)

        elif topology == 'geant':
            self.net = build_GEANT(
                n_ch=n_ch, swap_policy=SwapPolicy.FARTHEST,
                p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
                rng=self.rng)

        if heterogeneous:
            for rep in self.net.repeaters:
                rep.p_gen  = self.rng.uniform(0.3, 1.0)
                rep.p_swap = self.rng.uniform(0.3, 1.0)

        self.N  = self.net.N
        self.source = -1
        self.dest = -1
        self.steps = 0
        self.done = False
        self.topology = topology
        self._pick_targets()

                                           
# ▄▄▄▄▄▄▄▄▄                                  
# ▀▀▀███▀▀▀                       ██         
#    ███  ▀▀█▄ ████▄ ▄████ ▄█▀█▄ ▀██▀▀ ▄█▀▀▀ 
#    ███ ▄█▀██ ██ ▀▀ ██ ██ ██▄█▀  ██   ▀███▄ 
#    ███ ▀█▄██ ██    ▀████ ▀█▄▄▄  ██   ▄▄▄█▀ 
#                       ██                   
#                     ▀▀▀                    

    def _pick_targets(self):
        if self.N <= 2 or self.topology == 'chain':
            self.source, self.dest = 0, self.N - 1
            return
        while True:
            s, d = sorted(self.rng.choice(self.N, size=2, replace=False))
            # Ensure source and dest are not directly adjacent
            if not self.net.adj[s, d]:
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
            [1] mean_fidelity     — avg F of available (unlocked) qubits (0 if none)
            [2] is_source         — 0/1
            [3] is_dest           — 0/1
            [4] frac_available    — available (unlocked occupied) / n_ch
            [5] can_swap          — 1.0 if ≥2 available qubits to different partners
            [6] can_purify        — 1.0 if ≥2 available qubits to same partner
            [7] time_remaining    — (max_steps - steps) / max_steps

        Features [5] and [6] are forced to 0 for source / dest.
        #TODO add p_gen, p_s, tau as features for inhomogenious
        """
        feats = np.zeros((self.N, 8), dtype=np.float32)
        for i, rep in enumerate(self.net.repeaters):
            feats[i, 0] = rep.num_occupied() / rep.n_ch
            avail = rep.available_indices()
            feats[i, 1] = (float(np.mean(werner_to_fidelity(rep.werner_param[avail])))
                           if len(avail) > 0 else 0.0)
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
            return self.get_observation(), reward, True, info

        if self.steps >= self.max_steps:
            self.done = True
            shaping = -self._phi
            return self.get_observation(), self.STEP_COST + penalty + shaping, True, info

        # Phase 4: auto-entangle for next step's observation
        self._auto_entangle()

        # PBRS: γΦ(s') - Φ(s)
        if self.topology == "chain":
            phi_new = self._compute_chain_progress()
            shaping = self.gamma * phi_new - self._phi
            self._phi = phi_new
        else:
            shaping = 0

        return self.get_observation(), self.STEP_COST + penalty + shaping, False, info

                                                                     
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

    def _exec_swap(self, r: int) -> Dict:
        return self.net.swap(r)

    def _exec_purify(self, r: int) -> Dict:
        rep = self.net.repeaters[r]
        avail = rep.available_indices()
        if len(avail) < 2:
            return {"success": False, "reason": "insufficient_qubits"}
        partners = rep.partner_repeater[avail]
        unique, counts = np.unique(
            partners[partners != NO_PARTNER], return_counts=True)
        valid = [(int(p), c) for p, c in zip(unique, counts) if c >= 2]
        if not valid:
            return {"success": False, "reason": "no_valid_pair"}
        best_nb = max(valid, key=lambda x: x[1])[0]
        return self.net.purify(r, best_nb)

    def _check_e2e(self) -> Tuple[bool, float]:
        """Check whether source and dest share a direct entanglement link."""
        src_rep = self.net.repeaters[self.source]
        for qi in src_rep.occupied_indices():
            if int(src_rep.partner_repeater[qi]) == self.dest:
                return True, float(werner_to_fidelity(src_rep.werner_param[qi]))
        return False, 0.0

    def _compute_chain_progress(self) -> float:
        """BFS from source through entangled links; return farthest hop / total hops."""
        if self.topology != 'chain':
            return 0.0
        total_hops = self.dest - self.source
        if total_hops <= 0:
            return 0.0

        visited = {self.source}
        frontier = {self.source}
        farthest = self.source

        while frontier:
            next_frontier = set()
            for node in frontier:
                rep = self.net.repeaters[node]
                for qi in rep.occupied_indices():
                    partner = int(rep.partner_repeater[qi])
                    if partner != NO_PARTNER and partner not in visited:
                        visited.add(partner)
                        next_frontier.add(partner)
                        if partner > farthest:
                            farthest = partner
            frontier = next_frontier

        return min((farthest - self.source) / total_hops, 1.0)
    
                             
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
        self._phi = self._compute_chain_progress()
        return self.get_observation()

    @staticmethod
    def action_label(action: int, node: int) -> str:
        return f"{['W','S','P'][action]}({node})"
    
    def render(self, filepath=None, figsize=None, dpi=250):
        return self.net.render(filepath, figsize, dpi, source_dest=(self.source, self.dest))


