"""
Repeater module with qubit locking for classical communication delays.

Handles the intra-node logic.
"""

from __future__ import annotations
import enum
from typing import Optional, Tuple
import numpy as np

# ── helpers ───────────────────────────────────────────────────────
def fidelity_to_werner(f):
    return (4.0 * np.asarray(f, dtype=np.float64) - 1.0) / 3.0

def werner_to_fidelity(p):
    return (3.0 * np.asarray(p, dtype=np.float64) + 1.0) / 4.0

def bbpssw_success_prob(f1, f2):
    return (8/9 * f1 * f2) - 2/9 * (f1 + f2) +5/9

def bbpssw_new_werner(f1, f2):
    return (1 - (f1 + f2) + 10 * f1 * f2)/(5 - 2 * (f1 + f2) +8 * f1 * f2)

# ── enums / constants ─────────────────────────────────────────────
class SwapPolicy(enum.IntEnum):
    FARTHEST  = 0
    STRONGEST = 1
    RANDOM    = 2

QUBIT_FREE: np.int8 = np.int8(0)
QUBIT_OCCUPIED: np.int8 = np.int8(1)
NO_PARTNER: int = -1

# ── Repeater ──────────────────────────────────────────────────────
class Repeater:
    __slots__ = (
        "rid", 
        "n_ch", 
        "swap_policy", 
        "position",
        "p_gen", 
        "p_swap", 
        "cutoff",
        "status", 
        "partner_repeater", 
        "partner_qubit",
        "werner_param", 
        "initial_werner", 
        "age", 
        "link_cutoff",
        "locked",
    )

    def __init__(self, 
                 rid: int, 
                 n_ch: int,
                 swap_policy: SwapPolicy = SwapPolicy.FARTHEST,
                 position: Optional[np.ndarray] = None,
                 p_gen: float = 0.8, 
                 p_swap: float = 0.5,
                 cutoff: int = 20
                 ):
        self.rid = rid
        self.n_ch = n_ch
        self.swap_policy = swap_policy
        self.position = (np.array(position, dtype=np.float64)
                         if position is not None else np.zeros(2, dtype=np.float64))
        self.p_gen = p_gen
        self.p_swap = p_swap
        self.cutoff = cutoff
        self.status = np.full(n_ch, QUBIT_FREE, dtype=np.int8)
        self.partner_repeater = np.full(n_ch, NO_PARTNER, dtype=np.int32)
        self.partner_qubit = np.full(n_ch, NO_PARTNER, dtype=np.int32)
        self.werner_param = np.zeros(n_ch, dtype=np.float64)
        self.initial_werner = np.zeros(n_ch, dtype=np.float64)
        self.age = np.zeros(n_ch, dtype=np.int32)
        self.link_cutoff = np.full(n_ch, cutoff, dtype=np.int32)
        self.locked = np.zeros(n_ch, dtype=np.bool_)

    # ── raw queries (include locked, used internally) ─────────────
    def free_indices(self) -> np.ndarray:
        return np.flatnonzero(self.status == QUBIT_FREE)

    def occupied_indices(self) -> np.ndarray:
        return np.flatnonzero(self.status == QUBIT_OCCUPIED)

    def num_occupied(self) -> int:
        return int(np.count_nonzero(self.status == QUBIT_OCCUPIED))

    # ── agent-facing queries (exclude locked) ─────────────────────
    def available_indices(self) -> np.ndarray:
        """Occupied AND not locked."""
        return np.flatnonzero((self.status == QUBIT_OCCUPIED) & (~self.locked))

    def num_available(self) -> int:
        return int(np.count_nonzero((self.status == QUBIT_OCCUPIED) & (~self.locked)))

    def has_free_qubit(self) -> bool:
        return bool(np.any((self.status == QUBIT_FREE) & (~self.locked)))

    def can_swap(self) -> bool:
        return self.num_available() >= 2

    def qubits_to(self, partner_rid: int) -> np.ndarray:
        """Available (occupied, unlocked) qubits linked to partner_rid."""
        mask = ((self.status == QUBIT_OCCUPIED)
                & (self.partner_repeater == partner_rid)
                & (~self.locked))
        return np.flatnonzero(mask)
    
    def num_locked(self) -> int:
        return int(np.count_nonzero(self.locked))

    # ── state mutation ────────────────────────────────────────────
    def allocate_qubit(self) -> int:
        """Allocate the first available qubit"""
        free = np.flatnonzero((self.status == QUBIT_FREE) & (~self.locked))
        if len(free) == 0:
            return -1
        idx = int(free[0])
        self.status[idx] = QUBIT_OCCUPIED
        return idx

    def set_link(self, 
                 qidx: int, 
                 partner_rid: int, 
                 partner_qidx: int, 
                 p: float,
                 link_age: int=0, 
                 effective_cutoff: None | float=None):
        
        self.partner_repeater[qidx] = partner_rid
        self.partner_qubit[qidx] = partner_qidx
        self.initial_werner[qidx] = p
        self.age[qidx] = link_age
        # HACK mean cutoff calculation
        self.link_cutoff[qidx] = (effective_cutoff if effective_cutoff is not None
                                   else self.cutoff)
        lc = int(self.link_cutoff[qidx])
        # set the value for p. depending on age and effective cutoff
        self.werner_param[qidx] = (p * np.exp(-link_age / lc)
                                    if lc > 0 and link_age > 0 else p)

    def free_qubit(self, qidx):
        self.status[qidx] = QUBIT_FREE
        self.partner_repeater[qidx] = NO_PARTNER
        self.partner_qubit[qidx] = NO_PARTNER
        self.werner_param[qidx] = 0.0
        self.initial_werner[qidx] = 0.0
        self.age[qidx] = 0
        self.link_cutoff[qidx] = self.cutoff
        self.locked[qidx] = False

    def lock_qubit(self, qidx):
        self.locked[qidx] = True

    def unlock_qubit(self, qidx):
        self.locked[qidx] = False

    # ── aging ─────────────────────────────────────────────────────
    def age_occupied(self) -> np.ndarray:
        """Age all occupied qubits (including locked). Return expired indices."""
        occ_mask = (self.status == QUBIT_OCCUPIED)

        #empty if repeater is also empty
        if not np.any(occ_mask):
            return np.empty(0, dtype=np.intp)
        
        self.age[occ_mask] += 1
        lc = np.maximum(self.link_cutoff[occ_mask].astype(np.float64), 1e-30)

        self.werner_param[occ_mask] = (
            self.initial_werner[occ_mask]
            * np.exp(-self.age[occ_mask].astype(np.float64) / lc))
        
        # REVIEW return any 
        return np.flatnonzero(occ_mask & (self.age >= self.link_cutoff))

    # ── vectorised swap pair selection (uses available only) ──────
    def select_swap_pair(self, network_positions):
        occ = self.available_indices()
        k = len(occ)
        if k < 2:
            return None
        if self.swap_policy == SwapPolicy.RANDOM:
            chosen = np.random.choice(occ, size=2, replace=False)
            return int(chosen[0]), int(chosen[1])
        idx_i, idx_j = np.triu_indices(k, k=1)
        qa_all, qb_all = occ[idx_i], occ[idx_j]
        if self.swap_policy == SwapPolicy.FARTHEST:
            dists = np.linalg.norm(
                network_positions[self.partner_repeater[qa_all]]
                - network_positions[self.partner_repeater[qb_all]], axis=1)
            best = int(np.argmax(dists))
        else:
            products = self.werner_param[qa_all] * self.werner_param[qb_all]
            best = int(np.argmax(products))
        return int(qa_all[best]), int(qb_all[best])

    # ── features ──────────────────────────────────────────────────
    def feature_vector(self) -> np.ndarray:
        """
        REPEATER feature vector:
            [pos_x, pos_y, frac_occupied, mean_fidelity, p_gen, p_swap]
        """

        n_occ = self.num_occupied()
        frac = n_occ / self.n_ch #NOTE maybe use abs number of qubits instead?

        all_f = werner_to_fidelity(self.werner_param[self.status == QUBIT_OCCUPIED])
        mean_f = float(np.mean(all_f)
            if n_occ > 0 else 0.0)
        
        return np.array([self.position[0], 
                         self.position[1],
                         frac, 
                         mean_f, 
                         self.p_gen, 
                         self.p_swap], dtype=np.float64)

    def qubit_features(self) -> np.ndarray:
        """
        QUBIT feature vector:
            (n_ch, 6): [occupied, werner, fidelity, partner_rid, age_norm, locked]
        """
        is_occ = (self.status == QUBIT_OCCUPIED).astype(np.float64)
        fid = werner_to_fidelity(self.werner_param)
        pn = self.partner_repeater.astype(np.float64)

        #NOTE set unphisical links with no partenr to zero
        pn[pn == NO_PARTNER] = 0.0

        # normalized age
        age_norm = self.age.astype(np.float64) / max(self.cutoff, 1)
        is_locked = self.locked.astype(np.float64)

        return np.stack([is_occ, 
                         self.werner_param, 
                         fid, 
                         pn, 
                         age_norm, 
                         is_locked], axis=-1)

    def reset(self):
        """Resets the entire repeater"""
        self.status[:] = QUBIT_FREE
        self.partner_repeater[:] = NO_PARTNER
        self.partner_qubit[:] = NO_PARTNER
        self.werner_param[:] = 0.0
        self.initial_werner[:] = 0.0
        self.age[:] = 0
        self.link_cutoff[:] = self.cutoff
        self.locked[:] = False

    def __repr__(self):
        """Representation string for the repeater"""
        lk = self.num_locked()
        return (f"Repeater(rid={self.rid}, occ={self.num_occupied()}/{self.n_ch}"
                f"{f', locked={lk}' if lk else ''}, "
                f"p_gen={self.p_gen:.2f}, p_swap={self.p_swap:.2f}, "
                f"cutoff={self.cutoff}, policy={self.swap_policy.name})")