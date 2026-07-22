"""
--------------------------------------------------------------------------------
Repeater module: intra-node logic.

Qubits are split into two fixed ports: n_left LEFT-facing qubits (indices
[0, n_left)) that only entangle with a lower-index neighbour, and n_right
RIGHT-facing qubits (indices [n_left, n_left+n_right)) that only entangle with a
higher-index neighbour. A swap fuses one LEFT link with one RIGHT link. End
nodes have a single port (the other count is 0). This matches the multiplexed
chain of arXiv 2401.13168, where n_ch counts qubits *per side*.
--------------------------------------------------------------------------------
"""

from __future__ import annotations
import enum
from typing import Optional, Tuple
import numpy as np

# --- HELPERS -------------------------------------------------
def fidelity_to_werner(f):
    return (4.0 * np.asarray(f, dtype=np.float64) - 1.0) / 3.0

def werner_to_fidelity(p):
    return (3.0 * np.asarray(p, dtype=np.float64) + 1.0) / 4.0

def bbpssw_success_prob(f1, f2):
    """Canonical BBPSSW success probability on two Werner pairs.

    Bennett et al., PRL 76, 722 (1996). In fidelities the twirled BBPSSW
    density matrix gives P_succ = (8*F1*F2 - 2*(F1+F2) + 5)/9; in Werner
    parameters this is (p1*p2 + 1)/2. It is exactly 1/9 of the denominator
    of ``bbpssw_new_fidelity`` (both derive from the one BBPSSW density
    matrix), so the two are self-consistent. Anchors: 5/9 at F1=F2=0.5, 1 at
    F1=F2=1.
    """
    return (8 * f1 * f2 - 2 * (f1 + f2) + 5) / 9

def bbpssw_new_fidelity(f1, f2):
    """Post-purification fidelity given two input fidelities (BBPSSW protocol)."""
    return (1 - (f1 + f2) + 10 * f1 * f2)/(5 - 2 * (f1 + f2) + 8 * f1 * f2)

class SwapPolicy(enum.IntEnum):
    FARTHEST  = 0
    STRONGEST = 1
    RANDOM    = 2

QUBIT_FREE: np.int8 = np.int8(0)
QUBIT_OCCUPIED: np.int8 = np.int8(1)
NO_PARTNER: int = -1

LEFT: int = 0   # port facing lower-index neighbours
RIGHT: int = 1  # port facing higher-index neighbours

class Repeater:
    """                                                 
▄▄▄▄▄▄▄                                             
███▀▀███▄                          ██               
███▄▄███▀ ▄█▀█▄ ████▄ ▄█▀█▄  ▀▀█▄ ▀██▀▀ ▄█▀█▄ ████▄ 
███▀▀██▄  ██▄█▀ ██ ██ ██▄█▀ ▄█▀██  ██   ██▄█▀ ██ ▀▀ 
███  ▀███ ▀█▄▄▄ ████▀ ▀█▄▄▄ ▀█▄██  ██   ▀█▄▄▄ ██    
                ██                                  
                ▀▀                                  
    """
    __slots__ = (
        "rid",               # Unique ID of the repeater
        "n_ch",              # Number of qubits PER SIDE (left or right)
        "n_left",            # LEFT-facing qubit count (indices [0, n_left))
        "n_right",           # RIGHT-facing qubit count (indices [n_left, n_left+n_right))
        "swap_policy",       # The swap policy used by the repeater
        "position",          # The repeater position [x,y] in the network
        "p_gen",             # The elementary link generation probability
        "p_swap",            # The BSM probability
        "cutoff",            # Repeater specific cutoff
        "status",            # Status of the qubits (occupied=1 or FREE=0)
        "partner_repeater",  # rIDs of the partner repeaters for each qubit
        "partner_qubit",     # qIDs for the partner qubits
        "werner_param",      # The werner parameter p
        "initial_werner",    # Werner param to be used for ageing
        "age",               # The ages of the links
        "link_cutoff",       # Effective link cutoff (min(c1, c2))
        "locked",            # Locked qubits (invisible to the policy while an op is pending)
    )

    def __init__(self,
                 rid: int,
                 n_ch: int = 2,
                 swap_policy: SwapPolicy = SwapPolicy.FARTHEST,
                 position: Optional[np.ndarray] = None,
                 p_gen: float = 0.8,
                 p_swap: float = 0.5,
                 cutoff: int = 20,
                 n_left: Optional[int] = None,
                 n_right: Optional[int] = None,
                 ):

        # Repeater Attributes
        self.rid = rid
        self.n_ch = n_ch
        # Default: an interior node with both ports of width n_ch. End nodes pass
        # n_left/n_right explicitly (one is 0). Total qubits = n_left + n_right.
        self.n_left = n_ch if n_left is None else n_left
        self.n_right = n_ch if n_right is None else n_right
        n_total = self.n_left + self.n_right
        self.swap_policy = swap_policy
        self.position = (np.array(position, dtype=np.float64) if position is not None else np.zeros(2, dtype=np.float64))
        self.p_gen = p_gen
        self.p_swap = p_swap
        self.cutoff = cutoff

        #Qubit Attributes (sized to the two ports combined)
        self.status = np.full(n_total, QUBIT_FREE, dtype=np.int8)
        self.partner_repeater = np.full(n_total, NO_PARTNER, dtype=np.int32)
        self.partner_qubit = np.full(n_total, NO_PARTNER, dtype=np.int32)
        self.werner_param = np.zeros(n_total, dtype=np.float32)
        self.initial_werner = np.zeros(n_total, dtype=np.float64)
        self.age = np.zeros(n_total, dtype=np.int32)
        self.link_cutoff = np.full(n_total, cutoff, dtype=np.int32)
        self.locked = np.zeros(n_total, dtype=np.bool_)

    def __deepcopy__(self, memo):
        """Fast clone: the per-qubit arrays are the only mutable state, so copy
        those and share the immutable config (rid/rates/cutoff/policy/position
        are fixed at build and never written by step/age_links). Replaces the
        generic recursive ``copy.deepcopy`` (a measured ~50% of the exact-DP
        kernel build, 37M recursive calls) with 10 ``np.ndarray.copy`` calls."""
        new = object.__new__(Repeater)
        memo[id(self)] = new
        # immutable config -- shared by reference (scalars/enum are immutable;
        # position is read-only during a step)
        new.rid = self.rid
        new.n_ch = self.n_ch
        new.n_left = self.n_left
        new.n_right = self.n_right
        new.swap_policy = self.swap_policy
        new.p_gen = self.p_gen
        new.p_swap = self.p_swap
        new.cutoff = self.cutoff
        # mutable per-qubit state -- copied
        new.position = self.position.copy()
        new.status = self.status.copy()
        new.partner_repeater = self.partner_repeater.copy()
        new.partner_qubit = self.partner_qubit.copy()
        new.werner_param = self.werner_param.copy()
        new.initial_werner = self.initial_werner.copy()
        new.age = self.age.copy()
        new.link_cutoff = self.link_cutoff.copy()
        new.locked = self.locked.copy()
        return new

                                                                                 
# ▄▄▄▄▄▄▄                         ▄▄                                               
# ███▀▀███▄                      ██                     ██   ▀▀                    
# ███▄▄███▀  ▀▀█▄ ▄█▀▀▀ ▄█▀█▄   ▀██▀ ██ ██ ████▄ ▄████ ▀██▀▀ ██  ▄███▄ ████▄ ▄█▀▀▀ 
# ███  ███▄ ▄█▀██ ▀███▄ ██▄█▀    ██  ██ ██ ██ ██ ██     ██   ██  ██ ██ ██ ██ ▀███▄ 
# ████████▀ ▀█▄██ ▄▄▄█▀ ▀█▄▄▄    ██  ▀██▀█ ██ ██ ▀████  ██   ██▄ ▀███▀ ██ ██ ▄▄▄█▀ 
                                                                                                                                                            

    # --- Raw queries (include locked, used INTERNALLY) --------------------
    def free_indices(self) -> np.ndarray:
        return np.flatnonzero(self.status == QUBIT_FREE)

    def occupied_indices(self) -> np.ndarray:
        return np.flatnonzero(self.status == QUBIT_OCCUPIED)

    def num_occupied(self) -> int:
        return int(np.count_nonzero(self.status == QUBIT_OCCUPIED))

    # --- Network-facing queries (exclude locked) --------------------------

    def available_indices(self) -> np.ndarray:
        """Available FOR SWAP = Occupied AND not locked."""
        return np.flatnonzero((self.status == QUBIT_OCCUPIED) & (~self.locked))

    def num_available(self) -> int:
        return int(np.count_nonzero((self.status == QUBIT_OCCUPIED) & (~self.locked)))

    def _side_range(self, side: int) -> Tuple[int, int]:
        """[lo, hi) qubit-index range for a port."""
        return (0, self.n_left) if side == LEFT else (self.n_left, self.n_left + self.n_right)

    def available_on_side(self, side: int) -> np.ndarray:
        """Available (occupied, unlocked) qubit indices on one port."""
        lo, hi = self._side_range(side)
        if lo == hi:
            return np.empty(0, dtype=np.intp)
        rel = (self.status[lo:hi] == QUBIT_OCCUPIED) & (~self.locked[lo:hi])
        return np.flatnonzero(rel) + lo

    def has_free_qubit(self, side: Optional[int] = None) -> bool:
        free = (self.status == QUBIT_FREE) & (~self.locked)
        if side is None:
            return bool(np.any(free))
        lo, hi = self._side_range(side)
        return bool(np.any(free[lo:hi]))

    def can_swap(self) -> bool:
        """A BSM needs one available LEFT link AND one available RIGHT link."""
        return len(self.available_on_side(LEFT)) >= 1 and len(self.available_on_side(RIGHT)) >= 1

    def qubits_to(self, partner_rid: int) -> np.ndarray:
        """Available (occupied, unlocked) qubits linked to partner_rid."""
        isOccupied = (self.status == QUBIT_OCCUPIED)
        hasCorrectPartnerID = (self.partner_repeater == partner_rid)
        isFree = ~self.locked
        mask = isOccupied & hasCorrectPartnerID & isFree
        return np.flatnonzero(mask)
    
    def num_locked(self) -> int:
        return int(np.count_nonzero(self.locked))

                                                                                      
#  ▄▄▄▄▄▄▄                          ▄▄▄      ▄▄▄                                        
# █████▀▀▀  ██         ██           ████▄  ▄████        ██         ██   ▀▀              
#  ▀████▄  ▀██▀▀ ▀▀█▄ ▀██▀▀ ▄█▀█▄   ███▀████▀███ ██ ██ ▀██▀▀ ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
#    ▀████  ██  ▄█▀██  ██   ██▄█▀   ███  ▀▀  ███ ██ ██  ██  ▄█▀██  ██   ██  ██ ██ ██ ██ 
# ███████▀  ██  ▀█▄██  ██   ▀█▄▄▄   ███      ███ ▀██▀█  ██  ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 
                                                                                      
                                                                                      

    def allocate_qubit(self, side: int) -> int:
        """
        Allocate the first free qubit on the requested port (LEFT/RIGHT).
        Return: -1 if that port has no free qubit else the qubit idx.
        """
        lo, hi = self._side_range(side)
        if lo == hi:
            return -1
        rel = (self.status[lo:hi] == QUBIT_FREE) & (~self.locked[lo:hi])
        freeQubits = np.flatnonzero(rel)
        if len(freeQubits) == 0:
            return -1
        qubit = int(freeQubits[0]) + lo  # first free on this side
        self.status[qubit] = QUBIT_OCCUPIED
        return qubit

    def set_link(self, 
                 qubit: int, 
                 partner_rid: int, 
                 partner_qidx: int, 
                 p: float,
                 link_age: int=0, 
                 effective_cutoff: None | float=None):
        """
        Set link between two qubits between two repeaters
        Args:
            qidx.       : The qubit to include in the link
            partner_rid : the ID of the partner repeater
            p           : The Werner parameter at t=0
            link_age    : The age of the link on register (can be >0 due to CC)
        """
        if partner_rid == NO_PARTNER:
            raise ValueError('set_link called with NO_PARTNER partner_rid '
                             '(orphan qubit — would index repeaters[-1])')
        if partner_rid == self.rid:
            raise ValueError('Attempting to generate inter-node entanglement')
        
        # Point THIS repeater to the remote repeater
        self.partner_repeater[qubit] = partner_rid
        self.partner_qubit[qubit] = partner_qidx

        self.initial_werner[qubit] = p
        self.age[qubit] = link_age

        # HACK Set the cutoff for THIS LINK
        self.link_cutoff[qubit] = effective_cutoff if effective_cutoff is not None else self.cutoff

        # set the value for p. depending on age and effective cutoff
        linkCutoff = int(self.link_cutoff[qubit])
        if linkCutoff > 0 and link_age > 0:
            self.werner_param[qubit] = p * np.exp(-link_age / linkCutoff)
        else:
            self.werner_param[qubit] = p

    def free_qubit(self, qubit):
        """Set a qubit free by removing all internal and external pointers"""
        self.status[qubit] = QUBIT_FREE
        self.partner_repeater[qubit] = NO_PARTNER
        self.partner_qubit[qubit] = NO_PARTNER
        self.werner_param[qubit] = 0.0
        self.initial_werner[qubit] = 0.0
        self.age[qubit] = 0
        self.link_cutoff[qubit] = self.cutoff
        self.locked[qubit] = False

    def lock_qubit(self, qubit):
        self.locked[qubit] = True

    def unlock_qubit(self, qubit):
        self.locked[qubit] = False

    def age_occupied(self) -> np.ndarray:
        """
        Age all occupied qubits (including locked). Return expired indices.
        Returns:
            unaffectedQubits: List of qubits idx that either died or are occupied
        """
        occupationMask = (self.status == QUBIT_OCCUPIED)
        qubits = occupationMask

        # Return empty if no entanglements
        if not np.any(qubits):
            return np.empty(0, dtype=np.intp)
        
        self.age[qubits] += 1 #tick
        
        onlineCutoffs = self.link_cutoff[qubits]
        onlineP0s = self.initial_werner[qubits] 
        onlineAges = self.age[qubits]

        # Update OCCUPIED ONLY: \lambda = p0 e^(-m/m*)
        safe_cutoffs = np.maximum(onlineCutoffs, 1)
        self.werner_param[qubits] = (onlineP0s * np.exp(-onlineAges / safe_cutoffs))
        unaffectedQubits = np.flatnonzero(qubits & (self.age >= self.link_cutoff))
        return unaffectedQubits


                                                                   
#  ▄▄▄▄▄▄▄                        ▄▄▄▄▄▄▄       ▄▄                   
# █████▀▀▀                       █████▀▀▀       ██              ██   
#  ▀████▄  ██   ██  ▀▀█▄ ████▄    ▀████▄  ▄█▀█▄ ██ ▄█▀█▄ ▄████ ▀██▀▀ 
#    ▀████ ██ █ ██ ▄█▀██ ██ ██      ▀████ ██▄█▀ ██ ██▄█▀ ██     ██   
# ███████▀  ██▀██  ▀█▄██ ████▀   ███████▀ ▀█▄▄▄ ██ ▀█▄▄▄ ▀████  ██   
#                        ██                                          
#                        ▀▀     
                                     
    def select_swap_pair(self, network_positions: np.array,
                         network_cutoffs: np.ndarray,
                         rng: Optional[np.random.Generator] = None
                         ) -> (Tuple[int, int] | None):
        """Internal selection of the swap pair: one LEFT link + one RIGHT link,
        among VIABLE pairs only.

        A BSM fuses a left-facing link (partner < rid) with a right-facing link
        (partner > rid), so the two remote endpoints are always distinct.

        Viability (arXiv 2401.13168 protocol step 2, adapted to same-tick
        resolution): the fused link inherits age_a + age_b, and both parents
        age once more before the swap resolves this same tick, so it survives its
        cutoff iff age_a + age_b + 2 < ec with ec = min(remote endpoints'
        cutoffs). This gate is what lets _resolve_swap create the link
        unconditionally: no born-dead link can reach resolution.
        """
        left = self.available_on_side(LEFT)
        right = self.available_on_side(RIGHT)
        if len(left) == 0 or len(right) == 0:
            return None

        # every (left, right) combination
        qa_all = np.repeat(left, len(right))
        qb_all = np.tile(right, len(left))

        ra = self.partner_repeater[qa_all]
        rb = self.partner_repeater[qb_all]
        ec = np.minimum(network_cutoffs[ra], network_cutoffs[rb])
        viable = (self.age[qa_all] + self.age[qb_all] + 2) < ec
        if not bool(viable.any()):
            return None
        qa_v, qb_v = qa_all[viable], qb_all[viable]

        if self.swap_policy == SwapPolicy.RANDOM:
            _rng = rng if rng is not None else np.random.default_rng()
            k = int(_rng.integers(len(qa_v)))
            return int(qa_v[k]), int(qb_v[k])

        if self.swap_policy == SwapPolicy.FARTHEST:
            distanceAC = network_positions[self.partner_repeater[qa_v]]
            distanceCB = network_positions[self.partner_repeater[qb_v]]
            dists = np.linalg.norm(distanceAC - distanceCB, axis=1)
            best = int(np.argmax(dists))
        else:
            products = self.werner_param[qa_v] * self.werner_param[qb_v]
            best = int(np.argmax(products))
        return int(qa_v[best]), int(qb_v[best])


# ▄▄▄      ▄▄▄                 
# ████▄  ▄████ ▀▀              
# ███▀████▀███ ██  ▄█▀▀▀ ▄████ 
# ███  ▀▀  ███ ██  ▀███▄ ██    
# ███      ███ ██▄ ▄▄▄█▀ ▀████ 
                             
    def reset(self):
        """
        Resets the entire repeater
        """
        self.status[:] = QUBIT_FREE
        self.partner_repeater[:] = NO_PARTNER
        self.partner_qubit[:] = NO_PARTNER
        self.werner_param[:] = 0.0
        self.initial_werner[:] = 0.0
        self.age[:] = 0
        self.link_cutoff[:] = self.cutoff
        self.locked[:] = False

    def __repr__(self):
        """
        Representation string for the repeater
        """
        lk = self.num_locked()
        return (f"Repeater(rid={self.rid}, occ={self.num_occupied()}/{self.n_left + self.n_right}"
                f"{f', locked={lk}' if lk else ''}, "
                f"p_gen={self.p_gen:.2f}, p_swap={self.p_swap:.2f}, "
                f"cutoff={self.cutoff}, policy={self.swap_policy.name})")
    
