"""
Repeater module with qubit locking for classical communication delays.

Handles the intra-node logic.
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
    # Simplified BBPSSW success probability for Werner states.
    # Equivalent to (3*p1*p2 + 1)/4 in Werner parameters.
    # Yields 0.25 at F=0.25 (fully mixed), 1 at F=1 (perfect).
    return (4/3 * f1 * f2) - 1/3 * (f1 + f2) + 1/3

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
        "n_ch",              # Number of qubits on repeater 
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
        "locked",            # Locked qubits (used for CC)
        "generation_id",     # Monotonic counter incremented on each allocation
    )

    def __init__(self, 
                 rid: int, 
                 n_ch: int = 2,
                 swap_policy: SwapPolicy = SwapPolicy.FARTHEST,
                 position: Optional[np.ndarray] = None,
                 p_gen: float = 0.8, 
                 p_swap: float = 0.5,
                 cutoff: int = 20
                 ):
        
        # Repeater Attributes
        self.rid = rid
        self.n_ch = n_ch
        self.swap_policy = swap_policy
        self.position = (np.array(position, dtype=np.float64) if position is not None else np.zeros(2, dtype=np.float64))
        self.p_gen = p_gen
        self.p_swap = p_swap
        self.cutoff = cutoff

        #Qubit Attributes
        self.status = np.full(n_ch, QUBIT_FREE, dtype=np.int8)
        self.partner_repeater = np.full(n_ch, NO_PARTNER, dtype=np.int32)
        self.partner_qubit = np.full(n_ch, NO_PARTNER, dtype=np.int32)
        self.werner_param = np.zeros(n_ch, dtype=np.float32)
        self.initial_werner = np.zeros(n_ch, dtype=np.float64)
        self.age = np.zeros(n_ch, dtype=np.int32)
        self.link_cutoff = np.full(n_ch, cutoff, dtype=np.int32)
        self.locked = np.zeros(n_ch, dtype=np.bool_)
        self.generation_id = np.zeros(n_ch, dtype=np.uint32)

                                                                                 
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

    def has_free_qubit(self) -> bool:
        return bool(np.any((self.status == QUBIT_FREE) & (~self.locked)))

    def can_swap(self) -> bool:
        return self.num_available() >= 2

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
                                                                                      
                                                                                      

    def allocate_qubit(self) -> int:
        """
        Allocate the first available qubit > Set is to QUBIT_OCCUPIED
        Return: -1 if no free qubit else return qubit idx
        """
        freeQubits = np.flatnonzero((self.status == QUBIT_FREE) & (~self.locked))
        if len(freeQubits) == 0:
            return -1
        qubit = int(freeQubits[0]) # choose the first one in the list
        self.status[qubit] = QUBIT_OCCUPIED
        self.generation_id[qubit] += 1
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
                                     
    def select_swap_pair(self, network_positions:np.array, rng: Optional[np.random.Generator] = None) -> (Tuple[int, int] | None):
        """Internal selection of the swap pair"""
        occupiedQubits = self.available_indices()
        numQubitsReadyToSwap = len(occupiedQubits)

        if numQubitsReadyToSwap < 2:
            return None
        
        if self.swap_policy == SwapPolicy.RANDOM:
            _rng = rng if rng is not None else np.random.default_rng()
            chosen = _rng.choice(occupiedQubits, size=2, replace=False)
            return int(chosen[0]), int(chosen[1])
        
        idx_i, idx_j = np.triu_indices(numQubitsReadyToSwap, k=1)
        qa_all, qb_all = occupiedQubits[idx_i], occupiedQubits[idx_j]

        if self.swap_policy == SwapPolicy.FARTHEST:
            #calculate the distance to each remote qubit
            distanceAC = network_positions[self.partner_repeater[qa_all]]
            distanceCB = network_positions[self.partner_repeater[qb_all]]
            dists = np.linalg.norm(distanceAC - distanceCB, axis=1)
            # return the largest distance idx
            best = int(np.argmax(dists))
        else:
            products = self.werner_param[qa_all] * self.werner_param[qb_all]
            best = int(np.argmax(products))
        return int(qa_all[best]), int(qb_all[best])


                                                
#  ▄▄▄▄▄▄▄                                        
# ███▀▀▀▀▀           ██                           
# ███▄▄ ▄█▀█▄  ▀▀█▄ ▀██▀▀ ██ ██ ████▄ ▄█▀█▄ ▄█▀▀▀ 
# ███▀▀ ██▄█▀ ▄█▀██  ██   ██ ██ ██ ▀▀ ██▄█▀ ▀███▄ 
# ███   ▀█▄▄▄ ▀█▄██  ██   ▀██▀█ ██    ▀█▄▄▄ ▄▄▄█▀ 
                                                
                                                
    def feature_vector(self) -> np.ndarray:
        """
        REPEATER feature vector to be fed into the GNN
            `[pos_x, pos_y, frac_occupied, mean_fidelity, p_gen, p_swap]`
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
                         self.p_swap],)

    def qubit_features(self) -> np.ndarray:
        """
        QUBIT feature vector:
            (n_ch, 6): [occupied, werner, fidelity, partner_rid, age_norm, locked]
        """
        is_occ = (self.status == QUBIT_OCCUPIED).astype(np.float64)
        fid = werner_to_fidelity(self.werner_param)
        pn = self.partner_repeater.astype(np.float64)

        # Encode "no partner" as -1.0 to avoid collision with repeater ID 0
        pn[pn == NO_PARTNER] = -1.0

        # normalized age
        age_norm = self.age.astype(np.float64) / max(self.cutoff, 1)
        is_locked = self.locked.astype(np.float64)

        return np.stack([is_occ, 
                         self.werner_param, 
                         fid, 
                         pn, 
                         age_norm, 
                         is_locked], axis=-1)                           
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
        return (f"Repeater(rid={self.rid}, occ={self.num_occupied()}/{self.n_ch}"
                f"{f', locked={lk}' if lk else ''}, "
                f"p_gen={self.p_gen:.2f}, p_swap={self.p_swap:.2f}, "
                f"cutoff={self.cutoff}, policy={self.swap_policy.name})")
    
