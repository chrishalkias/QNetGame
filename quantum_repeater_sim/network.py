"""
RepeaterNetwork with asynchronous classical communication delays.

Handles the inter-node logic.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .repeater import (
    Repeater, SwapPolicy, NO_PARTNER, QUBIT_FREE, QUBIT_OCCUPIED,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_werner,
)

                                                                                           
# ▄▄▄▄▄▄▄▄▄                ▄▄                     ▄▄              ▄▄    ▄▄                   
# ▀▀▀███▀▀▀                ██                     ██          ▀▀  ██    ██                   
#    ███ ▄███▄ ████▄ ▄███▄ ██ ▄███▄ ▄████ ██ ██   ████▄ ██ ██ ██  ██ ▄████ ▄█▀█▄ ████▄ ▄█▀▀▀ 
#    ███ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██▄██   ██ ██ ██ ██ ██  ██ ██ ██ ██▄█▀ ██ ▀▀ ▀███▄ 
#    ███ ▀███▀ ████▀ ▀███▀ ██ ▀███▀ ▀████  ▀██▀   ████▀ ▀██▀█ ██▄ ██ ▀████ ▀█▄▄▄ ██    ▄▄▄█▀ 
#              ██                      ██   ██                                               
#              ▀▀                    ▀▀▀  ▀▀▀                                                

def build_chain(n_repeaters, 
                n_ch=4, 
                spacing=50.0,
                swap_policy=SwapPolicy.FARTHEST,
                p_gen=0.8, 
                p_swap=0.5, 
                cutoff=20, 
                **kw
                )-> RepeaterNetwork:
    """Creates a chain topology network"""
    reps = [
            Repeater(rid=i, 
                     n_ch=n_ch, 
                     swap_policy=swap_policy,
                     position=np.array([i * spacing, 0.0]),
                     p_gen=p_gen, 
                     p_swap=p_swap, 
                     cutoff=cutoff
                     )
            for i in range(n_repeaters)
            ]
    adj = np.zeros((n_repeaters, n_repeaters), dtype=np.float64)

    for i in range(n_repeaters - 1):
        adj[i, i+1] = adj[i+1, i] = 1.0

    return RepeaterNetwork(reps, adj, **kw)


def build_grid(rows, 
               cols, 
               n_ch=4, 
               spacing=50.0,
               swap_policy=SwapPolicy.FARTHEST,
               p_gen=0.8, 
               p_swap=0.5, 
               cutoff=20, 
               **kw
               )-> RepeaterNetwork:
    """Creates a grid topology network"""
    N = rows * cols
    reps = [
            Repeater(rid=idx, 
                     n_ch=n_ch, 
                     swap_policy=swap_policy,
                     position=np.array([c * spacing, r * spacing]),
                     p_gen=p_gen, 
                     p_swap=p_swap, 
                     cutoff=cutoff
                     ) for idx in range(N) for r, c in [divmod(idx, cols)]
            ]
    adj = np.zeros((N, N), dtype=np.float64)
    for idx in range(N):
        r, c = divmod(idx, cols)
        if c+1 < cols: adj[idx, idx+1] = adj[idx+1, idx] = 1.0
        if r+1 < rows: adj[idx, idx+cols] = adj[idx+cols, idx] = 1.0
    return RepeaterNetwork(reps, adj, **kw)

    def build_GEANT():
        ...


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


class RepeaterNetwork:
    """                                                           
  ▄▄▄▄▄   ▄▄▄    ▄▄▄                                        
▄███████▄ ████▄  ███        ██                       ▄▄     
███   ███ ███▀██▄███ ▄█▀█▄ ▀██▀▀ ██   ██ ▄███▄ ████▄ ██ ▄█▀ 
███▄█▄███ ███  ▀████ ██▄█▀  ██   ██ █ ██ ██ ██ ██ ▀▀ ████   
 ▀█████▀  ███    ███ ▀█▄▄▄  ██    ██▀██  ▀███▀ ██    ██ ▀█▄ 
      ▀▀                                                    
    """

    __slots__ = (
        "N",                # The total number of repeaters in the network
        "repeaters",        # A list of `Repeater` instances
        "adj",              # Adjecency matrix for the network
        "channel_loss",     # Fidelity damping coeff for distance dependent fibre loss
        "F0",               # Fidelity at zero distance
        "distance_dep_gen", # Have distance affect p_e
        "rng",              # Allow the use of user specified rng
        "time_step",        # The simulation timestep
        "dt_seconds",       # Simulation time to physical time
        "c_fiber",          # Speed of ligt in fibre
        "pending_events",   # List of pending events (for CC)
        "_positions",        # Array of repeater positions in space
        "_dist_matrix"       # Matrix of distances between repeaters
    )


    def __init__(self, 
                 repeaters: list[Repeater], 
                 adjacency: np.ndarray,
                 channel_loss: float = 0.02, 
                 F0: float = 1.0,
                 distance_dep_gen: bool = True,
                 rng: Optional[np.random.Generator] = None,
                 dt_seconds: float = 1e-4
                 ):
        
        #--- Base parameters
        self.repeaters = repeaters
        self.N = len(repeaters)
        self.adj = np.asarray(adjacency, dtype=np.float64)

        # import check: make sure provide adjecency is good
        assert self.adj.shape == (self.N, self.N)


        self.channel_loss = channel_loss
        self.F0 = F0
        self.distance_dep_gen = distance_dep_gen
        self.rng = rng if rng is not None else np.random.default_rng()
        self.time_step: int = 0

        # ---Classical delay parameters
        self.dt_seconds = dt_seconds
        self.c_fiber: float = 200_000.0  # km/s
        self.pending_events: List[dict] = []

        # ---Cached geometry
        self._positions = np.stack([r.position for r in self.repeaters], axis=0)
        #position differences
        diff = self._positions[:, None, :] - self._positions[None, :, :]
        self._dist_matrix = np.linalg.norm(diff, axis=-1)

    # ---------------- HELPER FUNCS ------------------------------

    def distance(self, r1: int, r2: int) -> float:
        return float(self._dist_matrix[r1, r2])

    def _classical_delay_steps(self, d_km: float) -> int:
        """Discrete steps for a classical signal to travel d_km through fibre."""
        if d_km <= 0.0 or self.dt_seconds <= 0.0:
            return 0
        simStepsForCC = int(np.ceil(d_km / (self.c_fiber * self.dt_seconds)))
        return simStepsForCC

    def _gen_prob(self, r1: int, r2: int) -> float:
        """Returns the effective p_e between two repeaters (can be distance dependent)"""
        p_avg = 0.5*(self.repeaters[r1].p_gen + self.repeaters[r2].p_gen)

        if self.distance_dep_gen:
            return p_avg * np.exp(-self.channel_loss * self.distance(r1, r2) / 2.0)
        return p_avg

    def _gen_fidelity(self, r1: int, r2: int) -> float:
        """Returns the fidelity of the generated elementary link"""
        return self.F0 * np.exp(-self.channel_loss * self.distance(r1, r2))

# ▄▄▄                          ▄▄▄▄▄▄▄                                                      
# ███      ▀▀        ▄▄       ███▀▀▀▀▀                                 ██   ▀▀              
# ███      ██  ████▄ ██ ▄█▀   ███       ▄█▀█▄ ████▄ ▄█▀█▄ ████▄  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███      ██  ██ ██ ████     ███  ███▀ ██▄█▀ ██ ██ ██▄█▀ ██ ▀▀ ▄█▀██  ██   ██  ██ ██ ██ ██ 
# ████████ ██▄ ██ ██ ██ ▀█▄   ▀██████▀  ▀█▄▄▄ ██ ██ ▀█▄▄▄ ██    ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 
                                                                                           

    # ── ACTION 1: entangle (instantaneous) ──────────────
    def entangle(self, r1: int, r2: int) -> Dict[str, Any]:
        """                                                                                                 
        Instantaneous EG between adjecent stations
        """
        result = {
                  "success": False, 
                  "fidelity": 0.0, 
                  "reason": ""
                  } 
        
        rep1, rep2 = self.repeaters[r1], self.repeaters[r2]

        if self.adj[r1, r2] == 0:
            result["reason"] = "not_adjacent"; return result
            
        if not rep1.has_free_qubit():
            result["reason"] = "no_free_qubit_r1"; return result
        
        if not rep2.has_free_qubit():
            result["reason"] = "no_free_qubit_r2"; return result
        
        if self.rng.random() > self._gen_prob(r1, r2):
            result["reason"] = "generation_failed"; return result
        
        # allocate one qubit on each repeater
        q1, q2 = rep1.allocate_qubit(), rep2.allocate_qubit()
        # give them the same shared fidelity
        fid = self._gen_fidelity(r1, r2)
        p = fidelity_to_werner(fid)
        # effective cutoff (min)
        ec = min(rep1.cutoff, rep2.cutoff)
        # register the link to the qubits
        rep1.set_link(q1, r2, q2, p, link_age=0, effective_cutoff=ec)
        rep2.set_link(q2, r1, q1, p, link_age=0, effective_cutoff=ec)
        result.update(success=True, fidelity=float(fid), reason="ok")
        return result


#  ▄▄▄▄▄▄▄                                     
# █████▀▀▀                     ▀▀              
#  ▀████▄  ██   ██  ▀▀█▄ ████▄ ██  ████▄ ▄████ 
#    ▀████ ██ █ ██ ▄█▀██ ██ ██ ██  ██ ██ ██ ██ 
# ███████▀  ██▀██  ▀█▄██ ████▀ ██▄ ██ ██ ▀████ 
#                        ██                 ██ 
#                        ▀▀               ▀▀▀  

    # ── ACTION 2: swap (deferred via event queue) ─────────────────
    def swap(self, r: int) -> Dict[str, Any]:
        """                                     
        Perform BSM at repeater r. On success, lock qubits and queue event.
        On failure, destroy both links immediately (no classical comm needed).
        """
        result = {"success": False, "new_fidelity": 0.0,"partners": None, "reason": ""} # Dict[str, Any]
        rep = self.repeaters[r]

        if not rep.can_swap():
            result["reason"] = "insufficient_qubits"; return result
        
        pair = rep.select_swap_pair(self._positions)

        if pair is None:
            result["reason"] = "no_valid_pair"; return result
        qa, qb = pair

        # Guard: both qubits point to the same remote repeater.
        # Swapping would try to create a self-link at the remote node.
        ra_check = int(rep.partner_repeater[qa])
        rb_check = int(rep.partner_repeater[qb])
        if ra_check == rb_check:
            result["reason"] = "same_partner"; return result

        # BSM outcome determined now
        if self.rng.random() > rep.p_swap:
            self._break_link(r, qa)
            self._break_link(r, qb)
            result["reason"] = "swap_failed"
            return result

        # -Success: compute p_new, free local qubits (BSM consumes them),
        #  lock remote qubits, queue event.
        # store the remote repeater and qubits
        ra, qa_r = int(rep.partner_repeater[qa]), int(rep.partner_qubit[qa])
        rb, qb_r = int(rep.partner_repeater[qb]), int(rep.partner_qubit[qb])
        p_new = rep.werner_param[qa] * rep.werner_param[qb]

        # The BSM physically destroys the local qubits — free them immediately
        # so the swapping repeater can reuse its memory slots.
        rep.free_qubit(qa)
        rep.free_qubit(qb)

        # Lock only the remote qubits (they must wait for classical notification).
        # Clear their stale back-pointers to the now-freed local qubits so that
        # an expiry during the delay does not corrupt reallocated local slots.
        rep_a, rep_b = self.repeaters[ra], self.repeaters[rb]
        rep_a.lock_qubit(qa_r)
        rep_b.lock_qubit(qb_r)
        rep_a.partner_repeater[qa_r] = NO_PARTNER
        rep_a.partner_qubit[qa_r]    = NO_PARTNER
        rep_b.partner_repeater[qb_r] = NO_PARTNER
        rep_b.partner_qubit[qb_r]    = NO_PARTNER

        # determine the max distance between the BSM station and the remote ones
        d_max = max(self.distance(r, ra), self.distance(r, rb))
        delay = self._classical_delay_steps(d_max)

        # append event to the queue
        self.pending_events.append({
            "type": "swap", "timer": delay,
            "r": r, "qa": qa, "qb": qb,
            "ra": ra, "qa_r": qa_r, "rb": rb, "qb_r": qb_r,
            "p_new": p_new,
        })

        result.update(success=True,
                      new_fidelity=float(werner_to_fidelity(p_new)),
                      partners=(ra, rb), reason="pending")
        return result

# ▄▄▄▄▄▄▄                    ▄▄                                       
# ███▀▀███▄             ▀▀  ██  ▀▀               ██   ▀▀              
# ███▄▄███▀ ██ ██ ████▄ ██ ▀██▀ ██  ▄████  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███▀▀▀▀   ██ ██ ██ ▀▀ ██  ██  ██  ██    ▄█▀██  ██   ██  ██ ██ ██ ██ 
# ███       ▀██▀█ ██    ██▄ ██  ██▄ ▀████ ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 

    # ── ACTION 3: purify (deferred via event queue) ───────────────
    def purify(self, r1: int, r2: int) -> Dict[str, Any]:
        """                                                                                                                                     
        BBPSSW purification. Lock all 4 qubits, queue event.
        Both success and failure are deferred (neither side knows outcome
        until classical message arrives).
        """
        result = {"success": False, "old_fidelity": 0.0, "new_fidelity": 0.0, "reason": ""} # Dict[str, Any]
        rep1, rep2 = self.repeaters[r1], self.repeaters[r2]
        q1s = rep1.qubits_to(r2)

        if len(q1s) < 2:
            result["reason"] = "insufficient_shared_pairs"
            return result

        werners = rep1.werner_param[q1s]
        si = np.argsort(werners)
        #QUESTION: is keeping the best and the worst good?
        q1_sac, q1_keep = int(q1s[si[0]]), int(q1s[si[-1]])
        q2_sac = int(rep1.partner_qubit[q1_sac])
        q2_keep = int(rep1.partner_qubit[q1_keep])
        p_keep, p_sac = rep1.werner_param[q1_keep], rep1.werner_param[q1_sac]

        result["old_fidelity"] = float(werner_to_fidelity(p_keep))

        success = self.rng.random() <= bbpssw_success_prob(p_keep, p_sac)
        p_new = bbpssw_new_werner(p_keep, p_sac) if success else 0.0

        # Lock all 4 qubits
        rep1.lock_qubit(q1_sac); rep1.lock_qubit(q1_keep)
        rep2.lock_qubit(q2_sac); rep2.lock_qubit(q2_keep)

        delay = self._classical_delay_steps(self.distance(r1, r2))

        self.pending_events.append({
            "type": "purify", "timer": delay, "success": success,
            "r1": r1, "r2": r2,
            "q1_sac": q1_sac, "q2_sac": q2_sac,
            "q1_keep": q1_keep, "q2_keep": q2_keep,
            "p_new": p_new,
        })

        result.update(success=success,
                      new_fidelity=float(werner_to_fidelity(p_new)) if success else 0.0,
                      reason="pending")
        return result


#   ▄▄▄▄                         
# ▄██▀▀██▄       ▀▀              
# ███  ███ ▄████ ██  ████▄ ▄████ 
# ███▀▀███ ██ ██ ██  ██ ██ ██ ██ 
# ███  ███ ▀████ ██▄ ██ ██ ▀████ 
#             ██              ██ 
#           ▀▀▀             ▀▀▀  
    # ── ACTION 4: age_links (+ event resolution) ─────────────────

    def age_links(self, discard_expired: bool = True) -> Dict[str, Any]:
        """                              
        Advance clock: age qubits, resolve pending events, expire old links."""
        self.time_step += 1

        # 1) age all occupied qubits (including locked)
        expired_pairs: List[Tuple[int, int]] = []

        for rep in self.repeaters:
            for qi in rep.age_occupied():
                expired_pairs.append((rep.rid, int(qi)))

        # 2) resolve pending events
        resolved = 0
        still_pending = []

        for ev in self.pending_events:
            ev["timer"] -= 1 # reduce timer by one step (event is closer to resolution)
            if ev["timer"] > 0:
                still_pending.append(ev)
                continue
            else:
                resolved += 1
                if ev["type"] == "swap":
                    self._resolve_swap(ev)

                elif ev["type"] == "purify":
                    self._resolve_purify(ev)

        self.pending_events = still_pending

        # 3) expire old links
        n_destroyed = 0
        if discard_expired:
            for rid, qidx in expired_pairs:
                rep = self.repeaters[rid]
                if rep.status[qidx] == QUBIT_OCCUPIED:
                    self._break_link(rid, qidx)
                    n_destroyed += 1

        return {"expired_count": n_destroyed,
                "over_cutoff_count": len(expired_pairs),
                "resolved_count": resolved,
                "pending_count": len(self.pending_events),
                "time_step": self.time_step}

                                                                                                  
#  ▄▄▄▄▄▄▄                           ▄▄▄▄▄▄▄                     ▄▄                             
# ███▀▀▀▀▀                    ██     ███▀▀███▄                   ██        ██   ▀▀              
# ███▄▄    ██ ██ ▄█▀█▄ ████▄ ▀██▀▀   ███▄▄███▀ ▄█▀█▄ ▄█▀▀▀ ▄███▄ ██ ██ ██ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███      ██▄██ ██▄█▀ ██ ██  ██     ███▀▀██▄  ██▄█▀ ▀███▄ ██ ██ ██ ██ ██  ██   ██  ██ ██ ██ ██ 
# ▀███████  ▀█▀  ▀█▄▄▄ ██ ██  ██     ███  ▀███ ▀█▄▄▄ ▄▄▄█▀ ▀███▀ ██ ▀██▀█  ██   ██▄ ▀███▀ ██ ██ 
                                                                                              
                                                                                              

    def _resolve_swap(self, ev: dict):
        """Resolve a deferred swap: rewrite remote qubits to point to each other.
        Local qubits were already freed at BSM time."""
        ra, qa_r, rb, qb_r = ev["ra"], ev["qa_r"], ev["rb"], ev["qb_r"]
        rep_a, rep_b = self.repeaters[ra], self.repeaters[rb]

        # [Guard] remote qubits may have been freed by expiry during the delay
        a_alive = rep_a.status[qa_r] == QUBIT_OCCUPIED
        b_alive = rep_b.status[qb_r] == QUBIT_OCCUPIED

        if not (a_alive and b_alive):
            # At least one side expired — clean up the survivor
            if a_alive:
                rep_a.free_qubit(qa_r)
            if b_alive:
                rep_b.free_qubit(qb_r)
            return

        ec = min(rep_a.cutoff, rep_b.cutoff)
        # Establish the new end-to-end link between the remote qubits
        rep_a.set_link(qa_r, rb, qb_r, ev["p_new"], link_age=0, effective_cutoff=ec)
        rep_b.set_link(qb_r, ra, qa_r, ev["p_new"], link_age=0, effective_cutoff=ec)
        rep_a.unlock_qubit(qa_r)
        rep_b.unlock_qubit(qb_r)



    def _resolve_purify(self, ev: dict):
        """Resolve a deferred purify: on success upgrade kept pair,
        on failure destroy both pairs."""
        r1, r2 = ev["r1"], ev["r2"]
        q1_sac, q2_sac = ev["q1_sac"], ev["q2_sac"]
        q1_keep, q2_keep = ev["q1_keep"], ev["q2_keep"]
        rep1, rep2 = self.repeaters[r1], self.repeaters[r2]

        if ev["success"]:
            # Destroy sacrifice pair
            self._break_link(r1, q1_sac)

            # Guard: kept qubits may have been freed by expiry
            if (rep1.status[q1_keep] != QUBIT_OCCUPIED or
                rep2.status[q2_keep] != QUBIT_OCCUPIED):
                if rep1.status[q1_keep] == QUBIT_OCCUPIED: self._break_link(r1, q1_keep)
                if rep2.status[q2_keep] == QUBIT_OCCUPIED: self._break_link(r2, q2_keep)
                return

            ec = min(rep1.cutoff, rep2.cutoff)
            rep1.set_link(q1_keep, r2, q2_keep, ev["p_new"],
                          link_age=0, effective_cutoff=ec)
            rep2.set_link(q2_keep, r1, q1_keep, ev["p_new"],
                          link_age=0, effective_cutoff=ec)
            rep1.unlock_qubit(q1_keep)
            rep2.unlock_qubit(q2_keep)
        else:
            # Failure: destroy both pairs (free_qubit clears locks)
            self._break_link(r1, q1_sac)
            self._break_link(r1, q1_keep)

                                                   
# ▄▄▄▄▄                                     ▄▄       
#  ███         ██                           ██       
#  ███  ████▄ ▀██▀▀ ▄█▀█▄ ████▄ ████▄  ▀▀█▄ ██ ▄█▀▀▀ 
#  ███  ██ ██  ██   ██▄█▀ ██ ▀▀ ██ ██ ▄█▀██ ██ ▀███▄ 
# ▄███▄ ██ ██  ██   ▀█▄▄▄ ██    ██ ██ ▀█▄██ ██ ▄▄▄█▀ 
                                                   
                                                   
    def _break_link(self, r: int, qidx: int):
        """Frees a qubit if it is pointing to nowhere"""
        rep = self.repeaters[r]
        pr, pq = int(rep.partner_repeater[qidx]), int(rep.partner_qubit[qidx])
        if pr != NO_PARTNER:
            self.repeaters[pr].free_qubit(pq)
        rep.free_qubit(qidx)


    def get_all_links(self) -> np.ndarray:
        """
        Get all the links in the network
            (L, 6): [r_a, q_a, r_b, q_b, fidelity, age], r_a < r_b.
            """
        links = []
        for rep in self.repeaters:
            for qi in rep.occupied_indices():
                pr = int(rep.partner_repeater[qi])
                if pr > rep.rid:
                    links.append([rep.rid, qi, pr, int(rep.partner_qubit[qi]),
                                  werner_to_fidelity(rep.werner_param[qi]),
                                  int(rep.age[qi])])
        return np.array(links, dtype=np.float64) if links else np.empty((0, 6), dtype=np.float64)
    

                                                                             
#   ▄▄▄▄                                 ▄▄▄      ▄▄▄                          
# ▄██▀▀██▄        ██   ▀▀                ████▄  ▄████             ▄▄           
# ███  ███ ▄████ ▀██▀▀ ██  ▄███▄ ████▄   ███▀████▀███  ▀▀█▄ ▄█▀▀▀ ██ ▄█▀ ▄█▀▀▀ 
# ███▀▀███ ██     ██   ██  ██ ██ ██ ██   ███  ▀▀  ███ ▄█▀██ ▀███▄ ████   ▀███▄ 
# ███  ███ ▀████  ██   ██▄ ▀███▀ ██ ██   ███      ███ ▀█▄██ ▄▄▄█▀ ██ ▀█▄ ▄▄▄█▀ 
                                                                             
                                                                             
    def action_mask_entangle(self) -> np.ndarray:
        """Entangle mask. Can only ask for entanglement in repeaters with free qubits"""
        mask = self.adj.copy().astype(bool)
        for i, rep in enumerate(self.repeaters):
            if not rep.has_free_qubit():
                mask[i, :] = False; mask[:, i] = False
        return mask

    def action_mask_swap(self) -> np.ndarray:
        """
        Swap mask: Can only swap if at least 2 qubits are connected 
        (outsourced to `Repeater.can_swap()`)
        """
        return np.array([rep.can_swap() for rep in self.repeaters], dtype=bool)

    def action_mask_purify(self) -> np.ndarray:
        mask = np.zeros((self.N, self.N), dtype=bool)
        for rep in self.repeaters:
            occ = rep.available_indices()
            if len(occ) < 2: continue
            partners = rep.partner_repeater[occ]
            unique, counts = np.unique(partners, return_counts=True)
            for pr, cnt in zip(unique, counts):
                if pr != NO_PARTNER and cnt >= 2:
                    mask[rep.rid, int(pr)] = True
        return mask
    
                             
# ▄▄▄      ▄▄▄                 
# ████▄  ▄████ ▀▀              
# ███▀████▀███ ██  ▄█▀▀▀ ▄████ 
# ███  ▀▀  ███ ██  ▀███▄ ██    
# ███      ███ ██▄ ▄▄▄█▀ ▀████ 
                             
                             
    def reset(self):
        self.time_step = 0
        self.pending_events.clear()
        for rep in self.repeaters:
            rep.reset()

    def __repr__(self) -> str:
        """Verbose summary of the state of the network (connections without idx)"""
        lines = [f"RepeaterNetwork N={self.N} t={self.time_step} "
                 f"pending={len(self.pending_events)}"]
        for rep in self.repeaters:
            lines.append(f"  {rep}")
        lk = self.get_all_links()
        lines.append(f"  Active links: {len(lk)}")
        for l in lk:
            lines.append(f"    R{int(l[0])}:q{int(l[1])}<->R{int(l[2])}:q{int(l[3])} "
                         f"F={l[4]:.4f} age={int(l[5])}")
        return "\n".join(lines)


                                                                                                        
                                                                                         
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄




#  ▄▄▄▄▄▄▄                         ▄▄                  
# ███▀▀███▄                        ██             ▀▀   
# ███▄▄███▀ ▄█▀█▄ ████▄  ▄███▄ ▄███▄ ▄█▀█▄ ████▄ ██  ████▄ ▄████ 
# ███▀▀██▄  ██▄█▀ ██ ██  ██ ██ ██ ██ ██▄█▀ ██ ▀▀ ██  ██ ██ ██ ██ 
# ███  ▀███ ▀█▄▄▄ ██ ██  ▀███▀ ▀███▀ ▀█▄▄▄ ██    ██▄ ██ ██ ▀████ 
#                                                               ██ 
#                                                             ▀▀▀  

    def render(self, filepath=None, figsize=None, dpi=250):
        """
        Render the current network state as a publication-quality PNG.

        Repeaters  → rounded boxes with an R_i label above.
        Qubits     → circles inside each box (white = free,
                      blue = occupied, red-orange = locked).
        Adjacency  → thin grey dashed lines between boxes.
        Entanglement links → curved arcs colour-coded by fidelity
                      that route around intermediate repeater boxes.

        Parameters
        ----------
        filepath : str or None
            Where to save the figure.  When *None* the matplotlib
            Figure is returned without saving.
        figsize  : tuple[float, float] or None
            Figure dimensions in inches.  Auto-computed when *None*.
        dpi : int
            Resolution of the saved PNG (default 250).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, Circle
        from matplotlib.path import Path as MplPath
        import numpy as np
        import networkx as nx

        plt.rcParams.update({
            "font.family":       "serif",
            "font.serif":        ["CMU Serif", "Computer Modern Roman",
                                  "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset":  "cm",
            "font.size":         8,
            "axes.linewidth":    0.4,
            "figure.facecolor":  "white",
            "savefig.facecolor": "white",
        })

        N      = self.N
        n_ch   = self.repeaters[0].n_ch
        pos    = self._positions.copy()


        #Build an nx.Graph for adjacency (used for edgedrawing and shortest-path hop counts)
        G_adj = nx.from_numpy_array(self.adj)
        pos_nx = {i: tuple(pos[i]) for i in range(N)}


        #Geometry parameters (scaled to inter-repeater spacing)
        if N > 1:
            nonzero = self._dist_matrix[self._dist_matrix > 0]
            scale = float(nonzero.min()) if len(nonzero) else 1.0
        else:
            scale = 1.0

        BOX_W     = scale * 0.55
        BOX_H     = scale * 0.30
        BOX_PAD   = scale * 0.012
        Q_RADIUS  = scale * 0.032
        C_BOX_FACE  = "#F2F2F2"
        C_BOX_EDGE  = "#2B2B2B"
        C_ADJ       = "#AAAAAA"
        C_FREE      = "#FFFFFF"
        C_FREE_E    = "#999999"
        C_OCC       = "#4477AA"
        C_OCC_E     = "#224488"
        C_LOCK      = "#EE6677"
        C_LOCK_E    = "#CC3311"


        #Compute qubit (x, y) positions inside each box
        # Dots sit in the lower half; labels go above them.
        qubit_xy = {}
        for rep in self.repeaters:
            cx, cy = rep.position
            for qi in range(n_ch):
                qx = cx - BOX_W / 2 + BOX_W * (qi + 1) / (n_ch + 1)
                qy = cy - BOX_H * 0.10
                qubit_xy[(rep.rid, qi)] = (qx, qy)


        #  Figure setup 
        all_x, all_y = pos[:, 0], pos[:, 1]
        margin_x = scale * 0.85
        margin_y = scale * 1.1          # extra vertical space for arcs

        x_lo = float(all_x.min()) - margin_x
        x_hi = float(all_x.max()) + margin_x
        y_lo = float(all_y.min()) - margin_y
        y_hi = float(all_y.max()) + margin_y

        if figsize is None:
            x_span = max(x_hi - x_lo, 1e-6)
            y_span = max(y_hi - y_lo, 1e-6)
            aspect = x_span / y_span
            fig_h  = max(3.5, min(10.0, 2.8 + N * 0.18))
            fig_w  = max(fig_h * aspect, fig_h)
            figsize = (fig_w, fig_h)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.axis("off")

        # Adjacency edges (dashed)
        nx.draw_networkx_edges(
            G_adj, pos_nx, ax=ax,
            style="dashed", width=0.7,
            edge_color=C_ADJ, alpha=0.75,
        )

        #  Entanglement arcs
        #      ─ colour-coded by fidelity (RdYlGn)
        #      ─ curvature scales with hop count so arcs
        #        clear all intermediate repeater boxes
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=0.25, vmax=1.0)

        links = self.get_all_links()
        from collections import defaultdict
        pair_idx = defaultdict(int)

        for lk in links:
            ra, qa, rb, qb = int(lk[0]), int(lk[1]), int(lk[2]), int(lk[3])
            fid = float(lk[4])

            x1, y1 = qubit_xy[(ra, qa)]
            x2, y2 = qubit_xy[(rb, qb)]
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < 1e-12:
                continue

            # perpendicular unit vector
            nx_v, ny_v = -dy / length, dx / length

            # ---- hop count via networkx shortest path ----
            try:
                n_hops = nx.shortest_path_length(G_adj, ra, rb)
            except nx.NetworkXNoPath:
                n_hops = 1

            # The arc's peak (at t = 0.5 of a quadratic Bézier) is
            #   peak_height = control_offset / 2.
            # We need peak_height > BOX_H / 2 + clearance for every
            # intermediate box, so control_offset > BOX_H + 2*clearance.
            clearance    = BOX_H * 0.45
            min_ctrl_off = BOX_H + 2 * clearance   # ≈ 1.9 × BOX_H
            hop_factor   = max(1, n_hops - 1)
            base_off     = max(min_ctrl_off, 0.15 * length) * hop_factor

            # ---- orient arc to avoid intermediate repeater boxes ----
            # Default: perpendicular with positive y component
            # (> "above" for horizontal chains).  For purely vertical
            # links, prefer positive x.
            if ny_v < 0 or (abs(ny_v) < 1e-9 and nx_v < 0):
                nx_v, ny_v = -nx_v, -ny_v

            # Override ONLY when an intermediate repeater is clearly
            # off-axis on the chosen side (not merely collinear).
            # Collinear intermediates are handled by the arc offset.
            rx1, ry1 = pos[ra]
            rx2, ry2 = pos[rb]
            rdx, rdy = rx2 - rx1, ry2 - ry1
            rl = np.hypot(rdx, rdy)
            if rl > 1e-12:
                rnx, rny = -rdy / rl, rdx / rl
                for k in range(N):
                    if k == ra or k == rb:
                        continue
                    pk = pos[k]
                    t_proj = ((pk[0]-rx1)*rdx + (pk[1]-ry1)*rdy) / (rl**2)
                    if 0.05 < t_proj < 0.95:
                        d_perp = (pk[0]-rx1)*rnx + (pk[1]-ry1)*rny
                        # Only flip for repeaters clearly on the arc
                        # side (off-axis), not for collinear ones.
                        if abs(d_perp) > BOX_H * 0.6:
                            same_side = (d_perp * (rnx*nx_v + rny*ny_v)) > 0
                            if same_side:
                                nx_v, ny_v = -nx_v, -ny_v
                                break

            # spread multiple links between the same repeater pair
            pair_key = (min(ra, rb), max(ra, rb))
            k = pair_idx[pair_key]
            pair_idx[pair_key] += 1
            sign = 1 if k % 2 == 0 else -1
            tier = (k // 2) + 1
            offset = sign * base_off * (0.85 + 0.40 * (tier - 1))

            # control point
            ctrl_x = (x1 + x2) / 2 + nx_v * offset
            ctrl_y = (y1 + y2) / 2 + ny_v * offset

            # Anchor the arc at the top of each qubit dot
            ax1 = x1 + nx_v * Q_RADIUS * np.sign(offset)
            ay1 = y1 + ny_v * Q_RADIUS * np.sign(offset)
            ax2 = x2 + nx_v * Q_RADIUS * np.sign(offset)
            ay2 = y2 + ny_v * Q_RADIUS * np.sign(offset)

            verts = [(ax1, ay1), (ctrl_x, ctrl_y), (ax2, ay2)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]

            colour = cmap(norm(fid))
            ax.add_patch(mpatches.PathPatch(
                MplPath(verts, codes),
                facecolor="none", edgecolor=colour,
                linewidth=1.15, zorder=3, capstyle="round"))

            # fidelity label at the Bézier midpoint (t = 0.5)
            lbl_x = 0.25 * ax1 + 0.5 * ctrl_x + 0.25 * ax2
            lbl_y = 0.25 * ay1 + 0.5 * ctrl_y + 0.25 * ay2
            ax.text(lbl_x, lbl_y, f"$F\\!=\\!{fid:.2f}$",
                    ha="center", va="center", fontsize=6,
                    color=colour, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.88, pad=0.8))

        # Repeater boxes
        for rep in self.repeaters:
            cx, cy = rep.position
            box = FancyBboxPatch(
                (cx - BOX_W / 2, cy - BOX_H / 2),
                BOX_W, BOX_H,
                boxstyle=f"round,pad={BOX_PAD:.6f}",
                facecolor=C_BOX_FACE, edgecolor=C_BOX_EDGE,
                linewidth=0.8, zorder=2)
            ax.add_patch(box)

            ax.text(cx, cy + BOX_H / 2 + Q_RADIUS * 2.5,
                    f"$R_{{{rep.rid}}}$",
                    ha="center", va="bottom", fontsize=9, zorder=6)

        # Qubit dots + labels
        for rep in self.repeaters:
            for qi in range(n_ch):
                qx, qy = qubit_xy[(rep.rid, qi)]

                if rep.locked[qi]:
                    fc, ec = C_LOCK, C_LOCK_E
                elif rep.status[qi] == QUBIT_OCCUPIED:
                    fc, ec = C_OCC, C_OCC_E
                else:
                    fc, ec = C_FREE, C_FREE_E

                ax.add_patch(Circle(
                    (qx, qy), Q_RADIUS,
                    facecolor=fc, edgecolor=ec,
                    linewidth=0.55, zorder=4))

                ax.text(qx, qy + Q_RADIUS * 2.2,
                        f"$q_{{{qi}}}$",
                        ha="center", va="bottom", fontsize=5.5,
                        color="#444444", zorder=6)


        # Legend & title
        legend_items = [
            mpatches.Patch(fc=C_FREE, ec=C_FREE_E, lw=0.6, label="Free"),
            mpatches.Patch(fc=C_OCC,  ec=C_OCC_E,  lw=0.6, label="Occupied"),
            mpatches.Patch(fc=C_LOCK, ec=C_LOCK_E,  lw=0.6, label="Locked"),
            plt.Line2D([], [], color=C_ADJ, ls="--", lw=0.7,
                       label="Adjacency"),
            plt.Line2D([], [], color=cmap(norm(0.85)), lw=1.15,
                       label="Entanglement"),
        ]
        ax.legend(handles=legend_items, loc="upper right",
                  fontsize=6.5, framealpha=0.92,
                  edgecolor="#CCCCCC", handlelength=1.4,
                  borderpad=0.6, labelspacing=0.35)

        ax.set_title(
            f"Network state at $t = {self.time_step}$,  "
            f"pending $= {len(self.pending_events)}$",
            fontsize=10, pad=8)

        fig.tight_layout(pad=0.5)

        if filepath is not None:
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

