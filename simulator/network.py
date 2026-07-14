"""
RepeaterNetwork with asynchronous classical communication delays.

Handles the inter-node logic.
"""

from __future__ import annotations
from math import sqrt
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .repeater import (
    Repeater, SwapPolicy, NO_PARTNER, QUBIT_FREE, QUBIT_OCCUPIED,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_fidelity,
)
from .snapshots import NodeState, Topology, _freeze


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


# ▄▄▄      ▄▄▄                 ▄▄
# ████▄  ▄████ ▀▀             ██  ▄█▀█▄
# ███▀████▀███ ██ ▄█▀▀▀ ▄████ ██▄█▀  ▄████  ▀▀█▄ ▄█▀█▄ ████▄
# ███  ▀▀  ███ ██ ▀███▄ ██    ██ ▀█▄ ██     ▄█▀██ ██▄█▀ ██ ▀▀
# ███      ███ ██▄ ▄▄▄█▀ ▀████ ██  ██ ▀████ ▀█▄██ ▀█▄▄▄ ██

def _sample_matched_uniform(mean, std, size, rng, lo=0.05, hi=1.0):
    """Per-repeater rates drawn from a uniform with variance ``std**2``, centred
    on ``mean`` and clipped to ``[lo, hi]``.

    A uniform on ``[mean - sqrt(3)*std, mean + sqrt(3)*std]`` has standard
    deviation exactly ``std`` (before clipping). ``std <= 0`` broadcasts the
    clipped ``mean`` and consumes NO rng draw, so the homogeneous path keeps the
    pre-inhomogeneity RNG stream bit-for-bit.

    Clipping bias: when ``mean`` sits near a bound, clipping to ``[lo, hi]``
    piles the truncated tail onto that bound, which pulls the REALIZED mean
    toward the interior and shrinks the REALIZED std below ``std`` (e.g. mean
    0.9, std 0.15 spans [0.64, 1.16], so the upper tail collapses onto 1.0).
    ``mean`` and ``std`` are therefore NOMINAL only; papers must report the
    realized per-node rate statistics measured from the drawn values, not these
    nominal parameters.
    """
    if std <= 0.0:
        return np.full(size, float(np.clip(mean, lo, hi)))
    hw = sqrt(3.0) * std
    return np.clip(rng.uniform(mean - hw, mean + hw, size=size), lo, hi)


def build_network(
    topology: str = "chain",
    *,
    n_repeaters: int = 5,
    n_ch: int = 4,
    spacing: float = 50.0,
    p_gen: float = 0.8,
    p_swap: float = 0.5,
    p_gen_std: float = 0.0,
    p_swap_std: float = 0.0,
    cutoff: int = 20,
    F0: float = 0.95,
    channel_loss: float = 0.02,
    dt_seconds: float = 1e-4,
    rng=None,
) -> RepeaterNetwork:
    """Build a RepeaterNetwork for the given topology.

    Inhomogeneity: `p_gen`/`p_swap` are the per-network MEANS; `p_gen_std`/
    `p_swap_std` spread per-repeater values via `_sample_matched_uniform`
    (std=0 -> homogeneous, no rng draw).
    """
    rng = rng if rng is not None else np.random.default_rng()
    # ponytail: chain is the only topology this project models; the `topology`
    # arg is kept as a validated constant so the ~40 call sites that pass
    # topology="chain" (and the --topology CLI flag) don't all need editing.
    if topology != "chain":
        raise ValueError(f"Unknown topology {topology!r}")
    net = build_chain(
        n_repeaters, n_ch=n_ch, spacing=spacing,
        p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
        F0=F0, channel_loss=channel_loss, dt_seconds=dt_seconds,
        distance_dep_gen=True, rng=rng)

    if p_gen_std > 0.0 or p_swap_std > 0.0:
        pg = _sample_matched_uniform(p_gen, p_gen_std, net.N, rng)
        ps = _sample_matched_uniform(p_swap, p_swap_std, net.N, rng)
        for i, rep in enumerate(net.repeaters):
            rep.p_gen, rep.p_swap = float(pg[i]), float(ps[i])
    return net


                                                                                           
"""                                                           
  ▄▄▄▄▄   ▄▄▄    ▄▄▄                                        
▄███████▄ ████▄  ███        ██                       ▄▄     
███   ███ ███▀██▄███ ▄█▀█▄ ▀██▀▀ ██   ██ ▄███▄ ████▄ ██ ▄█▀ 
███▄█▄███ ███  ▀████ ██▄█▀  ██   ██ █ ██ ██ ██ ██ ▀▀ ████   
 ▀█████▀  ███    ███ ▀█▄▄▄  ██    ██▀██  ▀███▀ ██    ██ ▀█▄ 
      ▀▀                                                    
 """

class RepeaterNetwork:
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
        "_dist_matrix",      # Matrix of distances between repeaters
        "_cutoffs"           # Array of per-repeater cutoffs for swap viability gate
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
        if self.adj.shape != (self.N, self.N):
            raise ValueError(f"Adjacency matrix shape {self.adj.shape} does not match "
                             f"number of repeaters ({self.N})")


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
        # Per-repeater cutoff vector for the swap viability gate (rid-indexed).
        self._cutoffs = np.array([r.cutoff for r in self.repeaters],
                                 dtype=np.int64)
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
        """Returns the fidelity of the generated elementary link.

        Modeling choice: ``channel_loss`` is one coefficient split across two
        physical effects on a fibre span, with different exponents:
          * generation RATE (``_gen_prob``): photon transmission ~ exp(-loss*d/2)
          * link FIDELITY (here): distance-dependent DEPOLARIZATION of the
            Werner parameter, p0 = w(F0)*exp(-loss*d).
        Damping p (not F) keeps F above the F=1/4 maximally-mixed floor at any
        distance, as a depolarizing channel must; F0*exp(-loss*d) would drive F
        below 1/4 (unphysical) past ~69 km. At loss=0 this reduces to F0 exactly.
        """
        p0 = fidelity_to_werner(self.F0) * np.exp(-self.channel_loss * self.distance(r1, r2))
        return float(werner_to_fidelity(p0))

# ▄▄▄                          ▄▄▄▄▄▄▄                                                      
# ███      ▀▀        ▄▄       ███▀▀▀▀▀                                 ██   ▀▀              
# ███      ██  ████▄ ██ ▄█▀   ███       ▄█▀█▄ ████▄ ▄█▀█▄ ████▄  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███      ██  ██ ██ ████     ███  ███▀ ██▄█▀ ██ ██ ██▄█▀ ██ ▀▀ ▄█▀██  ██   ██  ██ ██ ██ ██ 
# ████████ ██▄ ██ ██ ██ ▀█▄   ▀██████▀  ▀█▄▄▄ ██ ██ ▀█▄▄▄ ██    ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 
                                                                                           

    # -- ACTION 1: entangle (instantaneous) --------------
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

    # -- ACTION 2: swap (deferred via event queue) ------------
    def swap(self, r: int) -> Dict[str, Any]:
        """                                     
        Perform BSM at repeater r. On success, lock qubits and queue event.
        On failure, destroy both links immediately (no classical comm needed).
        """
        result = {"success": False, "new_fidelity": 0.0,"partners": None, "reason": ""} # Dict[str, Any]
        rep = self.repeaters[r]

        if not rep.can_swap():
            result["reason"] = "insufficient_qubits"; return result

        pair = rep.select_swap_pair(self._positions, self._cutoffs, rng=self.rng)

        if pair is None:
            result["reason"] = "no_valid_pair"; return result
        qa, qb = pair

        # Guard: both qubits point to the same remote repeater.
        # Swapping would try to create a self-link at the remote node.
        ra_check = int(rep.partner_repeater[qa])
        rb_check = int(rep.partner_repeater[qb])
        # Guard: an occupied but partner-less qubit (NO_PARTNER) is an orphan
        # (e.g. left by a resolution race); swapping it would queue a NO_PARTNER
        # endpoint that silently indexes repeaters[-1] at resolution. Never swap it.
        if ra_check == NO_PARTNER or rb_check == NO_PARTNER:
            result["reason"] = "orphan_qubit"; return result
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
        p_new = float(rep.werner_param[qa]) * float(rep.werner_param[qb])

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
            "gen_a": int(rep_a.generation_id[qa_r]),
            "gen_b": int(rep_b.generation_id[qb_r]),
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


    # -- ACTION 3: purify (deferred via event queue) ------------
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
        f_keep, f_sac = werner_to_fidelity(p_keep), werner_to_fidelity(p_sac)

        result["old_fidelity"] = float(f_keep)

        success = self.rng.random() <= bbpssw_success_prob(f_keep, f_sac)
        p_new = fidelity_to_werner(bbpssw_new_fidelity(f_keep, f_sac)) if success else 0.0

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
            "age_keep": int(rep1.age[q1_keep]),
            "gen_keep1": int(rep1.generation_id[q1_keep]),
            "gen_keep2": int(rep2.generation_id[q2_keep]),
            "gen_sac1": int(rep1.generation_id[q1_sac]),
            "gen_sac2": int(rep2.generation_id[q2_sac]),
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

    # -- ACTION 4: age_links (+ event resolution) -------------------

    def age_links(self, discard_expired: bool = True) -> Dict[str, Any]:
        """                              
        Advance clock: age qubits, resolve pending events, expire old links."""
        self.time_step += 1

        # 1) age all occupied qubits (including locked)
        expired_pairs: List[Tuple[int, int]] = []

        for rep in self.repeaters:
            for qi in rep.age_occupied():
                expired_pairs.append((rep.rid, int(qi)))

        # 2) resolve pending events (before expiring, so locked qubits
        #    involved in in-flight operations get resolved first)
        resolved = 0
        still_pending = []

        for ev in self.pending_events:
            ev["timer"] -= 1 # reduce timer by one step (event is closer to resolution)
            # A timer-k event resolves k age_links calls after the queuing step,
            # counting that step's own age_links as the first. delay 0 -> timer
            # 0 -> resolves in the same step (SOTA default); delay k>=1 defers to
            # k subsequent calls. Resolution fires only once timer goes negative.
            if ev["timer"] >= 0:
                still_pending.append(ev)
                continue
            else:
                resolved += 1
                if ev["type"] == "swap":
                    self._resolve_swap(ev)

                elif ev["type"] == "purify":
                    self._resolve_purify(ev)

        self.pending_events = still_pending

        # 3) expire old links (after resolving events)
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

        # [Guard] remote qubits may have been freed by expiry during the delay,
        # or reallocated to a new link (ghost link / dangling pointer check).
        a_alive = (rep_a.status[qa_r] == QUBIT_OCCUPIED and
                   int(rep_a.generation_id[qa_r]) == ev["gen_a"])
        b_alive = (rep_b.status[qb_r] == QUBIT_OCCUPIED and
                   int(rep_b.generation_id[qb_r]) == ev["gen_b"])

        if not (a_alive and b_alive):
            # At least one side expired or was reallocated — clean up the survivor
            if a_alive:
                rep_a.free_qubit(qa_r)
            if b_alive:
                rep_b.free_qubit(qb_r)
            return

        # [Guard] Both remote endpoints have collapsed onto the SAME node
        # (ra == rb): a deferred swap cannot form an inter-node link between two
        # qubits on one repeater. Immediate swap() refuses this via the
        # same_partner guard the deferred path must too (else set_link raises
        # "Attempting to generate inter-node entanglement"). Drop the void swap
        # and free both locked survivors, mirroring the expiry cleanup above.
        if int(ra) == int(rb):
            rep_a.free_qubit(qa_r)
            rep_b.free_qubit(qb_r)
            return

        ec = min(rep_a.cutoff, rep_b.cutoff)
        summed_age = int(rep_a.age[qa_r]) + int(rep_b.age[qb_r])
        # [Guard] Born-dead link: the fused state's inherited age already
        # reached the cutoff (in-flight CC accrual, or any path the decision
        # gate could not see). Creating it would leak a link past the fidelity
        # floor the cutoff exists to guarantee (expiry phase 3 only sweeps the
        # pre-resolution list, so it would survive to this step's delivery
        # check). Discard instead: free both remote endpoints, no link.
        if summed_age >= ec:
            rep_a.free_qubit(qa_r)
            rep_b.free_qubit(qb_r)
            return
        # Sum-ages history. The resolved value must be exactly the product of
        # the two remote links' already-decohered Werner values at resolution
        # (w_A*w_B); set_link must not re-apply decay on top (that would
        # double-count the pre-swap decoherence). Storing the baseline product
        # p0_A*p0_B with age = age_A + age_B reproduces w_A*w_B exactly and lets
        # future decay continue from it: for a shared cutoff tau,
        # p0_A*p0_B*exp(-(age_A+age_B)/tau) = (p0_A*e^{-age_A/tau})(p0_B*e^{-age_B/tau}).
        # Exact for the homogeneous per-link cutoffs used everywhere today; an
        # approximation only if the two links carried different cutoffs.
        base = float(rep_a.initial_werner[qa_r]) * float(rep_b.initial_werner[qb_r])
        rep_a.set_link(qa_r, rb, qb_r, base,
                       link_age=summed_age, effective_cutoff=ec)
        rep_b.set_link(qb_r, ra, qa_r, base,
                       link_age=summed_age, effective_cutoff=ec)
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
            # Destroy sacrifice pair — check both sides' generation-IDs before
            # calling _break_link, which follows partner pointers that may be stale.
            s1_valid = (rep1.status[q1_sac] == QUBIT_OCCUPIED and
                        int(rep1.generation_id[q1_sac]) == ev["gen_sac1"])
            s2_valid = (rep2.status[q2_sac] == QUBIT_OCCUPIED and
                        int(rep2.generation_id[q2_sac]) == ev["gen_sac2"])
            if s1_valid and s2_valid:
                self._break_link(r1, q1_sac)   # frees both sides via partner ptr
            else:
                # At least one side expired/reallocated — free survivors individually.
                if s1_valid:
                    rep1.free_qubit(q1_sac)
                if s2_valid:
                    rep2.free_qubit(q2_sac)
                # Release only OUR lingering lock. A slot whose generation no
                # longer matches this event was reallocated and re-locked by a
                # newer in-flight op — unlocking it there would orphan that op's
                # qubit (occupied + unlocked + NO_PARTNER), which then gets swapped
                # and queues a NO_PARTNER(-1) endpoint. Leave such locks untouched.
                if int(rep1.generation_id[q1_sac]) == ev["gen_sac1"] and rep1.locked[q1_sac]:
                    rep1.unlock_qubit(q1_sac)
                if int(rep2.generation_id[q2_sac]) == ev["gen_sac2"] and rep2.locked[q2_sac]:
                    rep2.unlock_qubit(q2_sac)

            # Guard: kept qubits may have been freed by expiry or reallocated
            k1_alive = (rep1.status[q1_keep] == QUBIT_OCCUPIED and
                        int(rep1.generation_id[q1_keep]) == ev["gen_keep1"])
            k2_alive = (rep2.status[q2_keep] == QUBIT_OCCUPIED and
                        int(rep2.generation_id[q2_keep]) == ev["gen_keep2"])
            if not (k1_alive and k2_alive):
                if k1_alive: self._break_link(r1, q1_keep)
                if k2_alive: self._break_link(r2, q2_keep)
                return

            ec = min(rep1.cutoff, rep2.cutoff)
            safe_ec = max(int(ec), 1)
            # Eq.(4) age semantics (arXiv 2401.13168): represent the purified
            # fidelity as an equivalent age on a fresh p0=1 baseline,
            # m' = ceil(-tau*ln(p_new)), plus ticks accrued since the decision
            # (CC delay + same-tick aging). Age is then an exact fidelity
            # proxy for every link and purification extends remaining
            # lifetime. Replaces the old sum-of-endpoint-ages bookkeeping
            # (doubled expiry clock) and its baseline>1 back-solve.
            p_new = float(ev["p_new"])
            accrued = max(int(rep1.age[q1_keep]) - int(ev["age_keep"]), 0)
            if p_new >= 1.0:
                m_equiv = 0
            elif p_new <= 0.0:
                # already at/below the depolarizing floor (F=1/4, p=0):
                # -ln(0) is undefined/inf, so force the discard branch below
                # rather than let ceil() overflow on int conversion.
                m_equiv = int(ec)
            else:
                m_equiv = int(np.ceil(-safe_ec * np.log(p_new)))
            new_age = m_equiv + accrued
            if new_age >= ec:
                # purified state already below the cutoff fidelity floor:
                # discard rather than create a link expiry can never police
                self._break_link(r1, q1_keep)
                return
            rep1.set_link(q1_keep, r2, q2_keep, 1.0,
                          link_age=new_age, effective_cutoff=ec)
            rep2.set_link(q2_keep, r1, q1_keep, 1.0,
                          link_age=new_age, effective_cutoff=ec)
            rep1.unlock_qubit(q1_keep)
            rep2.unlock_qubit(q2_keep)
        else:
            # Failure: destroy all four qubits, guarding each with generation-ID
            # to avoid corrupting a new link that reused the same qubit slot.
            for rep, q, gen_key in (
                (rep1, q1_sac,  "gen_sac1"),
                (rep2, q2_sac,  "gen_sac2"),
                (rep1, q1_keep, "gen_keep1"),
                (rep2, q2_keep, "gen_keep2"),
            ):
                if (rep.status[q] == QUBIT_OCCUPIED and
                        int(rep.generation_id[q]) == ev[gen_key]):
                    # Partner pointer still valid — use _break_link to free both sides.
                    # Determine which repeater index owns this qubit.
                    rid = rep.rid
                    self._break_link(rid, q)
                else:
                    # Slot no longer holds our link. Clear a lingering lock ONLY
                    # if the slot is still ours (e.g. freed by expiry, generation
                    # unchanged). If it was reallocated (generation differs) the
                    # lock now belongs to a newer in-flight op — leave it, else we
                    # orphan that op's qubit.
                    if int(rep.generation_id[q]) == ev[gen_key] and rep.locked[q]:
                        rep.unlock_qubit(q)

                                                   
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

    # ---- immutable read snapshots (engine-agnostic, F-domain) -------------
    def topology(self) -> Topology:
        return Topology(
            N=self.N,
            adjacency=_freeze(self.adj.copy()),
            positions=_freeze(self._positions.copy()),
        )

    def node_state(self, node: int) -> NodeState:
        rep = self.repeaters[node]
        occupied = (rep.status == QUBIT_OCCUPIED)
        fid = werner_to_fidelity(rep.werner_param).astype(np.float64)
        fid = np.where(occupied, fid, 0.0)
        return NodeState(
            node_id=node,
            n_ch=rep.n_ch,
            p_gen=float(rep.p_gen),
            p_swap=float(rep.p_swap),
            occupied=_freeze(occupied),
            locked=_freeze(rep.locked),
            partner_node=_freeze(rep.partner_repeater),
            partner_qubit=_freeze(rep.partner_qubit),
            fidelity=_freeze(fid),
            age=_freeze(rep.age.astype(np.int32)),
            link_cutoff=_freeze(rep.link_cutoff.astype(np.int32)),
        )


                                                                             
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


#  ▄▄▄▄▄▄▄                         ▄▄                  
# ███▀▀███▄                        ██             ▀▀   
# ███▄▄███▀ ▄█▀█▄ ████▄  ▄███▄ ▄███▄ ▄█▀█▄ ████▄ ██  ████▄ ▄████ 
# ███▀▀██▄  ██▄█▀ ██ ██  ██ ██ ██ ██ ██▄█▀ ██ ▀▀ ██  ██ ██ ██ ██ 
# ███  ▀███ ▀█▄▄▄ ██ ██  ▀███▀ ▀███▀ ▀█▄▄▄ ██    ██▄ ██ ██ ▀████ 
#                                                               ██ 
#                                                             ▀▀▀  

    def render(self, filepath=None, figsize=None, dpi=250,
            source_dest: tuple | None = None):
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
        source_dest : tuple[int, int] or None
            A ``(source, dest)`` pair of repeater indices.  When provided,
            both repeaters receive a soft yellow halo drawn behind their
            box to distinguish them visually from interior nodes.

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
        from collections import defaultdict
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

        N    = self.N
        n_ch = self.repeaters[0].n_ch
        pos  = self._positions.copy()

        G_adj  = nx.from_numpy_array(self.adj)
        pos_nx = {i: tuple(pos[i]) for i in range(N)}

        # -- Geometry constants ------------------------------
        if N > 1:
            nonzero = self._dist_matrix[self._dist_matrix > 0]
            scale   = float(nonzero.min()) if len(nonzero) else 1.0
        else:
            scale = 1.0

        BOX_W      = scale * 0.55
        BOX_H      = scale * 0.30
        BOX_PAD    = scale * 0.012
        Q_RADIUS   = scale * 0.032
        C_BOX_FACE = "#F2F2F2";  C_BOX_EDGE = "#2B2B2B"
        C_ADJ      = "#AAAAAA"
        C_FREE     = "#FFFFFF";  C_FREE_E   = "#999999"
        C_OCC      = "#4477AA";  C_OCC_E    = "#224488"
        C_LOCK     = "#EE6677";  C_LOCK_E   = "#CC3311"

        N_LAYERS    = 10
        GLOW_SPREAD = BOX_W * 0.7
        GLOW_COLOR  = "#FFD700"
        GLOW_ALPHA  = 0.18

        # -- Qubit dot positions ------------------------------------------
        qubit_xy = {}
        for rep in self.repeaters:
            cx, cy = rep.position
            for qi in range(n_ch):
                qubit_xy[(rep.rid, qi)] = (
                    cx - BOX_W / 2 + BOX_W * (qi + 1) / (n_ch + 1),
                    cy - BOX_H * 0.10,
                )

        # -- PASS 1: compute all arc geometry ------------------------------
        # A quadratic Bézier is always within its control-point convex hull,
        # so max(P0, ctrl, P2) gives a tight bound on the arc extent.
        # We collect every control point here so we can size the figure
        # correctly *before* drawing anything.
        links    = self.get_all_links()
        pair_idx = defaultdict(int)
        arc_data = []   # list of dicts consumed by the drawing pass

        ctrl_pts_x = []   # accumulate control-point coordinates for bbox
        ctrl_pts_y = []

        for lk in links:
            ra, qa, rb, qb = int(lk[0]), int(lk[1]), int(lk[2]), int(lk[3])
            fid = float(lk[4])

            x1, y1 = qubit_xy[(ra, qa)]
            x2, y2 = qubit_xy[(rb, qb)]
            dx, dy  = x2 - x1, y2 - y1
            length  = np.hypot(dx, dy)
            if length < 1e-12:
                continue

            nx_v, ny_v = -dy / length, dx / length

            clearance    = BOX_H * 0.45
            min_ctrl_off = BOX_H + 2 * clearance   # ≈ 1.9 × BOX_H — enough to clear any box
            # Arc height scales mildly with link length so short and long links look
            # proportional, but is hard-capped so long-range links never escape the
            # figure.  hop_factor was removed — arcs need not be taller just because
            # they span more hops; the cap alone keeps everything visible.
            base_off = min(max(min_ctrl_off, 0.10 * length), BOX_H * 5)

            if ny_v < 0 or (abs(ny_v) < 1e-9 and nx_v < 0):
                nx_v, ny_v = -nx_v, -ny_v

            rx1, ry1 = pos[ra];  rx2, ry2 = pos[rb]
            rdx, rdy  = rx2 - rx1, ry2 - ry1
            rl = np.hypot(rdx, rdy)
            if rl > 1e-12:
                rnx, rny = -rdy / rl, rdx / rl
                for k in range(N):
                    if k == ra or k == rb:
                        continue
                    pk     = pos[k]
                    t_proj = ((pk[0]-rx1)*rdx + (pk[1]-ry1)*rdy) / (rl**2)
                    if 0.05 < t_proj < 0.95:
                        d_perp = (pk[0]-rx1)*rnx + (pk[1]-ry1)*rny
                        if abs(d_perp) > BOX_H * 0.6:
                            if (d_perp * (rnx*nx_v + rny*ny_v)) > 0:
                                nx_v, ny_v = -nx_v, -ny_v
                                break

            pair_key = (min(ra, rb), max(ra, rb))
            k    = pair_idx[pair_key];  pair_idx[pair_key] += 1
            sign = 1 if k % 2 == 0 else -1
            tier = (k // 2) + 1
            offset = sign * base_off * (0.85 + 0.40 * (tier - 1))

            ctrl_x = (x1 + x2) / 2 + nx_v * offset
            ctrl_y = (y1 + y2) / 2 + ny_v * offset

            # anchor points on the qubit dot surface
            ax1 = x1 + nx_v * Q_RADIUS * np.sign(offset)
            ay1 = y1 + ny_v * Q_RADIUS * np.sign(offset)
            ax2 = x2 + nx_v * Q_RADIUS * np.sign(offset)
            ay2 = y2 + ny_v * Q_RADIUS * np.sign(offset)

            ctrl_pts_x.extend([ax1, ctrl_x, ax2])
            ctrl_pts_y.extend([ay1, ctrl_y, ay2])

            arc_data.append(dict(
                ax1=ax1, ay1=ay1, ax2=ax2, ay2=ay2,
                ctrl_x=ctrl_x, ctrl_y=ctrl_y,
                fid=fid,
            ))

        # -- Figure extent — driven by repeater positions AND arc control pts --
        all_x = list(pos[:, 0]) + ctrl_pts_x
        all_y = list(pos[:, 1]) + ctrl_pts_y

        margin_x = scale * 0.85
        margin_y = scale * 0.55   # smaller fixed padding; arcs now set the real ceiling

        x_lo = float(np.min(all_x)) - margin_x
        x_hi = float(np.max(all_x)) + margin_x
        y_lo = float(np.min(all_y)) - margin_y
        y_hi = float(np.max(all_y)) + margin_y

        if figsize is None:
            x_span  = max(x_hi - x_lo, 1e-6)
            y_span  = max(y_hi - y_lo, 1e-6)
            aspect  = x_span / y_span
            fig_h   = max(3.5, min(14.0, 2.8 + N * 0.18))
            fig_w   = max(fig_h * aspect, fig_h)
            figsize = (fig_w, fig_h)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.axis("off")

        # -- Yellow glow halos ------------
        if source_dest is not None:
            for rid in set(source_dest):
                cx, cy = self.repeaters[rid].position
                for layer in range(N_LAYERS, 0, -1):
                    t      = (N_LAYERS - layer) / (N_LAYERS - 1)
                    margin = GLOW_SPREAD * (layer / N_LAYERS)
                    alpha  = GLOW_ALPHA * (1.0 - t)
                    w, h   = BOX_W + 2*margin, BOX_H + 2*margin
                    pad    = BOX_PAD + margin * 0.25
                    ax.add_patch(FancyBboxPatch(
                        (cx - w/2, cy - h/2), w, h,
                        boxstyle=f"round,pad={pad:.6f}",
                        facecolor=GLOW_COLOR, edgecolor="none",
                        alpha=alpha, zorder=1, linewidth=0))

        # -- Adjacency edges (dashed) -------------------------------
        nx.draw_networkx_edges(
            G_adj, pos_nx, ax=ax,
            style="dashed", width=0.7, edge_color=C_ADJ, alpha=0.75)

        # -- PASS 2: draw entanglement arcs --------------------------
        cmap     = plt.cm.RdYlGn
        norm_col = plt.Normalize(vmin=0.25, vmax=1.0)

        for arc in arc_data:
            ax1, ay1   = arc["ax1"], arc["ay1"]
            ax2, ay2   = arc["ax2"], arc["ay2"]
            ctrl_x     = arc["ctrl_x"]
            ctrl_y     = arc["ctrl_y"]
            fid        = arc["fid"]
            colour     = cmap(norm_col(fid))

            verts = [(ax1, ay1), (ctrl_x, ctrl_y), (ax2, ay2)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            ax.add_patch(mpatches.PathPatch(
                MplPath(verts, codes),
                facecolor="none", edgecolor=colour,
                linewidth=1.15, zorder=3, capstyle="round"))

            # fidelity label at Bézier midpoint t = 0.5
            lbl_x = 0.25*ax1 + 0.5*ctrl_x + 0.25*ax2
            lbl_y = 0.25*ay1 + 0.5*ctrl_y + 0.25*ay2
            ax.text(lbl_x, lbl_y, f"$F\\!=\\!{fid:.2f}$",
                    ha="center", va="center", fontsize=6,
                    color=colour, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.88, pad=0.8))

        # -- Repeater boxes ---------------------------------
        for rep in self.repeaters:
            cx, cy = rep.position
            is_hl    = source_dest is not None and rep.rid in source_dest
            edge_col = "#B8860B" if is_hl else C_BOX_EDGE
            lw       = 1.4       if is_hl else 0.8
            ax.add_patch(FancyBboxPatch(
                (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                boxstyle=f"round,pad={BOX_PAD:.6f}",
                facecolor=C_BOX_FACE, edgecolor=edge_col,
                linewidth=lw, zorder=2))
            ax.text(cx, cy + BOX_H/2 + Q_RADIUS*2.5, f"$R_{{{rep.rid}}}$",
                    ha="center", va="bottom", fontsize=9, zorder=6)

        #  Qubit dots + labels ---------------------
        for rep in self.repeaters:
            for qi in range(n_ch):
                qx, qy = qubit_xy[(rep.rid, qi)]
                if rep.locked[qi]:
                    fc, ec = C_LOCK, C_LOCK_E
                elif rep.status[qi] == QUBIT_OCCUPIED:
                    fc, ec = C_OCC, C_OCC_E
                else:
                    fc, ec = C_FREE, C_FREE_E
                ax.add_patch(Circle((qx, qy), Q_RADIUS,
                                    facecolor=fc, edgecolor=ec,
                                    linewidth=0.55, zorder=4))
                ax.text(qx, qy + Q_RADIUS*2.2, f"$q_{{{qi}}}$",
                        ha="center", va="bottom", fontsize=5.5,
                        color="#444444", zorder=6)

        # -- Legend & title -----------------------------------------
        legend_items = [
            mpatches.Patch(fc=C_FREE, ec=C_FREE_E, lw=0.6, label="Free"),
            mpatches.Patch(fc=C_OCC,  ec=C_OCC_E,  lw=0.6, label="Occupied"),
            mpatches.Patch(fc=C_LOCK, ec=C_LOCK_E, lw=0.6, label="Locked"),
            plt.Line2D([], [], color=C_ADJ, ls="--", lw=0.7, label="Adjacency"),
            plt.Line2D([], [], color=cmap(norm_col(0.85)), lw=1.15,
                    label="Entanglement"),
        ]
        if source_dest is not None:
            legend_items.append(
                mpatches.Patch(fc=GLOW_COLOR, ec="#B8860B", lw=0.8,
                            alpha=0.7, label="Src / Dst"))
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

