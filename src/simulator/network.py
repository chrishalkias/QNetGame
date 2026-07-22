"""
--------------------------------------------------------------------------------
RepeaterNetwork: inter-node logic (entangle / swap / purify / age_links).

Swap and purify decide against the frozen start-of-tick state and apply at the
end of the tick (the synchronous-tick barrier); see age_links.
--------------------------------------------------------------------------------
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
        F0=F0, channel_loss=channel_loss,
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
        "pending_events",   # Within-step scratch list: swap/purify apply at end of age_links
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

        # Swap/purify decide against the frozen start-of-tick state, lock their
        # remote qubits, and record the outcome here; age_links applies the whole
        # list at end of tick. This deferral is the synchronous-tick barrier: it
        # stops swaps cascading within a tick, so long links build over rounds.
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

        # append event to the queue (applied at end of this tick's age_links)
        self.pending_events.append({
            "type": "swap",
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

        self.pending_events.append({
            "type": "purify", "success": success,
            "r1": r1, "r2": r2,
            "q1_sac": q1_sac, "q2_sac": q2_sac,
            "q1_keep": q1_keep, "q2_keep": q2_keep,
            "p_new": p_new,
            "age_keep": int(rep1.age[q1_keep]),
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

        # 2) apply every event queued this tick (before expiring, so the aged
        #    remote qubits are consumed into the fused/purified link first). All
        #    events resolve in the same age_links call that this step queued them.
        resolved = len(self.pending_events)
        for ev in self.pending_events:
            if ev["type"] == "swap":
                self._resolve_swap(ev)
            elif ev["type"] == "purify":
                self._resolve_purify(ev)

        self.pending_events = []

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
        """Resolve a swap queued earlier this tick: rewrite the two remote qubits
        to point at each other. Local qubits were freed at BSM time.

        Same-tick resolution means the remote qubits are always OCCUPIED, distinct
        nodes, and viable here: they were locked at decision (so no other node
        touched them), the swap() same_partner guard rules out ra==rb, and the
        select_swap_pair gate age_a+age_b+2 < ec keeps the fused age below cutoff
        even after this tick's aging. No in-flight guards are needed without CC.
        """
        ra, qa_r, rb, qb_r = ev["ra"], ev["qa_r"], ev["rb"], ev["qb_r"]
        rep_a, rep_b = self.repeaters[ra], self.repeaters[rb]

        ec = min(rep_a.cutoff, rep_b.cutoff)
        summed_age = int(rep_a.age[qa_r]) + int(rep_b.age[qb_r])
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

        # Same-tick resolution: all four qubits were locked at decision and are
        # still OCCUPIED and ours here (no CC flight window to expire/reallocate
        # them), so no generation-ID guards are needed.
        if ev["success"]:
            # Destroy the sacrifice pair (frees both sides via partner pointer).
            self._break_link(r1, q1_sac)

            ec = min(rep1.cutoff, rep2.cutoff)
            safe_ec = max(int(ec), 1)
            # Eq.(4) age semantics (arXiv 2401.13168): represent the purified
            # fidelity as an equivalent age on a fresh p0=1 baseline,
            # m' = ceil(-tau*ln(p_new)), plus the one tick accrued since the
            # decision (same-tick aging). Age is then an exact fidelity
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
            # Failure: destroy both pairs. Each _break_link frees a qubit and its
            # partner, so two calls clear all four (free_qubit also clears locks).
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

