"""
--------------------------------------------------------------------------------
RepeaterNetwork: inter-node logic (entangle / swap / purify / age_links).

Swap and purify apply their outcome immediately (sequential-sweep model, no
end-of-tick deferral); age_links only ages and expires links.
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from math import sqrt
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .repeater import (
    Repeater, SwapPolicy, NO_PARTNER, QUBIT_FREE, QUBIT_OCCUPIED,
    LEFT, RIGHT,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_fidelity,
)

def build_chain(n_repeaters, 
                n_ch=4, 
                spacing=50.0,
                swap_policy=SwapPolicy.FARTHEST,
                p_gen=0.8, 
                p_swap=0.5, 
                cutoff=20, 
                **kw
                )-> RepeaterNetwork:
    """Creates a chain topology network.

    Each node's ports face its neighbours: the leftmost node (rid 0) is
    RIGHT-only, the rightmost (rid n-1) is LEFT-only, interior nodes carry both
    ports of width n_ch (2*n_ch qubits total).
    """
    reps = [
            Repeater(rid=i,
                     n_ch=n_ch,
                     n_left=0 if i == 0 else n_ch,
                     n_right=0 if i == n_repeaters - 1 else n_ch,
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

        # Each endpoint uses the port facing the other: the lower-index node
        # generates on its RIGHT port, the higher-index node on its LEFT port.
        side1 = RIGHT if r2 > r1 else LEFT
        side2 = LEFT if r2 > r1 else RIGHT

        if not rep1.has_free_qubit(side1):
            result["reason"] = "no_free_qubit_r1"; return result

        if not rep2.has_free_qubit(side2):
            result["reason"] = "no_free_qubit_r2"; return result

        if self.rng.random() > self._gen_prob(r1, r2):
            result["reason"] = "generation_failed"; return result

        # allocate one qubit on the facing port of each repeater
        q1, q2 = rep1.allocate_qubit(side1), rep2.allocate_qubit(side2)
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

    # -- ACTION 2: swap (applies immediately) ------------
    def swap(self, r: int) -> Dict[str, Any]:
        """
        Perform BSM at repeater r. On success, the fused remote link exists
        immediately (sequential-sweep model, no deferral). On failure, destroy
        both links immediately (no classical comm needed).
        """
        result = {"success": False, "new_fidelity": 0.0,"partners": None, "reason": ""} # Dict[str, Any]
        rep = self.repeaters[r]

        # Structural precheck only. `can_swap` would ALSO apply the viability
        # gate, which select_swap_pair applies two lines down, and would then
        # report a born-dead pair as "insufficient_qubits" instead of
        # "no_valid_pair".
        if not rep.has_link_each_side():
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

        # -- Success: BSM consumes local qubits now; create the fused remote
        #    link IMMEDIATELY (sequential-sweep model, no deferral).
        ra, qa_r = int(rep.partner_repeater[qa]), int(rep.partner_qubit[qa])
        rb, qb_r = int(rep.partner_repeater[qb]), int(rep.partner_qubit[qb])
        rep.free_qubit(qa)
        rep.free_qubit(qb)

        rep_a, rep_b = self.repeaters[ra], self.repeaters[rb]
        ec = min(rep_a.cutoff, rep_b.cutoff)
        summed_age = int(rep_a.age[qa_r]) + int(rep_b.age[qb_r])
        base = float(rep_a.initial_werner[qa_r]) * float(rep_b.initial_werner[qb_r])
        rep_a.set_link(qa_r, rb, qb_r, base, link_age=summed_age, effective_cutoff=ec)
        rep_b.set_link(qb_r, ra, qa_r, base, link_age=summed_age, effective_cutoff=ec)

        result.update(success=True,
                      new_fidelity=float(werner_to_fidelity(base * np.exp(-summed_age / max(ec,1)))),
                      partners=(ra, rb), reason="ok")
        return result

# ▄▄▄▄▄▄▄                    ▄▄                                       
# ███▀▀███▄             ▀▀  ██  ▀▀               ██   ▀▀              
# ███▄▄███▀ ██ ██ ████▄ ██ ▀██▀ ██  ▄████  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄ 
# ███▀▀▀▀   ██ ██ ██ ▀▀ ██  ██  ██  ██    ▄█▀██  ██   ██  ██ ██ ██ ██ 
# ███       ▀██▀█ ██    ██▄ ██  ██▄ ▀████ ▀█▄██  ██   ██▄ ▀███▀ ██ ██ 


    # -- ACTION 3: purify (applies immediately) ------------
    def purify(self, r1: int, r2: int) -> Dict[str, Any]:
        """BBPSSW purification cascade over ALL links shared with r2.

        Sorted-adjacent multiplexed distillation (like in arXiv 2401.13168): sort the
        shared links by fidelity and fold up the list carrying a running
        survivor, purifying consecutive (closest-fidelity) links until one
        survives or none do. Beneficial-purify guard (Fig. 11 there): a pair is
        only put through the stochastic BBPSSW when the output beats the
        better input, F_new(F1,F2) > max(F1,F2). Otherwise we skip the coin flip
        and just keep the stronger link (purification must strictly beat
        discarding the weak link as Jan said). BBPSSW failure destroys both inputs.

        The whole cascade is decided AND applied now against the current state
        (all RNG drawn here); the outcome exists immediately, before any
        age_links call (sequential-sweep model, no deferral).
        """
        result = {"success": False, "old_fidelity": 0.0, "new_fidelity": 0.0, "reason": ""}
        rep1, rep2 = self.repeaters[r1], self.repeaters[r2]
        q1s = rep1.qubits_to(r2)

        if len(q1s) < 2:
            result["reason"] = "insufficient_shared_pairs"
            return result

        # sort shared links ascending by Werner p (== ascending fidelity)
        order = np.argsort(rep1.werner_param[q1s])
        q_sorted = [int(q1s[i]) for i in order]
        result["old_fidelity"] = float(werner_to_fidelity(rep1.werner_param[q_sorted[-1]]))

        # --- sequential-accumulate cascade + beneficial guard -----------
        keep = q_sorted[0]
        keep_p = float(rep1.werner_param[keep])
        keep_purified = False
        sacrificed: List[int] = []
        i = 1
        while i < len(q_sorted):
            L = q_sorted[i]
            i += 1
            pL = float(rep1.werner_param[L])
            F1 = float(werner_to_fidelity(keep_p))
            F2 = float(werner_to_fidelity(pL))
            f_new = float(bbpssw_new_fidelity(F1, F2))
            if f_new > max(F1, F2):
                # beneficial region -> attempt stochastic BBPSSW
                if self.rng.random() <= bbpssw_success_prob(F1, F2):
                    keep_p = float(fidelity_to_werner(f_new))
                    keep_purified = True
                    sacrificed.append(L)
                else:
                    # failure destroys both inputs; reseed from the next link
                    sacrificed.append(keep); sacrificed.append(L)
                    if i < len(q_sorted):
                        keep = q_sorted[i]; keep_p = float(rep1.werner_param[keep])
                        keep_purified = False; i += 1
                    else:
                        keep = None
            else:
                # not beneficial -> keep the stronger link, discard the weaker
                if F1 >= F2:
                    sacrificed.append(L)
                else:
                    sacrificed.append(keep)
                    keep = L; keep_p = pL; keep_purified = False

        # -- Apply immediately: destroy every sacrificed pair, then either
        #    leave a merely-kept survivor as-is or re-register a purified one
        #    via the Eq.(4) age proxy (arXiv 2401.13168).
        for q in sacrificed:
            self._break_link(r1, int(q))

        if keep is None:
            result.update(success=False, new_fidelity=0.0, reason="cascade_failed")
            return result
        keep = int(keep)
        q2_keep = int(rep1.partner_qubit[keep])

        if not keep_purified:
            # Survivor was only kept (weaker links discarded), never purified:
            # its existing (initial_werner, age) registration is still correct.
            result.update(success=True,
                          new_fidelity=float(werner_to_fidelity(keep_p)),
                          reason="ok")
            return result

        # Purified survivor: Eq.(4) age semantics (arXiv 2401.13168). Represent
        # the purified fidelity as an equivalent age on a fresh p0=1 baseline,
        # m' = ceil(-tau*ln(p_new)). Applied immediately, so zero ticks have
        # accrued since the decision (the old accrued-since-decision term is
        # therefore always 0 now).
        ec = min(rep1.cutoff, rep2.cutoff)
        safe_ec = max(int(ec), 1)
        p_new = float(keep_p)
        if p_new >= 1.0:
            m_equiv = 0
        elif p_new <= 0.0:
            # at/below the depolarizing floor (F=1/4, p=0): -ln(0) is undefined,
            # so force the discard branch rather than overflow ceil() -> int.
            m_equiv = int(ec)
        else:
            m_equiv = int(np.ceil(-safe_ec * np.log(p_new)))
        new_age = m_equiv
        if new_age >= ec:
            # purified state already below the cutoff fidelity floor: discard
            # rather than create a link expiry can never police
            self._break_link(r1, keep)
            result.update(success=False, new_fidelity=0.0, reason="purify_discarded_over_cutoff")
            return result
        rep1.set_link(keep, r2, q2_keep, 1.0,
                      link_age=new_age, effective_cutoff=ec)
        rep2.set_link(q2_keep, r1, keep, 1.0,
                      link_age=new_age, effective_cutoff=ec)
        result.update(success=True,
                      new_fidelity=float(werner_to_fidelity(p_new)),
                      reason="ok")
        return result


#   ▄▄▄▄                         
# ▄██▀▀██▄       ▀▀              
# ███  ███ ▄████ ██  ████▄ ▄████ 
# ███▀▀███ ██ ██ ██  ██ ██ ██ ██ 
# ███  ███ ▀████ ██▄ ██ ██ ▀████ 
#             ██              ██ 
#           ▀▀▀             ▀▀▀  

    # -- ACTION 4: age_links -------------------

    def age_links(self, discard_expired: bool = True) -> Dict[str, Any]:
        """
        Advance clock: age qubits, expire old links."""
        self.time_step += 1

        # 1) age all occupied qubits
        expired_pairs: List[Tuple[int, int]] = []

        for rep in self.repeaters:
            for qi in rep.age_occupied():
                expired_pairs.append((rep.rid, int(qi)))

        # 2) expire old links
        n_destroyed = 0
        if discard_expired:
            for rid, qidx in expired_pairs:
                rep = self.repeaters[rid]
                if rep.status[qidx] == QUBIT_OCCUPIED:
                    self._break_link(rid, qidx)
                    n_destroyed += 1

        return {"expired_count": n_destroyed,
                "over_cutoff_count": len(expired_pairs),
                "time_step": self.time_step}


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

    # ---- read handle -------------------------------------------------------
    def node(self, node: int) -> Repeater:
        """The LIVE Repeater for `node`. READ-ONLY BY CONVENTION: consumers
        (the RL env, the probes) must never mutate through this handle, and the
        arrays it exposes are the engine's own, so they keep changing under a
        caller that holds the object across an engine mutation.

        Returning the object rather than a frozen copy keeps 2N array copies per
        micro-step out of the hot loop (snapshots.py deleted 2026-07-26). The old
        node_state() froze six arrays per call and was the single largest cost in
        the env: over 30 episodes at N=10, n_ch=3 it took 48206 calls and 0.95 s
        of a 2.47 s profile, 38% of engine runtime. This handle costs 0.006 s
        over the same run, and the benchmark went 1.41 s to 0.84 s wall.

        Field names are the engine's: `status` / `partner_repeater` /
        `werner_param`, not `occupied` / `partner_node` / `fidelity`. Fidelity is
        no longer masked to 0.0 on free qubits, so callers must gate on
        `status == QUBIT_OCCUPIED` themselves before reading a link value.
        """
        return self.repeaters[node]


                             
# ▄▄▄      ▄▄▄                 
# ████▄  ▄████ ▀▀              
# ███▀████▀███ ██  ▄█▀▀▀ ▄████ 
# ███  ▀▀  ███ ██  ▀███▄ ██    
# ███      ███ ██▄ ▄▄▄█▀ ▀████ 
                             
                             
    def reset(self):
        self.time_step = 0
        for rep in self.repeaters:
            rep.reset()

    def __repr__(self) -> str:
        """Verbose summary of the state of the network (connections without idx)"""
        lines = [f"RepeaterNetwork N={self.N} t={self.time_step}"]
        for rep in self.repeaters:
            lines.append(f"  {rep}")
        lk = self.get_all_links()
        lines.append(f"  Active links: {len(lk)}")
        for l in lk:
            lines.append(f"    R{int(l[0])}:q{int(l[1])}<->R{int(l[2])}:q{int(l[3])} "
                         f"F={l[4]:.4f} age={int(l[5])}")
        return "\n".join(lines)

