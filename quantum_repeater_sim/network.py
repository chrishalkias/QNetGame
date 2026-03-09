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


class RepeaterNetwork:

    def __init__(self, 
                 repeaters: list[Repeater], 
                 adjacency: np.ndarray,
                 channel_loss: float = 0.02, 
                 F0: float = 0.98,
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
        #fidelity damping coefficient for distance dependent p_e
        self.channel_loss = channel_loss
        #Fidelity at 0 distance
        self.F0 = F0
        self.distance_dep_gen = distance_dep_gen
        # can specify rng of choice
        self.rng = rng if rng is not None else np.random.default_rng()
        self.time_step: int = 0

        # ---Classical delay parameters
        self.dt_seconds = dt_seconds #sim to physical time converter
        self.c_fiber: float = 200_000.0  # km/s
        self.pending_events: List[dict] = []

        # ---Cached geometry
        self._positions = np.stack([r.position for r in self.repeaters], axis=0)
        #position differences
        diff = self._positions[:, None, :] - self._positions[None, :, :]
        self._dist_matrix = np.linalg.norm(diff, axis=-1)

    # ── helpers ───────────────────────────────────────────────────

    def distance(self, r1: int, r2: int) -> float:
        return float(self._dist_matrix[r1, r2])

    def _classical_delay_steps(self, d_km: float) -> int:
        """Discrete steps for a classical signal to travel d_km through fibre."""
        if d_km <= 0.0 or self.dt_seconds <= 0.0:
            return 0
        return int(np.ceil(d_km / (self.c_fiber * self.dt_seconds)))

    def _gen_prob(self, r1: int, r2: int) -> float:
        """Returns the effective p_e between two repeaters (can be distance dependent)"""
        p_avg = 0.5*(self.repeaters[r1].p_gen + self.repeaters[r2].p_gen)

        if self.distance_dep_gen:
            return p_avg * np.exp(-self.channel_loss * self.distance(r1, r2) / 2.0)
        return p_avg

    def _gen_fidelity(self, r1: int, r2: int) -> float:
        """Returns the fidelity of the generated elementary link"""
        return self.F0 * np.exp(-self.channel_loss * self.distance(r1, r2))

    # ── ACTION 1: entangle (instantaneous) ──────────────

    def entangle(self, r1: int, r2: int) -> Dict[str, Any]:
        """Instantaneous EG between adjecent stations"""
        result = {"success": False, "fidelity": 0.0, "reason": ""} # Dict[str, Any]
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

        # BSM outcome determined now
        if self.rng.random() > rep.p_swap:
            self._break_link(r, qa)
            self._break_link(r, qb)
            result["reason"] = "swap_failed"
            return result

        # -Success: compute p_new, lock qubits, queue event
        # store the remote repeater and qubits
        ra, qa_r = int(rep.partner_repeater[qa]), int(rep.partner_qubit[qa])
        rb, qb_r = int(rep.partner_repeater[qb]), int(rep.partner_qubit[qb])
        p_new = rep.werner_param[qa] * rep.werner_param[qb]

        # lock remote qubits
        rep.lock_qubit(qa); rep.lock_qubit(qb)
        self.repeaters[ra].lock_qubit(qa_r)
        self.repeaters[rb].lock_qubit(qb_r)

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

    # ── ACTION 3: purify (deferred via event queue) ───────────────

    def purify(self, r1: int, r2: int) -> Dict[str, Any]:
        """BBPSSW purification. Lock all 4 qubits, queue event.
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

    # ── ACTION 4: age_links (+ event resolution) ─────────────────

    def age_links(self, discard_expired: bool = True) -> Dict[str, Any]:
        """Advance clock: age qubits, resolve pending events, expire old links."""
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

    # ── event resolution ──────────────────────────────────────────

    def _resolve_swap(self, ev: dict):
        """Resolve a deferred swap: free central qubits, rewrite remotes."""
        r, qa, qb = ev["r"], ev["qa"], ev["qb"]
        #qa_r = qubitA_remote, qb_r = qubitB_remote
        ra, qa_r, rb, qb_r = ev["ra"], ev["qa_r"], ev["rb"], ev["qb_r"]
        rep = self.repeaters[r]

        # [Guard]-> if qubits were freed (eg by expiry), clean up locks only
        if rep.status[qa] != QUBIT_OCCUPIED or rep.status[qb] != QUBIT_OCCUPIED:
            for rid, qi in [(ra, qa_r), (rb, qb_r)]:
                rr = self.repeaters[rid]
                if rr.status[qi] == QUBIT_OCCUPIED:
                    self._break_link(rid, qi)
            return

        rep.free_qubit(qa)
        rep.free_qubit(qb)

        rep_a, rep_b = self.repeaters[ra], self.repeaters[rb]
        # [Guard]-> remote qubits may also have been freed by expiry
        if rep_a.status[qa_r] != QUBIT_OCCUPIED or rep_b.status[qb_r] != QUBIT_OCCUPIED:
            if rep_a.status[qa_r] == QUBIT_OCCUPIED: 
                self._break_link(ra, qa_r)
            if rep_b.status[qb_r] == QUBIT_OCCUPIED: 
                self._break_link(rb, qb_r)
            return

        ec = min(rep_a.cutoff, rep_b.cutoff)
        # set the remote qubits and unlock them
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

    # ── internal ──────────────────────────────────────────────────

    def _break_link(self, r: int, qidx: int):
        """Frees a qubit if it is pointing to nowhere"""
        rep = self.repeaters[r]
        pr, pq = int(rep.partner_repeater[qidx]), int(rep.partner_qubit[qidx])
        if pr != NO_PARTNER:
            self.repeaters[pr].free_qubit(pq)
        rep.free_qubit(qidx)

    # ── observation helpers ───────────────────────────────────────

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


# ── factory helpers ───────────────────────────────────────────────

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