"""
LegacyBackend — wraps the existing numpy RepeaterNetwork engine.
"""
from __future__ import annotations
import numpy as np

from .base import PhysicsBackend, NodeState, LinkState, Topology, _freeze
from ..network import RepeaterNetwork
from ..repeater import werner_to_fidelity, QUBIT_OCCUPIED


class LegacyBackend(PhysicsBackend):
    def __init__(self, net: RepeaterNetwork):
        self._net = net

    @property
    def net(self) -> RepeaterNetwork:
        """Direct access to the wrapped engine (legacy-only convenience)."""
        return self._net

    # ---- topology ----
    def topology(self) -> Topology:
        return Topology(
            N=self._net.N,
            adjacency=_freeze(self._net.adj.copy()),
            positions=_freeze(self._net._positions.copy()),
        )

    # ---- read ----
    def node_state(self, node: int) -> NodeState:
        rep = self._net.repeaters[node]
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
        )

    def all_links(self) -> list[LinkState]:
        raw = self._net.get_all_links()  # (L,6): r_a,q_a,r_b,q_b,fid,age
        return [
            LinkState(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                      float(r[4]), int(r[5]))
            for r in raw
        ]

    @property
    def n_pending(self) -> int:
        return len(self._net.pending_events)

    @property
    def time(self) -> float:
        return float(self._net.time_step)

    # ---- write ----
    def entangle(self, r1: int, r2: int) -> dict:
        return self._net.entangle(r1, r2)

    def swap(self, r: int) -> dict:
        return self._net.swap(r)

    def purify(self, r1: int, r2: int) -> dict:
        return self._net.purify(r1, r2)

    def advance(self) -> dict:
        res = self._net.age_links(discard_expired=True)
        return {
            "expired": res["expired_count"],
            "resolved": res["resolved_count"],
            "pending": res["pending_count"],
            "time": float(res["time_step"]),
        }

    def reset(self) -> None:
        self._net.reset()

    def render(self, **kw):
        return self._net.render(**kw)
