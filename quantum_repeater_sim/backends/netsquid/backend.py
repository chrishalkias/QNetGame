"""Analytic NetSquid backend (M1): chain topology, scalar-fidelity records,
pydynaa-driven classical-comm timing. Implements the PhysicsBackend interface.
"""
from __future__ import annotations
import numpy as np

from ..base import PhysicsBackend, NodeState, LinkState, Topology, _freeze
from ...repeater import fidelity_to_werner, werner_to_fidelity
from .timing import SimClock

NO_PARTNER = -1


class NetSquidBackend(PhysicsBackend):
    """Analytic-mode physics on NetSquid's pydynaa engine. Chain only (M1)."""

    def __init__(self, N, n_ch=4, spacing=50.0, p_gen=0.8, p_swap=0.5,
                 cutoff=20, F0=0.95, channel_loss=0.02, dt_seconds=0.0,
                 distance_dep_gen=True, rng=None, c_fiber=200_000.0):
        self._N = int(N)
        self._n_ch = int(n_ch)
        self._F0 = float(F0)
        self._channel_loss = float(channel_loss)
        self._distance_dep_gen = bool(distance_dep_gen)
        self._rng = rng if rng is not None else np.random.default_rng()

        self._p_gen = np.full(self._N, float(p_gen), dtype=np.float64)
        self._p_swap = np.full(self._N, float(p_swap), dtype=np.float64)
        self._cutoff = np.full(self._N, int(cutoff), dtype=np.int32)

        self._positions = np.stack(
            [np.array([i * spacing, 0.0]) for i in range(self._N)], axis=0)
        self._adj = np.zeros((self._N, self._N), dtype=np.float64)
        for i in range(self._N - 1):
            self._adj[i, i + 1] = self._adj[i + 1, i] = 1.0
        diff = self._positions[:, None, :] - self._positions[None, :, :]
        self._dist = np.linalg.norm(diff, axis=-1)

        shape = (self._N, self._n_ch)
        self._occupied = np.zeros(shape, dtype=bool)
        self._locked = np.zeros(shape, dtype=bool)
        self._partner_node = np.full(shape, NO_PARTNER, dtype=np.int32)
        self._partner_qubit = np.full(shape, NO_PARTNER, dtype=np.int32)
        self._p0 = np.zeros(shape, dtype=np.float64)
        self._age = np.zeros(shape, dtype=np.int32)
        self._link_cutoff = np.full(shape, int(cutoff), dtype=np.int32)
        self._generation = np.zeros(shape, dtype=np.uint32)

        self._clock = SimClock(c_fiber=c_fiber, dt_seconds=dt_seconds)
        self._clock.reset()
        self._resolved_this_advance = 0

    def _distance(self, a, b):
        return float(self._dist[a, b])

    def _gen_prob(self, a, b):
        p_avg = 0.5 * (self._p_gen[a] + self._p_gen[b])
        if self._distance_dep_gen:
            return p_avg * np.exp(-self._channel_loss * self._distance(a, b) / 2.0)
        return p_avg

    def _gen_fidelity(self, a, b):
        return self._F0 * np.exp(-self._channel_loss * self._distance(a, b))

    def _current_werner(self, node, q):
        c = max(int(self._link_cutoff[node, q]), 1)
        return float(self._p0[node, q] * np.exp(-int(self._age[node, q]) / c))

    def topology(self) -> Topology:
        return Topology(N=self._N,
                        adjacency=_freeze(self._adj.copy()),
                        positions=_freeze(self._positions.copy()))

    def node_state(self, node: int) -> NodeState:
        occ = self._occupied[node]
        fid = np.zeros(self._n_ch, dtype=np.float64)
        for q in np.flatnonzero(occ):
            fid[q] = werner_to_fidelity(self._current_werner(node, int(q)))
        return NodeState(
            node_id=int(node),
            n_ch=self._n_ch,
            p_gen=float(self._p_gen[node]),
            p_swap=float(self._p_swap[node]),
            occupied=_freeze(occ),
            locked=_freeze(self._locked[node]),
            partner_node=_freeze(self._partner_node[node]),
            partner_qubit=_freeze(self._partner_qubit[node]),
            fidelity=_freeze(fid),
            age=_freeze(self._age[node]),
        )

    def all_links(self) -> list[LinkState]:
        out = []
        for a in range(self._N):
            for q in np.flatnonzero(self._occupied[a]):
                b = int(self._partner_node[a, q])
                if b > a:
                    out.append(LinkState(
                        a, int(q), b, int(self._partner_qubit[a, q]),
                        werner_to_fidelity(self._current_werner(a, int(q))),
                        int(self._age[a, q])))
        return out

    @property
    def n_pending(self) -> int:
        return self._clock.n_pending

    @property
    def time(self) -> float:
        return float(self._clock.tick)

    def entangle(self, r1: int, r2: int) -> dict:
        raise NotImplementedError("entangle lands in Task 3")

    def swap(self, r: int) -> dict:
        raise NotImplementedError("swap lands in Task 4")

    def purify(self, r1: int, r2: int) -> dict:
        return {"success": False, "reason": "not_implemented_m1",
                "old_fidelity": 0.0, "new_fidelity": 0.0}

    def advance(self) -> dict:
        raise NotImplementedError("advance lands in Task 3")

    def reset(self) -> None:
        self._occupied[:] = False
        self._locked[:] = False
        self._partner_node[:] = NO_PARTNER
        self._partner_qubit[:] = NO_PARTNER
        self._p0[:] = 0.0
        self._age[:] = 0
        self._link_cutoff[:] = self._cutoff[:, None]
        self._generation[:] = 0
        self._clock.reset()
