"""PhysicsBackend interface + immutable state snapshots.

The environment depends only on this module: it issues mutating ops and reads
immutable snapshots, never touching a concrete engine's internals. All values
are reported in the fidelity domain (F in [0, 1]); Werner parameters are a
legacy-internal detail and never cross this boundary.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


def _freeze(a: np.ndarray) -> np.ndarray:
    """Return an independent read-only copy of *a* (never mutates the caller's array)."""
    a = np.ascontiguousarray(a).copy()
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class NodeState:
    """Immutable per-node snapshot. Arrays are read-only, length n_ch."""
    node_id: int
    n_ch: int
    p_gen: float
    p_swap: float
    occupied: np.ndarray       # bool  (n_ch,)
    locked: np.ndarray         # bool  (n_ch,)
    partner_node: np.ndarray   # int32 (n_ch,)  -1 = none
    partner_qubit: np.ndarray  # int32 (n_ch,)  -1 = none
    fidelity: np.ndarray       # float (n_ch,)  F-domain, 0.0 if free
    age: np.ndarray            # int32 (n_ch,)

    def __post_init__(self):
        for field in ("occupied", "locked", "partner_node",
                      "partner_qubit", "fidelity", "age"):
            arr = getattr(self, field)
            if arr.flags.writeable:
                object.__setattr__(self, field, _freeze(arr))


@dataclass(frozen=True)
class LinkState:
    """Immutable snapshot of one entanglement link."""
    node_a: int
    qubit_a: int
    node_b: int
    qubit_b: int
    fidelity: float
    age: int


@dataclass(frozen=True)
class Topology:
    """Immutable network topology: adjacency + node positions."""
    N: int
    adjacency: np.ndarray   # (N, N) float, 0/weight
    positions: np.ndarray   # (N, 2)  float

    def __post_init__(self):
        for field in ("adjacency", "positions"):
            arr = getattr(self, field)
            if arr.flags.writeable:
                object.__setattr__(self, field, _freeze(arr))


class PhysicsBackend(ABC):
    """Engine-agnostic physics interface consumed by QRNEnv."""

    # ---- topology (static after build) ----
    @abstractmethod
    def topology(self) -> Topology: ...

    # ---- read: immutable snapshots ----
    @abstractmethod
    def node_state(self, node: int) -> NodeState: ...

    @abstractmethod
    def all_links(self) -> list[LinkState]: ...

    @property
    @abstractmethod
    def n_pending(self) -> int: ...

    @property
    @abstractmethod
    def time(self) -> float: ...

    # ---- write: mutating ops (result dicts mirror the legacy shape) ----
    @abstractmethod
    def entangle(self, r1: int, r2: int) -> dict: ...

    @abstractmethod
    def swap(self, r: int) -> dict: ...

    @abstractmethod
    def purify(self, r1: int, r2: int) -> dict: ...

    @abstractmethod
    def advance(self) -> dict: ...

    @abstractmethod
    def reset(self) -> None: ...

    # ---- optional: rendering delegated to the backend ----
    def render(self, **kw):
        raise NotImplementedError
