"""
Immutable state snapshots returned by RepeaterNetwork read methods.

Consumers (the RL env, strategies, the adversary) read these read-only,
fidelity-domain snapshots instead of poking the engine's mutable arrays. All
fidelities are F in [0, 1]; Werner parameters are an engine-internal detail and
never cross this boundary.
"""
from __future__ import annotations
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
    link_cutoff: np.ndarray    # int32 (n_ch,)  effective per-link cutoff

    def __post_init__(self):
        for field in ("occupied", "locked", "partner_node",
                      "partner_qubit", "fidelity", "age", "link_cutoff"):
            arr = getattr(self, field)
            if arr.flags.writeable:
                object.__setattr__(self, field, _freeze(arr))


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
