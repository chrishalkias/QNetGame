"""
--------------------------------------------------------------------------------
Topology-general PBRS potential: the single entanglement link that shortcuts
the most of the source->dest path. Pure (no env/torch deps).

Phi(s) = max over entangled links (a, b) of
         max(0, d_total - d_src[a] - d_dst[b], d_total - d_src[b] - d_dst[a]) / d_total
where d_src/d_dst are physical shortest-path hops from source / to dest.

SINGLE longest link, NOT a component union, ON PURPOSE: crediting a connected
component's span let a blob of ADJACENT links — which background auto-
entanglement creates for free — span the geodesic and max out Phi with zero
swaps, so the agent learned to do nothing (NOOP) and collect that free reward.
Adjacent links span only 1 hop each, so a single-longest-link potential is only
raised by SWAP-built long links — the progress the agent actually controls. The
cost: it under-credits a chain of swapped links (credited by their longest
member, not their union). See docs/superpowers for the debugging trail.
--------------------------------------------------------------------------------
"""
from __future__ import annotations
from collections import deque

import numpy as np


def bfs_hops(adjacency: np.ndarray, start: int) -> np.ndarray:
    """
    Shortest-path hop distance from `start` to every node over the unweighted
    graph (edge where adjacency != 0). Unreachable nodes are np.inf.
    """
    n = adjacency.shape[0]
    dist = np.full(n, np.inf)
    dist[start] = 0.0
    q = deque([int(start)])
    while q:
        u = q.popleft()
        for v in np.flatnonzero(adjacency[u] != 0):
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                q.append(int(v))
    return dist


def path_progress(d_src: np.ndarray, d_dst: np.ndarray, d_total: float,
                  entangled_edges) -> float:
    """
    PBRS potential Phi in [0, 1] (see module docstring): the single entangled
    link offering the largest source->dest shortcut. 0 if no edges or d_total is
    non-positive / non-finite.
    """
    if not np.isfinite(d_total) or d_total <= 0:
        return 0.0
    best = 0.0
    for a, b in entangled_edges:
        a, b = int(a), int(b)
        for x, y in ((a, b), (b, a)):
            dx, dy = d_src[x], d_dst[y]
            if np.isfinite(dx) and np.isfinite(dy):
                span = d_total - dx - dy
                if span > best:
                    best = span
    return float(min(max(best, 0.0) / d_total, 1.0))
