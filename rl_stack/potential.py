"""Topology-general PBRS potential: the best 'shortcut' any entanglement
component offers toward connecting source and dest. Pure (no env/torch deps).

Phi(s) = max over entanglement-connected components C of
         max(0, d_total - min_{v in C} d_src[v] - min_{v in C} d_dst[v]) / d_total
where d_src/d_dst are physical shortest-path hops from source / to dest, and a
component is a set of nodes joined by entanglement links (post-swap long-range
links included). Reduces exactly to the old chain potential when a component
touches the source. See the design doc for the derivation.
"""
from __future__ import annotations
from collections import deque

import numpy as np


def bfs_hops(adjacency: np.ndarray, start: int) -> np.ndarray:
    """Shortest-path hop distance from `start` to every node over the unweighted
    graph (edge where adjacency != 0). Unreachable nodes are np.inf."""
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


def _components(n_nodes: int, edges):
    """Connected components (lists of node ids) over undirected `edges`.
    Nodes with no edge are omitted."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for a, b in edges:
        a, b = int(a), int(b)
        seen.add(a)
        seen.add(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = {}
    for x in seen:
        comp.setdefault(find(x), []).append(x)
    return list(comp.values())


def path_progress(d_src: np.ndarray, d_dst: np.ndarray, d_total: float,
                  entangled_edges) -> float:
    """PBRS potential Phi in [0, 1] (see module docstring). 0 if no edges or
    d_total is non-positive / non-finite."""
    if not np.isfinite(d_total) or d_total <= 0:
        return 0.0
    edges = list(entangled_edges)
    if not edges:
        return 0.0
    best = 0.0
    for comp in _components(len(d_src), edges):
        ds = min(d_src[v] for v in comp)
        dd = min(d_dst[v] for v in comp)
        if not (np.isfinite(ds) and np.isfinite(dd)):
            continue
        span = d_total - ds - dd
        if span > best:
            best = span
    return float(min(max(best, 0.0) / d_total, 1.0))
