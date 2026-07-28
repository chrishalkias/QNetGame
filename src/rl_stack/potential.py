"""
--------------------------------------------------------------------------------
PBRS potential for the linear repeater chain: the single entanglement link that
shortcuts the most of the source->dest path. Pure (no env/torch deps).

On a chain, node i sits i hops from the source and N-1-i hops from the dest, so
the shortcut a link (a, b) offers collapses to |a - b| hops out of N-1:

    Phi(s) = max over entangled links (a, b) of |a - b| / (N - 1)

(This is the topology-general BFS formula
 max(0, d_total - d_src[a] - d_dst[b], ...sym...) / d_total specialised to the
 chain, which is the only geometry the project models since f0bb963. The values
 are identical, see
 tests/test_potential.py::test_chain_closed_form_matches_the_bfs_formula.)

SINGLE longest link, NOT a component union, ON PURPOSE: crediting a connected
component's span let a blob of ADJACENT links, which background auto-
entanglement creates for free, span the geodesic and max out Phi with zero
swaps, so the agent learned to do nothing (NOOP) and collect that free reward.
Adjacent links span only 1 hop each, so a single-longest-link potential is only
raised by SWAP-built long links, the progress the agent actually controls. The
cost: it under-credits a chain of swapped links (credited by their longest
member, not their union). See docs/superpowers for the debugging trail.
--------------------------------------------------------------------------------
"""
from __future__ import annotations


def path_progress(entangled_edges, n_nodes: int) -> float:
    """
    PBRS potential Phi in [0, 1] (see module docstring): the single entangled
    link offering the largest source->dest shortcut on an `n_nodes` chain. 0
    when there are no edges or the chain is degenerate (n_nodes < 2).
    """
    if n_nodes < 2:
        return 0.0
    span = max((abs(int(a) - int(b)) for a, b in entangled_edges), default=0)
    return min(span / (n_nodes - 1), 1.0)
