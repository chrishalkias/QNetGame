'''
Heuristic strategies for baseline comparison against the RL agent.

Each strategy takes a QRNEnv and returns an (N,) action array
containing only NOOP, SWAP, or PURIFY.  All strategies respect the
action mask (source/dest are always NOOP).

Entanglement is handled automatically by the environment step.
'''

from __future__ import annotations
import numpy as np
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY
from simulator.repeater import NO_PARTNER, SwapPolicy


def belief_propagation_policy(env: QRNEnv, n_iters: int = 8,
                              damping: float = 0.5,
                              min_score: float = 1e-9) -> np.ndarray:
    """Compatibility wrapper for the BP baseline strategy."""
    return BeliefPropagationPolicy(n_iters=n_iters, damping=damping,
                                   min_score=min_score)(env)


def swap_asap(env: QRNEnv) -> np.ndarray:
    """Swap at every interior node that can, immediately.

    If a node has =>2 available qubits linked to distinct partners,
    assign SWAP.  The network's swap function itself handles
    contention gracefully (returns failure if qubits became
    locked by an earlier swap in the same timestep).
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if mask[i, SWAP]:
            actions[i] = SWAP
    return actions


def purify_then_swap(env: QRNEnv) -> np.ndarray:
    """Purify if possible, otherwise swap if possible, else noop."""
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if mask[i, PURIFY]:
            actions[i] = PURIFY
        elif mask[i, SWAP]:
            actions[i] = SWAP
    return actions


def fidelity_gated_swap(env: QRNEnv, f_threshold: float = 0.5) -> np.ndarray:
    """Swap only when the node's mean link fidelity exceeds a threshold.

    Approximates the learned RL policy from cluster_004: wait for fresh,
    high-quality links before swapping; never purify.

    Reads the RepeaterNetwork node_state snapshot rather than the engine's
    mutable internals.
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        if not mask[i, SWAP]:
            continue
        ns = env.net.node_state(i)
        occ = ns.occupied
        if not bool(occ.any()):
            continue
        mean_f = float(ns.fidelity[occ].mean())
        if mean_f >= f_threshold:
            actions[i] = SWAP
    return actions


def random_policy(env: QRNEnv, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random valid action per node.

    IMPORTANT: *rng* must be independent of env.rng, otherwise drawing
    action choices perturbs the environment's own random stream
    (link generation, swap outcomes) and invalidates the comparison.
    """
    mask = env.get_action_mask()
    actions = np.full(env.N, NOOP, dtype=np.int32)
    for i in range(env.N):
        valid = np.flatnonzero(mask[i])
        actions[i] = rng.choice(valid) if len(valid) > 0 else NOOP
    return actions


class BeliefPropagationPolicy:
    """Training-free BP-style swap scheduler.

    On tree topologies such as chains, reachability messages are exact: each
    node's belief is the product of edge probabilities on the unique path from
    the endpoint. Cyclic topologies keep the loopy iterative fallback.
    """

    def __init__(self, n_iters: int = 8, damping: float = 0.5,
                 min_score: float = 1e-9):
        self.n_iters = n_iters
        self.damping = damping
        self.min_score = min_score

    def __call__(self, env: QRNEnv) -> np.ndarray:
        mask = env.get_action_mask()
        actions = np.full(env.N, NOOP, dtype=np.int32)

        edge_prob = self.physical_edge_prob(env)
        src_msg = self.reachability(env._topo.adjacency, edge_prob, env.source)
        dst_msg = self.reachability(env._topo.adjacency, edge_prob, env.dest)

        for node in range(env.N):
            if not mask[node, SWAP]:
                continue
            cand = self.predicted_swap_candidate(env, node)
            if cand is None:
                continue
            a, b, qa, qb = cand
            progress = self.shortcut_progress(env, a, b)
            if progress <= 0.0:
                continue
            support = max(float(src_msg[a]) * float(dst_msg[b]),
                          float(src_msg[b]) * float(dst_msg[a]))
            p_swap = float(np.clip(env.net.node_state(node).p_swap, 0.0, 1.0))
            score = support * progress * p_swap * float(np.sqrt(max(qa * qb, 0.0)))
            if score > self.min_score:
                actions[node] = SWAP
        return actions

    def reachability(self, adjacency: np.ndarray, edge_prob: np.ndarray,
                     root: int) -> np.ndarray:
        """Endpoint reachability messages over the physical topology."""
        parent, order = self.tree_parent_order(adjacency, root)
        if parent is not None:
            return self.tree_reachability(edge_prob, parent, order, root)
        return self.loopy_reachability(adjacency, edge_prob, root)

    @staticmethod
    def tree_parent_order(adjacency: np.ndarray, root: int):
        """Return DFS parents/order if adjacency is a connected tree, else None."""
        n = adjacency.shape[0]
        if n == 0:
            return None, None
        undirected_edges = int(np.count_nonzero(np.triu(adjacency != 0, k=1)))
        if undirected_edges != n - 1:
            return None, None

        parent = np.full(n, -1, dtype=np.int32)
        seen = np.zeros(n, dtype=bool)
        order = []
        stack = [int(root)]
        seen[int(root)] = True

        while stack:
            u = stack.pop()
            order.append(u)
            for v in np.flatnonzero(adjacency[u] != 0):
                v = int(v)
                if seen[v]:
                    continue
                seen[v] = True
                parent[v] = u
                stack.append(v)

        if not bool(np.all(seen)):
            return None, None
        return parent, order

    @staticmethod
    def tree_reachability(edge_prob: np.ndarray, parent: np.ndarray,
                          order: list[int], root: int) -> np.ndarray:
        """Exact tree reachability: product of edge probabilities on root paths."""
        reach = np.zeros(len(parent), dtype=np.float64)
        reach[int(root)] = 1.0
        for v in order:
            if v == root:
                continue
            u = int(parent[v])
            reach[v] = reach[u] * float(edge_prob[u, v])
        return np.clip(reach, 0.0, 1.0)

    def loopy_reachability(self, adjacency: np.ndarray, edge_prob: np.ndarray,
                           root: int) -> np.ndarray:
        """Loopy BP-style reachability marginals for cyclic topologies."""
        n = adjacency.shape[0]
        reach = np.zeros(n, dtype=np.float64)
        reach[int(root)] = 1.0
        damping = float(np.clip(self.damping, 0.0, 1.0))

        for _ in range(max(int(self.n_iters), 1)):
            new = np.zeros(n, dtype=np.float64)
            new[int(root)] = 1.0
            for v in range(n):
                if v == root:
                    continue
                miss = 1.0
                for u in np.flatnonzero(adjacency[v] != 0):
                    miss *= 1.0 - float(edge_prob[v, u]) * float(reach[u])
                new[v] = 1.0 - miss
            reach = damping * reach + (1.0 - damping) * new
            reach[int(root)] = 1.0
        return np.clip(reach, 0.0, 1.0)

    @staticmethod
    def physical_edge_prob(env: QRNEnv) -> np.ndarray:
        """Per-physical-edge generation probability used by BP messages."""
        adj = env._topo.adjacency
        probs = np.zeros_like(adj, dtype=np.float64)
        for i, j in zip(*np.nonzero(np.triu(adj, k=1))):
            if hasattr(env.net, "_gen_prob"):
                p = float(env.net._gen_prob(int(i), int(j)))
            else:
                ni = env.net.node_state(int(i))
                nj = env.net.node_state(int(j))
                p = 0.5 * (float(ni.p_gen) + float(nj.p_gen))
            probs[i, j] = probs[j, i] = float(np.clip(p, 0.0, 1.0))
        return probs

    @staticmethod
    def qubit_quality(ns, q: int) -> float:
        """Fidelity weighted by remaining lifetime for an available link."""
        cutoff = max(float(ns.link_cutoff[q]), 1.0)
        freshness = 1.0 - float(ns.age[q]) / cutoff
        return float(np.clip(ns.fidelity[q], 0.0, 1.0) *
                     np.clip(freshness, 0.0, 1.0))

    def predicted_swap_candidate(self, env: QRNEnv, node: int):
        """Return the partner nodes and link qualities the env is likely to swap."""
        ns = env.net.node_state(node)
        avail = np.flatnonzero(ns.occupied & (~ns.locked))
        if len(avail) < 2:
            return None

        idx_i, idx_j = np.triu_indices(len(avail), k=1)
        qa_all, qb_all = avail[idx_i], avail[idx_j]
        pa_all, pb_all = ns.partner_node[qa_all], ns.partner_node[qb_all]
        valid = ((pa_all != NO_PARTNER) & (pb_all != NO_PARTNER) &
                 (pa_all != pb_all))
        if not bool(np.any(valid)):
            return None

        qa_all, qb_all = qa_all[valid], qb_all[valid]
        pa_all, pb_all = pa_all[valid], pb_all[valid]
        rep = env.net.repeaters[node]

        if rep.swap_policy == SwapPolicy.STRONGEST:
            scores = ns.fidelity[qa_all] * ns.fidelity[qb_all]
        else:
            pos = env.net._positions
            scores = np.linalg.norm(pos[pa_all] - pos[pb_all], axis=1)

        best = int(np.argmax(scores))
        qa, qb = int(qa_all[best]), int(qb_all[best])
        return (int(pa_all[best]), int(pb_all[best]),
                self.qubit_quality(ns, qa), self.qubit_quality(ns, qb))

    @staticmethod
    def shortcut_progress(env: QRNEnv, a: int, b: int) -> float:
        """Source-destination progress created by a candidate shortcut edge."""
        total = float(env._d_total)
        if not np.isfinite(total) or total <= 0.0:
            return 0.0

        best = 0.0
        for x, y in ((a, b), (b, a)):
            dx, dy = float(env._d_src[x]), float(env._d_dst[y])
            if np.isfinite(dx) and np.isfinite(dy):
                best = max(best, total - dx - dy)
        return float(np.clip(best / total, 0.0, 1.0))
