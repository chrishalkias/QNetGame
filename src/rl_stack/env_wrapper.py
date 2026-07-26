"""
--------------------------------------------------------------------------------
RL environment wrapper for the quantum repeater network simulator.

Serialized left-to-right sweep
-------------------------------
One env "step" is now ONE micro-decision for env.active_node, applied
immediately (the engine applies swap/purify without deferral). Interior
nodes 1..N-2 act in a fixed left-to-right order each tick; source (0) and
dest (N-1) never act (they are never members of `_interior`, so they can
never be active). After the LAST interior node's micro-step, the tick
boundary runs: age_links (decohere, expire) then auto-entangle (background
link generation). End-to-end is checked after EVERY micro-step, so a swap
may cascade end-to-end within one sweep and the episode can terminate
mid-sweep on the closing node.

  reset()       -> auto-entangle once, cursor = first interior node, return obs
  step(action):
    1. apply `action` to env.active_node immediately
    2. check end-to-end -> terminate here if connected (mid-sweep close)
    3. advance the cursor to the next interior node (intra-tick), OR if this
       was the last interior node: age_links, check end-to-end again,
       auto-entangle, wrap the cursor back to the first interior node
       (tick boundary)

Physical time/cost attach to the TICK, not the micro-step: STEP_COST is
charged once, on the boundary micro-step; intra-tick micro-steps carry pure
PBRS shaping with gamma_eff=1.0 (telescoping over the tick), the boundary
micro-step uses gamma_eff=self.gamma. See `step()` for the exact formulas.

Action space
-------------
    0 = NOOP   (wait)
    1 = SWAP   (BSM at this node)
    2 = PURIFY (BBPSSW on the best shared pair at this node)

Entanglement generation is **not** an agent action - it is handled
entirely by the automatic background generation step.

Source and destination nodes never act (structurally excluded from
`_interior`).
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from simulator.network import build_network
from simulator.repeater import NO_PARTNER, QUBIT_OCCUPIED, werner_to_fidelity
from rl_stack import potential

# --- action constants ----------------------------------------------------
NOOP    = 0
SWAP    = 1
PURIFY  = 2
N_ACTIONS = 3
ACTION_NAMES = ["noop", "swap", "purify"]

"""

      ▄▄▄▄▄   ▄▄▄▄▄▄▄   ▄▄▄    ▄▄▄        ▄▄▄▄▄▄▄
    ▄███████▄ ███▀▀███▄ ████▄  ███       ███▀▀▀▀▀
    ███   ███ ███▄▄███▀ ███▀██▄███       ███▄▄    ████▄ ██ ██
    ███▄█▄███ ███▀▀██▄  ███  ▀████ ▀▀▀▀▀ ███      ██ ██ ██▄██
     ▀█████▀  ███  ▀███ ███    ███       ▀███████ ██ ██  ▀█▀
          ▀▀

Gym-like wrapper around RepeaterNetwork for RL training.

The agent decides SWAP / PURIFY / NOOP at each interior node.
Source and destination nodes are always forced to NOOP.
"""


class QRNEnv:
    __slots__ = (
        "rng",        # RNG: per-episode physics + action sampling
        "max_steps",  # truncation horizon (steps before time-limit)
        "gamma",      # PBRS discount (threaded to equal the DQN gamma)
        "topology",   # topology tag (always 'chain')
        "net",        # RepeaterNetwork physics engine
        "N",          # number of repeaters
        "source",     # source node id (0)
        "dest",       # destination node id (N-1)
        "steps",      # steps elapsed in the current episode
        "done",       # episode terminated or truncated
        "_phi",       # last PBRS potential Φ(s)
        "_d_src",     # BFS hop distances from source
        "_d_dst",     # BFS hop distances from dest
        "_d_total",   # source→dest hop distance (path length)
        "_interior",  # fixed left-to-right sweep order (source/dest excluded)
        "_active",    # current active interior node (-1 if _interior is empty)
    )

    STEP_COST       = -0.01
    SUCCESS_REWARD  =  1.0

    def __init__(self,
                 n_repeaters = 5,
                 n_ch = 4,
                 spacing = 50.0,
                 p_gen = 0.8,
                 p_swap = 0.5,
                 p_gen_std = 0.0,
                 p_swap_std = 0.0,
                 cutoff = 20,
                 F0 = 0.95,
                 channel_loss = 0.02,
                 max_steps = 50,
                 rng: Optional[np.random.Generator] = None,
                 topology = 'chain',
                 gamma = 0.99):

        if topology != 'chain':
            raise ValueError(f'Topology {topology} not supported')

        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_steps = max_steps
        self.gamma = gamma
        self._phi = 0.0
        self.topology = topology

        self.net = build_network(
            topology=topology, n_repeaters=n_repeaters, n_ch=n_ch,
            spacing=spacing, p_gen=p_gen, p_swap=p_swap,
            p_gen_std=p_gen_std, p_swap_std=p_swap_std, cutoff=cutoff,
            F0=F0, channel_loss=channel_loss,
            rng=self.rng)

        self.N = self.net.N
        self.source = -1
        self.dest = -1
        self.steps = 0
        self.done = False
        self._pick_targets()



# ▄▄▄▄▄▄▄▄▄
# ▀▀▀███▀▀▀                       ██
#    ███  ▀▀█▄ ████▄ ▄████ ▄█▀█▄ ▀██▀▀ ▄█▀▀▀
#    ███ ▄█▀██ ██ ▀▀ ██ ██ ██▄█▀  ██   ▀███▄
#    ███ ▀█▄██ ██    ▀████ ▀█▄▄▄  ██   ▄▄▄█▀
#                       ██
#                     ▀▀▀

    def _pick_targets(self):
        # Chain endpoints are the two ends; also cache BFS hop distances that
        # the PBRS potential (_progress) reads every step.
        self.source, self.dest = 0, self.N - 1
        adj = self.net.adj
        self._d_src = potential.bfs_hops(adj, self.source)
        self._d_dst = potential.bfs_hops(adj, self.dest)
        self._d_total = float(self._d_src[self.dest])

    def _entangled_edges(self):
        """Undirected (a, b) edges of the current entanglement graph."""
        edges = set()
        for i in range(self.N):
            rep = self.net.node(i)
            for qi in np.flatnonzero(rep.status == QUBIT_OCCUPIED):
                p = int(rep.partner_repeater[qi])
                if p != NO_PARTNER and p != i:
                    edges.add((min(i, p), max(i, p)))
        return edges

    def _progress(self) -> float:
        """Topology-general PBRS potential in [0, 1]."""
        return potential.path_progress(
            self._d_src, self._d_dst, self._d_total, self._entangled_edges())

    def is_target(self, node: int) -> bool:
        return node == self.source or node == self.dest

    @property
    def active_node(self) -> int:
        """Interior node deciding the current micro-step (-1 if none, N<3)."""
        return self._active

#   ▄▄▄▄▄   ▄▄
# ▄███████▄ ██                                   ██   ▀▀
# ███   ███ ████▄ ▄█▀▀▀ ▄█▀█▄ ████▄ ██ ██  ▀▀█▄ ▀██▀▀ ██  ▄███▄ ████▄
# ███▄▄▄███ ██ ██ ▀███▄ ██▄█▀ ██ ▀▀ ██▄██ ▄█▀██  ██   ██  ██ ██ ██ ██
#  ▀█████▀  ████▀ ▄▄▄█▀ ▀█▄▄▄ ██     ▀█▀  ▀█▄██  ██   ██▄ ▀███▀ ██ ██



    def get_observation(self) -> Dict[str, np.ndarray]:
        """Build size-agnostic node features + topology.

        Features per node (8):
            [0] frac_occupied      occupied / physical capacity (2*n_ch interior, n_ch ends)
            [1] can_swap           1.0 if a viable swap pair exists: one available LEFT link (partner<node) + one available RIGHT link (partner>node) whose fused link survives the tick boundary (age_i + age_j + 1 < min cutoff)
            [2] can_purify         1.0 if >=2 available qubits to same partner
            [3] p_gen              per-repeater link-generation prob. (inhomogeneity)
            [4] p_swap             per-repeater BSM success prob. (inhomogeneity)
            [5] link_urgency       mean(age/link_cutoff) over occupied qubits (0 if none)
            [6] relative_position  i / (N-1): 0.0 at source, 1.0 at dest
            [7] is_active          1.0 at env.active_node, the node deciding this micro-step (exactly one node; all 0 if N<3 has no interior nodes)

        Features [1] and [2] are forced to 0 for source / dest. Features [3]/[4]
        are constant across nodes when the network is homogeneous (std=0); they
        carry node-quality signal only under per-repeater inhomogeneity.
        Feature [5] is 0 for nodes with no occupied qubits; ~1 means links are
        about to expire.
        """
        feats = np.zeros((self.N, 8), dtype=np.float32)
        for i in range(self.N):
            rep = self.net.node(i)
            occ = rep.status == QUBIT_OCCUPIED
            capacity = occ.size  # physical qubit count (2*n_ch interior, n_ch ends)
            feats[i, 0] = int(occ.sum()) / capacity
            if self.is_target(i):
                feats[i, 1] = 0.0
                feats[i, 2] = 0.0
            else:
                feats[i, 1] = 1.0 if rep.can_swap() else 0.0
                feats[i, 2] = 1.0 if rep.can_purify() else 0.0
            feats[i, 3] = rep.p_gen
            feats[i, 4] = rep.p_swap
            if bool(occ.any()):
                lc = np.maximum(rep.link_cutoff[occ], 1)
                feats[i, 5] = float(np.clip(np.mean(rep.age[occ] / lc), 0.0, 1.0))
            else:
                feats[i, 5] = 0.0
        feats[:, 6] = np.arange(self.N, dtype=np.float32) / (self.N - 1)
        if self._active != -1:
            feats[self._active, 7] = 1.0
        src, dst = np.nonzero(self.net.adj)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        return {"x": feats, "edge_index": edge_index}

# ▄▄▄      ▄▄▄
# ████▄  ▄████             ▄▄
# ███▀████▀███  ▀▀█▄ ▄█▀▀▀ ██ ▄█▀
# ███  ▀▀  ███ ▄█▀██ ▀███▄ ████
# ███      ███ ▀█▄██ ▄▄▄█▀ ██ ▀█▄


    def action_mask(self, node: int) -> np.ndarray:
        """(3,) bool mask for ONE node: the decision-path primitive.

        Only env.active_node ever decides, so building an (N,3) grid to read one
        row was an O(N) pass per micro-step for O(1) of information. NOOP is
        always legal so a masked argmax can never be all -inf; source and dest
        are NOOP-only. Legality itself is defined once, on the node
        (`Repeater.can_swap` / `can_purify`), so the mask cannot drift from the
        engine's own gate.

        Returns a fresh array every call, so callers may edit it in place (the
        agent's `disable_actions`) without touching anyone else's mask.
        """
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[NOOP] = True
        if self.is_target(node):
            return m
        rep = self.net.node(node)
        m[SWAP] = rep.can_swap()
        m[PURIFY] = rep.can_purify()
        return m

    def get_action_mask(self) -> np.ndarray:
        """(N, 3) bool mask for every node. Kept for the offline probes and the
        tests, which want the whole grid; built FROM `action_mask` so there is
        exactly one definition of what is legal. The decision path should call
        `action_mask(node)` directly."""
        return np.stack([self.action_mask(i) for i in range(self.N)])

#  ▄▄▄▄▄▄▄
# █████▀▀▀  ██
#  ▀████▄  ▀██▀▀ ▄█▀█▄ ████▄
#    ▀████  ██   ██▄█▀ ██ ██
# ███████▀  ██   ▀█▄▄▄ ████▀
#                      ██
#                      ▀▀

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute one micro-decision for env.active_node. See the module
        docstring for the serialized-sweep model; see the class docstring
        note above for the credit-assignment (gamma_eff / STEP_COST) rules.
        """
        if not self._interior:
            return self._step_no_interior(action)

        r = self._active
        info = {"active_node": r, "fidelity": 0.0}
        phi_before = self._phi

        # apply the active node's action immediately (source/dest never active)
        if action == PURIFY:   self._exec_purify(r)
        elif action == SWAP:   self._exec_swap(r)

        # mid-sweep delivery check (immediate apply -> link exists now)
        connected, fidelity = self._check_e2e()
        if connected:
            self.done = True
            info.update(fidelity=fidelity, terminated=True, truncated=False,
                        tick_boundary=False, next_active_node=-1,
                        gamma_eff=self.gamma, ticks=self.steps + 1)
            reward = fidelity * self.SUCCESS_REWARD - phi_before  # Phi(terminal)=0
            self._phi = 0.0
            return self.get_observation(), reward, True, info

        # advance the sweep cursor
        idx = self._interior.index(r)
        if idx + 1 < len(self._interior):          # intra-tick micro-step
            self._active = self._interior[idx + 1]
            phi_new = self._progress()
            reward = phi_new - phi_before           # gamma_eff = 1, no step cost
            self._phi = phi_new
            info.update(terminated=False, truncated=False, tick_boundary=False,
                        next_active_node=self._active, gamma_eff=1.0, ticks=self.steps)
            return self.get_observation(), reward, False, info

        # tick boundary: physics resolves, then background generation
        self.net.age_links(discard_expired=True)
        self.steps += 1
        connected, fidelity = self._check_e2e()     # a link may have expired/formed
        if connected:
            self.done = True
            info.update(fidelity=fidelity, terminated=True, truncated=False,
                        tick_boundary=True, next_active_node=-1,
                        gamma_eff=self.gamma, ticks=self.steps)
            reward = fidelity * self.SUCCESS_REWARD - phi_before
            self._phi = 0.0
            return self.get_observation(), reward, True, info
        self._auto_entangle()
        self._active = self._interior[0]
        phi_new = self._progress()
        reward = self.STEP_COST + (self.gamma * phi_new - phi_before)
        self._phi = phi_new
        truncated = self.steps >= self.max_steps
        self.done = truncated
        info.update(terminated=False, truncated=truncated, tick_boundary=True,
                    next_active_node=self._active, gamma_eff=self.gamma, ticks=self.steps)
        return self.get_observation(), reward, truncated, info

    def _step_no_interior(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """N < 3: there are no interior nodes to act on, so every call is
        directly a tick boundary (defensive edge case; no caller trains on
        chains this short, but reset()/step() must not crash on them)."""
        info = {"active_node": -1, "fidelity": 0.0}
        phi_before = self._phi

        # "mid-sweep" equivalent (no action to apply, but mirrors the check
        # right after the main step()'s action-execution point).
        connected, fidelity = self._check_e2e()
        if connected:
            self.done = True
            info.update(fidelity=fidelity, terminated=True, truncated=False,
                        tick_boundary=True, next_active_node=-1,
                        gamma_eff=self.gamma, ticks=self.steps + 1)
            reward = fidelity * self.SUCCESS_REWARD - phi_before
            self._phi = 0.0
            return self.get_observation(), reward, True, info

        # tick boundary: physics resolves, then background generation
        self.net.age_links(discard_expired=True)
        self.steps += 1
        connected, fidelity = self._check_e2e()
        if connected:
            self.done = True
            info.update(fidelity=fidelity, terminated=True, truncated=False,
                        tick_boundary=True, next_active_node=-1,
                        gamma_eff=self.gamma, ticks=self.steps)
            reward = fidelity * self.SUCCESS_REWARD - phi_before
            self._phi = 0.0
            return self.get_observation(), reward, True, info

        self._auto_entangle()
        phi_new = self._progress()
        reward = self.STEP_COST + (self.gamma * phi_new - phi_before)
        self._phi = phi_new
        truncated = self.steps >= self.max_steps
        self.done = truncated
        info.update(terminated=False, truncated=truncated, tick_boundary=True,
                    next_active_node=-1, gamma_eff=self.gamma, ticks=self.steps)
        return self.get_observation(), reward, truncated, info


#  ▄▄▄▄▄▄▄
# ███▀▀▀▀▀                                  ██   ▀▀
# ███▄▄    ██ ██ ▄█▀█▄ ▄████    ▀▀█▄ ▄████ ▀██▀▀ ██  ▄███▄ ████▄ ▄█▀▀▀
# ███       ███  ██▄█▀ ██      ▄█▀██ ██     ██   ██  ██ ██ ██ ██ ▀███▄
# ▀███████ ██ ██ ▀█▄▄▄ ▀████   ▀█▄██ ▀████  ██   ██▄ ▀███▀ ██ ██ ▄▄▄█▀



    def _auto_entangle(self):
        """Background entanglement: one pass over all adjacent pairs."""
        pairs = list(zip(*np.nonzero(np.triu(self.net.adj, k=1))))
        self.rng.shuffle(pairs)
        for r1, r2 in pairs:
            self.net.entangle(int(r1), int(r2))

    def _exec_swap(self, r: int) -> Dict:
        return self.net.swap(r)

    def _exec_purify(self, r: int) -> Dict:
        """One PURIFY action runs the distillation cascade on EVERY partner with
        which node r shares >=2 available links (each cascade leaves one survivor
        or none).

        ``valid`` is built ONCE, before the first ``purify()`` call, and ``rep``
        is the live engine object (snapshots.py deleted 2026-07-26), so the list
        is deliberately stale by the time later partners are processed. That is
        safe because of DISJOINTNESS, not because of how the list was copied:
        the links to a lower-index partner and to a higher-index partner sit on
        different ports of r, and ``purify(r, p)`` only touches qubits in
        ``rep.qubits_to(p)``, so cascading on one partner cannot change how many
        links r shares with another. ``valid`` also holds partner IDs, never
        qubit indices, and ``purify()`` re-reads the qubits live and returns
        ``insufficient_shared_pairs`` if a partner has since dropped below two,
        so a stale ID degrades to a no-op rather than to wrong physics.
        Guarded by tests/test_rl_stack.py::TestMultiPartnerPurify.
        """
        rep = self.net.node(r)
        avail = rep.status == QUBIT_OCCUPIED
        if int(avail.sum()) < 2:
            return {"success": False, "reason": "insufficient_qubits"}
        partners = rep.partner_repeater[avail]
        unique, counts = np.unique(partners[partners != NO_PARTNER],
                                   return_counts=True)
        valid = [int(p) for p, c in zip(unique, counts) if c >= 2]
        if not valid:
            return {"success": False, "reason": "no_valid_pair"}
        any_ok = False
        for p in valid:
            res = self.net.purify(r, p)
            any_ok = any_ok or res["success"]
        return {"success": any_ok, "reason": "ok"}

    def _check_e2e(self) -> Tuple[bool, float]:
        """Check whether source and dest share a direct entanglement link."""
        rep = self.net.node(self.source)
        for qi in np.flatnonzero(rep.status == QUBIT_OCCUPIED):
            if int(rep.partner_repeater[qi]) == self.dest:
                return True, float(werner_to_fidelity(rep.werner_param[qi]))
        return False, 0.0


# ▄▄▄      ▄▄▄
# ████▄  ▄████ ▀▀
# ███▀████▀███ ██  ▄█▀▀▀ ▄████
# ███  ▀▀  ███ ██  ▀███▄ ██
# ███      ███ ██▄ ▄▄▄█▀ ▀████



    def reset(self) -> Dict[str, np.ndarray]:
        """Reset, position the sweep cursor at the first interior node,
        auto-entangle once, return observation."""
        self.net.reset()
        self._pick_targets()
        self.steps = 0
        self.done  = False
        self._interior = list(range(1, self.N - 1))
        self._active = self._interior[0] if self._interior else -1
        self._auto_entangle()
        self._phi = self._progress()
        return self.get_observation()
