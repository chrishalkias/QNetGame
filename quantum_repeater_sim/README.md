# `quantum_repeater_sim`

A discrete-time quantum repeater network simulator with classical
communication delays, designed as the environment for a Reinforcement
Learning pipeline. Pure Python/NumPy with optional `torch_geometric`
integration for GNN-based agents.

**Dependencies:** NumPy.  `torch` and `torch_geometric` are optional
(enables `HeteroData` output; falls back to a NumPy-based container
otherwise).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Physics Model](#2-physics-model)
3. [Architecture Overview](#3-architecture-overview)
4. [`repeater.py` — The Repeater Class](#4-repeaterpy--the-repeater-class)
5. [`network.py` — The Network and Its Actions](#5-networkpy--the-network-and-its-actions)
6. [`graph_builder.py` — Graph Observations](#6-graph_builderpy--graph-observations)
7. [Classical Communication Delays](#7-classical-communication-delays)
8. [Runnable Examples](#8-runnable-examples)
9. [RL Integration Guide](#9-rl-integration-guide)
10. [Known Limitations](#10-known-limitations)

---

## 1. Quick Start

```python
import numpy as np
from quantum_repeater_sim import build_chain, network_to_heterodata

net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.7, cutoff=15,
                  F0=0.95, channel_loss=0.02,
                  dt_seconds=1e-4,          # enables classical delays
                  rng=np.random.default_rng(42))

# RL loop
for step in range(100):
    net.age_links()                          # tick clock, resolve events
    obs = network_to_heterodata(net)         # graph observation
    # ... agent picks action from obs ...
    net.entangle(0, 1)                       # or net.swap(2), net.purify(0, 1)
```

---

## 2. Physics Model

### 2.1 Werner States

Every entangled pair is described by a single scalar, the Werner
parameter $p \in [0, 1]$, corresponding to the two-qubit state

$$
\rho(p) = p\,|\Phi^+\rangle\!\langle\Phi^+| + \frac{1-p}{4}\,I_4
$$

Fidelity with respect to the target Bell state is

$$
F = \frac{3p+1}{4}
\qquad\Longleftrightarrow\qquad
p = \frac{4F-1}{3}
$$

### 2.2 Entanglement Generation

Two adjacent repeaters $i, j$ attempt to create a Bell pair.

**Success probability** (per-repeater, averaged):

$$
P_{\text{gen}}(i,j) = \frac{p_{\text{gen}}^{(i)} + p_{\text{gen}}^{(j)}}{2}
\times
\begin{cases}
e^{-\alpha\, d_{ij}/2} & \text{if distance\_dep\_gen} \\
1 & \text{otherwise}
\end{cases}
$$

**Initial fidelity** of the new pair:

$$
F_0(i,j) = F_0 \cdot e^{-\alpha\, d_{ij}}
$$

where $\alpha$ is the fibre attenuation and $d_{ij}$ is the Euclidean
distance between the two repeaters.

### 2.3 Entanglement Swapping

A Bell-state measurement (BSM) at repeater $C$ holding links to $A$
(Werner $p_1$) and $B$ (Werner $p_2$) produces a direct $A$-$B$ pair:

$$
p_{\text{new}} = p_1 \cdot p_2
$$

The BSM succeeds with probability $p_{\text{swap}}^{(C)}$.  On failure,
both links through $C$ are destroyed.

### 2.4 BBPSSW Purification

When two repeaters share two or more Bell pairs, they can sacrifice one
pair to improve another:

$$
P_{\text{succ}} = \frac{3\,p_1\,p_2+1}{4},
\qquad
p_{\text{new}} = \frac{2\,p_1\,p_2+p_1+p_2}{3\,p_1\,p_2+1}
$$

The pair with the lower Werner parameter is sacrificed. On failure, both
pairs are destroyed.

### 2.5 Memory Decoherence

Every occupied qubit carries a discrete age $m$ (time-steps since link
creation). Each tick:

$$
p(m) = p_0 \cdot e^{-m/c_{\text{eff}}}
$$

where $c_{\text{eff}} = \min(c_A, c_B)$ is the per-link effective cutoff,
determined by the weaker memory of the two endpoints. Links whose age
reaches $c_{\text{eff}}$ are optionally destroyed.

### 2.6 Classical Communication Delay

After a successful BSM or purification measurement, the classical
outcome must travel through fibre at $c_{\text{fiber}} = 2 \times 10^5$
km/s before remote qubits can be updated. The delay in discrete steps
is

$$
\Delta = \left\lceil \frac{d}{c_{\text{fiber}} \cdot \Delta t} \right\rceil
$$

where $d$ is the relevant distance and $\Delta t$ is the physical
duration of one simulator tick. During the delay, involved qubits are
**locked** and invisible to the agent.

### 2.7 Parameter Summary

| Symbol | Scope | Default | Description |
|---|---|---|---|
| `n_ch` | per repeater | 4 | Qubit memory slots |
| `p_gen` | per repeater | 0.8 | Generation success probability |
| `p_swap` | per repeater | 0.5 | BSM success probability |
| `cutoff` | per repeater | 20 | Memory coherence time (steps) |
| `F0` | network | 0.98 | Zero-distance fidelity |
| `channel_loss` | network | 0.02 | Fibre attenuation (km$^{-1}$) |
| `dt_seconds` | network | $10^{-4}$ | Physical time per tick (s) |
| `distance_dep_gen` | network | True | Scale $P_{\text{gen}}$ by distance? |

Setting `dt_seconds=0.0` disables all classical delays (events resolve
on the next `age_links()` call).

---

## 3. Architecture Overview

```
  RL Agent ───────────────────┐
    │ observes graph          │ picks action
    ▼                         ▼
┌────────────────┐   ┌───────────────────────────────────────┐
│ graph_builder  │◄──│          RepeaterNetwork              │
│                │   │                                       │
│ network_to_    │   │  .entangle(r1, r2)  → instant         │
│ heterodata()   │   │  .swap(r)           → deferred (lock) │
│                │   │  .purify(r1, r2)    → deferred (lock) │
│  → HeteroData  │   │  .age_links()       → tick + resolve  │
│  or numpy dict │   │                                       │
└────────────────┘   └───────────────┬───────────────────────┘
                                     │ delegates to
                                     ▼
                          ┌───────────────────────┐
                          │   Repeater (×N)       │
                          │  per-qubit arrays     │
                          │  locking, aging       │
                          │  swap pair selection  │
                          └───────────────────────┘
```

**One RL step:**

1. `net.age_links()` — advance clock, resolve pending events, expire
   old links.
2. `network_to_heterodata(net)` — build the graph observation.
3. Agent selects an action using the action masks.
4. Call the chosen action (`entangle`, `swap`, or `purify`).
5. Compute reward from the result dict.

---

## 4. `repeater.py` — The Repeater Class

### 4.1 Per-Qubit Data Layout

Each `Repeater` stores its state in nine parallel NumPy arrays of length
`n_ch`, using `__slots__` to eliminate per-instance overhead.

| Array | dtype | Free value | Description |
|---|---|---|---|
| `status` | `int8` | `0` | `QUBIT_FREE=0` or `QUBIT_OCCUPIED=1` |
| `partner_repeater` | `int32` | `-1` | Remote repeater's `rid` |
| `partner_qubit` | `int32` | `-1` | Remote qubit index |
| `werner_param` | `float64` | `0.0` | Current Werner $p$ |
| `initial_werner` | `float64` | `0.0` | Werner $p_0$ at creation |
| `age` | `int32` | `0` | Steps since creation |
| `link_cutoff` | `int32` | `cutoff` | Per-link $c_{\text{eff}}$ |
| `locked` | `bool` | `False` | In-flight classical comm |

### 4.2 Two Query Layers

**Raw queries** (include locked qubits — used internally by aging,
symmetry checks, graph builder):

- `occupied_indices()`, `free_indices()`, `num_occupied()`

**Agent-facing queries** (exclude locked qubits — used by action
functions and masks):

- `available_indices()`, `has_free_qubit()`, `can_swap()`, `qubits_to(rid)`

This split ensures the agent never sees or acts on qubits that are
waiting for a classical message.

### 4.3 Swap Pair Selection

`select_swap_pair(network_positions)` is fully vectorised.
`np.triu_indices(k, k=1)` generates all $\binom{k}{2}$ pair indices
over the available qubits, then a single NumPy call evaluates the
objective (distance for `FARTHEST`, Werner product for `STRONGEST`).

### 4.4 Feature Vectors

| Method | Shape | Columns |
|---|---|---|
| `feature_vector()` | `(6,)` | `pos_x, pos_y, frac_occupied, mean_fidelity, p_gen, p_swap` |
| `qubit_features()` | `(n_ch, 6)` | `is_occupied, werner_p, fidelity, partner_rid, age_norm, is_locked` |

---

## 5. `network.py` — The Network and Its Actions

### 5.1 Constructor

```python
RepeaterNetwork(
    repeaters: list[Repeater],
    adjacency: np.ndarray,       # (N, N) symmetric
    channel_loss=0.02,
    F0=0.98,
    distance_dep_gen=True,
    rng=None,                    # np.random.Generator for reproducibility
    dt_seconds=1e-4,             # physical time per tick; 0 disables delays
)
```

Pre-computes and caches `_positions` $(N, 2)$ and `_dist_matrix` $(N, N)$.

### 5.2 Action: `entangle(r1, r2) → dict`

Instantaneous. Checks adjacency, free qubits, probabilistic generation.
On success, allocates one qubit on each side and writes link metadata
with `link_cutoff = min(cutoff_r1, cutoff_r2)`.

Returns: `{"success": bool, "fidelity": float, "reason": str}`

Possible reasons: `"not_adjacent"`, `"no_free_qubit_r1"`,
`"no_free_qubit_r2"`, `"generation_failed"`, `"ok"`.

### 5.3 Action: `swap(r) → dict`

Two-phase (deferred). The BSM outcome is determined immediately.

**On failure:** both links destroyed on the spot, no event queued, no
locks. Reason: `"swap_failed"`.

**On success:** `p_new = p1 * p2` is frozen, all four qubits are
locked, an event is pushed with `timer =
ceil(d_max / (c_fiber * dt))`. Reason: `"pending"`.

Returns: `{"success": bool, "new_fidelity": float,
"partners": (ra, rb)|None, "reason": str}`

### 5.4 Action: `purify(r1, r2) → dict`

Two-phase (deferred). Requires $\geq 2$ shared unlocked pairs between
`r1` and `r2`. The pair with the lowest Werner parameter is sacrificed;
the highest is kept.

Both success and failure are deferred because neither party knows the
measurement outcome until the classical message arrives. All four
qubits are locked for the delay duration.

Returns: `{"success": bool, "old_fidelity": float,
"new_fidelity": float, "reason": str}`

### 5.5 Action: `age_links(discard_expired=True) → dict`

The clock tick. Executes three phases in order:

1. **Age:** increment age of every occupied qubit (including locked),
   recompute Werner parameters via the decoherence model.
2. **Resolve:** decrement timers on all pending events. Events whose
   timer reaches 0 are resolved: swap events rewrite remote partners
   and free central qubits; purify events destroy the sacrifice pair
   and upgrade the kept pair (or destroy both on failure). Locks are
   cleared.
3. **Expire:** destroy links whose age $\geq$ `link_cutoff` (if
   `discard_expired=True`).

Resolution functions include **guards** for the edge case where a locked
qubit was freed by cutoff expiry before the event resolved. In that
case, remaining locks are cleaned up and the resolution is skipped.

Returns: `{"expired_count": int, "over_cutoff_count": int,
"resolved_count": int, "pending_count": int, "time_step": int}`

### 5.6 Action Masks

| Method | Shape | Semantics |
|---|---|---|
| `action_mask_entangle()` | `(N, N)` bool | True where `entangle(i,j)` is valid |
| `action_mask_swap()` | `(N,)` bool | True where `swap(i)` is valid ($\geq 2$ available qubits) |
| `action_mask_purify()` | `(N, N)` bool | True where `purify(i,j)` is valid ($\geq 2$ available shared pairs) |

All masks exclude locked qubits, so the agent cannot interact with
in-flight operations.

### 5.7 Factory Functions

```python
net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.5, cutoff=20,
                  channel_loss=0.02, F0=0.98, dt_seconds=1e-4)

net = build_grid(rows=3, cols=3, n_ch=4, spacing=50.0, ...)
```

`build_chain` creates a 1-D line. `build_grid` creates a 2-D lattice
with 4-connectivity. Both forward `**kwargs` to `RepeaterNetwork`.

---

## 6. `graph_builder.py` — Graph Observations

### 6.1 `network_to_heterodata(net, force_numpy=False)`

Converts the network state into a hierarchical heterogeneous graph.
Returns `torch_geometric.data.HeteroData` when PyG is installed, or a
`HeteroGraphDict` fallback (same key structure, NumPy arrays).

### 6.2 Graph Schema

**Node types:**

| Type | Shape | Features |
|---|---|---|
| `repeater` | `(N, 6)` | `pos_x, pos_y, frac_occupied, mean_fidelity, p_gen, p_swap` |
| `qubit` | `(N*n_ch, 6)` | `is_occupied, werner_p, fidelity, partner_rid, age_norm, is_locked` |

**Edge types:**

| Triplet | Description | Attr |
|---|---|---|
| `(repeater, adjacent, repeater)` | Physical topology (both dirs) | normalised distance |
| `(repeater, has, qubit)` | Ownership hierarchy | — |
| `(qubit, belongs_to, repeater)` | Reverse ownership | — |
| `(qubit, entangled, qubit)` | Active entanglement (both dirs) | fidelity |

**Attached masks and state:**

- `data["repeater"].swap_mask` — `(N,)` bool
- `data["entangle_mask"]` — `(2, K)` int, valid entangle pairs
- `data["purify_mask"]` — `(2, K)` int, valid purify pairs
- `data["repeater"].network_state` — `[time_step, pending_count]`

### 6.3 Global Qubit Indexing

Qubit $j$ on repeater $r$ has global index $r \cdot n_{\text{ch}} + j$.
This is implicit in the ownership edges.

---

## 7. Classical Communication Delays

The key design principle is that actions never block execution. Instead,
`swap` and `purify` use a two-phase protocol:

**Initiation** (same tick as the agent's call):
- The local quantum measurement is performed.
- The outcome (success/failure) and all computed values ($p_{\text{new}}$)
  are frozen.
- All involved qubits are **locked** (invisible to action masks and
  agent-facing queries).
- A deferred event is pushed to `pending_events` with a countdown timer.

**Resolution** (inside `age_links()`, when `timer` reaches 0):
- Swap: free central qubits, rewrite remote qubits with frozen
  $p_{\text{new}}$, clear locks.
- Purify success: destroy sacrifice pair, upgrade kept pair, clear locks.
- Purify failure: destroy both pairs (which clears locks via `free_qubit`).

This preserves the Markov property: at every tick the agent can observe
which qubits are locked (via the `is_locked` feature column) and how
many events are pending (via `network_state`), and is free to act on
unrelated parts of the network.

**Failed BSMs resolve immediately** (no event queued, no locks) because
the measurement is local and no classical communication is needed to
know it failed.

**Edge case — expiry during delay:** if a locked qubit's age exceeds its
cutoff before the event resolves, `age_links()` frees it. When the
event later resolves, guard checks detect the freed qubit and clean up
any remaining locks without corrupting state.

**Disabling delays:** set `dt_seconds=0.0`. All events get `timer=0`
and resolve on the very next `age_links()` call.

---

## 8. Runnable Examples

### 8.1 Entangle, Swap, Observe

```python
import numpy as np
from quantum_repeater_sim import build_chain, network_to_heterodata

net = build_chain(5, n_ch=4, spacing=50.0, p_gen=1.0, p_swap=1.0,
                  cutoff=20, F0=0.95, channel_loss=0.02,
                  distance_dep_gen=False, dt_seconds=0.0,
                  rng=np.random.default_rng(42))

for i in range(4):
    print(net.entangle(i, i+1))

net.swap(1); net.age_links()    # resolve
net.swap(3); net.age_links()
net.swap(2); net.age_links()

links = net.get_all_links()
print(f"End-to-end: R{int(links[0][0])}<->R{int(links[0][2])} F={links[0][4]:.4f}")
```

### 8.2 Classical Delay in Action

```python
import numpy as np
from quantum_repeater_sim import build_chain

# 100 km spacing, dt=1e-4 => 5-step delay
net = build_chain(3, n_ch=4, spacing=100.0, p_gen=1.0, p_swap=1.0,
                  cutoff=999, F0=1.0, channel_loss=0.0,
                  dt_seconds=1e-4, distance_dep_gen=False,
                  rng=np.random.default_rng(0))

net.entangle(0, 1); net.entangle(1, 2)
res = net.swap(1)
print(f"Swap initiated: {res['reason']}")          # "pending"
print(f"Locked qubits: R0={net.repeaters[0].num_locked()}, "
      f"R1={net.repeaters[1].num_locked()}, R2={net.repeaters[2].num_locked()}")

for step in range(1, 7):
    ar = net.age_links()
    print(f"  t={step}: resolved={ar['resolved_count']}, pending={ar['pending_count']}")
```

### 8.3 Purification

```python
import numpy as np
from quantum_repeater_sim import build_chain

net = build_chain(3, n_ch=6, spacing=0.0, p_gen=1.0, p_swap=1.0,
                  cutoff=999, F0=0.90, channel_loss=0.0,
                  dt_seconds=0.0,
                  rng=np.random.default_rng(0))

# Create 2 links R0-R1
net.entangle(0, 1); net.entangle(0, 1)
print(f"Before purify: {net.repeaters[0].num_occupied()} links")

res = net.purify(0, 1)
net.age_links()  # resolve
print(f"Purify success={res['success']}")
print(f"After purify: {net.repeaters[0].num_occupied()} links")
if res['success']:
    print(f"Fidelity: {res['old_fidelity']:.4f} -> {res['new_fidelity']:.4f}")
```

### 8.4 Heterogeneous Network

```python
import numpy as np
from quantum_repeater_sim import Repeater, RepeaterNetwork, SwapPolicy

rng = np.random.default_rng(0)
repeaters = [
    Repeater(rid=0, n_ch=4, position=np.array([0., 0.]),
             p_gen=0.9, p_swap=0.8, cutoff=30),
    Repeater(rid=1, n_ch=8, position=np.array([40., 0.]),  # more memory
             p_gen=0.3, p_swap=0.95, cutoff=10),             # worse gen
    Repeater(rid=2, n_ch=4, position=np.array([80., 0.]),
             p_gen=0.7, p_swap=0.5, cutoff=25),
]
adj = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float64)
net = RepeaterNetwork(repeaters, adj, channel_loss=0.01, F0=0.99,
                      dt_seconds=1e-4, rng=rng)
print(net)
```

### 8.5 RL Loop Skeleton

```python
import numpy as np
from quantum_repeater_sim import build_chain, network_to_heterodata

rng = np.random.default_rng(2024)
net = build_chain(5, n_ch=4, spacing=25.0, p_gen=0.7, p_swap=0.5,
                  cutoff=10, F0=0.95, channel_loss=0.02,
                  dt_seconds=1e-4, rng=rng)

for step in range(100):
    net.age_links()
    obs = network_to_heterodata(net)
    # obs["repeater"].x, obs["qubit"].x -> feed to GNN
    sm = net.action_mask_swap()
    em = net.action_mask_entangle()
    pm = net.action_mask_purify()

    rv = rng.random()
    if pm.any() and rv < 0.15:
        pairs = np.argwhere(np.triu(pm))
        i = rng.integers(len(pairs))
        net.purify(int(pairs[i,0]), int(pairs[i,1]))
    elif sm.any() and rv < 0.45:
        net.swap(int(rng.choice(np.flatnonzero(sm))))
    elif em.any():
        pairs = np.argwhere(np.triu(em))
        if len(pairs):
            i = rng.integers(len(pairs))
            net.entangle(int(pairs[i,0]), int(pairs[i,1]))
```

---

## 9. RL Integration Guide

### 9.1 Zero-Shot Generalisation (Train Small, Test Large)

The graph observation has **no fixed-size component** that depends on $N$
or $n_{\text{ch}}$. A GNN processes variable-size graphs natively. To
enable transfer:

- **Never flatten** the graph into a fixed vector.
- **Normalise positions** to zero mean per graph (or use only the
  relative distances in edge attributes).
- Use `mean` or `attention` aggregation in GNN layers, not `sum`.
- Keep the model shallow (2-3 message-passing layers).

### 9.2 Action Space

Score each valid action with a head that operates on node/edge
embeddings, then apply the action mask and sample:

- **Swap head:** score each repeater $s(r) = \text{MLP}(h_r)$
- **Entangle head:** score each adjacent edge
  $s(i,j) = \text{MLP}([h_i \| h_j])$
- **Purify head:** score each pair with shared links

Concatenate all scores, mask, softmax, sample. The dimensionality scales
with the graph, not with a fixed constant.

### 9.3 Training Protocol

Randomise network parameters every episode:

```python
def sample_env(rng):
    N = rng.integers(3, 12)
    net = build_chain(N, n_ch=rng.integers(2, 9),
                      spacing=rng.uniform(10, 100),
                      p_gen=rng.uniform(0.3, 1.0),
                      p_swap=rng.uniform(0.3, 1.0),
                      cutoff=rng.integers(5, 30),
                      dt_seconds=1e-4, rng=rng)
    for rep in net.repeaters:
        rep.p_gen = rng.uniform(0.1, 1.0)
        rep.p_swap = rng.uniform(0.3, 1.0)
    return net
```

### 9.4 Reward Design

The simulator returns raw dicts. A simple reward template:

```python
def reward(action_type, result):
    if action_type == "entangle":
        return 0.05 if result["success"] else -0.01
    elif action_type == "swap":
        return result["new_fidelity"] if result["success"] else -0.1
    elif action_type == "purify":
        if result["success"]:
            return result["new_fidelity"] - result["old_fidelity"]
        return -0.15
```

---

## 10. Known Limitations

### 10.1 Noise Model

Werner states model isotropic depolarising noise only. Real quantum
memories exhibit anisotropic dephasing ($T_2$ processes). Extending to
the Bell-diagonal model (4 parameters per link) would change the swap
formula from scalar multiplication to a $4 \times 4$ matrix product.

### 10.2 Entanglement Generation Delay

Generation is currently instantaneous (heralding signal absorbed into
one time-step). For long links where this is unrealistic, the same
event queue mechanism used by `swap` and `purify` can be applied to
`entangle`.

### 10.3 Swap Pair Selection Scaling

`select_swap_pair` is vectorised over all $\binom{k}{2}$ pairs. For
$n_{\text{ch}} \leq 16$ the cost is negligible. Scaling to very large
memories would benefit from approximate nearest-neighbour methods.

### 10.4 Batch Environments

The simulator is single-instance. For vectorised RL training, run $B$
instances in a `multiprocessing.Pool` or refactor state into batched
$(B, N, n_{\text{ch}}, \ldots)$ arrays.

### 10.5 Static Topology

Positions and adjacency are fixed at construction time. If the RL
application requires dynamic topology (e.g., satellite networks), the
`_positions` and `_dist_matrix` caches must be invalidated on change.

---

## Test Suite

Run all 109 checks across 30 test groups:

```
python -m quantum_repeater_sim.test_demo
```

The suite covers: repeater unit tests, entangle/swap/purify correctness
across parameter sweeps, policy optimality verification, aging and
decoherence, conditional discard, generation and swap rate statistics,
graph shape consistency, BBPSSW formula properties, locking invariants,
delay calculation, deferred swap and purify with known delays, locked
qubit invisibility to masks, concurrent events, expiry during delay
windows, purify-then-swap interaction chains, RL stress tests (72
configs x 30 steps with delays), and grid stress tests (2x2 through
4x4 with random per-repeater parameters).