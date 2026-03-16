# QNetGame

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?logo=latex&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white)

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
9. [Known Limitations](#10-known-limitations)
10. [RL stack](#10.-`rl_stack`-RL-Agent-Module)
11. [QRN test suite](#12-qrn-simulator-test-suite---test_rlpy)
12. [RL test suite](#13-rl-stack-test-suite-test_rlpy)

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
\rho(p) = p\,|\Phi^+\rangle\!\langle\Phi^+| + \frac{1-p}{4}I_4
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
\begin{Bmatrix}
e^{-\alpha\, d_{ij}/2} & \text{if distance\_dep\_gen} \\
1 & \text{otherwise}
\end{Bmatrix}
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

**On success:** `p_new = p1 * p2` is frozen, both remote qubits are
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
Returns `torch_geometric.data.HeteroData`.

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
- Swap: rewrite remote qubits with frozen
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
and resolve on the very **next** `age_links()` call (this in turn disables a "end-to-end" state in 1 step even for perfect operations).

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


---


## 9. Known Limitations

### 9.1 Noise Model

Werner states model isotropic depolarising noise only. Real quantum
memories exhibit anisotropic dephasing ($T_2$ processes). Extending to
the Bell-diagonal model (4 parameters per link) would change the swap
formula from scalar multiplication to a $4 \times 4$ matrix product.

### 9.2 Entanglement Generation Delay

Generation is currently instantaneous (heralding signal absorbed into
one time-step). For long links where this is unrealistic, the same
event queue mechanism used by `swap` and `purify` can be applied to
`entangle`.

### 9.3 Swap Pair Selection Scaling

`select_swap_pair` is vectorised over all $\binom{k}{2}$ pairs. For
$n_{\text{ch}} \leq 16$ the cost is negligible. Scaling to very large
memories would benefit from approximate nearest-neighbour methods.

### 9.4 Batch Environments

The simulator is single-instance. For vectorised RL training, run $B$
instances in a `multiprocessing.Pool` or refactor state into batched
$(B, N, n_{\text{ch}}, \ldots)$ arrays.

### 9.5 Static Topology

Positions and adjacency are fixed at construction time. If the RL
application requires dynamic topology (e.g., satellite networks), the
`_positions` and `_dist_matrix` caches must be invalidated on change.

---

## 10. `rl_stack` — RL Agent Module

A Double-DQN agent that learns multi-node routing policies on quantum
repeater chains and generalises zero-shot to larger, differently-parameterised
networks.

**Requires:** `torch`, `torch_geometric`, `matplotlib` (in addition to the
base simulator's `numpy`).

---

### 10.1 Architecture

```
                  ┌─────────────────────────────────┐
                  │        QRNAgent                 │
                  │  Double DQN + ReplayBuffer      │
                  │  .train()   .validate()         │
                  └──────────┬──────────────────────┘
                             │ selects N actions
                             ▼
                  ┌─────────────────────────────────┐
                  │         QRNEnv                  │
                  │  1. auto-entangle all pairs     │
                  │  2. execute agent actions       │
                  │  3. age_links (resolve events)  │
                  │  4. check end-to-end            │
                  └──────────┬──────────────────────┘
                             │ wraps
                             ▼
                  ┌─────────────────────────────────┐
                  │     RepeaterNetwork             │
                  │     (from base simulator)       │
                  └─────────────────────────────────┘
```

### 10.2 Step Semantics

Each environment step proceeds as:

1. **Auto-entangle:** every adjacent pair attempts entanglement generation
   (the "background" physical process).
2. **Agent actions:** the agent provides one action per node from
   `{noop=0, entangle=1, swap=2, purify=3}`. Actions are masked so
   only valid actions can be chosen.
3. **Age links:** advance the discrete clock by one tick — resolve pending
   classical communication events, apply decoherence, expire old links.
4. **Check e-e:** if the source and destination repeaters share a direct
   entangled link, the episode terminates with success.

The reward is `+1.0` on end-to-end success and `-0.1` per step.

### 10.3 Action Space

Per node, 4 discrete actions:

| Index | Name | Condition |
|---|---|---|
| 0 | `noop` | Always valid |
| 1 | `entangle` | Node and some neighbour both have free unlocked qubits |
| 2 | `swap` | Node has ≥ 2 available (occupied, unlocked) qubits |
| 3 | `purify` | Node shares ≥ 2 available pairs with some neighbour |

The agent outputs `(N,)` actions simultaneously — the system is "frozen
in time" while the agent decides, then all actions execute in one step.

## 10.4 Observation Space

**Node features** `(N, 7)`, all in [0, 1]:

| Column | Feature |
|---|---|
| 0 | `frac_occupied` - occupied qubits / n_ch |
| 1 | `mean_fidelity` - average Werner fidelity of occupied qubits |
| 2 | `is_source` - 1 if this node is the e-e source |
| 3 | `is_dest` - 1 if this node is the e-e destination |
| 4 | `frac_available` - available (unlocked) qubits / n_ch |
| 5 | `can_swap` - 1 if swap is possible |
| 6 | `has_purify_option` - 1 if purification is possible |
| 7 | `time_remaining` - Steps until episode termination

**Edges:** the repeater adjacency graph (both directions).

Because all features are normalised and topology-agnostic, the GNN processes any chain length / graph size without retraining.


### 10.5 Replay Buffer (`buffer.py`)

Circular buffer storing `(state, actions, reward, next_state, done)`:

```python
from quantum_repeater_sim.rl import ReplayBuffer

buf = ReplayBuffer(max_size=50_000)
buf.add(obs, actions, reward, next_obs, done)
batch = buf.sample(64)
```

### 10.6 GNN Model (`model.py`)

Two-layer GraphSAGE with a 2-layer MLP head:

```               
Input: (N, 7)     |<------------------------
      ↓           | System Representation
SAGEConv(7→64)    |<------------------------         
      ↓           |          GNN 
    ReLU          |         Block
      ↓           |       (Encoder)
SAGEConv(64→64)   |<------------------------
      ↓           |
    ReLU          |          MLP
      ↓           |         Block
Linear(64→64)     |        (Latent)
      ↓           |
    ReLU          |<------------------------
      ↓           |        Linear
Linear(64→4)      |       (Decoder)
      ↓           |<------------------------
(N, 4) Q-values   |    Value estimation
```

All layers are local (message-passing + per-node linear), so the model
is size-agnostic by construction.


## 11. Training / Testing
### 11.1 Training (`agent.py: QRNAgent.train`)

```python
from quantum_repeater_sim.rl import QRNAgent

agent = QRNAgent(lr=5e-4, gamma=0.99, batch_size=64, buffer_size=1e4)
metrics = agent.train(
    episodes=3000,
    max_steps=50,
    n_range=[4, 5, 6],       # train on small chains
    curriculum=True,          # progressive difficulty
    heterogeneous=True,       # randomise per-repeater params
    p_gen=0.8, p_swap=0.7,
    cutoff=15, F0=1.0,
    topolog='chain', dt_seconds=1e4
    save_path="checkpoints/",
    plot=True,
)
```

**Key training features:**

- **Curriculum learning:** starts with small chains (N=4), gradually
  introduces larger ones. Controlled by `n_range` and `curriculum=True`.
- **Domain randomisation:** `heterogeneous=True` randomises p_gen and
  p_swap per repeater each episode.
- **Epsilon schedule:** cosine annealing from 1.0 to 0.05.
- **Double DQN:** policy net selects actions, target net evaluates them.
- **Polyak averaging:** target net updated with τ=0.005 each step.
- **Gradient clipping:** max_norm=10.

### 11.2 Validation (`agent.py: QRNAgent.validate`)

```python
results = agent.validate(
    model_path="checkpoints/policy.pth",
    n_episodes=100,
    n_repeaters=10,           # test on larger chain than trained
    p_gen=0.6, p_swap=0.5,   # different params than training
    plot_actions=True,
)
```

Compares the trained agent against three heuristic baselines:

| Strategy | Description |
|---|---|
| **SwapASAP** | Swap wherever possible, entangle elsewhere |
| **PurifyThenSwap** | Purify if possible, otherwise swap, then entangle |
| **Random** | Uniform random valid action per node |

**Output:** a results table and a colour-coded action timeline:

```
======================================================================
Validation: N=10, p_gen=0.6, p_swap=0.5, cutoff=15
======================================================================
Strategy       |    Avg Steps |   Avg Fidelity | S%    | Succ
----------------------------------------------------------------------
Agent          |  12.3±4.2   | 0.6521±0.1200  | 100%  |  87%
SwapASAP       |  15.1±5.8   | 0.5102±0.0980  | 123%  |  72%
PurifySwap     |  18.4±6.1   | 0.7210±0.0850  | 150%  |  65%
Random         |  45.2±8.3   | 0.3100±0.1500  | 367%  |  12%
```

The timeline plot uses:
- Solid colours for each node's action
- `///` hatching for swaps
- `...` hatching for purifications
- Black for terminal "Done"

### 11.3 Heuristic Strategies (`strategies.py`)

Available as standalone functions:

```python
from quantum_repeater_sim.rl import strategies

actions = strategies.swap_asap(env)           # (N,) int array
actions = strategies.purify_then_swap(env)
actions = strategies.entangle_only(env)
actions = strategies.random_policy(env, rng)
```

Each function respects the current action mask and returns valid actions.


### 11.4 Zero-Shot Generalisation

The design enables training on small chains (N=4–6) and testing on larger ones (N=10-20+):

1. **Node features are normalised** — fractions and binary flags.
2. **GNN is local** — SAGEConv aggregates from 1-hop neighbours (N-independent). 
3. **Action space is per-node** — 4 actions for any topology/size.
4. **Domain randomisation** — heterogeneous p_gen/p_swap during training prevents overfitting to specific parameter regimes (experimental).
5. **Curriculum** — progressive difficulty teaches general patterns before scaling up.

## 12. QRN Simulator test suite - `test_rl.py`

2 AI generated test suites available. Human testing soon to follow...

`test_simulator.py` is a `unittest`-based test suite for the QRN simulator, covering physical correctness, core mechanics, and RL safety. It contains **67 tests across 14 test classes** organised into three sections.

### 12.3 Structure

#### 12.3.1 Physical Validation
Verifies the simulator's fidelity to quantum networking theory:
- **Werner ↔ Fidelity** round-trips and boundary values (`F = (3p+1)/4`)
- **Decoherence** exponential decay `p(t) = p₀ exp(−t/τ)` checked numerically at each tick
- **BBPSSW purification** success probability and fidelity improvement against closed-form formulas
- **Entanglement swapping** product rule `p_new = p_a × p_b` verified end-to-end in a 3-node chain
- **Classical delay** `ceil(d / c_fiber·dt)` and remote qubit locking
- **Distance-dependent generation** probability and fidelity scaling laws

#### 12.3.2 Core Functionality
Exercises standard simulation mechanics:
- Entanglement (FREE→OCCUPIED transitions, partner back-pointers, adjacency guards)
- Swapping (BSM qubit destruction, event queueing, long-range link resolution)
- Purification (fidelity upgrade on success, both pairs destroyed on failure)
- Ageing (timestep increment, per-tick decay, cutoff expiry)
- Cross-module wiring (network↔repeater↔env_wrapper references, topology builders)

### 12.4 Edge Cases & RL Loophole Tests
Catches bugs that could allow an RL agent to exploit unphysical behaviour:
- **Ghost link resolution** — remote qubit expires during classical delay; verifies clean abort with no dangling state
- **Asymmetric cutoff** — enforces `min(c₁, c₂)` as the effective link lifetime
- **Zero-distance operations** — no division-by-zero; delay is exactly 0
- **Double-booking / locking integrity** — locked qubits are invisible to swap, purify, and all action masks
- **Self-swapping** — rejected when both qubits point to the same remote repeater

### 12.5 Running
```bash
python -m pytest test_simulator.py -v
# or
python -m unittest test_simulator -v
```

### 12.6 Dependencies
`numpy`, `torch`, `torch_geometric` (only for `graph_builder` imports; all network tests are pure NumPy).

## 13. RL stack test suite `test_rl.py`

Comprehensive `unittest` suite for the Double-DQN RL stack of the Quantum Repeater Network Simulator. Contains **58 tests across 11 test classes**.

### 13.1 What Is Tested

#### 13.1.1 Architecture & Logic Validation
- **Double-DQN update rule** — verifies that the policy net selects the next action and the target net evaluates it, and that `(1 − done)` correctly zeros out future reward on terminal transitions.
- **Polyak averaging** — tests τ=0 (frozen target), τ=1 (full copy), and τ∈(0,1) (interpolation) cases of `θ_target ← τ·θ_policy + (1−τ)·θ_target`.
- **Action masking in target computation** — confirms that invalid actions are assigned −∞ *before* `argmax` in the target Q calculation, preventing the agent from learning Q-values for physically impossible states.
- **Graph batching** — verifies that `Batch.from_data_list` correctly sums node counts, assigns the `batch` index vector, and that per-graph rewards broadcast to per-node rewards without shape mismatches.

#### 13.1.2 Environment (`QRNEnv`)
- `reset()` returns a correctly shaped observation, reinitialises counters, runs auto-entanglement, and sets valid source/dest nodes.
- All 8 node features (`frac_occupied`, `mean_fidelity`, `is_source`, `is_dest`, `frac_available`, `can_swap`, `can_purify`, `time_remaining`) are checked for range, mutual exclusivity, and per-step decay.
- `step()` ordering (purify → swap → age → check e2e → auto-entangle), step cost, done-on-max-steps, and success reward.

#### 13.1.3 Agent (`QRNAgent`)
- `select_actions` never chooses a masked action under ε-greedy exploration **or** greedy exploitation.
- NOOP-only mask forces NOOP from both modes without throwing an empty-sequence error.
- Output shape and dtype are correct; greedy actions are deterministic across calls.
- `train_step` tensor shapes: `current_q` and `target_q` are 1-D with length equal to total nodes in the batch.

#### 13.1.4 Buffer (`ReplayBuffer`)
- `add`, `size`, ring-buffer rollover at `max_size`, oldest-entry overwrite.
- `sample` returns the correct batch size (or all entries if buffer is smaller).
- All transition keys (`s`, `a`, `r`, `s_`, `d`, `m_`) are present with correct shapes.
- `clear` resets both the list and the position pointer.

### 13.2 Edge Cases & RL Loopholes
| Test | Failure mode prevented |
|---|---|
| **Target-node action injection** | Agent assigns SWAP/PURIFY to source or dest; env must silently overwrite to NOOP |
| **Heterogeneous graph batching** | Curriculum mixes chain sizes 4 and 7 in one batch; tensor shapes must still align |
| **All-actions-masked fallback** | Node with zero available qubits; `argmax` on all-−∞ must not raise a runtime error |
| **NOOP column always True** | Env mask must guarantee at least one valid action per node at all times |

### 13.3 Running

```bash
# pytest (recommended)
python -m pytest test_rl_stack.py -v

# standard unittest
python -m unittest test_rl_stack -v
```

### 13.4 Dependencies

```
torch
torch_geometric
numpy
quantum_repeater_sim   # network / repeater / env_wrapper
rl_stack               # model / buffer / agent / strategies
```

### 13.5 DISCLAIMER:

The tests as written give structural and crash-safety guarantees, and the subset derived from physics formulas gives physical correctness guarantees for those specific mechanics. HOWEVER, they do not guarantee the RL agent is learning anything useful, and they cannot detect bugs that were present in the code when the tests were written.
