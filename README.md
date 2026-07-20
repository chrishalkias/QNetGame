<p align="center">
  <img src="logo.svg" width="180" alt="QNetGame logo">
</p>

# QNetGame

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?logo=latex&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A discrete-time quantum repeater network simulator with classical
communication delays, designed as the environment for a Reinforcement
Learning pipeline. The core simulator is pure Python/NumPy; an optional
`torch` / `torch_geometric` stack provides a GNN-based Double-DQN agent.

The RL agent trains on small chains and **generalises zero-shot** to larger /
differently-parameterised networks, including a 24-node GÉANT topology. `QRNEnv`
drives a `RepeaterNetwork` directly and reads state only through immutable,
fidelity-domain snapshots, there is no backend-abstraction layer.

**Dependencies:** NumPy (core). `torch` + `torch_geometric` for the RL
stack.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Physics Model](#2-physics-model)
3. [Architecture Overview](#3-architecture-overview)
4. [`repeater.py`, The Repeater Class](#4-repeaterpy--the-repeater-class)
5. [`network.py`, The Network and Its Actions](#5-networkpy--the-network-and-its-actions)
6. [Physics Engine & Read Snapshots](#6-physics-engine--read-snapshots)
7. [Classical Communication Delays](#7-classical-communication-delays)
8. [Runnable Examples](#8-runnable-examples)
9. [Known Limitations](#9-known-limitations)
10. [`rl_stack`, RL Agent Module](#10-rl_stack--rl-agent-module)
11. [Training / Validation](#11-training--validation)
12. [PBRS Reward Shaping (`potential.py`)](#12-pbrs-reward-shaping-potentialpy)
13. [Optimal-Policy Benchmark](#13-optimal-policy-benchmark)
14. [Repository Layout & Runners](#14-repository-layout--runners)
15. [Test Suites](#15-test-suites)

---

## 1. Quick Start

**Low-level simulator** (pure NumPy, no torch needed):

```python
import numpy as np
from simulator import build_chain

net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.7, cutoff=15,
                  F0=0.95, channel_loss=0.02,
                  dt_seconds=1e-4,          # enables classical delays
                  rng=np.random.default_rng(42))

for step in range(100):
    net.age_links()              # tick clock, resolve events
    net.entangle(0, 1)           # or net.swap(2), net.purify(0, 1)
    # ... inspect net.get_all_links() ...
```

**RL environment** (the agent's view, node features + topology):

```python
import numpy as np
from rl_stack import QRNEnv

env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.8, p_swap=0.7,
             cutoff=15, topology='chain',
             rng=np.random.default_rng(0))

obs  = env.get_observation()          # {"x": (N,9), "edge_index": (2,E)}
mask = env.get_action_mask()          # (N, 3) bool: NOOP / SWAP / PURIFY
actions = np.zeros(env.N, dtype=int)  # one action per node
obs, reward, done, info = env.step(actions)
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
\begin{cases}
e^{-\alpha\; d_{ij}/2} & \text{if distance-dep-gen} \\
1 & \text{otherwise}
\end{cases}
$$

**Initial fidelity** of the new pair:

$$
F_0(i,j) = F_0 \cdot e^{-\alpha\; d_{ij}}
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
pair to improve another. In terms of fidelities $F_1, F_2 \in [0.25, 1]$:

$$
P_{\text{succ}} = \tfrac{4}{3}F_1 F_2 - \tfrac{1}{3}(F_1+F_2) + \tfrac{1}{3},
\qquad
F_{\text{new}} = \frac{1 - (F_1+F_2) + 10\,F_1 F_2}{5 - 2(F_1+F_2) + 8\,F_1 F_2}
$$

The pair with the lower fidelity is sacrificed. On failure, both
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
| `F0` | network | 0.95 | Zero-distance fidelity |
| `channel_loss` | network | 0.02 | Fibre attenuation (km$^{-1}$) |
| `dt_seconds` | network | $10^{-4}$ | Physical time per tick (s) |
| `distance_dep_gen` | network | True | Scale $P_{\text{gen}}$ by distance? |
| `p_gen_std` / `p_swap_std` | network | 0.0 | Per-repeater inhomogeneity spread |

Setting `dt_seconds=0.0` disables all classical delays (events resolve
on the next `age_links()` call).

---

## 3. Architecture Overview

```
  RL Agent (rl_stack) ─────────────┐
    │ observes {x, edge_index}      │ picks (N,) actions
    ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│                        QRNEnv                              │
│  get_observation() → {"x": (N,9), "edge_index": (2,E)}     │
│  get_action_mask() → (N, 3)  NOOP / SWAP / PURIFY          │
│  step(actions): purify → swap → age → e2e → auto-entangle  │
└───────────────────────────┬────────────────────────────────┘
                            │ drives
                            ▼
                ┌───────────────────────────┐
                │      RepeaterNetwork        │   build_network(...)
                │  node_state / topology /    │
                │  entangle / swap / purify / │
                │  age_links · Repeater (×N)  │
                └───────────────────────────┘
```

**One RL step (`QRNEnv.step`):**

1. Execute agent actions, **purify first, then swap** (purify-before-swap
   ensures freshly-improved links get swapped).
2. `net.age_links()`, tick the clock, resolve pending classical events,
   decohere, expire old links.
3. Check end-to-end; compute reward (step cost + PBRS shaping; success pays
   `fidelity × SUCCESS_REWARD`). Failed actions are **not** penalised
   (`FAILED_ACTION = 0.0`; see §10.2).
4. Auto-entangle every adjacent pair so the next observation is fresh.

---

## 4. `repeater.py`, The Repeater Class

### 4.1 Per-Qubit Data Layout

Each `Repeater` stores its state in parallel NumPy arrays of length
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

**Raw queries** (include locked qubits, used internally by aging,
symmetry checks, observation builder):

- `occupied_indices()`, `free_indices()`, `num_occupied()`

**Agent-facing queries** (exclude locked qubits, used by action
functions and masks):

- `available_indices()`, `has_free_qubit()`, `can_swap()`, `qubits_to(rid)`

This split ensures the agent never sees or acts on qubits that are
waiting for a classical message.

### 4.3 Swap Pair Selection

`select_swap_pair(network_positions, rng=None)` is fully vectorised.
`np.triu_indices(k, k=1)` generates all $\binom{k}{2}$ pair indices
over the available qubits, then a single NumPy call evaluates the
objective (distance for `FARTHEST`, Werner product for `STRONGEST`);
`RANDOM` draws a uniform pair from `rng`.

> Note: `Repeater` no longer builds observation features. The agent's node
> features are assembled entirely in `rl_stack/env_wrapper.py::get_observation`
> (see §10.4), reading immutable `NodeState` snapshots.

---

## 5. `network.py`, The Network and Its Actions

### 5.1 Constructor

```python
RepeaterNetwork(
    repeaters: list[Repeater],
    adjacency: np.ndarray,       # (N, N) symmetric
    channel_loss=0.02,
    F0=1.0,
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
from simulator import build_chain, build_grid   # top-level
from simulator.network import build_GEANT        # network module

net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.5, cutoff=20,
                  channel_loss=0.02, F0=0.98, dt_seconds=1e-4)

net = build_grid(rows=3, cols=3, n_ch=4, spacing=50.0, ...)

net = build_GEANT(n_ch=4, p_gen=0.8, p_swap=0.5, cutoff=20, ...)
```

- `build_chain`, 1-D line.
- `build_grid`, 2-D lattice with 4-connectivity.
- `build_GEANT`, the GÉANT pan-European research network (24 nodes,
  37 links), node positions from member-state capital lat/lon projected
  to a flat km plane.

All forward `**kwargs` to `RepeaterNetwork`.

---

## 6. Physics Engine & Read Snapshots

`QRNEnv` drives a `RepeaterNetwork` (`simulator/network.py`) directly. The
engine exposes immutable, fidelity-domain read snapshots so the agent reads
state without being able to mutate it. (There is no `PhysicsBackend` layer ,
that abstraction was removed; older docs referencing `simulator/backends/` are
stale.)

### 6.1 Read snapshots (`simulator/snapshots.py`)

| Object | Role |
|---|---|
| `RepeaterNetwork` | the engine: `node_state(i)`, `topology()`, `get_all_links()`, `entangle/swap/purify`, `age_links()` |
| `NodeState` | frozen per-node snapshot: `occupied`, `locked`, `partner_node`, `partner_qubit`, `fidelity`, `age`, `link_cutoff`, `n_ch`, `p_gen`, `p_swap` |
| `Topology` | `N`, `adjacency`, `positions`, the static graph |

`NodeState` and `Topology` are the only two snapshot types: frozen dataclasses
with read-only arrays, so the agent cannot mutate physics state through the
observation. Werner `p` is engine-internal; only fidelity `F` crosses this
boundary. Raw link rows come from `RepeaterNetwork.get_all_links()`, not a
snapshot dataclass.

### 6.2 `build_network(topology, ...)`

Builds a `RepeaterNetwork` for `topology` in `{chain, grid, geant}` (the
factory that dispatches to `build_chain` / `build_grid` / `build_GEANT` and
applies per-repeater inhomogeneity).

**Inhomogeneity:** `p_gen` / `p_swap` are per-network *means*;
`p_gen_std` / `p_swap_std` spread per-repeater values (std = 0 →
homogeneous, no RNG draw).

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
which qubits are locked and how many events are pending, and is free to
act on unrelated parts of the network.

**Failed BSMs resolve immediately** (no event queued, no locks) because
the measurement is local and no classical communication is needed to
know it failed.

**Edge case, expiry during delay:** if a locked qubit's age exceeds its
cutoff before the event resolves, `age_links()` frees it. When the
event later resolves, guard checks detect the freed qubit and clean up
any remaining locks without corrupting state.

**Disabling delays:** set `dt_seconds=0.0`. All events get `timer=0`
and resolve on the very **next** `age_links()` call (this in turn allows
an "end-to-end" state in 1 step even for perfect operations).

---

## 8. Runnable Examples

### 8.1 Entangle, Swap, Observe

```python
import numpy as np
from simulator import build_chain

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
from simulator import build_chain

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
from simulator import build_chain

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
from simulator import Repeater, RepeaterNetwork, SwapPolicy

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
arrays.

---

## 10. `rl_stack`, RL Agent Module

A Double-DQN agent (`rl_stack`) that learns multi-node routing policies
on quantum repeater networks and generalises zero-shot to larger,
differently-parameterised networks.

**Requires:** `torch`, `torch_geometric`, `matplotlib` (in addition to the
base simulator's `numpy`).

### 10.1 Step Semantics

Each `QRNEnv.step(actions)`:

1. **Agent actions:** one action per node from `{NOOP=0, SWAP=1,
   PURIFY=2}`, masked so only valid actions are chosen. Purifications
   execute before swaps. Actions at source/dest are forced to NOOP.
2. **Age links:** `net.age_links()`, resolve pending classical
   events, apply decoherence, expire old links.
3. **Check e-e:** if source and dest share a direct entangled link, the
   episode terminates successfully.
4. **Auto-entangle:** every adjacent pair attempts generation (the
   background physical process) so the next observation is fresh.

**Entanglement generation is not an agent action**, it is handled
entirely by the automatic background step.

### 10.2 Reward

| Term | Value | When |
|---|---|---|
| `STEP_COST` | `-0.01` | every non-success step |
| `SUCCESS_REWARD` | `+1.0` × end-to-end fidelity | on connection |
| `FAILED_ACTION` | `0.0` (disabled) | see note below |
| PBRS shaping | $\gamma\Phi(s') - \Phi(s)$ | every step (see §12) |

`FAILED_ACTION` is **deliberately zero**. It was previously `-0.05` and caused a
**swap-shy** policy: the penalty fired on *stochastic* BSM failures
($\text{rng} > p_{\text{swap}}$), punishing the agent for the environment's coin
flip on its only productive action, so the policy collapsed to NOOP. Masking
already blocks invalid actions and lost links are costed via PBRS, so the penalty
was double-counting. Do not reintroduce it.

The potential $\Phi$ is a topology-general path-progress signal in
$[0,1]$; by PBRS convention $\Phi(s_{\text{terminal}}) = 0$, so the
optimal policy is unchanged by the shaping.

### 10.3 Action Space

Per node, 3 discrete actions:

| Index | Name | Condition |
|---|---|---|
| 0 | `NOOP` | Always valid |
| 1 | `SWAP` | $\geq 2$ available qubits to **distinct** partners |
| 2 | `PURIFY` | $\geq 2$ available qubits to the **same** partner |

The agent outputs `(N,)` actions simultaneously, the system is "frozen
in time" while the agent decides, then all actions execute in one step.

### 10.4 Observation Space

`get_observation()` returns `{"x": (N, 9) float32, "edge_index": (2, E)
int64}`, a **homogeneous** graph.

**Node features** `(N, 9)`:

| Col | Feature |
|---|---|
| 0 | `frac_occupied`, occupied / n_ch |
| 1 | `mean_fidelity`, avg F of available (unlocked) qubits (0 if none) |
| 2 | `in_endnode`, 1.0 if source OR dest (endpoints are symmetric) |
| 3 | `frac_available`, available (unlocked occupied) / n_ch |
| 4 | `can_swap`, 1.0 if $\geq 2$ available qubits to distinct partners |
| 5 | `can_purify`, 1.0 if $\geq 2$ available qubits to same partner |
| 6 | `p_gen`, per-repeater generation prob. (inhomogeneity signal) |
| 7 | `p_swap`, per-repeater BSM prob. (inhomogeneity signal) |
| 8 | `link_urgency`, mean(age / link_cutoff) over occupied qubits (0 if none) |

Columns 4/5 are forced to 0 for source/dest. Columns 6/7 are constant
across nodes when the network is homogeneous (std = 0). Column 8 is 0
for nodes with no occupied qubits and approaches 1 as links near expiry.
`edge_index` is the repeater adjacency (both directions).

Because all features are normalised and topology-agnostic, the GNN
processes any chain length / graph size without retraining.

### 10.5 Replay Buffer (`buffer.py`)

Circular buffer storing `(state, actions, reward, next_state, done,
next_mask)`. The stored **next-state action mask** is what lets the DQN target
avoid selecting physically-impossible successor actions (§11):

```python
from rl_stack import ReplayBuffer

buf = ReplayBuffer(max_size=50_000)
buf.add(obs, actions, reward, next_obs, done, next_mask)
batch = buf.sample(64)
```

### 10.6 GNN Model (`model.py`)

`QNetwork`: three `SAGEConv` layers + a 2-layer MLP head.

```
Input: (N, 9)
      ↓
SAGEConv(9→64) → ReLU
      ↓
SAGEConv(64→64) → ReLU         GNN encoder
      ↓
SAGEConv(64→64) → ReLU
      ↓
Linear(64→64)   → ReLU         MLP head
      ↓
Linear(64→3)
      ↓
(N, 3) Q-values
```

All layers are local (message-passing + per-node linear), so the model
is size-agnostic by construction. `load_qnet(path)` infers `(node_dim, hidden)`
from the checkpoint's `conv1.lin_l.weight`, so you never have to specify the
architecture (all real checkpoints use `hidden=64`; the class default is 32).

---

## 11. Training / Validation

### 11.1 Training (`QRNAgent.train`)

```python
from rl_stack import QRNAgent

agent = QRNAgent(lr=5e-4, gamma=0.99, batch_size=64, buffer_size=1e4)
metrics = agent.train(
    episodes=3000,
    max_steps=50,
    n_range=[4, 5, 6, 7],     # train on small chains
    curriculum=True,           # progressive difficulty
    curriculum_frac=0.5,       # widen the size pool over the first half
    p_gen=0.8, p_swap=0.7,
    p_gen_std=0.0, p_swap_std=0.0,   # per-repeater inhomogeneity
    cutoff=30, F0=0.95,
    topology='chain',
    dt_seconds=1e-3,
    disable_actions=(),        # e.g. (PURIFY,) → pure swap-scheduler
    save_path="checkpoints/",
    plot=True,
)
```

**Key training features:**

- **Curriculum learning:** linearly widens the eligible chain size to the
  full `n_range` over the first `curriculum_frac` of training.
- **Domain randomisation:** `p_gen_std` / `p_swap_std` spread per-repeater
  parameters each episode.
- **Best-checkpointing:** `policy.pth` holds the *best* agent (judged by a
  held-out greedy probe via `eval_fn`, else rolling-mean reward in the
  settled late window); final weights go to `policy_final.pth`.
- **Early stopping:** with `eval_fn` + `eval_every` + `eval_patience`,
  training stops when the probe stops improving.
- **Winnability pruning:** `--prune_unwinnable` resamples each episode's cell
  until swap-asap can deliver it (`WinnabilityCache`), so no episodes are wasted
  on unsolvable configs.
- **Double DQN + Polyak averaging (τ=0.005) + gradient clipping (max-norm 10).**
- **`disable_actions`:** mask action indices in both selection and the
  Double-DQN target (e.g. ablate PURIFY).
- **`compare=True`:** log greedy-agent vs swap-asap vs random returns,
  steps and success each episode (+ a 3-panel `training_compare.png`).

### 11.2 Validation (`QRNAgent.validate`)

```python
results = agent.validate(
    model_path="checkpoints/policy.pth",
    n_episodes=100,
    n_repeaters=10,           # test on larger chain than trained
    p_gen=0.6, p_swap=0.5,   # different params than training
    topology='chain',
    plot_actions=True,
)
```

Compares the trained agent against heuristic baselines:

| Strategy | Description |
|---|---|
| **SwapASAP** | Swap wherever possible |
| **BeliefProp** | Swap scheduler using exact tree reachability messages on chains and loopy fallback on cyclic topologies |
| **FidGatedSwap** | Swap only above a fidelity threshold, else hold |
| **PurifySwap** | Purify if possible, otherwise swap |
| **Random** | Uniform random valid action per node |

**Output:** a results table and a colour-coded action timeline.

### 11.3 Heuristic Strategies (`strategies.py`)

Available as standalone functions, each respecting the current action
mask and returning valid `(N,)` int arrays:

```python
from rl_stack import strategies

actions = strategies.swap_asap(env)
actions = strategies.belief_propagation_policy(env)
actions = strategies.purify_then_swap(env)
actions = strategies.fidelity_gated_swap(env, f_threshold=0.5)
actions = strategies.random_policy(env, rng)
```

### 11.4 Zero-Shot Generalisation

Train on small chains (N=4–7), test on larger ones (N=10–20+) or other
topologies:

1. **Node features are normalised**, fractions and binary flags.
2. **GNN is local**, `SAGEConv` aggregates from 1-hop neighbours
   (N-independent).
3. **Action space is per-node**, 3 actions for any topology/size.
4. **Domain randomisation**, heterogeneous `p_gen`/`p_swap` prevents
   overfitting to a parameter regime.
5. **Curriculum**, progressive difficulty teaches general patterns
   before scaling up.

---

## 12. PBRS Reward Shaping (`potential.py`)

Potential-Based Reward Shaping keeps the optimal policy invariant while
giving the agent a dense progress signal toward an end-to-end link.

| Function | Role |
|---|---|
| `bfs_hops(adjacency, start)` | hop-distance from `start` to every node |
| `path_progress(d_src, d_dst, d_total, edges)` | potential $\Phi \in [0,1]$: how far the current entanglement graph has stitched a source→dest path |

`QRNEnv` precomputes `bfs_hops` from source and dest, then each step adds
$\gamma\Phi(s') - \Phi(s)$ to the reward (with $\Phi=0$ at the terminal
state). The potential is topology-general (works on chain / grid /
GÉANT).

---

## 13. Optimal-Policy Benchmark (retired)

The exact-DP swap-only optimum and its consumers were retired on 2026-07-18
(code + pickles + heatmap results now under `.local/legacy/optimal_dp/`,
recoverable from git history). The canonical delivery-time evaluator
`mc_eval` survives in `experiments/mc_eval.py`.

---

## 14. Repository Layout & Runners

```
src/
  simulator/                 core simulator (NumPy only, imports without torch)
    repeater.py, network.py    engine + build_network()
    snapshots.py               frozen NodeState / Topology (F-domain read boundary)
  rl_stack/                  Double-DQN RL stack (torch imports guarded)
    env_wrapper.py  agent.py  model.py  buffer.py  strategies.py  potential.py
    winnability.py             swap-asap winnability oracle (training cell pruning)
experiments/                 entry-point scripts (argparse at top of file)
  mc_eval.py                 THE canonical censored delivery-time evaluator
  training/                  train.py  validation.py  batch_validate.py  replot.py
  comparisons/               paper figure suite (delivery_vs_* , _common.py)
  q_heuristic/               stochastic q-heuristic control experiment
  policy_probes/             feature_importance.py  decision_map.py  _collect.py
  scripts/
    local/                     local train.sh / test.sh
    SLURM/                     cluster submit_*.sh
    sync/                      upload.sh (code up) / download.sh (artifacts down)
tests/                       pytest suites (see §15)
```

Runner scripts in `experiments/` put their `argparse` at the top of the
file so the available flags are visible at a glance; they are invoked (with
`PYTHONPATH=src:.`) by the `scripts/` shell wrappers.

---

## 15. Test Suites

`pytest` suites live in `tests/`.

| File | Covers |
|---|---|
| `test_simulator.py` | physics: Werner↔fidelity, decoherence, BBPSSW, swap product rule, classical (CC) delay timing, distance scaling; RL-loophole edge cases (ghost links, asymmetric cutoff, locking integrity, self-swap) |
| `test_rl_stack.py` | Double-DQN update rule, Polyak averaging, masked-target argmax, done-mask, graph batching + reward broadcast, `QRNEnv` reset/step/features, `QRNAgent.select_actions`, `ReplayBuffer` |
| `test_backends.py` | `RepeaterNetwork` read snapshots + `build_network`, inhomogeneity sampling (incl. `std=0` RNG-stream neutrality), frozen/read-only `NodeState`/`Topology`, urgency feature |
| `test_potential.py` | `bfs_hops` and `path_progress` PBRS potential (incl. the adjacent-blob anti-exploit case) |
| `test_game.py` | curriculum pool, rate/cutoff/n_ch draws, best/final checkpointing, early stopping, run-config manifest |
| `test_winnability.py` | the swap-asap winnability oracle |

### Running

```bash
# all tests
PYTHONPATH=src:. python -m pytest tests -v

# a single suite
PYTHONPATH=src:. python -m pytest tests/test_simulator.py -v
```

### Dependencies

`numpy` (core); `torch`, `torch_geometric` (RL-stack tests).

### Disclaimer

The tests give structural and crash-safety guarantees, and the subset
derived from physics formulas gives physical-correctness guarantees for
those specific mechanics. They do **not** guarantee the RL agent is
learning anything useful, nor can they detect bugs that were present in
the code when the tests were written.
