# Technical Note: Quantum Repeater Network Simulator

## 1. Module Overview

The `simulator` package is a discrete-time simulator for quantum repeater networks.
It models the generation, ageing, purification and swapping of bipartite entangled links
across a graph of quantum repeater nodes, with optional classical communication (CC) delays.

The simulator is designed as a backend for reinforcement learning (RL) pipelines: an external
agent observes the network state (via feature vectors or a heterogeneous graph representation),
selects per-node actions, and the simulator advances one time step.

### Module map

| File | Scope | Role |
|---|---|---|
| `repeater.py` | Intra-node | Single repeater: qubit bookkeeping, link state, swap-pair selection, two query layers |
| `network.py` | Inter-node | Network of repeaters: entanglement generation, swap, purification, ageing, event queue, topology builders, rendering |
| `snapshots.py` | Read boundary | Frozen, fidelity-domain `NodeState` / `Topology` dataclasses the RL side reads |
| `optimal_policy/` | DP baseline | Consumers of the pickled swap-only optimal policy + comparison / report helpers |
| `__init__.py` | Re-exports | Public API surface |

> There is **no** `backends/` package, the historical `PhysicsBackend` / NetSquid
> abstraction was removed. `RepeaterNetwork` is the engine, and consumers read it
> only through the immutable `snapshots.py` dataclasses.
>
> The RL observation is **not** built in this package, it lives in
> `rl_stack/` (`env_wrapper.get_observation` + `agent._obs_to_data`). See §4.

### Physical model in brief

Each entangled link is described by a single scalar: the **Werner parameter** `p` of a Werner
state `rho = p |Phi+><Phi+| + (1-p)/4 I`. The fidelity with respect to the maximally
entangled state is `F = (3p + 1)/4`. All noise channels (fibre loss, decoherence,
imperfect BSM) act through this scalar, so the simulator never constructs density matrices.

Time is discrete. Each simulation step increments a global clock. Links age deterministically;
all stochastic outcomes (generation success, BSM success, purification success) are sampled at
the moment the operation is requested. When CC delays are nonzero, the *outcome* is determined
immediately but its *effect* is deferred: the involved qubits are locked and a pending event is
placed on a queue that counts down until the classical notification arrives.


## 2. File: `repeater.py` - Intra-Node Logic

### 2.1 Module-level helpers

**`fidelity_to_werner(f)`** and **`werner_to_fidelity(p)`**

Convert between the Bell-diagonal fidelity F and the Werner parameter p of a Werner state.
Both accept scalars or numpy arrays.

```
p = (4F - 1) / 3
F = (3p + 1) / 4
```

**`bbpssw_success_prob(f1, f2)`**

Returns the probability that a BBPSSW purification round succeeds, given two input fidelities.
The formula implemented (`repeater.py:19-23`) is:

```
P_succ = (4/3) f1 f2 - (1/3)(f1 + f2) + 1/3
```

This corresponds to the bilateral CNOT purification protocol (Bennett, Brassard, Popescu,
Schumacher, Smolin, Wootters 1996) applied to two Werner states, and equals `(3 p1 p2 + 1)/4`
in Werner parameters. It yields `0.25` at `F=0.25` (fully mixed) and `1` at `F=1` (perfect).
Both input fidelities enter symmetrically. Note that the arguments are *fidelities*, not
Werner parameters.

**`bbpssw_new_fidelity(f1, f2)`**

Returns the **fidelity** of the purified state upon a successful BBPSSW round
(`repeater.py:25-27`):

```
F_new = (1 - (f1 + f2) + 10 f1 f2) / (5 - 2(f1 + f2) + 8 f1 f2)
```

Both arguments and the return value are *fidelities* (not Werner parameters). The caller
(`network.purify`) converts `F_new` back to a Werner parameter via `fidelity_to_werner`
before storing it.


### 2.2 Constants

| Name | Value | Meaning |
|---|---|---|
| `QUBIT_FREE` | `np.int8(0)` | Qubit slot is unoccupied |
| `QUBIT_OCCUPIED` | `np.int8(1)` | Qubit slot holds one half of an entangled pair |
| `NO_PARTNER` | `-1` (int) | Sentinel for "no remote partner assigned" |


### 2.3 `SwapPolicy` enum

Determines how the repeater selects which two of its occupied qubits to use when performing
an entanglement swap. Three policies are defined:

| Policy | Value | Selection rule |
|---|---|---|
| `FARTHEST` | 0 | Maximise Euclidean distance between the two remote partners |
| `STRONGEST` | 1 | Maximise the product of the two Werner parameters |
| `RANDOM` | 2 | Uniform random selection among all occupied pairs |


### 2.4 `Repeater` class

Represents a single quantum repeater node with `n_ch` qubit slots (communication channels).
Uses `__slots__` for memory efficiency.

#### 2.4.1 Attributes

**Node-level (set at construction, persist across resets of qubit state):**

- `rid` (int): Unique repeater ID, used as the index into the network's repeater list.
- `n_ch` (int): Number of qubit slots. All repeaters in a network are assumed to share the
  same `n_ch` (the supported, tested case).
- `swap_policy` (SwapPolicy): Policy for selecting swap pairs.
- `position` (ndarray, shape `(2,)`): Spatial coordinates `[x, y]` in km.
- `p_gen` (float): Probability that an elementary link generation attempt succeeds at this
  node. The network averages `p_gen` of both endpoints to obtain the effective generation
  probability for a given link.
- `p_swap` (float): Probability that a Bell-state measurement (BSM) succeeds at this node.
- `cutoff` (int): Maximum link age (in time steps) before the link is considered expired.

**Per-qubit arrays (shape `(n_ch,)`):**

- `status` (int8): `QUBIT_FREE` or `QUBIT_OCCUPIED`.
- `partner_repeater` (int32): The `rid` of the remote repeater this qubit is entangled with,
  or `NO_PARTNER` if free.
- `partner_qubit` (int32): The qubit index on the remote repeater, or `NO_PARTNER`.
- `werner_param` (float32): Current Werner parameter `p(t)` of the link held by this qubit.
  Decays over time according to the ageing model.
- `initial_werner` (float64): Werner parameter at the moment the link was established
  (`p_0`). Used as the reference value in the exponential decay model.
- `age` (int32): Number of time steps since the link was created (or last refreshed by a
  swap/purify resolution).
- `link_cutoff` (int32): Per-link effective cutoff, set to `min(cutoff_A, cutoff_B)` where
  A and B are the two endpoint repeaters. This allows heterogeneous cutoff values.
- `locked` (bool): Whether the qubit is locked due to an in-flight classical communication.
  Locked qubits cannot be used for new swaps or purifications but continue to age.
- `generation_id` (uint32): A monotonic counter incremented on every `allocate_qubit()`.
  Deferred swap/purify events record the generation of each qubit they touch, and resolution
  refuses to act on a slot whose generation no longer matches, this is the engine's defence
  against ghost links and stranded locks when a locked qubit expires and is reallocated
  during the CC delay (see §3.6).


#### 2.4.2 Query methods

Two tiers of queries exist, distinguished by whether they include locked qubits:

**Raw queries (include locked qubits, used internally):**
- `free_indices()`: Indices of all `QUBIT_FREE` slots.
- `occupied_indices()`: Indices of all `QUBIT_OCCUPIED` slots (including locked ones).
- `num_occupied()`: Count of occupied qubits.

**Network-facing queries (exclude locked qubits):**
- `available_indices()`: Occupied AND not locked. These are the qubits eligible for swap or
  purification.
- `num_available()`: Count of available qubits.
- `has_free_qubit()`: Whether at least one free, unlocked slot exists (for entanglement
  generation).
- `can_swap()`: Whether at least 2 qubits are available (the minimum for a swap).
- `qubits_to(partner_rid)`: Available qubits that are entangled with a specific remote
  repeater. Used by purification to find shared pairs.
- `num_locked()`: Count of locked qubits.


#### 2.4.3 State mutation methods

**`allocate_qubit() -> int`**

Finds the first free, unlocked qubit slot and marks it as `QUBIT_OCCUPIED`.
Returns the qubit index, or `-1` if no slot is available.
Note: this only changes the status flag. The link metadata (`partner_repeater`, `werner_param`,
etc.) must be set separately via `set_link`.

**`set_link(qubit, partner_rid, partner_qidx, p, link_age=0, effective_cutoff=None)`**

Registers a link on qubit `qubit`. Writes:
- `partner_repeater[qubit] = partner_rid`
- `partner_qubit[qubit] = partner_qidx`
- `initial_werner[qubit] = p`
- `age[qubit] = link_age`
- `link_cutoff[qubit] = effective_cutoff` (or `self.cutoff` if None)

If `link_age > 0` and the effective cutoff is positive, the current Werner parameter is set to
the time-decayed value `p * exp(-link_age / cutoff)`. Otherwise it is set to `p` directly.
This handles the case where a link is registered with a nonzero initial age (e.g. after a
deferred swap resolution where the link was created some steps ago).

Raises `ValueError` if `partner_rid == self.rid` (self-entanglement is unphysical).

**`free_qubit(qubit)`**

Resets all metadata for qubit `qubit` to the default free state: status to `QUBIT_FREE`,
partner fields to `NO_PARTNER`, Werner parameter and age to zero, lock to False, and
`link_cutoff` back to `self.cutoff`.

**`lock_qubit(qubit)` / `unlock_qubit(qubit)`**

Set or clear the `locked` flag. Locked qubits are excluded from `available_indices()` and
therefore cannot participate in new swap or purification operations. They continue to age and
can be expired.


#### 2.4.4 `age_occupied() -> ndarray`

Advances the age of all occupied qubits by one step and recomputes their Werner parameters
according to an exponential decay model:

```
p(t) = p_0 * exp(-t / m*)
```

where `p_0` is `initial_werner`, `t` is the current `age`, and `m*` is the per-link
`link_cutoff`. Division by zero is guarded: `m*` is clamped to `max(link_cutoff, 1)`.

Returns the indices of qubits whose age has reached or exceeded their `link_cutoff`. These
are candidates for expiry but are not freed here; the network layer decides when to break them.


#### 2.4.5 `select_swap_pair(network_positions, rng=None) -> Tuple[int, int] | None`

Selects which two available qubits on this repeater should be used for a swap. The selection
depends on `self.swap_policy`:

**FARTHEST**: Enumerates all C(k,2) pairs of available qubits (where k is the number of
available qubits). For each pair `(qa, qb)`, looks up the spatial positions of their respective
remote partners using `network_positions[partner_repeater[qa]]` and
`network_positions[partner_repeater[qb]]`, then computes the Euclidean distance between those
two remote positions. Selects the pair that maximises this distance. The rationale is that
swapping qubits whose partners are far apart extends the entangled link across the greatest
spatial span.

**STRONGEST**: Enumerates all C(k,2) pairs and selects the one maximising the product
`werner_param[qa] * werner_param[qb]`. Since the post-swap Werner parameter equals this
product (see Section 3.4), this policy maximises the fidelity of the resulting swapped link.

**RANDOM**: Selects two available qubits uniformly at random using the provided `rng`
generator.

Returns `None` if fewer than 2 qubits are available.


#### 2.4.6 `reset()`

**`reset()`**

Restores all per-qubit arrays to their default (free) state. Node-level attributes
(`rid`, `p_gen`, etc.) are not modified.


## 3. File: `network.py` - Inter-Node Logic

### 3.1 `RepeaterNetwork` class

Manages a collection of `Repeater` instances connected by an adjacency matrix. Provides the
four core operations (entangle, swap, purify, age) and an event queue for deferred resolution
of operations subject to classical communication delays.

#### 3.1.1 Constructor attributes

- `repeaters` (list[Repeater]): The list of repeater nodes. Indexed by `rid`.
- `N` (int): Number of repeaters (`len(repeaters)`).
- `adj` (ndarray, shape `(N, N)`, float64): Adjacency matrix. Nonzero entry `adj[i,j]`
  indicates that repeaters `i` and `j` are physically connected by a fibre link. For chain
  and grid topologies the entries are 1.0; for the GEANT topology they store the Haversine
  distance in km between the two nodes.
- `channel_loss` (float): Attenuation coefficient for distance-dependent link generation
  probability and fidelity. Units: per km (specifically, used in the exponent as
  `exp(-channel_loss * d / 2)` for generation probability and `exp(-channel_loss * d)` for
  fidelity).
- `F0` (float): Fidelity at zero distance. The generated elementary link fidelity is
  `F0 * exp(-channel_loss * d)`.
- `distance_dep_gen` (bool): Whether the generation probability depends on distance. When
  False, the generation probability is simply the average of the two endpoints' `p_gen`.
- `rng` (numpy Generator): Seeded random number generator for reproducibility.
- `time_step` (int): Global discrete clock, incremented by `age_links()`.
- `dt_seconds` (float): Duration of one simulation step in physical seconds. Used together
  with `c_fiber` to compute classical communication delays.
- `c_fiber` (float): Speed of light in fibre, fixed at 200,000 km/s.
- `pending_events` (list[dict]): Queue of deferred swap/purify resolutions.
- `_positions` (ndarray, shape `(N, 2)`): Cached array of repeater positions.
- `_dist_matrix` (ndarray, shape `(N, N)`): Cached pairwise Euclidean distance matrix,
  computed at construction from `_positions`.


#### 3.1.2 Helper methods

**`distance(r1, r2) -> float`**

Returns the cached Euclidean distance between repeaters `r1` and `r2`.

**`_classical_delay_steps(d_km) -> int`**

Computes the number of simulation steps required for a classical signal to travel `d_km`
kilometres through fibre:

```
steps = ceil(d_km / (c_fiber * dt_seconds))
```

Returns 0 if `d_km <= 0` or `dt_seconds <= 0` (the zero-delay regime, where all operations
resolve instantaneously).

**`_gen_prob(r1, r2) -> float`**

Effective elementary link generation probability between two adjacent repeaters:

```
p_e = avg(r1.p_gen, r2.p_gen) * exp(-channel_loss * d / 2)    [if distance_dep_gen]
p_e = avg(r1.p_gen, r2.p_gen)                                  [otherwise]
```

The factor of 2 in the exponent follows the convention that `channel_loss` is a two-way
attenuation coefficient and generation involves a one-way photon transmission (midpoint source
model).

**`_gen_fidelity(r1, r2) -> float`**

Fidelity of a newly generated elementary link:

```
F = F0 * exp(-channel_loss * d)
```

No factor of 2 here: the full round-trip attenuation applies to the fidelity.


### 3.2 ACTION 1: `entangle(r1, r2)` - Elementary Link Generation

Attempts to generate an entangled link between adjacent repeaters `r1` and `r2`.
This operation is instantaneous (no CC delay).

**Procedure:**
1. Check adjacency: `adj[r1, r2] != 0`.
2. Check that both repeaters have at least one free, unlocked qubit.
3. Sample generation success: draw `u ~ Uniform(0,1)`, succeed if `u <= _gen_prob(r1, r2)`.
4. On success: allocate one qubit on each repeater, compute the initial fidelity via
   `_gen_fidelity(r1, r2)`, convert to Werner parameter, compute the effective cutoff as
   `min(r1.cutoff, r2.cutoff)`, and register the link on both sides via `set_link`.

**Returns** a dict with keys `success` (bool), `fidelity` (float), `reason` (str).


### 3.3 ACTION 2: `swap(r)` - Entanglement Swapping

Performs a Bell-state measurement (BSM) at repeater `r` to extend entanglement between
the two remote partners of the selected qubit pair.

**Procedure:**
1. Check that the repeater has at least 2 available qubits (`can_swap()`).
2. Select a pair `(qa, qb)` using the repeater's `select_swap_pair` method.
3. Guard against same-partner swap: if both qubits point to the same remote repeater,
   swapping would create a self-link. Reject.
4. Sample BSM success: draw `u ~ Uniform(0,1)`, succeed if `u <= rep.p_swap`.
5. On failure: break both links immediately (no CC delay needed, since the repeater knows
   locally that the BSM failed).
6. On success:
   - Compute the post-swap Werner parameter as the product of the two input Werner
     parameters: `p_new = p_a * p_b`. This follows from the Werner state swap formula.
   - Free the two local qubits at the swapping repeater (the BSM physically consumes them).
   - Lock the two remote qubits (one on each remote partner). Clear their back-pointers to
     the now-freed local qubits to prevent stale references.
   - Compute the CC delay: `max(distance(r, ra), distance(r, rb))` converted to steps.
   - Enqueue a `"swap"` event with `timer = delay`.

**Post-swap Werner parameter derivation:**

For two Werner states with parameters `p_a` and `p_b`, entanglement swapping via a perfect
BSM produces a Werner state with parameter `p_new = p_a * p_b`. This is exact for Werner
states and follows from the depolarising channel composition.

**Returns** a dict with `success`, `new_fidelity`, `partners` (tuple of the two remote rids),
and `reason`.


### 3.4 ACTION 3: `purify(r1, r2)` - BBPSSW Purification

Applies bilateral CNOT purification (BBPSSW protocol) to two entangled pairs shared between
repeaters `r1` and `r2`.

**Pair selection:**
Among all available (occupied, unlocked) qubits on `r1` that are linked to `r2`, select
the pair with the lowest and highest Werner parameters (`argsort`). The highest-quality link
is designated "keep" and the lowest-quality link is "sacrifice". The rationale is that the
sacrifice pair provides the most information gain for the keep pair.

**Procedure:**
1. Require at least 2 shared pairs between `r1` and `r2`.
2. Select sacrifice and keep qubits on both sides.
3. Compute the BBPSSW success probability using the Werner parameters of the keep and
   sacrifice pairs (converted to fidelities internally by the BBPSSW formulas).
4. Sample the outcome immediately.
5. Lock all 4 qubits (2 on each side).
6. Compute CC delay from `distance(r1, r2)`.
7. Enqueue a `"purify"` event carrying the outcome, involved qubit indices, and (if
   successful) the new Werner parameter.

**Both success and failure are deferred** because neither side knows the outcome until the
classical measurement results are exchanged.


### 3.5 ACTION 4: `age_links(discard_expired=True)` - Clock Advance

This is the "tick" method that advances the simulation by one time step. It performs three
sub-steps in order:

**Step 1: Age all occupied qubits.**
Calls `rep.age_occupied()` on every repeater. This increments ages by 1 and recomputes
Werner parameters via the decay model `p(t) = p_0 * exp(-t / m*)`. Collects indices of
qubits that have reached their cutoff ("expired candidates").

**Step 2: Resolve pending events.**
Iterates through `pending_events`, decrements each event's `timer` by 1. Events whose timer
reaches 0 are resolved by calling `_resolve_swap` or `_resolve_purify`. Events still pending
are retained.

Events are resolved *before* expired links are broken. This ordering ensures that qubits
locked by in-flight operations get a chance to be resolved before the expiry sweep frees them.

**Step 3: Expire old links.**
If `discard_expired` is True, each expired candidate is checked: if still occupied (it may
have already been freed during event resolution), its link is broken via `_break_link`.

**Returns** a dict with `expired_count`, `over_cutoff_count`, `resolved_count`,
`pending_count`, and `time_step`.


### 3.6 Event resolution

#### `_resolve_swap(ev)`

Resolves a deferred swap event. The local qubits at the swapping repeater were already freed
at BSM time. This method rewrites the two remote qubits to point to each other, establishing
the new end-to-end link.

**Guard:** Either remote qubit may have been freed by expiry during the CC delay. If at least
one side is no longer occupied, the survivor (if any) is also freed and no link is
established.

On success: calls `set_link` on both remote qubits with `link_age=0` (the age counter
restarts for the new virtual link), the pre-computed `p_new`, and effective cutoff
`min(cutoff_a, cutoff_b)`. Then unlocks both qubits.

#### `_resolve_purify(ev)`

Resolves a deferred purification event.

**On success:**
1. Break the sacrifice pair via `_break_link`.
2. Guard: check that both "keep" qubits are still occupied (they may have expired during
   delay). If not, clean up and return.
3. Re-register the kept link with the upgraded Werner parameter `p_new` and `link_age=0`.
   Unlock both keep qubits.

**On failure:**
Break all involved links. Each break is guarded by checking occupancy first, since qubits
may have already been freed by expiry or by the sacrifice-pair break propagating through
`_break_link`.


### 3.7 `_break_link(r, qidx)`

Frees qubit `qidx` on repeater `r` and, if that qubit has a valid partner, also frees the
corresponding qubit on the remote repeater. This ensures link breakage is always bilateral.


### 3.8 Link and mask queries

**`get_all_links() -> ndarray`** (shape `(L, 6)`)

Returns all active links as rows `[r_a, q_a, r_b, q_b, fidelity, age]`, deduplicated so that
`r_a < r_b` (each physical link appears once).

**`action_mask_entangle() -> ndarray`** (shape `(N, N)`, bool)

Boolean matrix where `mask[i,j] = True` iff repeaters `i` and `j` are adjacent and both
have at least one free qubit.

**`action_mask_swap() -> ndarray`** (shape `(N,)`, bool)

Per-repeater mask: True if the repeater has at least 2 available (occupied, unlocked) qubits.

**`action_mask_purify() -> ndarray`** (shape `(N, N)`, bool)

`mask[i,j] = True` iff repeater `i` has at least 2 available qubits linked to repeater `j`.


### 3.9 Topology builders

Three factory functions create `RepeaterNetwork` instances with standard topologies. All
accept `**kw` which is forwarded to the `RepeaterNetwork` constructor (allowing
`channel_loss`, `F0`, `distance_dep_gen`, `rng`, `dt_seconds` to be set).

#### `build_chain(n_repeaters, n_ch, spacing, swap_policy, p_gen, p_swap, cutoff, **kw)`

Creates a linear chain of `n_repeaters` nodes spaced `spacing` km apart along the x-axis.
Adjacency connects each node to its immediate neighbours: `adj[i, i+1] = 1.0`.

#### `build_grid(rows, cols, n_ch, spacing, swap_policy, p_gen, p_swap, cutoff, **kw)`

Creates a `rows x cols` rectangular grid. Node `idx` is placed at position
`(col * spacing, row * spacing)` where `row, col = divmod(idx, cols)`. Adjacency connects
horizontal and vertical neighbours.

#### `build_GEANT(n_ch, swap_policy, p_gen, p_swap, cutoff, **kw)`

Creates the GEANT pan-European research network topology with 24 nodes and 37 links.
Node positions are derived from capital-city coordinates (latitude, longitude) projected onto
a 2D plane via equirectangular projection centered on the mean latitude (~50 deg N). Positions
are in km.

The adjacency matrix entries store **Haversine great-circle distances** (in km) between
connected nodes, computed by `_haversine_km`. This means `adj[i,j]` serves double duty:
nonzero indicates connectivity, and the value itself is the fibre distance.

**`_haversine_km(lat1, lon1, lat2, lon2) -> float`**

Standard Haversine formula for great-circle distance on a sphere of radius 6371 km.

### 3.10 Rendering

The `render()` method produces a publication-quality matplotlib visualization of the network
state. It draws repeater boxes, qubit circles (colored by status: white=free, blue=occupied,
red-orange=locked), adjacency edges, and entanglement arcs colored by fidelity. This method is
purely visual and performs no simulation logic.


## 4. The RL observation interface

The observation is assembled in the RL layer, **not** in the `simulator` package.
`QRNEnv.get_observation()` (`rl_stack/env_wrapper.py`) reads each node's `NodeState`
snapshot directly from the `RepeaterNetwork` engine and returns a flat, size-agnostic dict:

| Key | Shape | Content |
|---|---|---|
| `x` | `(N, 9)` | per-repeater node features (table below) |
| `edge_index` | `(2, E)` | directed adjacency from `Topology.adjacency` |

`rl_stack/agent.py::_obs_to_data` wraps this into a **homogeneous** PyTorch Geometric
`Data(x, edge_index)`, a single node type (repeaters); there are **no** qubit nodes and
**no** `HeteroData`. The GraphSAGE `QNetwork` (`rl_stack/model.py`) message-passes over it
and emits per-node Q-values for the 3 actions (NOOP / SWAP / PURIFY); validity is enforced
by a separate action mask, not by edges in the graph.

**The 9 node features** (from `get_observation`), all in `[0, 1]`:

| Index | Name | Content |
|---|---|---|
| 0 | `frac_occupied` | occupied / `n_ch` |
| 1 | `mean_fidelity` | avg F of available (unlocked) qubits, 0 if none |
| 2 | `in_endnode` | 1.0 if source **or** dest (endpoints are symmetric) |
| 3 | `frac_available` | available (unlocked occupied) / `n_ch` |
| 4 | `can_swap` | 1.0 if ≥2 available qubits to different partners (forced 0 at endpoints) |
| 5 | `can_purify` | 1.0 if ≥2 available qubits to the same partner (forced 0 at endpoints) |
| 6 | `p_gen` | per-repeater link-generation prob. (inhomogeneity signal) |
| 7 | `p_swap` | per-repeater BSM success prob. (inhomogeneity signal) |
| 8 | `link_urgency` | mean(age / link_cutoff) over occupied qubits, 0 if none; →1 near expiry |

Columns 6/7 are constant across nodes when the network is homogeneous
(`p_gen_std = p_swap_std = 0`); they carry node-quality signal only under per-repeater
inhomogeneity, produced by `build_network` (see §3.9 and `network._sample_matched_uniform`).


## 5. Simulation Loop (typical usage)

A typical RL training step proceeds as:

```
1. obs  = env.reset()                       # reset network, pick source/dest
2. mask = env.get_action_mask()             # per-node valid-action mask (N, 3)
3. actions = agent.select_actions(obs, mask)# per-node action (NOOP/SWAP/PURIFY)
4. obs, reward, done, info = env.step(actions)   # one step (see below)
5. Repeat from step 2 until done
```

`env.step(actions)` does the per-step work, talking directly to the `RepeaterNetwork`
engine: purifies first, then swaps (`net.purify` / `net.swap`), then `net.age_links()`
(advance clock, age/decohere, resolve pending events), then background entanglement, then the
end-to-end check that produces the reward.

The order within a step matters: purifications run before swaps (so swapped links benefit
from freshly purified inputs), and ageing happens after all actions (so newly created links
are not immediately aged). The full step ordering, including the termination-vs-truncation
distinction and PBRS shaping, is defined in `rl_stack/env_wrapper.py::QRNEnv.step`.
