# Technical Note: Quantum Repeater Network Simulator

## 1. Module Overview

The `simulator` package is a discrete-time simulator for quantum repeater chains.
It models the generation, ageing, purification and swapping of bipartite entangled links
across a chain of quantum repeater nodes.

The simulator is designed as a backend for reinforcement learning (RL) pipelines: an external
agent observes the network state, one node decides, and the simulator applies that decision
immediately.

### Module map

| File | Scope | Role |
|---|---|---|
| `repeater.py` | Intra-node | Single repeater: qubit bookkeeping, link state, swap-pair selection, legality gates (`can_swap` / `can_purify`), ONE query layer |
| `network.py` | Inter-node | Chain of repeaters: entanglement generation, swap, purification, ageing, `build_chain`/`build_network` |
| `__init__.py` | Re-exports | Public API surface |

> The exact-DP swap-only optimal-policy benchmark (formerly `simulator/optimal_policy/`) was
> retired 2026-07-18; recoverable from git history if needed.
>
> **Chain-only**: `build_grid`/`build_GEANT` and the `'grid'`/`'geant'` topology strings were
> removed in the chain-only refactor. The `topology` argument itself was removed from
> `build_network` on 2026-07-27, so a stale caller now gets a `TypeError` naming
> `build_network`.

> There is **no** `backends/` package, the historical `PhysicsBackend` / NetSquid
> abstraction was removed. `RepeaterNetwork` is the engine.
>
> `snapshots.py` was **deleted 2026-07-27**. It froze six arrays per call, roughly `2N` times
> per micro-step, for 38% of env runtime. The read boundary is now `net.node(i)`, which
> returns the **live** `Repeater` and is read-only by convention. The `NodeState` and
> `Topology` dataclasses are gone.
>
> The RL observation is **not** built in this package, it lives in
> `rl_stack/` (`env_wrapper.get_observation` + `agent._obs_to_data`). See §4.

### Physical model in brief

Each entangled link is described by a single scalar: the **Werner parameter** `p` of a Werner
state `rho = p |Phi+><Phi+| + (1-p)/4 I`. The fidelity with respect to the maximally
entangled state is `F = (3p + 1)/4`. All noise channels (fibre loss, decoherence,
imperfect BSM) act through this scalar, so the simulator never constructs density matrices.

Time is discrete. Each tick increments a global clock. Links age deterministically; all
stochastic outcomes (generation success, BSM success, purification success) are sampled at
the moment the operation is requested, and **applied immediately**, against the live state.
There is no lock, no pending-event queue and no deferred resolution: the classical-
communication delay model was removed from the engine on 2026-07-22, and the synchronous
apply barrier on 2026-07-25 (the sequential-sweep rewrite). Recover either from git history
if multi-tick CC is ever needed again.


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
The canonical formula implemented (Bennett et al., PRL 76, 722 (1996)) is:

```
P_succ = (8 f1 f2 - 2(f1 + f2) + 5) / 9
```

which equals `(p1 p2 + 1)/2` in Werner parameters and is exactly 1/9 of the denominator of
`bbpssw_new_fidelity` below (both derive from the same BBPSSW density matrix, so the two are
self-consistent). Anchors: `5/9` at `F1=F2=0.5`, `1` at `F1=F2=1`. Both input fidelities enter
symmetrically; arguments and return value are *fidelities*, not Werner parameters.

> Fixed 2026-07-10: the previously implemented `(4/3)f1f2 - (1/3)(f1+f2) + 1/3` under-counted
> success probability by 24-50% across the working range. Any results/checkpoints predating
> that fix used the wrong rate.

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

Represents a single quantum repeater node with two fixed **ports**: `n_left` LEFT-facing
qubits (indices `[0, n_left)`, entangle only with a lower-index neighbour) and `n_right`
RIGHT-facing qubits (indices `[n_left, n_left+n_right)`, higher-index neighbour). `n_ch` counts
qubits **per side**, so an interior chain node holds `2*n_ch` qubits and an end node `n_ch`
(the other port width is 0). A swap fuses one LEFT link with one RIGHT link.
Uses `__slots__` for memory efficiency.

`Repeater.__deepcopy__` overrides the generic recursive `copy.deepcopy` with flat
`ndarray.copy()` calls (config fields are immutable and shared by reference). It has no
consumer in the current tree, it is kept as a cheap clone for anyone who needs one.

#### 2.4.1 Attributes

**Node-level (set at construction, persist across resets of qubit state):**

- `rid` (int): Unique repeater ID, used as the index into the network's repeater list.
- `n_ch` (int): Number of qubit slots **per side**. All repeaters share the same `n_ch` (the
  supported, tested case).
- `n_left` / `n_right` (int): LEFT / RIGHT port widths (total qubits = `n_left + n_right`).
  `build_chain` sets interior nodes to `n_left = n_right = n_ch` and end nodes to a single port.
- `swap_policy` (SwapPolicy): Policy for selecting swap pairs.
- `position` (ndarray, shape `(2,)`): Spatial coordinates `[x, y]` in km.
- `p_gen` (float): Probability that an elementary link generation attempt succeeds at this
  node. The network averages `p_gen` of both endpoints to obtain the effective generation
  probability for a given link.
- `p_swap` (float): Probability that a Bell-state measurement (BSM) succeeds at this node.
- `cutoff` (int): Maximum link age (in time steps) before the link is considered expired.

**Per-qubit arrays (shape `(n_left + n_right,)`):**

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

The `locked` and `generation_id` arrays are **gone**. They existed only to hide qubits held
by an in-flight classical-communication event; with CC removed (2026-07-22) and apply made
immediate (2026-07-25) there is no in-flight state to hide.


#### 2.4.2 Query methods

There is now **one** query layer, not two. The old raw-versus-agent-facing split existed
only to mask locked qubits; under immediate apply there is no locking, so
`available_indices()` simply aliases `occupied_indices()` and every caller sees the same,
exact view. A node's action mask is therefore correct at the instant it acts, because every
earlier node's action this tick has already landed.

- `occupied_indices()` / `available_indices()`: indices of all `QUBIT_OCCUPIED` slots.
- `num_occupied()`: count of occupied qubits.
- `available_on_side(side)`: occupied qubits on the LEFT or RIGHT port.
- `has_free_qubit(side=None)`: whether a free slot exists (optionally on one side), for
  entanglement generation.
- `has_link_each_side()`: whether the node holds at least one link on each port.
- `qubits_to(partner_rid)`: occupied qubits entangled with a specific remote repeater. Used
  by purification to find shared pairs.

**Legality gates** (moved onto the node 2026-07-27, so `env.action_mask` cannot drift from
the engine):
- `can_swap()`: true iff a VIABLE pair exists, one occupied LEFT link (partner `<` rid) and
  one occupied RIGHT link (partner `>` rid) whose fused link survives its first tick
  boundary. This is observation feature [1] and the SWAP bit of the action mask.
- `can_purify()`: true iff >=2 occupied qubits point at the SAME partner, the BBPSSW
  precondition. This is feature [2] and the PURIFY bit.
- `_pair_survives_tick(qa, qb, ec)`: **the** viability gate, `age_a + age_b + 1 < ec`, and
  the single place the `+ 1` lives. `can_swap` and `select_swap_pair` both call it, so the
  mask and the engine can never disagree.


#### 2.4.3 State mutation methods

**`allocate_qubit(side) -> int`**

Finds the first free qubit slot **on the given port** (LEFT or RIGHT) and marks it as
`QUBIT_OCCUPIED`. Returns the qubit index, or `-1` if no slot is available on that side.
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
This is what carries the sum-ages semantics: a fused link is registered with
`link_age = age_a + age_b` on the baseline product, never at `age = 0` (§3.6).

Raises `ValueError` if `partner_rid == self.rid` (self-entanglement is unphysical).

**`free_qubit(qubit)`**

Resets all metadata for qubit `qubit` to the default free state: status to `QUBIT_FREE`,
partner fields to `NO_PARTNER`, Werner parameter and age to zero, and `link_cutoff` back to
`self.cutoff`.

There are no `lock_qubit` / `unlock_qubit` methods: locking existed only for the removed
classical-communication delays.


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


#### 2.4.5 `select_swap_pair(network_positions, network_cutoffs, rng=None) -> Tuple[int, int] | None`

Selects which two occupied qubits on this repeater should be used for a swap, one from each
port, among the **viable** pairs only.

**Viability gate (2026-07-12 cutoff-invariant fix)**: for candidate pair `(qa, qb)` with remote
endpoints `ra, rb`, the fused link is created **immediately**, inheriting `age_a + age_b`,
and then ages exactly once at the tick boundary, so the pair is only offered if
`age_a + age_b + 1 < min(network_cutoffs[ra], network_cutoffs[rb])`. This closes a leak where
over-age pairs were swapped and then resolved past their cutoff anyway (pre-fix repro on
swap-asap, N=10: 49% of deliveries were over-age links). Returns `None` if no LEFT/RIGHT
pair is viable.

The constant was `+ 2` until 2026-07-26, inherited from the synchronous-barrier model in
which both parents aged once more before the swap resolved. Under immediate apply that extra
tick does not exist, and the over-strict gate overstated delivery time by up to 25% under
memory pressure. The expression lives in exactly one place,
`Repeater._pair_survives_tick`, which `can_swap` (§2.4.2) also calls.

Among the viable pairs, the choice depends on `self.swap_policy`:

**FARTHEST**: looks up the spatial positions of each pair's remote partners via
`network_positions[partner_repeater[...]]` and selects the pair maximising the Euclidean
distance between those two remote positions — swapping qubits whose partners are far apart
extends the entangled link across the greatest spatial span.

**STRONGEST**: selects the pair maximising the product `werner_param[qa] * werner_param[qb]`.
Since the post-swap Werner parameter equals this product (see §3.3), this policy maximises the
fidelity of the resulting swapped link.

**RANDOM**: selects uniformly at random among the viable pairs using the provided `rng`.

Decision and application are now the same instant, so "viable at resolution" collapses to
"viable now": there is no separate resolution-time guard left to hold.


#### 2.4.6 `reset()`

**`reset()`**

Restores all per-qubit arrays to their default (free) state. Node-level attributes
(`rid`, `p_gen`, etc.) are not modified.


## 3. File: `network.py` - Inter-Node Logic

### 3.1 `RepeaterNetwork` class

Manages a collection of `Repeater` instances connected by an adjacency matrix. Provides the
four core operations (entangle, swap, purify, age). Swap and purify apply their outcome
**immediately**; there is no event queue and no deferred resolution.

#### 3.1.1 Constructor attributes

- `repeaters` (list[Repeater]): The list of repeater nodes. Indexed by `rid`.
- `N` (int): Number of repeaters (`len(repeaters)`).
- `adj` (ndarray, shape `(N, N)`, float64): Adjacency matrix. Nonzero entry `adj[i,j]`
  indicates that repeaters `i` and `j` are physically connected by a fibre link; `build_chain`
  sets these to 1.0 (chain-only, §3.9).
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
- `_positions` (ndarray, shape `(N, 2)`): Cached array of repeater positions.
- `_dist_matrix` (ndarray, shape `(N, N)`): Cached pairwise Euclidean distance matrix,
  computed at construction from `_positions`.
- `_cutoffs` (ndarray, shape `(N,)`, int64): Per-repeater cutoff, rid-indexed. Fed to
  `Repeater.select_swap_pair` as the swap viability gate (§2.4.5).


#### 3.1.2 Helper methods

**`distance(r1, r2) -> float`**

Returns the cached Euclidean distance between repeaters `r1` and `r2`.

**`node(node) -> Repeater`**

The read boundary the RL side uses. Returns the **live** `Repeater` object, not a copy and
not a frozen snapshot: `snapshots.py` was deleted 2026-07-27 because freezing six arrays per
call, about `2N` times per micro-step, was 38% of env runtime. Callers treat it as
**read-only by convention**; mutating it mutates the engine.

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

**Procedure:**
1. Check adjacency: `adj[r1, r2] != 0`.
2. Check that both repeaters have a free qubit **on the port facing the partner**.
3. Sample generation success: draw `u ~ Uniform(0,1)`, succeed if `u <= _gen_prob(r1, r2)`.
4. On success: allocate one qubit on each repeater, compute the initial fidelity via
   `_gen_fidelity(r1, r2)`, convert to Werner parameter, compute the effective cutoff as
   `min(r1.cutoff, r2.cutoff)`, and register the link on both sides via `set_link`.

**Returns** a dict with keys `success` (bool), `fidelity` (float), `reason` (str).


### 3.3 ACTION 2: `swap(r)` - Entanglement Swapping

Performs a Bell-state measurement (BSM) at repeater `r` to extend entanglement between
the two remote partners of the selected qubit pair.

**Procedure:**
1. Structural precheck only: `has_link_each_side()`. Deliberately not `can_swap()`, which
   would also apply the viability gate and make a born-dead pair report
   `insufficient_qubits` instead of `no_valid_pair`.
2. Select a pair `(qa, qb)` via `select_swap_pair` (§2.4.5), already filtered to viable
   pairs only (`no_valid_pair` if the precheck passed but no pair survives the gate).
3. Guards, rejected before sampling: `orphan_qubit` if either qubit's `partner_repeater` is
   `NO_PARTNER` (would silently index `repeaters[-1]`); `same_partner` if both qubits point
   at the same remote repeater (would create a self-link).
4. Sample BSM success: draw `u ~ Uniform(0,1)`, succeed if `u <= rep.p_swap`.
5. On failure: break both links immediately.
6. On success, all of it **immediately**, no deferral:
   - Free the two local qubits at the swapping repeater (the BSM physically consumes them,
     and the slots are available for reuse right away).
   - Create the fused remote link on both remote partners via `set_link`, with effective
     cutoff `min(cutoff_a, cutoff_b)`, **sum-ages** inheritance
     `link_age = age_a + age_b`, and baseline `p_0 = initial_werner_a * initial_werner_b`.
     Evaluating the decay at the summed age reproduces the product of the two
     already-decohered values, `w_a * w_b`, without double-counting the pre-swap
     decoherence.

Because apply is immediate, the fused link is visible to every later node in the same
left-to-right sweep, which is what makes an intra-tick swap cascade possible.

**Post-swap Werner parameter derivation:**

For two Werner states with parameters `p_a` and `p_b`, entanglement swapping via a perfect
BSM produces a Werner state with parameter `p_new = p_a * p_b`. This is exact for Werner
states and follows from the depolarising channel composition.

**Returns** a dict with `success`, `new_fidelity`, `partners` (tuple of the two remote rids),
and `reason`.


### 3.4 ACTION 3: `purify(r1, r2)` - BBPSSW distillation cascade

Applies a **sorted-adjacent distillation cascade** (multiplexed BBPSSW, arXiv 2401.13168) to
ALL entangled pairs shared between repeaters `r1` and `r2`, not a single best/worst round.

**Cascade (sequential accumulate):**
Sort the shared links ascending by Werner parameter (== ascending fidelity) and fold up the
list carrying a running survivor. For each next link, purify it with the survivor; on success
the survivor improves, on BBPSSW failure both inputs are destroyed and the cascade restarts
from the next link. It ends with one survivor or none. (Pairing sorted-adjacent links keeps
the two inputs at the closest available fidelities, which is where BBPSSW works best.)

**Beneficial-purify guard (Fig. 11 there):** a pair is only put through the *stochastic*
BBPSSW when the twirled output beats the better input, `F_new(F1,F2) > max(F1,F2)`. Otherwise
the coin flip is skipped and the stronger link is kept (purification must strictly beat
discarding the weak link); this branch consumes no RNG.

**Procedure:**
1. Require at least 2 shared pairs between `r1` and `r2` (`insufficient_shared_pairs`).
2. Sort the shared links ascending by Werner parameter.
3. Run the guarded cascade against the CURRENT state, all RNG drawn here, breaking each
   sacrificed or failed link as it happens.
4. Register the outcome **immediately**, before any `age_links` call.

A survivor that was only kept (never purified) retains its original registration. A purified
survivor is re-registered via the Eq.(4) proxy of arXiv 2401.13168: its fidelity is expressed
as an equivalent age on a fresh `p_0 = 1` baseline, `age' = ceil(-ec * ln(p_new))`. Because
apply is immediate, nothing has accrued since the decision, so the old
accrued-since-decision term is always 0 now. If `age' >= ec` the survivor is already below
the cutoff fidelity floor and is **discarded** (`purify_discarded_over_cutoff`) rather than
creating a link expiry could never police.

One PURIFY *action* runs this cascade on every partner with which the node shares >=2 links
(env `_exec_purify`).


### 3.5 ACTION 4: `age_links(discard_expired=True)` - Clock Advance

This is the "tick" method that advances the simulation by one time step. Since the
2026-07-25 sequential-sweep rewrite it is **only the clock**: there is nothing queued left
to resolve, because swap and purify already applied their outcome when the node acted. Two
sub-steps, in order:

**Step 1: Age all occupied qubits.**
Calls `rep.age_occupied()` on every repeater. This increments ages by 1 and recomputes
Werner parameters via the decay model `p(t) = p_0 * exp(-t / m*)`. Collects indices of
qubits that have reached their cutoff ("expired candidates").

**Step 2: Expire old links.**
If `discard_expired` is True, each expired candidate is checked: if still occupied, its link
is broken via `_break_link`.

**Returns** a dict with `expired_count`, `over_cutoff_count` and `time_step`.


### 3.6 Sum-ages resolution semantics

`_resolve_swap` and `_resolve_purify` were **deleted** with the pending-event queue
(2026-07-25). Their VALUE semantics survive unchanged inside `swap()` and `purify()`; only
the moment of application moved, from the next tick boundary to the instant the node acts.

**Swap (sum-ages).** The fused Werner value must equal the product of the two parents'
*already-decohered* values, `w_a * w_b`, not a fresh `p_new` planted at `age = 0` (that
would double-count the pre-swap decoherence, and is what the project's docs from before the
2026-07-10 / 07-12 physics fixes describe). This is reproduced by storing the **baseline
product** `p0_a * p0_b` (`initial_werner` of each side) as the new `initial_werner`, with
`age = age_a + age_b` and `effective_cutoff = min(cutoff_a, cutoff_b)`: since
`p0_a*p0_b*exp(-(age_a+age_b)/tau) = (p0_a*e^{-age_a/tau})(p0_b*e^{-age_b/tau}) = w_a*w_b`
for a shared `tau`, this is exact for homogeneous per-link cutoffs (the only regime in use)
and an approximation only if the two links carried different cutoffs.

**Purify (Eq.(4) proxy).** Rather than reset the kept link's age to 0 (the pre-2026-07-12
behaviour, which double-counted decoherence via sum-of-endpoint-ages bookkeeping), the
purified fidelity `p_new` is represented as an equivalent age on a fresh `p0 = 1` baseline:
`m_equiv = ceil(-cutoff * ln(p_new))`. Ticks accrued since the decision are always 0 now,
so `new_age = m_equiv`. If `new_age >= cutoff` the purified state is below the fidelity
floor the cutoff exists to guarantee, so it is discarded (`_break_link`) rather than
creating a link expiry can never police.

The generation-ID liveness guards, the collapsed-endpoint guard and the born-dead guard at
resolution time are all gone with the queue: with no delay between decision and application
there is no window in which a qubit can be expired and reallocated underneath an in-flight
operation.


### 3.7 `_break_link(r, qidx)`

Frees qubit `qidx` on repeater `r` and, if that qubit has a valid partner, also frees the
corresponding qubit on the remote repeater. This ensures link breakage is always bilateral.


### 3.8 Link queries

**`get_all_links() -> ndarray`** (shape `(L, 6)`)

Returns all active links as rows `[r_a, q_a, r_b, q_b, fidelity, age]`, deduplicated so that
`r_a < r_b` (each physical link appears once).

The `action_mask_entangle` / `action_mask_swap` / `action_mask_purify` helpers are gone.
Legality is defined once, per node, by `Repeater.can_swap()` / `can_purify()` (§2.4.2), and
the RL side reads them through `QRNEnv.action_mask(node)`.


### 3.9 Topology builders

> **Chain-only.** `build_grid`/`build_GEANT` and Haversine positioning were removed in the
> chain-only refactor (`5544d26`); chain is the only topology this project models.

**`build_chain(n_repeaters, n_ch, spacing, swap_policy, p_gen, p_swap, cutoff, **kw)`**

Creates a linear chain of `n_repeaters` nodes spaced `spacing` km apart along the x-axis.
Adjacency connects each node to its immediate neighbours: `adj[i, i+1] = 1.0`. Ports are set
from node position: interior nodes get `n_left = n_right = n_ch`, the two end nodes get a
single port. `**kw` forwards to the `RepeaterNetwork` constructor (`channel_loss`, `F0`,
`distance_dep_gen`, `rng`).

**`build_network(*, n_repeaters, n_ch, spacing, p_gen, p_swap, p_gen_std,
p_swap_std, cutoff, F0, channel_loss, rng)`**

The public entry point (used by `rl_stack/env_wrapper.py`): builds via `build_chain`, then, if
`p_gen_std > 0` or `p_swap_std > 0`, overwrites each repeater's `p_gen`/`p_swap` with per-node
values drawn by `_sample_matched_uniform(mean, std, N, rng)`, a uniform on
`[mean - sqrt(3)*std, mean + sqrt(3)*std]` clipped to `[0.05, 1]`, so its *pre-clip* standard
deviation is exactly `std`. `std <= 0` broadcasts the (clipped) mean and consumes **no** RNG
draw, keeping the homogeneous RNG stream bit-identical to a run with inhomogeneity code paths
compiled out. There is no `topology` argument (removed 2026-07-27); passing one raises
`TypeError`.

### 3.10 Rendering

`render()` and the `--verbose` state dumps were removed (commit `c4ee745`). The engine draws
nothing; all matplotlib in this project lives in `rl_stack/plots.py` and the
`experiments/` figure scripts.


## 4. The RL observation interface

The observation is assembled in the RL layer, **not** in the `simulator` package.
`QRNEnv.get_observation()` (`rl_stack/env_wrapper.py`) reads each node through `net.node(i)`,
the live `Repeater` (read-only by convention), and returns a flat, size-agnostic dict:

| Key | Shape | Content |
|---|---|---|
| `x` | `(N, 8)` | per-repeater node features (table below) |
| `edge_index` | `(2, E)` | directed adjacency, `np.nonzero(net.adj)` |

`rl_stack/agent.py::_obs_to_data` wraps this into a **homogeneous** PyTorch Geometric
`Data(x, edge_index)`, a single node type (repeaters); there are **no** qubit nodes and
**no** `HeteroData`. The GraphSAGE `QNetwork` (`rl_stack/model.py`) message-passes over it
and emits per-node Q-values for the 3 actions (NOOP / SWAP / PURIFY); validity is enforced
by a separate action mask, not by edges in the graph.

**The 8 node features** (from `get_observation`), all in `[0, 1]`:

| Index | Name | Content |
|---|---|---|
| 0 | `frac_occupied` | occupied / physical capacity (`2*n_ch` interior, `n_ch` ends) |
| 1 | `can_swap` | 1.0 if a *viable* swap pair exists: one available LEFT link (partner < node) + one available RIGHT link (partner > node) whose fused link survives the tick boundary (`age_i + age_j + 1 < min cutoff`, mirrors §2.4.5); forced 0 at endpoints |
| 2 | `can_purify` | 1.0 if ≥2 available qubits to the same partner (forced 0 at endpoints) |
| 3 | `p_gen` | per-repeater link-generation prob. (inhomogeneity signal) |
| 4 | `p_swap` | per-repeater BSM success prob. (inhomogeneity signal) |
| 5 | `normalized_age` | mean(age / link_cutoff) over occupied qubits, 0 if none; →1 near expiry |
| 6 | `relative_position` | `i / (N-1)`: 0.0 at source, 1.0 at dest |
| 7 | `is_active` | 1.0 at `env.active_node`, the node deciding this micro-step |

The layout was reduced from 11 features to 8 on 2026-07-25 (`193e0fc`): `mean_fidelity`,
`in_endnode` and `frac_available` were dropped as redundant with `normalized_age`,
`relative_position` and `frac_occupied`.

Columns 3/4 are constant across nodes when the network is homogeneous
(`p_gen_std = p_swap_std = 0`); they carry node-quality signal only under per-repeater
inhomogeneity, produced by `build_network` (see §3.9 and `network._sample_matched_uniform`).


## 5. Simulation Loop (typical usage)

A typical RL training loop proceeds as:

```
1. obs  = env.reset()                                 # reset, auto-entangle, cursor at node 1
2. r    = env.active_node                             # the ONE node deciding now
3. mask = env.action_mask(r)                          # (3,) valid-action mask
4. a    = agent.select_action(obs, mask, r)           # ONE scalar action
5. obs, reward, done, info = env.step(a)              # apply it immediately
6. Repeat from step 2 until done
```

`env.step(a)` applies the action at `env.active_node` **immediately** (`net.purify` /
`net.swap`), checks end to end (the episode can terminate mid-sweep), then advances the
sweep cursor. When the last interior node has acted, the **tick boundary** runs
`net.age_links()` (advance clock, decohere, expire), re-checks end to end, and runs one
shuffled pass of background entanglement.

There is no purify-before-swap phase ordering any more: nodes act one at a time, in fixed
left-to-right order, and each action lands against everything the earlier nodes already did.
The full step ordering, including the termination-versus-truncation distinction, the
per-transition `gamma_eff` and PBRS shaping, is defined in
`rl_stack/env_wrapper.py::QRNEnv.step`.
