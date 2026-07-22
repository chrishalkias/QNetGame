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
| `network.py` | Inter-node | Network of repeaters: entanglement generation, swap, purification, ageing, event queue, `build_chain`/`build_network`, rendering |
| `snapshots.py` | Read boundary | Frozen, fidelity-domain `NodeState` / `Topology` dataclasses the RL side reads |
| `__init__.py` | Re-exports | Public API surface |

> The exact-DP swap-only optimal-policy benchmark (formerly `simulator/optimal_policy/`) was
> retired 2026-07-18; recoverable from git history if needed.
>
> **Chain-only**: `build_grid`/`build_GEANT` and the `'grid'`/`'geant'` topology strings were
> removed in the chain-only refactor. `build_network(topology=...)` now accepts only
> `'chain'` and raises `ValueError` otherwise.

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

`Repeater.__deepcopy__` overrides the generic recursive `copy.deepcopy` with 10 flat
`ndarray.copy()` calls (config fields are immutable and shared by reference). It was added for
the exact-DP optimal-policy kernel (§1); that consumer was retired 2026-07-18, so this is
currently unused but harmless if a future consumer needs cheap `Repeater` clones again.

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


#### 2.4.5 `select_swap_pair(network_positions, network_cutoffs, rng=None) -> Tuple[int, int] | None`

Selects which two available qubits on this repeater should be used for a swap, among the
**viable** pairs only.

**Viability gate (2026-07-12 cutoff-invariant fix)**: for candidate pair `(qa, qb)` with remote
endpoints `ra, rb`, the fused link would inherit `age_a + age_b`, and both parents age once
more before a same-tick event resolves, so the pair is only offered if
`age_a + age_b + 2 < min(network_cutoffs[ra], network_cutoffs[rb])`. This closes a leak where
over-age pairs were swapped and then resolved past their cutoff anyway (pre-fix repro on
swap-asap, N=10: 49% of deliveries were over-age links). Returns `None` if fewer than 2 qubits
are available, or if none of the C(k,2) candidate pairs are viable.

Among the viable pairs, the choice depends on `self.swap_policy`:

**FARTHEST**: looks up the spatial positions of each pair's remote partners via
`network_positions[partner_repeater[...]]` and selects the pair maximising the Euclidean
distance between those two remote positions — swapping qubits whose partners are far apart
extends the entangled link across the greatest spatial span.

**STRONGEST**: selects the pair maximising the product `werner_param[qa] * werner_param[qb]`.
Since the post-swap Werner parameter equals this product (see §3.3), this policy maximises the
fidelity of the resulting swapped link.

**RANDOM**: selects uniformly at random among the viable pairs using the provided `rng`.

With CC delays the in-flight accrual can exceed what this pre-swap gate sees; `network.py`'s
`_resolve_swap` holds an additional born-dead guard at resolution time (§3.6).


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
- `dt_seconds` (float): Duration of one simulation step in physical seconds. Used together
  with `c_fiber` to compute classical communication delays.
- `c_fiber` (float): Speed of light in fibre, fixed at 200,000 km/s.
- `pending_events` (list[dict]): Queue of deferred swap/purify resolutions.
- `_positions` (ndarray, shape `(N, 2)`): Cached array of repeater positions.
- `_dist_matrix` (ndarray, shape `(N, N)`): Cached pairwise Euclidean distance matrix,
  computed at construction from `_positions`.
- `_cutoffs` (ndarray, shape `(N,)`, int64): Per-repeater cutoff, rid-indexed. Fed to
  `Repeater.select_swap_pair` as the swap viability gate (§2.4.5).


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
2. Select a pair `(qa, qb)` via `select_swap_pair` (§2.4.5) — already filtered to viable
   pairs only (`no_valid_pair` if `can_swap()` passed but no pair survives the viability gate).
3. Guards, rejected before sampling: `orphan_qubit` if either qubit's `partner_repeater` is
   `NO_PARTNER` (would silently index `repeaters[-1]` at resolution); `same_partner` if both
   qubits point at the same remote repeater (would create a self-link).
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
   - Enqueue a `"swap"` event with `timer = delay` and the two remote qubits'
     `generation_id`s (`gen_a`, `gen_b`), used by `_resolve_swap` to detect reallocation
     during the delay (§3.6).

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
1. Require at least 2 shared pairs between `r1` and `r2`.
2. Sort shared links by fidelity; lock every involved qubit (both ends).
3. Run the guarded cascade against the frozen start-of-tick state (all RNG drawn now).
4. Enqueue ONE `"purify"` event: the sacrificed qubit list, the surviving qubit (or `None`),
   whether it was actually purified (`keep_purified`), the survivor's Werner parameter, and
   its age at decision time (`age_keep`, for the Eq.(4) rebase at resolution, §3.6).

The whole cascade is deferred to end-of-tick (the synchronous-tick barrier); a survivor that
was only kept (never purified) retains its original registration, while a purified survivor is
re-registered via Eq.(4) age semantics. One PURIFY *action* runs this cascade on every partner
with which the node shares >=2 links (env `_exec_purify`).


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

**Guard 1 — liveness:** a remote qubit is only "alive" if it is still `QUBIT_OCCUPIED` **and**
its current `generation_id` matches the value recorded on the event (`gen_a`/`gen_b`). Occupancy
alone is not enough: a locked qubit can be freed by expiry and then *reallocated* to an
unrelated link during the CC delay, and the generation check is what stops the resolution from
corrupting that new occupant (a same-index-different-link "ghost link" bug). If either side is
dead, the live survivor (if any) is freed and no link is established.

**Guard 2 — collapsed endpoints:** if both remote endpoints turned out to be the same repeater
(`ra == rb`, e.g. a chain of re-swaps collapsed onto one node during the delay), forming the
link would create a self-loop; both survivors are freed instead.

**Guard 3 — born-dead:** the fused link would inherit `age = age_a + age_b` (see below). If
`age_a + age_b >= min(cutoff_a, cutoff_b)`, the link is already past its cutoff at the moment
it would be created — the pre-swap viability gate (§2.4.5) cannot always see this (CC-delay
accrual happens after the gate ran), and the plain expiry sweep in `age_links()` only scans
qubits that existed *before* this resolution, so it would never police a link born expired.
Both remote endpoints are freed instead of creating the link.

**On success (sum-ages resolution):** the resolved Werner value must equal the product of the
two remote links' *already-decohered* values at resolution time, `w_a * w_b`, not a fresh
`p_new` planted at `age=0` (that would double-count the pre-swap decoherence, and is what the
project's docs from before the 2026-07-10/07-12 physics fixes describe). This is reproduced by
storing the **baseline product** `p0_a * p0_b` (`initial_werner` of each side) as the new
`initial_werner`, with `age = age_a + age_b` and `effective_cutoff = min(cutoff_a, cutoff_b)`:
since `p0_a*p0_b*exp(-(age_a+age_b)/tau) = (p0_a*e^{-age_a/tau})(p0_b*e^{-age_b/tau}) = w_a*w_b`
for a shared `tau`, this is exact for homogeneous per-link cutoffs (the only regime in use) and
an approximation only if the two links carried different cutoffs. Both qubits are then unlocked.

#### `_resolve_purify(ev)`

Resolves a deferred purification event. All liveness checks use the generation-ID guard
described above, not bare occupancy.

**On success:**
1. Break the sacrifice pair, guarded by each side's generation-ID (`gen_sac1`/`gen_sac2`)
   against `q1_sac`/`q2_sac`). If a side is no longer live, free the live survivor and (if our
   lock is still the current one) release it — a lock whose generation no longer matches was
   re-taken by a newer in-flight op and must be left alone, or that op's qubit gets orphaned.
2. Guard: check that both "keep" qubits are still live (generation-ID + occupancy). If not,
   break the live survivor(s) and return.
3. **Eq.(4) age semantics** (arXiv 2401.13168): rather than reset the kept link's age to 0 (the
   pre-2026-07-12 behaviour, which double-counted decoherence via sum-of-endpoint-ages
   bookkeeping), the purified fidelity `p_new` is represented as an equivalent age on a fresh
   `p0=1` baseline: `m_equiv = ceil(-cutoff * ln(p_new))`, plus ticks accrued since the decision
   (`accrued = current_age - age_keep`, covering CC delay + same-tick aging). If the resulting
   `new_age = m_equiv + accrued` has already reached the cutoff, the purified state is below
   the fidelity floor the cutoff exists to guarantee — discard (`_break_link`) rather than
   create a link expiry can never police, same rationale as swap's born-dead guard. Otherwise
   `set_link(..., p=1.0, link_age=new_age, effective_cutoff=cutoff)` and unlock both qubits.

**On failure:**
Break all four qubits, each guarded by its own generation-ID against the event's recorded
value (a qubit may have already been freed by expiry, reallocated, or freed as a side effect
of another qubit's `_break_link` on the same physical pair).


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

> **Chain-only.** `build_grid`/`build_GEANT` and Haversine positioning were removed in the
> chain-only refactor (`5544d26`); chain is the only topology this project models.

**`build_chain(n_repeaters, n_ch, spacing, swap_policy, p_gen, p_swap, cutoff, **kw)`**

Creates a linear chain of `n_repeaters` nodes spaced `spacing` km apart along the x-axis.
Adjacency connects each node to its immediate neighbours: `adj[i, i+1] = 1.0`. `**kw` forwards
to the `RepeaterNetwork` constructor (`channel_loss`, `F0`, `distance_dep_gen`, `rng`,
`dt_seconds`).

**`build_network(topology="chain", *, n_repeaters, n_ch, spacing, p_gen, p_swap, p_gen_std,
p_swap_std, cutoff, F0, channel_loss, dt_seconds, rng)`**

The public entry point (used by `rl_stack/env_wrapper.py`): builds via `build_chain`, then, if
`p_gen_std > 0` or `p_swap_std > 0`, overwrites each repeater's `p_gen`/`p_swap` with per-node
values drawn by `_sample_matched_uniform(mean, std, N, rng)` — a uniform on
`[mean - sqrt(3)*std, mean + sqrt(3)*std]` clipped to `[0.05, 1]`, so its *pre-clip* standard
deviation is exactly `std`. `std <= 0` broadcasts the (clipped) mean and consumes **no** RNG
draw, keeping the homogeneous RNG stream bit-identical to a run with inhomogeneity code paths
compiled out. Raises `ValueError` for any `topology` other than `"chain"`.

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
| 0 | `frac_occupied` | occupied / physical capacity (`2*n_ch` interior, `n_ch` ends) |
| 1 | `mean_fidelity` | avg F of available (unlocked) qubits, 0 if none |
| 2 | `in_endnode` | 1.0 if source **or** dest (endpoints are symmetric) |
| 3 | `frac_available` | available (unlocked occupied) / physical capacity |
| 4 | `can_swap` | 1.0 if a *viable* swap pair exists: one available LEFT link (partner < node) + one available RIGHT link (partner > node) whose fused link would survive same-tick resolution (`age_i + age_j + 2 < min cutoff`, mirrors §2.4.5); forced 0 at endpoints |
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
