# Technical Note: RL Stack for Quantum Repeater Network Routing

## 1. Module Overview

The `rl_stack` package implements a reinforcement learning pipeline for learning per-node
routing policies on quantum repeater networks. It wraps the physics simulator
(`simulator`) in a Gym-like environment, defines a Graph Neural Network (GNN)
that outputs per-node Q-values, and provides a Double-DQN training loop with curriculum
learning and linear epsilon annealing (`cosine` is a flag away, §5.5).

The package is designed for size-agnostic generalisation: the agent trains on small chain
topologies and is evaluated zero-shot on larger or differently parameterised networks.

### Module map

| File | Scope | Role |
|---|---|---|
| `env_wrapper.py` | Environment | Gym-like wrapper: observation construction, action masking, reward shaping, step logic |
| `model.py` | Neural network | 3-layer GraphSAGE Q-network producing per-node action values |
| `buffer.py` | Experience storage | Fixed-size ring buffer of PER-DECISION transitions (s, a, ai, r, s', nai, mask', terminated, gamma_eff) |
| `agent.py` | Training and evaluation | Double-DQN agent: action selection, loss computation, training loop, validation |
| `policies.py` | Baselines + training helper | Heuristic policies (SwapASAP, PurifyThenSwap, Random) for benchmarking, plus the winnability oracle used to prune unsolvable episode cells (it rolls out purify-then-swap, §6) |
| `potential.py` | Reward shaping | PBRS potential (`path_progress`, chain closed form), pure, no torch |
| `plots.py` | Figures | All matplotlib (`Agg` backend); `agent.py` imports it lazily, never at module level |
| `__init__.py` | Re-exports | Public API surface with guarded torch imports |


### RL formulation in brief

The agent controls the interior nodes of a repeater chain. Since the 2026-07-25
sequential-sweep rewrite, **one `env.step` is one micro-decision by ONE node**, the current
`env.active_node`: interior nodes `1..N-2` are visited in a fixed left-to-right order each
tick and each picks one of NOOP (wait), SWAP (entanglement swap via BSM) or PURIFY (BBPSSW
distillation). The chosen operation is applied **immediately**, against the live state, so a
link built by an earlier node in the sweep can be swapped again by a later one in the same
tick (an intra-tick cascade). Entanglement generation is not an agent decision; it runs
automatically at the tick boundary. Source and destination never act.

The reward signal is sparse and **per-decision**, not broadcast: a `STEP_COST` of -0.01
charged once per tick (at the tick-boundary micro-step only, so longer chains are not
penalised more per tick), and a terminal reward of `+1.0 × delivered fidelity` attributed to
whichever node's action closed the chain, plus potential-based shaping (§5.5,
`potential.py`). There is no failed-action penalty: the constant `FAILED_ACTION` was
**deleted** on 2026-07-27, it was already 0.0 and unreferenced, and a nonzero value trained
the agent swap-shy. Do not reintroduce it. An episode ends on delivery (`terminated`) or
when `max_steps` ticks are reached (`truncated`), a distinction the DQN target relies on
(§5.4).


## 2. File: `env_wrapper.py` - Environment

### 2.1 Class `QRNEnv`

`QRNEnv` wraps a `RepeaterNetwork` instance and exposes a Gym-like interface (`reset`,
`step`, `get_observation`, `action_mask`). It does not subclass `gym.Env` but follows
the same contract.

#### Constructor

```
QRNEnv(n_repeaters, n_ch, spacing, p_gen, p_swap, p_gen_std, p_swap_std,
       cutoff, F0, channel_loss, max_steps, rng, gamma)
```

The constructor builds the underlying network via `build_network(...)`. **Chain-only**: a
linear chain of `n_repeaters` nodes, source=0 / dest=N-1. There is no `topology` argument
any more (removed 2026-07-27; passing one raises `TypeError`). Grid and GEANT support
(`build_grid`, `build_GEANT`, random non-adjacent source/destination selection) was removed
in the chain-only refactor, see `src/simulator/README.md` §3.9. `n_repeaters >= 3` is
enforced: the sweep needs at least one interior node, so a smaller chain raises
`ValueError`.

Physics defaults are IDEALISED (`F0=1.0`, `channel_loss=0.0`); the lossy machinery is
untouched, pass the arguments explicitly to exercise it. All physics parameters (`p_gen`,
`p_swap`, `cutoff`, `F0`, `channel_loss`)
are forwarded to the chain builder. **Inhomogeneity** is controlled by `p_gen_std` /
`p_swap_std`: `p_gen` / `p_swap` are per-network *means*, and each `*_std > 0` spreads
per-repeater values via a variance-matched uniform clipped to `[0.05, 1]`
(`network._sample_matched_uniform`). `std = 0` broadcasts the mean and consumes **no** RNG
draw, keeping the homogeneous stream bit-identical. `gamma` is threaded into the PBRS
shaping so the env's shaping discount matches the DQN discount.

After construction the environment calls `_pick_targets()` to select the source-destination
pair for the current episode.

#### Constants

| Name | Value | Meaning |
|---|---|---|
| `NOOP` | 0 | Wait (no operation) |
| `SWAP` | 1 | Entanglement swap via BSM |
| `PURIFY` | 2 | BBPSSW purification |
| `N_ACTIONS` | 3 | Size of per-node action space |
| `STEP_COST` | -0.01 | Reward penalty charged once per TICK, at the tick-boundary micro-step only |
| `SUCCESS_REWARD` | +1.0 | Terminal reward on end-to-end delivery (× delivered fidelity) |

(`FAILED_ACTION` was deleted 2026-07-27, see §1.)


### 2.2 Target selection: `_pick_targets()`

Chain-only: source and destination are always fixed to the first and last node (indices `0`
and `N-1`). The PBRS potential needs no cached hop distances: on a chain the shortcut a
link `(a, b)` offers is just `|a - b| / (N - 1)`.


### 2.3 Observation: `get_observation()`

Returns a dictionary with two keys:

- `"x"`: an `(N, 8)` float32 feature matrix (`NODE_DIM = 8` in `agent.py` must match)
- `"edge_index"`: a `(2, E)` int64 adjacency list (both directions)

The eight per-node features (all in `[0, 1]`) are:

| Index | Name | Formula |
|---|---|---|
| 0 | `frac_occupied` | `num_occupied / physical capacity` (`2*n_ch` interior, `n_ch` at the ends) |
| 1 | `can_swap` | 1.0 if a *viable* swap pair exists: one available LEFT link (partner `<` node) plus one available RIGHT link (partner `>` node) whose fused link survives the tick boundary (`age_i + age_j + 1 < min cutoff`) |
| 2 | `can_purify` | 1.0 if node has >=2 available qubits linked to the same partner |
| 3 | `p_gen` | per-repeater generation probability (inhomogeneity signal) |
| 4 | `p_swap` | per-repeater BSM success probability (inhomogeneity signal) |
| 5 | `normalized_age` | `mean(age / link_cutoff)` over occupied qubits; 0 if none, →1 near expiry |
| 6 | `relative_position` | `i / (N-1)`: 0.0 at source, 1.0 at dest, 0.5 at the true middle for any N |
| 7 | `is_active` | 1.0 at `env.active_node`, the node deciding THIS micro-step; exactly one node, always |

Features 1 and 2 are forced to 0 for source and destination nodes (they may only NOOP).
Features 3/4 are constant across nodes when the network is homogeneous (`std = 0`).

The layout was reduced from 11 features to 8 on 2026-07-25 (commit `193e0fc`):
`mean_fidelity`, `in_endnode` and `frac_available` were dropped as redundant with
`normalized_age`, `relative_position` and `frac_occupied`. Any checkpoint not built on this
layout is incompatible and lives under `checkpoints/legacy/`.

The edge index is derived from the topology adjacency matrix via `np.nonzero`, producing
directed edges in both directions (the adjacency matrix is symmetric). The observation is a
**homogeneous** graph, one node type (repeaters), no qubit nodes, no `HeteroData`.


### 2.4 Action mask: `action_mask(node)` and `get_action_mask()`

`action_mask(node)` returns a fresh `(3,)` boolean array and is **the** decision-path
primitive: only `env.active_node` ever decides, so building an `(N, 3)` grid to read one row
was an O(N) pass per micro-step for O(1) of information. NOOP is always valid (so a masked
argmax can never be all `-inf`); source and destination are NOOP-only. `get_action_mask()`
is literally `np.stack([action_mask(i) for i in range(N)])`, kept for the offline probes and
the tests that want the whole grid, so there is exactly one definition of legality.

Legality itself is defined **on the node**, by `Repeater.can_swap()` and
`Repeater.can_purify()` (moved there 2026-07-27), so the mask cannot drift from the engine's
own gate. There are no `_can_swap_from` / `_can_purify_from` helpers and no `NodeState`
snapshot any more.

**`Repeater.can_swap()`**: true when the node has one available LEFT link (partner index `<`
the node) and one available RIGHT link (partner index `>` the node), **and** the pair is
viable: `age_a + age_b + 1 < min(link_cutoff_a, link_cutoff_b)`. The left/right split is the
2026-07-22 ports change, it makes a same-partner swap structurally impossible. The `+ 1` is
the fused link's single ageing step at the tick boundary: apply is immediate, so the link is
created now and ages once before expiry is checked. (It was `+ 2` until 2026-07-26, a
leftover from the old synchronous-barrier model in which both parents aged again before the
swap resolved; the over-strict gate overstated delivery time by up to 25% under memory
pressure. See `simulator/README.md` §2.4.5.)

**`Repeater.can_purify()`**: true when the node has at least 2 available qubits linked to the
same partner, so two copies of a link exist for BBPSSW distillation.


### 2.5 Step logic: `step(action)`

The step function executes ONE micro-decision at `env.active_node`. It takes a **scalar**
`int` action (not an `(N,)` array; the batched form was deleted with the 2026-07-25
sequential-sweep rewrite) and returns `(observation, reward, done, info)`.

1. **Apply immediately.** PURIFY or SWAP mutate the live state now, against everything every
   earlier node in this sweep already did. There is no lock, no `pending_events`, no
   deferred `_resolve_swap` / `_resolve_purify`. Source and dest are structurally excluded
   from the sweep, so there is nothing to clamp.

2. **Mid-sweep delivery check.** `_check_e2e()` looks for an occupied source qubit whose
   partner is the destination. If found the episode is **terminated right here**, mid-sweep,
   at the closing node: `reward = fidelity × SUCCESS_REWARD − Φ(s)` (`Φ(terminal) = 0`),
   `done = True`, `info["terminated"] = True`, and the step returns **before**
   auto-entangle. Because apply is immediate, a swap earlier in this same sweep may have
   built the long link this node's swap just completed (an intra-tick cascade).

3. **Advance the cursor.** If another interior node is still to act this tick, this is a
   pure shaping micro-step: `gamma_eff = 1.0`, no step cost, reward is just
   `Φ(s') − Φ(s)`.

4. **Tick boundary** (this was the LAST interior node): `net.age_links(discard_expired=True)`
   decoheres and expires links (there is nothing queued to resolve), the end-to-end check is
   repeated (a link may have expired, or auto-entanglement may have formed a direct
   source-dest edge), then `_auto_entangle()` runs one shuffled pass over adjacent pairs.
   The reward is `STEP_COST + [gamma·Φ(s') − Φ(s)]` and the cursor wraps back to the first
   interior node. `env.steps` counts completed TICKS, so `truncated = steps >= max_steps`.

**Discount.** `gamma_eff` is 1.0 for intra-tick micro-steps and `self.gamma` at the tick
boundary only: physical time advances in ticks, not in node-decisions, so discounting every
micro-step would make the effective per-tick discount `gamma^(N-2)`, too aggressive and
N-dependent. `info["gamma_eff"]` carries it per transition, and the DQN target consumes it
(§5.4). The same per-transition `gamma_eff` is used for the PBRS shaping, which makes the
sweep telescope exactly to the old per-tick shaping.

The `info` dictionary contains exactly: `active_node`, `next_active_node`, `fidelity`
(end-to-end fidelity if connected, else 0.0), `age`, `gamma_eff`, `tick_boundary`, `ticks`
(the tick-accurate count to log for `mc_eval`), `terminated` and `truncated`.


### 2.6 Auxiliary methods

**`_exec_purify(r)`**: Finds the neighbour with the most available shared links at node `r`
(breaking ties by whichever `np.unique` returns first) and calls `self.net.purify(r,
best_nb)`. If the node has fewer than 2 available qubits or no neighbour shares 2 or more
links, the call is a no-op.

**`_check_e2e()`**: Iterates over the source repeater's occupied qubits and checks whether
any has `partner_repeater == dest`. Returns `(connected: bool, fidelity: float, age: int)`.
Fidelity is computed via `werner_to_fidelity` on the matching qubit's Werner parameter; age
is the sum-ages inheritance from every swap that built the link.

**`reset()`**: Resets the underlying network, resets the tick counter, performs one round of
auto-entanglement, and positions the sweep cursor at the first interior node. The agent
always sees the post-entanglement state so it can act immediately.

**`active_node`** (property): the node deciding the current micro-step, provably in
`[1, N-2]`.


## 3. File: `model.py` - Q-Network Architecture

### 3.1 Class `QNetwork`

A 3-layer GraphSAGE network followed by a 2-layer MLP head.

**Architecture**:

```
Input: x in R^{N x node_dim}     (node features)
       edge_index in Z^{2 x E}   (adjacency)

Layer 1:  SAGEConv(node_dim -> hidden) + ReLU
Layer 2:  SAGEConv(hidden -> hidden)   + ReLU
Layer 3:  SAGEConv(hidden -> hidden)   + ReLU
Head:     Linear(hidden -> hidden) + ReLU + Linear(hidden -> n_actions)

Output: Q in R^{N x n_actions}
```

**Design rationale**: Three message-passing layers give a 3-hop receptive field. In a linear
chain of repeaters this means each node can "see" information from up to 3 hops away in each
direction when computing its Q-values, which is sufficient for local coordination of swap
and purify decisions.

GraphSAGE (`SAGEConv`) uses mean aggregation over neighbour features. It is permutation-
invariant across nodes and handles variable-size graphs, enabling the model to train on small
networks and generalise to larger ones without architectural changes.

**Default parameters**: `node_dim=9`, `hidden=32`, `n_actions=3`. Both class defaults are
**stale in practice and never used**: the live observation is 8 features
(`agent.py::NODE_DIM = 8`) and all real checkpoints use `hidden=64`. `load_qnet(path)`
infers `(node_dim, hidden)` from `conv1.lin_l.weight`, so you never construct the
architecture by hand; use it.

When used with PyTorch Geometric batching (`Batch.from_data_list`), multiple graphs of
different sizes are concatenated into a single disconnected graph. The output tensor has
shape `(total_nodes_in_batch, n_actions)` and is indexed back to individual graphs via the
`batch` attribute.


## 4. File: `buffer.py` - Replay Buffer

### 4.1 Class `ReplayBuffer`

A fixed-size ring buffer that stores transitions as Python dictionaries containing numpy
arrays.

**Storage format**: Each entry is a dictionary:

| Key | Type | Shape | Content |
|---|---|---|---|
Since the 2026-07-25 sequential-sweep rewrite an entry is ONE node's micro-decision, added
via `add(s, a, active_idx, r, s_, next_active_idx, next_mask_row, terminated, gamma_eff)`:

| Key | Type | Shape | Content |
|---|---|---|---|
| `"s"` | dict | `{"x": (N,8), "edge_index": (2,E)}` | Current observation |
| `"a"` | int | scalar | Action taken by the active node |
| `"ai"` | int | scalar | Index of the active node in `s` |
| `"r"` | float | scalar | Reward for this micro-decision |
| `"s_"` | dict | `{"x": (N,8), "edge_index": (2,E)}` | Next observation |
| `"nai"` | int | scalar | Index of the active node in `s_` |
| `"m_"` | ndarray | `(3,)` bool | Next ACTIVE NODE's action mask row |
| `"d"` | bool | scalar | Episode `terminated` flag (not raw `done`, so truncations bootstrap) |
| `"g"` | float | scalar | `gamma_eff` for this transition (1.0 intra-tick, `gamma` at the tick boundary) |

The next-state action mask `"m_"` is a key design element. It is stored alongside each
transition so that during training, the target Q-value computation can mask out physically
impossible actions in the successor state. Without this, the agent could learn inflated
Q-values for actions that would never be available.

PyG conversion of a replayed state is cached in place on the transition dictionary, so a
transition is converted at most once no matter how often it is resampled.

**Ring buffer mechanics**: When `len(buffer) < max_size`, new entries are appended. Once
full, entries overwrite at position `self.pos` (modulo `max_size`). Sampling uses
`random.sample` for uniform random selection without replacement.


## 5. File: `agent.py` - Double-DQN Agent

### 5.1 Helper functions

**`_obs_to_data(obs, device)`**: Converts a numpy observation dictionary to a PyTorch
Geometric `Data` object on the specified device. This bridges the numpy-based environment
interface with the torch-based model.

(`_running_avg`, the causal backward-looking moving average used to smooth the training
curves, moved to `plots.py` with the rest of the figure code.)


### 5.2 Class `QRNAgent`

The agent maintains two copies of the Q-network: `policy_net` (trained) and `target_net`
(slowly tracked via Polyak averaging). This is the standard Double-DQN architecture.

#### Constructor

```
QRNAgent(node_dim=NODE_DIM, hidden=64, lr=3e-4, gamma=0.99,
         buffer_size=80_000, batch_size=64, tau=0.005,
         epsilon=1.0, rng=None, seed=None)
```

| Parameter | Role |
|---|---|
| `node_dim` | Input feature dimension (must match `env_wrapper`'s feature count: `NODE_DIM = 8`) |
| `hidden` | Hidden dimension of both `QNetwork` instances |
| `lr` | Adam learning rate |
| `gamma` | Discount factor |
| `buffer_size` | Maximum replay buffer capacity |
| `batch_size` | Mini-batch size for SGD updates |
| `tau` | Polyak averaging coefficient for target network |
| `epsilon` | Initial exploration rate for epsilon-greedy |
| `rng` | Seeded numpy Generator for reproducible exploration |
| `seed` | If given, also seeds the replay buffer's sampler (`ReplayBuffer(seed=...)`) so a run is bit-reproducible end to end |

The loss function is `SmoothL1Loss` (Huber loss), and gradient norms are clipped to 10.0.


### 5.3 Action selection: `select_action(obs, mask_row, active_node, training)`

Epsilon-greedy action selection with action masking, for ONE node. This is the only
selection entry point; the batched `select_actions(obs, mask, training)` was deleted with
the 2026-07-25 sequential-sweep rewrite.

**Exploration** (`training=True` and `rng.random() < epsilon`): sample uniformly from the
valid actions in `mask_row`, the `(3,)` mask of the active node. Uses the seeded `rng` for
reproducibility.

**Exploitation** (otherwise): the observation is converted to a `Data` object and passed
through `policy_net` to get `(N, 3)` Q-values; only the `active_node` row is read, invalid
actions in it are set to `-inf`, and `argmax` picks the greedy action.

Returns a scalar `int` action.


### 5.4 Training step: `train_step()`

Performs one gradient update using a mini-batch from the replay buffer. Returns the scalar
loss value, or `None` if the buffer has fewer samples than `batch_size`.

The computation proceeds as follows:

1. **Sample batch**: `batch_size` transitions are drawn uniformly from the buffer.

2. **Batch construction**: Current states and next states are each assembled into a single
   PyTorch Geometric `Batch` via `Batch.from_data_list`. This concatenates all graphs into
   one large disconnected graph. The `ptr` attribute gives each graph's node offset.

3. **Per-decision indexing (no broadcasting)**: since the sequential-sweep rewrite each
   transition is ONE node's decision, so there is **exactly one Q-value per transition**,
   not one per node per graph. Transition `b`'s active node sits at the batched-graph global
   index `Batch.ptr[b] + ai_b`, where `ai` is the stored active-node index. Rewards and done
   flags are per transition and are never broadcast to the other nodes.

4. **Current Q-values**: `policy_net(states)` is evaluated, the active-node rows are
   selected by `act_idx`, and the taken action is picked out via `gather`, yielding
   `current_q` of shape `(batch_size,)`.

5. **Target Q-values (Double DQN with mask)**:
   - The **policy net** evaluates next states, at the NEXT active node (`nai`). Invalid
     next-state actions in that node's stored mask row (`m_`) are set to `-inf`. The argmax
     over these masked Q-values selects the best valid action. This is the "double" part:
     the policy net selects, but...
   - The **target net** evaluates the same next states, and the Q-value at the
     policy-selected action is extracted via `gather`.
   - The Bellman target uses the **per-transition** discount, not a fixed scalar:
     `target_q = reward + gamma_eff * next_q * (1 - done)`, with `gamma_eff = 1.0`
     intra-tick and `self.gamma` at the tick boundary.

6. **Loss and update**: SmoothL1 loss between `current_q` and `target_q`. Gradients are
   clipped to max norm 10.0. Adam step. Then Polyak update of target network:
   `theta_target <- tau * theta_policy + (1 - tau) * theta_target`.


### 5.5 Training loop: `train(episodes, ...)`

The main training loop creates a fresh environment each episode, runs up to `max_steps`
steps, and accumulates experience in the replay buffer.

**Curriculum learning** (`_curriculum_pool`): the eligible chain-size cap widens **linearly**
from `min(n_range)` to `max(n_range)` over the first `curriculum_frac` (default 0.5) of
training (round-half-up), then holds the full range. Reaching the full range by mid-training
ensures the largest sizes still get a substantial share of episodes. At each episode the chain
size is sampled uniformly from the current eligible pool. `curriculum=False` always uses the
full range.

**Domain randomisation**: per episode, `p_gen` / `p_swap` / `cutoff` / `n_ch` are drawn from
the ranges/pools passed to `train`, and per-repeater `p_gen_std` / `p_swap_std` scatter node
rates. With `--prune_unwinnable`, a `WinnabilityCache` resamples the cell until purify-then-swap
can deliver it, so no episode is spent on an unsolvable config.

**Epsilon schedule** (`eps_schedule`, default `'linear'` since commit `fa7e038`, the
Mnih / SB3 standard; `'cosine'` is the other option): annealed from `eps_init=1.0` to
`eps_fin=0.05` over the first 90% of training, then held constant at `eps_fin`.

```
For ep < 0.9 * episodes:
    linear: epsilon = eps_init + (eps_fin - eps_init) * ep / (0.9 * episodes)
    cosine: epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (1 + cos(pi * ep / episodes))

For ep >= 0.9 * episodes:
    epsilon = eps_fin
```

**Model saving (best-checkpointing)**: `save_path/policy.pth` always holds the **best** agent
seen, and the final weights go to `save_path/policy_final.pth`, so late-training degradation
can never clobber the best. "Best" is judged by a held-out greedy probe (`eval_fn`, with
`eval_every` / `eval_patience` early stopping and `eval_mode='min'` for delivery time) when
provided, else by rolling-mean reward over `best_window` episodes, **gated** to the settled
late window (`_ckpt_window_start`: curriculum fully open and epsilon at its floor) so the easy
early phase can't freeze the checkpoint. Raw per-episode metrics are written to
`save_path/metrics.json` (replot via `experiments/training/replot.py`).

**Checkpoint pool and runoff** (2026-07-27): `ckpt_pool=True` also saves a checkpoint at
every eval probe into `<save_path>/pool/`, and `runoff(pool_dir, eval_fn, n_repeats)`
re-scores every candidate at the end on PAIRED seeds (the probe seeds each rollout from
`(probe_seed, episode index)` alone, so nothing draws from `self.rng`) and promotes the
winner to `policy.pth`. Motivation: training length is non-monotonic and the rolling-reward
criterion missed it. Lower is better throughout, this is the delivery-time convention.

The method returns a dictionary of per-episode metrics: `reward`, `loss`, `steps` (in
TICKS, `info["ticks"]`), `success` (plus `eval` and, under `compare=True`, per-policy
comparison columns).


### 5.6 Validation: `validate(model_path, ...)`

Evaluates the trained agent against three baseline strategies on a fixed network
configuration. If `model_path` is provided, the policy network weights are loaded from disk.
During validation, epsilon is set to 0 (pure exploitation).

The strategies compared are:

| Strategy | Source | Behaviour |
|---|---|---|
| Agent | `select_action(training=False)` | Greedy Q-value policy |
| SwapASAP | `policies.swap_asap` | Swap whenever the active node's mask allows it |
| PurifySwap | `policies.purify_then_swap` | Purify if possible, else swap if possible |
| Random | `policies.random_policy` | Uniform random valid action |

> `BeliefPropagationPolicy` and `fidelity_gated_swap` were deliberately removed 2026-07-09
> (out of scope for the paper; recoverable from git history).

Each strategy runs `n_episodes` episodes on **paired** episode seeds. Results (average steps
to success with standard deviation, average end-to-end fidelity, success rate) are printed
in a table. The `plot_actions` / `save_dir` arguments and the action-timeline figure were
dropped 2026-07-27, as was the per-step network rendering.


### 5.7 Plotting (module `plots.py`)

The figure code lives in `rl_stack/plots.py`, not in `agent.py`, so importing the agent
does not pull in matplotlib. `agent.py` imports these lazily, at the call site, only when
a figure is actually requested. `plots.py` selects the non-interactive `Agg` backend
before importing `pyplot`, so it works on a headless cluster node.

**`plot_training(metrics, save_path, window=None)`**: Generates a 3-panel figure showing episode return
(with running average), loss (log scale), and success rate over training. Saved as
`training_metrics.png`.

**`print_results_table(results, N, pg, ps, c)`**: Prints a formatted ASCII table of
validation results (average steps with standard deviation, average fidelity with standard
deviation, success percentage) for each strategy.

**`config_caption(cfg)`** and **`_running_avg(vals, window)`** are the remaining helpers.
`plot_timeline_grid` was deleted with the action-timeline figure (2026-07-27).


## 6. File: `policies.py` - Baseline Policies and the Winnability Oracle

Three heuristic strategies are provided for benchmarking. Each takes the env and returns a
**scalar** `int` action for `env.active_node` (they were `(N,)` arrays before the
2026-07-25 sequential-sweep rewrite), and each respects that node's action mask.

**`swap_asap(env)`**: SWAP whenever the active node's mask allows it, else NOOP. This is the
most aggressive strategy: it extends entanglement reach as fast as possible but does not
improve fidelity through purification.

**`purify_then_swap(env)`**: prefer PURIFY if legal; otherwise SWAP if legal; otherwise
NOOP. This prioritises link quality over speed. It is also what `WinnabilityCache` rolls
out as its feasibility oracle (`policies.py`, the pilot loop), so `--prune_unwinnable`
keeps exactly the cells purify-then-swap can deliver.

> ⚠ The choice of oracle is **not settled**. The project docs long claimed swap-asap was
> the oracle, and justified it with "purify-then-swap can livelock at `n_ch=4`". The code
> has done the opposite since before that claim was written, and the livelock has never
> been measured. If it is real, pruning has been discarding cells swap-asap could deliver
> and quietly narrowing the training distribution. Settle it before the next full-scale
> retrain, not after.

**`random_policy(env, rng)`**: sample uniformly from the valid actions. Takes an **explicit
RNG that must be independent of `env.rng`**, sharing it would perturb the environment's own
generation/BSM coin flips and invalidate the comparison. Lower-bound baseline.

> `fidelity_gated_swap` and `BeliefPropagationPolicy` were deliberately removed 2026-07-09
> (out of scope for the paper; recoverable from git history).


## 7. Typical Training and Evaluation Flow

A standard usage of the RL stack follows these steps:

1. **Instantiate agent**: `agent = QRNAgent(hidden=64, lr=3e-4, gamma=0.99, ...)`.

2. **Train**: `metrics = agent.train(episodes=3000, n_range=[4,5,6,7],
   curriculum=True, p_gen_std=0.15, p_swap_std=0.15, prune_unwinnable=True,
   save_path='runs/exp1/')`. The agent trains on randomised chain networks of increasing size
   with per-repeater inhomogeneous link parameters.

3. **Validate**: `results = agent.validate(n_repeaters=8, n_episodes=100)`. The agent is
   tested on a fixed 8-node chain (larger than any training size) against the three
   baselines.

4. **Inspect**: Review `training_metrics.png` for learning curves. For delivery time, the
   canonical metric, use `experiments/mc_eval.py` and the figure suite under
   `experiments/comparisons/`.


## 8. Key Design Decisions

**Per-node Q-values, per-decision credit**: rather than a single centralised action, the
network emits Q-values for every node and one node acts per env step, in a fixed
left-to-right sweep. Credit is assigned to that node alone: there is no shared-reward
broadcast (that was the pre-2026-07-25 barrier model). The GNN still enables implicit
coordination through message passing, and the shared weights make the policy a
parameter-shared multi-agent policy.

**Immediate apply, not a synchronous barrier**: a node's action lands against the live state
the instant it acts, so its action mask is exact and a link built earlier in the sweep can
be swapped again in the same tick. The motivation is Inesta, Vardoyan, Scavuzzo and Wehner,
npj QI 9, 46 (2023): the global-knowledge MDP optimum beats swap-asap by withholding swaps,
a timing strategy the stale start-of-tick observation may have blocked the agent from
learning.

**Action masking at both selection and training time**: Invalid actions are masked during
both epsilon-greedy selection (only valid actions are sampled) and target Q-value computation
(invalid successor actions are set to -inf). This prevents the agent from learning Q-values
for physically impossible actions and accelerates convergence.

**Background entanglement**: Entanglement generation is removed from the action space and
handled automatically. This reduces the action space from 4 to 3 actions per node and
eliminates the need for the agent to learn the trivial policy of "always try to generate
links". The agent focuses on the non-trivial routing decisions: when to swap, when to
purify, and when to wait.

**Curriculum learning**: Training on progressively larger networks prevents the agent from
being overwhelmed by large action spaces early in training. Small networks allow rapid
exploration and reward collection, establishing basic swap/purify intuitions before
graduating to harder configurations.

**Size-agnostic architecture**: The GNN operates on arbitrary graph sizes. Combined with
normalised features (fractions and binary flags rather than absolute counts), this enables
zero-shot transfer to networks larger than those seen during training.

**Linear epsilon annealing**: the DQN standard (Mnih 2015, SB3), and the default since
commit `fa7e038`. `'cosine'` holds exploration higher through early training and is kept
behind `--eps_schedule` for reproducing the runs that used it. Either way the last 10% of
training holds epsilon constant at the minimum to stabilise final performance.
