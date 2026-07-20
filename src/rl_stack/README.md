# Technical Note: RL Stack for Quantum Repeater Network Routing

## 1. Module Overview

The `rl_stack` package implements a reinforcement learning pipeline for learning per-node
routing policies on quantum repeater networks. It wraps the physics simulator
(`simulator`) in a Gym-like environment, defines a Graph Neural Network (GNN)
that outputs per-node Q-values, and provides a Double-DQN training loop with curriculum
learning and cosine epsilon annealing.

The package is designed for size-agnostic generalisation: the agent trains on small chain
topologies and is evaluated zero-shot on larger or differently parameterised networks.

### Module map

| File | Scope | Role |
|---|---|---|
| `env_wrapper.py` | Environment | Gym-like wrapper: observation construction, action masking, reward shaping, step logic |
| `model.py` | Neural network | 3-layer GraphSAGE Q-network producing per-node action values |
| `buffer.py` | Experience storage | Fixed-size ring buffer for (s, a, r, s', done, mask') transitions |
| `agent.py` | Training and evaluation | Double-DQN agent: action selection, loss computation, training loop, validation |
| `strategies.py` | Baselines | Heuristic policies (SwapASAP, PurifyThenSwap, Random) for benchmarking |
| `potential.py` | Reward shaping | PBRS potential (`bfs_hops`, `path_progress`), pure, no torch |
| `winnability.py` | Training helper | Purify-then-swap winnability oracle (2026-07-12; was swap-asap) used to prune unsolvable episode cells |
| `__init__.py` | Re-exports | Public API surface with guarded torch imports |


### RL formulation in brief

The agent controls all interior nodes of a quantum repeater network. At each discrete time
step every node independently selects one of three actions: NOOP (wait), SWAP (entanglement
swap via BSM), or PURIFY (BBPSSW distillation). Entanglement generation is not an agent
decision; it runs automatically as a background process after each step. Source and
destination nodes are always forced to NOOP.

The reward signal is sparse: a fixed penalty of -0.01 per step (encouraging speed) and a
terminal reward of `+1.0 × delivered fidelity` when an end-to-end entangled link between
source and destination is established, plus potential-based shaping (§5.5, `potential.py`).
Failed actions are **not** penalised (`FAILED_ACTION = 0.0`; a nonzero value trained the
agent swap-shy). An episode ends on success (`terminated`) or when `max_steps` is reached
(`truncated`), a distinction the DQN target relies on (§5.4).


## 2. File: `env_wrapper.py` - Environment

### 2.1 Class `QRNEnv`

`QRNEnv` wraps a `RepeaterNetwork` instance and exposes a Gym-like interface (`reset`,
`step`, `get_observation`, `get_action_mask`). It does not subclass `gym.Env` but follows
the same contract.

#### Constructor

```
QRNEnv(n_repeaters, n_ch, spacing, p_gen, p_swap, p_gen_std, p_swap_std,
       cutoff, F0, channel_loss, dt_seconds, max_steps, rng, topology, gamma)
```

The constructor builds the underlying network via `build_network(topology, ...)`.
**Chain-only**: `topology` must be `'chain'` (a linear chain of `n_repeaters` nodes, source=0 /
dest=N-1); any other value raises `ValueError`. Grid and GÉANT topology support (`build_grid`,
`build_GEANT`, random non-adjacent source/destination selection) was removed in the chain-only
refactor — see `src/simulator/README.md` §3.9.

All physics parameters (`p_gen`, `p_swap`, `cutoff`, `F0`, `channel_loss`, `dt_seconds`)
are forwarded to the topology builder. **Inhomogeneity** is controlled by `p_gen_std` /
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
| `STEP_COST` | -0.01 | Per-step reward penalty |
| `SUCCESS_REWARD` | +1.0 | Terminal reward on end-to-end success (× delivered fidelity) |
| `FAILED_ACTION` | 0.0 | Penalty for a failed swap/purify, **disabled** (see §1) |


### 2.2 Target selection: `_pick_targets()`

Chain-only: source and destination are always fixed to the first and last node (indices `0`
and `N-1`). Also caches the BFS hop distances (`_d_src`, `_d_dst`, `_d_total`) the PBRS
potential reads every step (§5.5).


### 2.3 Observation: `get_observation()`

Returns a dictionary with two keys:

- `"x"`: an `(N, 9)` float32 feature matrix
- `"edge_index"`: a `(2, E)` int64 adjacency list (both directions)

The nine per-node features (all in `[0, 1]`) are:

| Index | Name | Formula |
|---|---|---|
| 0 | `frac_occupied` | `num_occupied / n_ch` |
| 1 | `mean_fidelity` | Mean of `werner_to_fidelity(p)` over available (unlocked) qubits; 0 if none |
| 2 | `in_endnode` | 1.0 if this node is the episode source **or** destination (endpoints are symmetric) |
| 3 | `frac_available` | `num_available / n_ch` (available = occupied and not locked) |
| 4 | `can_swap` | 1.0 if a *viable* swap pair exists: >=2 available qubits linked to distinct partners whose fused link survives same-tick resolution (`age_i + age_j + 2 < min cutoff`) |
| 5 | `can_purify` | 1.0 if node has >=2 available qubits linked to the same partner |
| 6 | `p_gen` | per-repeater generation probability (inhomogeneity signal) |
| 7 | `p_swap` | per-repeater BSM success probability (inhomogeneity signal) |
| 8 | `link_urgency` | `mean(age / link_cutoff)` over occupied qubits; 0 if none, →1 near expiry |

Features 4 and 5 are forced to 0 for source and destination nodes (they may only NOOP).
Features 6/7 are constant across nodes when the network is homogeneous (`std = 0`).

The edge index is derived from the topology adjacency matrix via `np.nonzero`, producing
directed edges in both directions (the adjacency matrix is symmetric). The observation is a
**homogeneous** graph, one node type (repeaters), no qubit nodes, no `HeteroData`.


### 2.4 Action mask: `get_action_mask()`

Returns an `(N, 3)` boolean array. NOOP is always valid for every node (so a masked argmax
can never be all `-inf`). For interior nodes, SWAP is valid when `_can_swap_from(ns)` is true
and PURIFY when `_can_purify_from(ns)` is true. Source and destination nodes have only NOOP
enabled. Both helpers operate on a node's immutable `NodeState` snapshot.

**`_can_swap_from(ns)`**: Returns true when the node has at least 2 available (occupied and
unlocked) qubits whose `partner_node` values point to at least 2 distinct neighbours, **and**
at least one such pair is viable: `age_i + age_j + 2 < min(link_cutoff_i, link_cutoff_j)`
(2026-07-12 fix, mirrors the engine's swap decision gate in `Repeater.select_swap_pair`,
`simulator/README.md` §2.4.5). This ensures a BSM can bridge two different link segments
*and* that the fused link wouldn't be born past its cutoff.

**`_can_purify_from(ns)`**: Returns true when the node has at least 2 available qubits linked
to the same partner. This ensures two copies of a link to the same neighbour exist for
BBPSSW distillation. Both helpers share `_partner_counts`, a `np.bincount` over partner ids.


### 2.5 Step logic: `step(actions)`

The step function executes one discrete time step. It takes an `(N,)` integer action array
and returns `(observation, reward, done, info)`.

**Defensive copy**: The input `actions` array is copied immediately to prevent mutation of
the caller's data.

**Source/destination clamping**: Any non-NOOP action assigned to source or destination nodes
is silently overwritten to NOOP.

The step proceeds in four phases:

1. **Phase 1a - Purifications**: All nodes with `action == PURIFY` execute purification
   first. This ordering ensures that if a node purifies a link and another node subsequently
   swaps through it, the swapped link benefits from the improved fidelity.

2. **Phase 1b - Swaps**: All nodes with `action == SWAP` execute swaps. Internally this
   calls `self.net.swap(r)`, which uses the repeater's `select_swap_pair` method to choose
   which two qubits to Bell-measure.

3. **Phase 2 - Age links**: Calls `self.net.age_links(discard_expired=True)`. This resolves
   any pending classical communication events, applies decoherence (Werner parameter decay),
   increments qubit ages, and expires links that have exceeded their cutoff lifetime.

4. **Phase 3 - End-to-end check**: The step counter increments. `_check_e2e()` scans the
   source node's occupied qubits for one whose `partner_node` equals the destination. If
   found, the episode succeeds with `reward = fidelity × SUCCESS_REWARD + shaping`,
   `done = True`, and `info["terminated"] = True`; the step returns **before** auto-entangle.
   Otherwise, if `steps >= max_steps`, the episode is **truncated** (`done = True`,
   `info["terminated"] = False`), it still takes the normal non-terminal path so `V(s')`
   stays bootstrappable in the DQN target.

5. **Phase 4 - Auto-entangle & PBRS**: If the episode is not done, `_auto_entangle()` performs
   one round of background entanglement over all adjacent pairs (ordering shuffled each step
   via the seeded RNG), then the reward is `STEP_COST + [γ·Φ(s') − Φ(s)]` (potential-based
   shaping; `Φ = 0` at the terminal state).

The `info` dictionary returned contains: `fidelity` (end-to-end fidelity if connected, else
0.0), `swaps`, `purifies`, `noops`, `failed_actions` (action counts), `actions` (the clamped
action array), and `terminated` / `truncated` flags.


### 2.6 Auxiliary methods

**`_exec_purify(r)`**: Finds the neighbour with the most available shared links at node `r`
(breaking ties by whichever `np.unique` returns first) and calls `self.net.purify(r,
best_nb)`. If the node has fewer than 2 available qubits or no neighbour shares 2 or more
links, the call is a no-op.

**`_check_e2e()`**: Iterates over the source repeater's occupied qubits and checks whether
any has `partner_repeater == dest`. Returns `(connected: bool, fidelity: float)`. Fidelity
is computed via `werner_to_fidelity` on the matching qubit's Werner parameter.

**`reset()`**: Resets the underlying network, picks new source/destination targets, resets
the step counter, performs one round of auto-entanglement, and returns the initial
observation. The agent always sees the post-entanglement state so it can act immediately.


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

**Default parameters**: `node_dim=9` (matching the 9 observation features), `hidden=32`,
`n_actions=3`. All real checkpoints use `hidden=64`; `load_qnet(path)` infers `(node_dim,
hidden)` from `conv1.lin_l.weight`, so you never construct the architecture by hand.

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
| `"s"` | dict | `{"x": (N,9), "edge_index": (2,E)}` | Current observation |
| `"a"` | ndarray | `(N,)` int32 | Actions taken |
| `"r"` | float | scalar | Reward received |
| `"s_"` | dict | `{"x": (N,9), "edge_index": (2,E)}` | Next observation |
| `"d"` | bool | scalar | Episode `terminated` flag (not raw `done`, so truncations bootstrap) |
| `"m_"` | ndarray | `(N, 3)` bool | Next-state action mask |

The next-state action mask `"m_"` is a key design element. It is stored alongside each
transition so that during training, the target Q-value computation can mask out physically
impossible actions in the successor state. Without this, the agent could learn inflated
Q-values for actions that would never be available.

**Ring buffer mechanics**: When `len(buffer) < max_size`, new entries are appended. Once
full, entries overwrite at position `self.pos` (modulo `max_size`). Sampling uses
`random.sample` for uniform random selection without replacement.


## 5. File: `agent.py` - Double-DQN Agent

### 5.1 Helper functions

**`_obs_to_data(obs, device)`**: Converts a numpy observation dictionary to a PyTorch
Geometric `Data` object on the specified device. This bridges the numpy-based environment
interface with the torch-based model.

**`_running_avg(vals, window)`**: Computes a causal (backward-looking) moving average with
the given window size. Used exclusively for smoothing training metric plots.


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
| `node_dim` | Input feature dimension (must match `env_wrapper` feature count = 9, `NODE_DIM`) |
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


### 5.3 Action selection: `select_actions(obs, mask, training)`

Implements epsilon-greedy action selection with action masking.

**Exploration** (`training=True` and `rng.random() < epsilon`): For each node, the agent
samples uniformly from the set of valid actions defined by the mask. Uses the seeded `rng`
for reproducibility.

**Exploitation** (otherwise): The observation is converted to a `Data` object, passed
through `policy_net` to get `(N, 3)` Q-values, invalid actions are set to `-inf`, and
`argmax` selects the greedy action per node.

Returns an `(N,)` int32 array of actions.


### 5.4 Training step: `train_step()`

Performs one gradient update using a mini-batch from the replay buffer. Returns the scalar
loss value, or `None` if the buffer has fewer samples than `batch_size`.

The computation proceeds as follows:

1. **Sample batch**: `batch_size` transitions are drawn uniformly from the buffer.

2. **Batch construction**: Current states and next states are each assembled into a single
   PyTorch Geometric `Batch` via `Batch.from_data_list`. This concatenates all graphs into
   one large disconnected graph. The `batch` attribute maps each node to its source graph.

3. **Reward/done broadcasting**: Per-graph scalar rewards and done flags are broadcast to
   every node in the corresponding graph using the `node_to_graph = states.batch` index.
   This implements shared global reward: every node in the network receives the same reward
   signal.

4. **Current Q-values**: `policy_net(states)` produces Q-values for all nodes. The actually
   taken action is used to index via `gather`, yielding `current_q` of shape
   `(total_nodes,)`.

5. **Target Q-values (Double DQN with mask)**:
   - The **policy net** evaluates next states. Invalid next-state actions are masked to
     `-inf` using the stored `next_masks`. The argmax over these masked Q-values selects the
     best valid action. This is the "double" part: the policy net selects, but...
   - The **target net** evaluates the same next states, and the Q-value at the
     policy-selected action is extracted via `gather`.
   - The Bellman target is: `target_q = reward + gamma * next_q * (1 - done)`.

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

**Epsilon schedule**: Cosine annealing from `eps_init=1.0` to `eps_fin=0.05` over the first
90% of training, then held constant at `eps_fin`:

```
For ep < 0.9 * episodes:
    epsilon = eps_fin + 0.5 * (eps_init - eps_fin) * (1 + cos(pi * ep / episodes))

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

The method returns a dictionary of per-episode metrics: `reward`, `loss`, `steps`, `success`
(plus `eval` and, under `compare=True`, per-policy comparison columns).


### 5.6 Validation: `validate(model_path, ...)`

Evaluates the trained agent against three baseline strategies on a fixed network
configuration. If `model_path` is provided, the policy network weights are loaded from disk.
During validation, epsilon is set to 0 (pure exploitation).

The strategies compared are:

| Strategy | Source | Behaviour |
|---|---|---|
| Agent | `select_actions(training=False)` | Greedy Q-value policy |
| SwapASAP | `strategies.swap_asap` | Swap at every node that can, every step |
| PurifySwap | `strategies.purify_then_swap` | Purify if possible, else swap if possible |
| Random | `strategies.random_policy` | Uniform random valid action per node |

> `BeliefPropagationPolicy` and `fidelity_gated_swap` were deliberately removed 2026-07-09
> (out of scope for the paper; recoverable from git history).

Each strategy runs `n_episodes` episodes on identically configured environments. Results
(average steps to success, average end-to-end fidelity, success rate) are printed in a table.

When `plot_actions=True`, the first episode of each strategy is recorded as an action
timeline and plotted as a grid (see plotting methods below). When `verbose=1`, the agent's
episodes additionally produce per-step geometric renderings of the network state, saved to
`save_dir/visual/state_{step}.png`.


### 5.7 Plotting methods (brief)

**`_plot_training(metrics, save_path)`**: Generates a 3-panel figure showing episode return
(with running average), loss (log scale), and success rate over training. Saved as
`training_metrics.png`.

**`_print_results_table(results, N, pg, ps, c)`**: Prints a formatted ASCII table of
validation results (average steps with standard deviation, average fidelity with standard
deviation, success percentage) for each strategy.

**`_plot_timeline_grid(timelines, N, pg, ps, c, save_dir)`**: Visualises action sequences
from the first validation episode. Each cell represents one node at one timestep. The cell
colour encodes the repeater identity (from a colourmap). Hatching distinguishes actions:
solid = NOOP, `///` = SWAP, `...` = PURIFY. A black patch marks the terminal step. Saved as
`validation_actions.png`.


## 6. File: `strategies.py` - Baseline Policies

Three heuristic strategies are provided for benchmarking. All respect the action mask
(source/destination are NOOP) and return an `(N,)` int32 action array.

**`swap_asap(env)`**: At every interior node where the mask allows SWAP, assign SWAP. This
is the most aggressive strategy: it extends entanglement reach as fast as possible but does
not improve fidelity through purification. Contention (multiple swaps competing for the same
qubit) is handled gracefully by the underlying `network.swap()` method.

**`purify_then_swap(env)`**: At each node, prefer PURIFY if available; otherwise SWAP if
available; otherwise NOOP. This prioritises link quality over speed. It is also the
winnability-feasibility oracle used by `WinnabilityCache` (2026-07-12; swap-asap can livelock
at `n_ch=4`, purify-then-swap does not).

**`random_policy(env, rng)`**: At each node, sample uniformly from the set of valid actions.
Takes an **explicit RNG that must be independent of `env.rng`**, sharing it would perturb the
environment's own generation/BSM coin flips and invalidate the comparison. Lower-bound baseline.

> `fidelity_gated_swap` and `BeliefPropagationPolicy` were deliberately removed 2026-07-09
> (out of scope for the paper; recoverable from git history).


## 7. Typical Training and Evaluation Flow

A standard usage of the RL stack follows these steps:

1. **Instantiate agent**: `agent = QRNAgent(hidden=64, lr=3e-4, gamma=0.99, ...)`.

2. **Train**: `metrics = agent.train(episodes=3000, n_range=[4,5,6,7], topology='chain',
   curriculum=True, p_gen_std=0.15, p_swap_std=0.15, prune_unwinnable=True,
   save_path='runs/exp1/')`. The agent trains on randomised chain networks of increasing size
   with per-repeater inhomogeneous link parameters.

3. **Validate**: `results = agent.validate(n_repeaters=8, n_episodes=100, topology='chain',
   save_dir='runs/exp1/')`. The agent is tested on a fixed 8-node chain (larger than any
   training size) against the three baselines.

4. **Inspect**: Review `training_metrics.png` for learning curves and
   `validation_actions.png` for qualitative action patterns.


## 8. Key Design Decisions

**Per-node Q-values with shared reward**: Rather than a single centralised action, the agent
selects an action for each node independently. All nodes share the same scalar reward. This
is a form of parameter-shared multi-agent RL where the GNN enables implicit coordination
through message passing.

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

**Cosine epsilon annealing**: Smoother than linear decay. The cosine schedule maintains
higher exploration in early training (where the value function is unreliable) and decays
smoothly to the final exploration rate. The last 10% of training holds epsilon constant at
the minimum to stabilise final performance.
