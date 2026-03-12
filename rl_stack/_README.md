# `quantum_repeater_sim.rl` — RL Agent Module

A Double-DQN agent that learns multi-node routing policies on quantum
repeater chains and generalises zero-shot to larger, differently-parameterised
networks.

**Requires:** `torch`, `torch_geometric`, `matplotlib` (in addition to the
base simulator's `numpy`).

---

## Architecture

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

## Step Semantics

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

## Action Space

Per node, 4 discrete actions:

| Index | Name | Condition |
|---|---|---|
| 0 | `noop` | Always valid |
| 1 | `entangle` | Node and some neighbour both have free unlocked qubits |
| 2 | `swap` | Node has ≥ 2 available (occupied, unlocked) qubits |
| 3 | `purify` | Node shares ≥ 2 available pairs with some neighbour |

The agent outputs `(N,)` actions simultaneously — the system is "frozen
in time" while the agent decides, then all actions execute in one step.

## Observation Space

**Node features** `(N, 7)`, all in [0, 1]:

| Column | Feature |
|---|---|
| 0 | `frac_occupied` — occupied qubits / n_ch |
| 1 | `mean_fidelity` — average Werner fidelity of occupied qubits |
| 2 | `is_source` — 1 if this node is the e-e source |
| 3 | `is_dest` — 1 if this node is the e-e destination |
| 4 | `frac_available` — available (unlocked) qubits / n_ch |
| 5 | `can_swap` — 1 if swap is possible |
| 6 | `has_purify_option` — 1 if purification is possible |

**Edges:** the repeater adjacency graph (both directions).

Because all features are normalised and topology-agnostic, the GNN
processes any chain length without retraining.

## GNN Model (`model.py`)

Two-layer GraphSAGE with a 2-layer MLP head:

```
Input: (N, 7) → SAGEConv(7→64) → ReLU → SAGEConv(64→64) → ReLU
     → Linear(64→64) → ReLU → Linear(64→4) → (N, 4) Q-values
```

All layers are local (message-passing + per-node linear), so the model
is size-agnostic by construction.

## Training (`agent.py: QRNAgent.train`)

```python
from quantum_repeater_sim.rl import QRNAgent

agent = QRNAgent(lr=5e-4, gamma=0.99, batch_size=64, buffer_size=50000)
metrics = agent.train(
    episodes=3000,
    max_steps=50,
    n_range=[4, 5, 6],       # train on small chains
    jitter=1,                 # re-sample N every episode
    curriculum=True,          # progressive difficulty
    heterogeneous=True,       # randomise per-repeater params
    p_gen=0.8, p_swap=0.7,
    cutoff=15,
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

## Validation (`agent.py: QRNAgent.validate`)

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
| **PurifyThenSwap** | Purify first, then swap, then entangle |
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

## Heuristic Strategies (`strategies.py`)

Available as standalone functions:

```python
from quantum_repeater_sim.rl import strategies

actions = strategies.swap_asap(env)           # (N,) int array
actions = strategies.purify_then_swap(env)
actions = strategies.entangle_only(env)
actions = strategies.random_policy(env, rng)
```

Each function respects the current action mask and returns valid actions.

## Replay Buffer (`buffer.py`)

Circular buffer storing `(state, actions, reward, next_state, done)`:

```python
from quantum_repeater_sim.rl import ReplayBuffer

buf = ReplayBuffer(max_size=50_000)
buf.add(obs, actions, reward, next_obs, done)
batch = buf.sample(64)
```

## Zero-Shot Generalisation

The design enables training on small chains (N=4–6) and testing on
larger ones (N=10–20+) because:

1. **Node features are normalised** — fractions and binary flags, no
   absolute counts or positions.
2. **GNN is local** — SAGEConv aggregates from 1-hop neighbours. The
   number of parameters is independent of N.
3. **Action space is per-node** — 4 actions regardless of topology size.
4. **Domain randomisation** — heterogeneous p_gen/p_swap during training
   prevents overfitting to specific parameter regimes.
5. **Curriculum** — progressive difficulty teaches general patterns before
   scaling up.

## Test Suite

```
python -m quantum_repeater_sim.rl.test_rl
```

39 checks across 18 test groups, covering: environment creation across
N=3..15, action mask shapes and validity, step execution, end-to-end
detection, reward structure, max-steps termination, observation feature
normalisation, all action types, reset idempotency, strategy validity
and solvability, replay buffer operations, heterogeneous parameters,
classical delays, and a 50-episode stress test with random configs.