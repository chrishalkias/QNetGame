<p align="center">
  <img src="logo.svg" width="180" alt="QNetGame logo">
</p>

# QNetGame

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?logo=latex&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A discrete-time quantum repeater network simulator with classical communication
delays, built as the training environment for a reinforcement-learning agent
that schedules **swap** / **purify** / **wait** decisions along a repeater
chain. The simulator (`src/simulator`) is pure NumPy; the RL stack
(`src/rl_stack`) is a GraphSAGE Double-DQN agent (PyTorch + PyTorch Geometric)
that trains on small, randomized chains and generalises zero-shot to larger
and out-of-distribution ones.

**Dependencies:** NumPy (core). `torch` + `torch_geometric` for the RL stack.

---

## Quick Start

```python
import numpy as np
from simulator import build_chain

net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.7, cutoff=15,
                  F0=0.95, channel_loss=0.02, dt_seconds=1e-4,
                  rng=np.random.default_rng(42))

for step in range(100):
    net.age_links()      # tick the clock, resolve pending events
    net.entangle(0, 1)   # or net.swap(2), net.purify(0, 1)
```

```python
from rl_stack import QRNEnv

env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.8, p_swap=0.7,
             cutoff=15, topology='chain', rng=np.random.default_rng(0))

obs  = env.get_observation()          # {"x": (N,9), "edge_index": (2,E)}
mask = env.get_action_mask()          # (N,3) bool: NOOP / SWAP / PURIFY
obs, reward, done, info = env.step(np.zeros(env.N, dtype=int))
```

Run from the repo root with `PYTHONPATH=src:.` (or use the wrappers in
`experiments/scripts/`).

---

## Physics Model

Every entangled pair is a **Werner state**, a single scalar `p ∈ [0,1]`,
fidelity `F = (3p+1)/4`.

| Operation | Rule |
|---|---|
| Generation | succeeds w.p. `(p_gen_i+p_gen_j)/2 · e^{-αd/2}` (if distance-dependent); new pair starts at `p₀ = F0 · e^{-αd}` |
| Swap (BSM) | succeeds w.p. `p_swap`; on success `p_new = p₁·p₂`; on failure both links are destroyed immediately |
| Purify (BBPSSW) | needs ≥2 shared pairs; sacrifices the lower-fidelity one to raise the other, `P_succ = (p₁p₂+1)/2`; failure destroys both |
| Decoherence | each tick `p(age) = p₀·e^{-age/c_eff}`; link destroyed at `age ≥ cutoff` |
| Classical comm delay | BSM/purify outcomes are computed instantly but only take effect after `⌈d / (c_fiber·dt_seconds)⌉` ticks; involved qubits are **locked** (invisible to the agent) until then |

`dt_seconds=0.0` disables classical delays (events resolve on the very next
tick). See `src/simulator/network.py` and `src/simulator/repeater.py` for the
exact implementation, and `tests/test_simulator.py` for the invariants these
formulas are held to.

---

## Architecture

```
  RL Agent (rl_stack) ─────────────┐
    │ observes {x, edge_index}      │ picks (N,) actions
    ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│                        QRNEnv                             │
│  step(actions): purify → swap → age_links → check e2e     │
│                 → auto-entangle                            │
└───────────────────────────┬───────────────────────────────┘
                             │ drives, reads via frozen snapshots
                             ▼
                ┌──────────────────────────┐
                │      RepeaterNetwork       │  build_network(...)
                │  entangle / swap / purify  │
                │  / age_links · Repeater(×N)│
                └──────────────────────────┘
```

The agent picks one of `{NOOP, SWAP, PURIFY}` per node from a 9-feature,
per-node observation (occupancy, fidelity, availability, can-swap/purify,
`p_gen`/`p_swap`, link urgency) over a GraphSAGE Q-network, so it is
size-agnostic and transfers to chains it never trained on. Reward is a small
per-step cost plus fidelity-weighted success plus PBRS shaping
(`src/rl_stack/potential.py`) toward the current best source→dest shortcut.
Entanglement generation is never an agent action, it's an automatic
background process the agent schedules around.

---

## Repository Layout

```
src/
  simulator/                 core physics engine (NumPy only, imports without torch)
  rl_stack/                  Double-DQN RL stack (torch imports guarded)
experiments/                 entry-point scripts (argparse at top of file)
  mc_eval.py                 canonical censored delivery-time evaluator
  training/                  train.py, validation.py, batch_validate.py, replot.py
  comparisons/                paper figure suite (delivery_vs_*)
  policy_probes/              interpretability probes
  q_heuristic/                 stochastic q-heuristic control experiment
  scripts/                    local/ SLURM/ sync/ shell wrappers
tests/                        pytest suites (physics, RL stack, env, potential, ...)
```

---

## Running Tests

```bash
PYTHONPATH=src:. python -m pytest tests -v
```

Tests give structural + physics-formula correctness guarantees; they do not
guarantee the RL agent is learning anything useful.

---

## Known Limitations

- **Noise model** is isotropic depolarising (Werner) only, no anisotropic
  dephasing.
- **Generation is instantaneous**; the CC-delay machinery used for swap/purify
  isn't applied to entanglement generation itself.
- **Single-instance simulator**; vectorised training runs `B` copies via
  `multiprocessing`, there's no batched-array backend.
