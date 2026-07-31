# QNetGame

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?logo=latex&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A discrete-time quantum repeater chain simulator, built as the training
environment for a reinforcement-learning agent that schedules **swap** /
**purify** / **wait** decisions along a repeater
chain. The simulator (`src/simulator`) is pure NumPy; the RL stack
(`src/rl_stack`) is a GraphSAGE Double-DQN agent (PyTorch + PyTorch Geometric)
that trains on small, randomized chains and generalises zero-shot to larger
and out-of-distribution ones.

**Dependencies:** `requirements.txt` is everything needed to train, evaluate and
test. `requirements-extra.txt` holds the cluster-only analysis extras (pandas,
seaborn, scikit-learn) that only `batch_validate.py` and
`q_heuristic/fit_q_conditional.py` import.

---

## Quick Start

```python
import numpy as np
from simulator import build_chain

net = build_chain(n_repeaters=5, n_ch=4, spacing=50.0,
                  p_gen=0.8, p_swap=0.7, cutoff=15,
                  F0=0.95, channel_loss=0.02,
                  rng=np.random.default_rng(42))

for step in range(100):
    net.age_links()      # tick the clock: decohere + expire
    net.entangle(0, 1)   # or net.swap(2), net.purify(0, 1) -- applied immediately
```

```python
from rl_stack import QRNEnv

env = QRNEnv(n_repeaters=5, n_ch=4, p_gen=0.8, p_swap=0.7,
             cutoff=15, rng=np.random.default_rng(0))

obs  = env.reset()                        # {"x": (N,8), "edge_index": (2,E)}
mask = env.action_mask(env.active_node)   # (3,) bool: NOOP / SWAP / PURIFY
obs, reward, done, info = env.step(0)     # ONE scalar action, for env.active_node
```

One `step` is one micro-decision by the current **active** interior node.
Interior nodes are visited left to right; after the last one the tick boundary
runs `age_links` then background auto-entanglement.

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
| Purify (BBPSSW) | needs ≥2 shared pairs; a sorted-adjacent distillation cascade folds all of them into one survivor, `P_succ = (p₁p₂+1)/2`, failure destroys both inputs; a round is only attempted when it beats keeping the stronger link |
| Decoherence | each tick `p(age) = p₀·e^{-age/c_eff}`; link destroyed at `age ≥ cutoff` |

Swap and purify outcomes are drawn and applied **immediately**, the instant the
node acts. Classical-communication delays were removed from the engine on
2026-07-22 (`dt_seconds` and the pending-event queue are gone). A swap is only
offered if the fused link would survive the tick boundary,
`age_a + age_b + 1 < cutoff`.
See `src/simulator/network.py` and `src/simulator/repeater.py` for the
exact implementation, and `tests/test_simulator.py` for the invariants these
formulas are held to.

---

## Architecture

```
  RL Agent (rl_stack) ─────────────┐
    │ observes {x: (N,8),           │ picks ONE scalar action
    │           edge_index}         │ for env.active_node
    ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│                        QRNEnv                            │
│  step(action): apply at active_node (immediately)        │
│                → check e2e → advance the sweep cursor    │
│  tick boundary: age_links → check e2e → auto-entangle    │
└───────────────────────────┬──────────────────────────────┘
                            │ drives, reads via net.node(i)
                            ▼
                ┌────────────────────────────┐
                │     RepeaterNetwork        │  build_network(...)
                │  entangle / swap / purify  │
                │  / age_links · Repeater(xN)│
                └────────────────────────────┘
```

`src/rl_stack/` is `agent.py` (Double-DQN training loop), `buffer.py` (replay),
`env_wrapper.py` (`QRNEnv`), `model.py` (the GraphSAGE Q-network),
`plots.py` (all matplotlib), `policies.py` (heuristic baselines + the
winnability oracle) and `potential.py` (the PBRS potential).

The agent picks one of `{NOOP, SWAP, PURIFY}` for the active node from an
8-feature, per-node observation over a GraphSAGE Q-network, so it is
size-agnostic and transfers to chains it never trained on:

| idx | feature | definition |
|---|---|---|
| 0 | `frac_occupied` | occupied / physical capacity (`2*n_ch` interior, `n_ch` ends) |
| 1 | `can_swap` | one available LEFT link + one available RIGHT link, fused link viable (0 at endpoints) |
| 2 | `can_purify` | >=2 available qubits to the same partner (0 at endpoints) |
| 3 | `p_gen` | per-repeater generation prob. (the inhomogeneity signal) |
| 4 | `p_swap` | per-repeater BSM success prob. |
| 5 | `normalized_age` | mean(`age/link_cutoff`) over occupied qubits, ~1 near expiry |
| 6 | `relative_position` | `i / (N-1)`: 0.0 at source, 1.0 at dest |
| 7 | `is_active` | 1.0 at `env.active_node`, the node deciding this micro-step |

Reward is a per-tick step cost plus fidelity-weighted success plus PBRS shaping
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
  comparisons/               paper figure suite (shared _common.py, merge_json.py)
    policy_vs_agent/         one agent vs the fixed heuristics (delivery_vs_*)
    agent_vs_agent/          checkpoint vs checkpoint (seeds, training length)
  policy_probes/             interpretability probes
  q_heuristic/               stochastic q-heuristic control experiment
  scripts/                   _setup.sh, comparison.sh, _local_run.sh, sync.sh, train_*.sh
tests/                       pytest suites (physics, RL stack, env, potential, ...)
```

---

## Running Tests

```bash
PYTHONPATH=src:. python -m pytest tests/ -q              # everything
PYTHONPATH=src:. python -m pytest tests/ -q -m "not slow" # skip the golden-numbers gate
```

The `PYTHONPATH=src:.` prefix is mandatory; without it every test module fails
to import.

Tests give structural + physics-formula correctness guarantees; they do not
guarantee the RL agent is learning anything useful.

---

## Known Limitations

- **Noise model** is isotropic depolarising (Werner) only, no anisotropic
  dephasing.
- **No classical-communication delays.** Swap and purify outcomes apply
  immediately; the CC event queue was removed from the engine on 2026-07-22.
- **Chain topology only.** The grid / GEANT builders were removed; they were
  never used by the paper.
- **Single-instance simulator**, no batched-array or vectorised-env backend.
