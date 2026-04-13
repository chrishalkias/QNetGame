# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QNetGame is a **discrete-time quantum repeater network simulator** (pure NumPy) plus a **Double-DQN RL pipeline** (PyTorch + `torch_geometric`) that learns per-node routing policies (swap / purify / wait) on chain topologies and is evaluated for zero-shot generalisation to larger networks.

## Commands

All Python entry points are CLI-driven with `argparse` and launched from shell wrappers in `scripts/`. Always run from the repo root with `PYTHONPATH=.` (the `scripts/*.sh` wrappers set this).

```bash
# Training (local)
./scripts/train.sh              # small local run into checkpoints/local/
./scripts/submit.sh             # SLURM job (ALICE HPC) — 50k episodes, GPU

# Validation
./scripts/test.sh               # agent vs heuristic baselines on a fixed config
./scripts/validate.sh           # SLURM sweep via train-test/batch_validate.py
./scripts/partial_validate.sh

# Direct python invocation (mirror what the .sh wrappers do)
PYTHONPATH=. python -u train-test/train.py      --run_id <id> [flags]
PYTHONPATH=. python -u train-test/validation.py --run_id <id> --path checkpoints/<id>/ --dict policy.pth [flags]

# Tests (pytest or unittest; run from repo root)
python -m pytest diagnostics/unittests/test_simulator.py -v    # physics engine
python -m pytest diagnostics/unittests/test_rl_stack.py  -v    # RL pipeline
python -m pytest diagnostics/unittests/test_simulator.py::TestSwap::test_swap_product_rule -v   # single test

# Mutation testing
cd diagnostics/mutations && make         # mutmut run / results

# Diagnostics (policy probes)
PYTHONPATH=. python diagnostics/policy_probes/policy_interpretation.py   <ckpt>
PYTHONPATH=. python diagnostics/policy_probes/generate_policy_explorer.py <ckpt>
PYTHONPATH=. python diagnostics/policy_probes/policy_physics_check.py    <ckpt>
```

Dependencies are pinned in `requirements.txt` (numpy 2.4, torch 2.6, torch_geometric 2.7, matplotlib, networkx, pytest, mutmut). A `.venv/` exists at the repo root (Python 3.13).

## Architecture

Three layers, each built on the one below:

```
quantum_repeater_sim/   pure-NumPy physics engine
    repeater.py         Repeater: per-qubit __slots__ arrays (status, partner,
                        werner_param, age, link_cutoff, locked)
    network.py          RepeaterNetwork: entangle / swap / purify / age_links,
                        action masks, build_chain, build_grid
    graph_builder.py    network_to_heterodata() → PyG HeteroData observation
                        (falls back to numpy dict if torch_geometric missing)

rl_stack/               Double-DQN agent over the simulator
    env_wrapper.py      QRNEnv: reset/step, 8 per-node features, auto-entangle
                        background, action space {NOOP=0, SWAP=1, PURIFY=2}
    model.py            QNetwork: 3-layer GraphSAGE + 2-layer MLP head,
                        size-agnostic per-node Q-values
    agent.py            QRNAgent: Double-DQN, Polyak τ=0.005, cosine ε-schedule,
                        curriculum, domain randomisation; .train() / .validate()
    buffer.py           ReplayBuffer (ring buffer over transitions)
    strategies.py       Heuristic baselines: swap_asap, purify_then_swap,
                        entangle_only, random_policy

train-test/             CLI entry points that wire the above together
    train.py            Argparse → QRNAgent.train() → policy.pth + plots
    validation.py       Argparse → QRNAgent.validate() vs baselines
    batch_validate.py   Parameter sweeps for HPC evaluation
    partial_validate.py

diagnostics/
    unittests/          test_simulator.py (67 tests), test_rl_stack.py (58 tests)
    mutations/          mutmut config + runner
    policy_probes/      Q-value interpretability tools (static + interactive HTML)
```

### Physics model (critical invariants)

- Every entangled pair = one Werner scalar `p`; fidelity `F = (3p+1)/4`.
- **Swap:** `p_new = p1 * p2`; **BBPSSW purify:** closed-form success prob and fidelity update. Both are derived in §2 of `README.md`; tests in `test_simulator.py` assert the formulas.
- **Decoherence:** `p(m) = p0 * exp(-m / c_eff)` per tick, where `c_eff = min(cutoff_A, cutoff_B)`.
- **Classical-communication delays** are the key design constraint. `swap`/`purify` are **two-phase**: the measurement outcome is frozen immediately, involved qubits are **locked**, and an event is queued with `timer = ceil(d / (c_fiber * dt_seconds))`. `age_links()` resolves the event when the timer hits 0. Failed BSMs resolve on the spot with no lock. Setting `dt_seconds=0.0` makes all events resolve on the next `age_links()` call — this is the mode used for RL training.
- There are **two query layers** on `Repeater`: raw (`occupied_indices`, …) includes locked qubits for internal bookkeeping; agent-facing (`available_indices`, `can_swap`, action masks) excludes locked qubits so the agent never sees in-flight qubits.
- Resolution routines include guards for the edge case where a locked qubit expires via cutoff before its event resolves — clean up locks, skip the resolution, do not corrupt state. Any change to `age_links`, `swap`, `purify`, or expiry handling must keep this guard intact.

### RL step semantics (`QRNEnv.step`)

Order per step: `purify → swap → age_links (resolve events, decohere, expire) → check end-to-end → auto-entangle all adjacent pairs → build observation`. Source and destination nodes are forced to NOOP. The action mask is guaranteed to have NOOP True on every node so `argmax` over masked Q-values never hits all −∞.

**Observation — 8 per-node features** (all in `[0,1]`):
`frac_occupied, mean_fidelity, is_source, is_dest, frac_available, can_swap, can_purify, time_remaining`.

**Reward:** `SUCCESS_REWARD = 1.0` (weighted by e2e fidelity), `STEP_COST = -0.01`, `FAILED_ACTION = -0.05` per invalid attempt, plus **PBRS** shaping `γ·Φ(s') − Φ(s)` where `Φ = farthest_hop_from_source / total_hops` (BFS over entangled links). PBRS potential is **chain-specific** — grid/GEANT topologies need a different potential.

### Design decisions to respect

1. **Entanglement is not an agent action.** The env auto-entangles every adjacent pair at the start of each step; the agent only picks swap/purify/noop. Do not move entangle into the action space without updating masks, reward, and `test_rl_stack.py`.
2. **Action masking happens at selection AND inside the Double-DQN target computation** (invalid actions set to −∞ before `argmax`). If you edit agent update logic, preserve both.
3. **GNN must stay local.** `model.py` uses SAGEConv + per-node linear layers only, so weights transfer across chain sizes. Do not introduce global pooling or fixed-N layers.
4. **Curriculum and domain randomisation** (`n_range`, `heterogeneous=True`) are part of why training generalises — keep them when tweaking `agent.train()`.

## Conventions

- Use the module names from `quantum_repeater_sim/__init__.py` (`Repeater`, `RepeaterNetwork`, `SwapPolicy`, `build_chain`, `build_grid`, `network_to_heterodata`, the `*_werner` helpers) and `rl_stack/__init__.py` (`QRNEnv`, `QRNAgent`, `ReplayBuffer`, `strategies`, action constants `NOOP/SWAP/PURIFY/N_ACTIONS`).
- `network.py` caches `_positions` and `_dist_matrix` at construction — invalidate them if you ever add mutable topology.
- Checkpoints live in `checkpoints/<run_id>/policy.pth` (plus training plots). Validation defaults to `--path checkpoints/<run_id>/ --dict policy.pth`.
- Reproducibility: pass `rng=np.random.default_rng(seed)` into `build_chain` / `RepeaterNetwork`. Tests rely on this.
- Background notes, findings, and cluster details live in `a_project_information/` (notably `guidelines.md`, `QRN_note.md`, `RL_note.md`, `cluster_info.md`). Read these before making non-trivial design changes.

## Known quirks / open items

- `TODO.md` flags an open question: what happens when `swap(1,2)` and `swap(2,3)` touch the same qubit in the same tick? Behaviour is not yet pinned down — check before relying on it.
- Werner model is isotropic depolarising only (no anisotropic T₂). Moving to Bell-diagonal would change swap from scalar multiply to a 4×4 matrix product.
- Entanglement generation is instantaneous (no event queue). The `pending_events` machinery exists and could be reused if long-link heralding delay is needed.
- Purification is underused by the trained agent — the sparse binary reward does not strongly incentivise fidelity beyond the minimum needed for e2e connectivity.
- CC delays (`dt_seconds > 0`) are exercised by the simulator tests but **disabled in training** (`dt_seconds=0.0`). Enabling them changes task difficulty substantially.
