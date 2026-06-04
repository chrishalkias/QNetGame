"""Phase configuration for the curriculum trainer (pure data, no RL imports)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PhaseConfig:
    """Immutable description of one curriculum phase.

    `n_range` and `n_ch` are tuples sampled from per-episode by the trainer
    (n_range via the size curriculum, n_ch uniformly)."""
    name: str
    topology: str
    n_range: Tuple[int, ...]
    n_ch: Tuple[int, ...]
    p_gen: float
    p_swap: float
    cutoff: int
    F0: float
    channel_loss: float
    dt_seconds: float
    backend: str
    fidelity_mode: str
    episodes: int
    max_steps: int
    heterogeneous: bool


# Concrete Phase 1: small chains, slight imperfections -> learns near-optimal
# fast. cutoff=5 and (p_gen,p_swap)=(0.9,0.9) match the optimal-policy pickle
# produced by train-test/optimal_baseline.py, so gap-to-optimal is apples-to-apples.
PHASE1 = PhaseConfig(
    name="phase1",
    topology="chain",
    n_range=(4, 5),
    n_ch=(2, 3),
    p_gen=0.9,
    p_swap=0.9,
    cutoff=5,
    F0=0.95,
    channel_loss=0.0,
    dt_seconds=0.0,
    backend="legacy",
    fidelity_mode="analytic",
    episodes=8000,
    max_steps=30,
    heterogeneous=False,
)

# Phase 2: square grids (n_range = grid side length), random non-adjacent
# source/dest, topology-general PBRS. Grid training is cluster-only.
PHASE2 = PhaseConfig(
    name="phase2",
    topology="grid",
    n_range=(3, 4),          # 3x3 and 4x4 grids
    n_ch=(2, 3),
    p_gen=(0.5, 0.9),        # (lo, hi) -> per-episode domain randomization
    p_swap=(0.6, 0.9),
    cutoff=20,
    F0=0.95,
    channel_loss=0.0,
    dt_seconds=0.0,
    backend="legacy",
    fidelity_mode="analytic",
    episodes=20000,          # cap; early-stopping + best-checkpoint trim it
    max_steps=60,
    heterogeneous=False,
)
