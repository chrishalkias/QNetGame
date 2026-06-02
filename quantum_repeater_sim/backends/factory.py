"""Backend factory: selects and builds a PhysicsBackend."""
from __future__ import annotations
from typing import Optional
import numpy as np

from .legacy import LegacyBackend
from ..network import build_chain, build_grid, build_GEANT
from ..repeater import SwapPolicy


def make_backend(
    backend: str = "legacy",
    *,
    topology: str = "chain",
    n_repeaters: int = 5,
    n_ch: int = 4,
    spacing: float = 50.0,
    p_gen: float = 0.8,
    p_swap: float = 0.5,
    cutoff: int = 20,
    F0: float = 0.95,
    channel_loss: float = 0.02,
    dt_seconds: float = 1e-4,
    heterogeneous: bool = False,
    rng: Optional[np.random.Generator] = None,
    fidelity_mode: str = "analytic",
):
    """Build a PhysicsBackend. `fidelity_mode` is reserved for NetSquid (M1+)."""
    rng = rng if rng is not None else np.random.default_rng()
    if backend == "legacy":
        net = _build_legacy_net(topology, n_repeaters, n_ch, spacing,
                                p_gen, p_swap, cutoff, F0, channel_loss,
                                dt_seconds, rng)
        if heterogeneous:
            for rep in net.repeaters:
                rep.p_gen = rng.uniform(0.3, 1.0)
                rep.p_swap = rng.uniform(0.3, 1.0)
        return LegacyBackend(net)
    if backend == "netsquid":
        raise NotImplementedError("NetSquidBackend lands in M1")
    raise ValueError(f"Unknown backend {backend!r}")


def _build_legacy_net(topology, n_repeaters, n_ch, spacing, p_gen, p_swap,
                      cutoff, F0, channel_loss, dt_seconds, rng):
    if topology == "chain":
        return build_chain(
            n_repeaters, n_ch=n_ch, spacing=spacing,
            p_gen=p_gen, p_swap=p_swap, cutoff=cutoff,
            F0=F0, channel_loss=channel_loss, dt_seconds=dt_seconds,
            distance_dep_gen=True, rng=rng)
    if topology == "grid":
        return build_grid(
            rows=n_repeaters, cols=n_repeaters, n_ch=n_ch, spacing=spacing,
            swap_policy=SwapPolicy.FARTHEST, p_gen=p_gen, p_swap=p_swap,
            cutoff=cutoff, rng=rng)
    if topology == "geant":
        return build_GEANT(
            n_ch=n_ch, swap_policy=SwapPolicy.FARTHEST,
            p_gen=p_gen, p_swap=p_swap, cutoff=cutoff, rng=rng)
    raise ValueError(f"Unknown topology {topology!r}")
