"""
Backend factory: selects and builds a PhysicsBackend.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np

from .legacy import LegacyBackend
from ..network import build_chain, build_grid, build_GEANT
from ..repeater import SwapPolicy


def _sample_matched_uniform(mean, std, size, rng, lo=0.05, hi=1.0):
    """Per-repeater rates drawn from a uniform with variance ``std**2``, centred
    on ``mean`` and clipped to ``[lo, hi]``.

    A uniform on ``[mean - sqrt(3)*std, mean + sqrt(3)*std]`` has standard
    deviation exactly ``std`` (before clipping). ``std <= 0`` broadcasts the
    clipped ``mean`` and consumes NO rng draw, so the homogeneous path keeps the
    pre-inhomogeneity RNG stream bit-for-bit.
    """
    if std <= 0.0:
        return np.full(size, float(np.clip(mean, lo, hi)))
    hw = math.sqrt(3.0) * std
    return np.clip(rng.uniform(mean - hw, mean + hw, size=size), lo, hi)


def make_backend(
    backend: str = "legacy",
    *,
    topology: str = "chain",
    n_repeaters: int = 5,
    n_ch: int = 4,
    spacing: float = 50.0,
    p_gen: float = 0.8,
    p_swap: float = 0.5,
    p_gen_std: float = 0.0,
    p_swap_std: float = 0.0,
    cutoff: int = 20,
    F0: float = 0.95,
    channel_loss: float = 0.02,
    dt_seconds: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
):
    """Build a PhysicsBackend.

    Inhomogeneity: `p_gen`/`p_swap` are the per-network MEANS; `p_gen_std`/
    `p_swap_std` spread per-repeater values via `_sample_matched_uniform`
    (std=0 -> homogeneous, no rng draw).
    """
    rng = rng if rng is not None else np.random.default_rng()
    if backend == "legacy":
        net = _build_legacy_net(topology, n_repeaters, n_ch, spacing,
                                p_gen, p_swap, cutoff, F0, channel_loss,
                                dt_seconds, rng)
        if p_gen_std > 0.0 or p_swap_std > 0.0:
            pg = _sample_matched_uniform(p_gen, p_gen_std, net.N, rng)
            ps = _sample_matched_uniform(p_swap, p_swap_std, net.N, rng)
            for i, rep in enumerate(net.repeaters):
                rep.p_gen, rep.p_swap = float(pg[i]), float(ps[i])
        return LegacyBackend(net)
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
