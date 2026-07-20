"""Lazy purify-then-swap winnability oracle.

A cell `(p_gen, p_swap, cutoff, N, n_ch)` is *winnable* if purify-then-swap
delivers end-to-end at least `min_deliveries` times over `n_pilots` rollouts at a
generous `probe_steps` horizon, **under the same (idealized) physics as
training** — otherwise "unwinnable" would conflate physics with a regime the
agent never trains in. Results are cached per coarse bin so each region of the
parameter space is probed only once; pruning the training distribution then
costs essentially nothing after warmup.

purify-then-swap is the oracle (user decision 2026-07-12): under the repaired
cutoff physics purification extends link lifetimes, so it dominates swap-asap as
a feasibility oracle; its n_ch=4 livelock risk is bounded by `probe_steps` and a
livelocked pilot just marks the cell unwinnable (conservative).
"""
from __future__ import annotations
import numpy as np

from rl_stack.env_wrapper import QRNEnv
from rl_stack import strategies


def _bin(p_gen, p_swap, cutoff, n_repeaters, n_ch):
    """Coarse bin: rates to 0.1, cutoff to buckets of 5, N and n_ch exact."""
    return (round(float(p_gen), 1), round(float(p_swap), 1),
            int(cutoff) // 5, int(n_repeaters), int(n_ch))


class WinnabilityCache:
    def __init__(self, n_pilots=5, probe_steps=400, min_deliveries=1, seed=0,
                 dt_seconds=0.0, channel_loss=0.0, F0=1.0):
        self.n_pilots = n_pilots
        self.probe_steps = probe_steps
        self.min_deliveries = min_deliveries
        self.rng = np.random.default_rng(seed)
        # physics must match the training regime (idealized by default)
        self.dt_seconds = dt_seconds
        self.channel_loss = channel_loss
        self.F0 = F0
        self._cache: dict = {}
        self.pilot_calls = 0

    def winnable(self, p_gen, p_swap, cutoff, n_repeaters, n_ch) -> bool:
        key = _bin(p_gen, p_swap, cutoff, n_repeaters, n_ch)
        if key in self._cache:
            return self._cache[key]
        deliveries = 0
        for _ in range(self.n_pilots):
            self.pilot_calls += 1
            env = QRNEnv(
                n_repeaters=int(n_repeaters), n_ch=int(n_ch),
                p_gen=float(p_gen), p_swap=float(p_swap), cutoff=int(cutoff),
                max_steps=self.probe_steps, dt_seconds=self.dt_seconds,
                channel_loss=self.channel_loss, F0=self.F0, topology="chain",
                rng=np.random.default_rng(self.rng.integers(2**32)))
            env.reset()
            for _ in range(self.probe_steps):
                _, _, done, info = env.step(strategies.purify_then_swap(env))
                if info.get("terminated"):   # delivered (a win, not a timeout)
                    deliveries += 1
                    break
                if done:
                    break
            if deliveries >= self.min_deliveries:
                break
        result = deliveries >= self.min_deliveries
        self._cache[key] = result
        return result
