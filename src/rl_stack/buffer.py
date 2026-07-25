"""
--------------------------------------------------------------------------------
Simple replay buffer storing transitions as dicts of numpy arrays.
--------------------------------------------------------------------------------
"""

from __future__ import annotations
import random
from typing import List, Dict


class ReplayBuffer:
    """Fixed-size ring buffer of PER-DECISION transitions (s, a, ai, r, s',
    ai', mask', terminated, gamma_eff) -- one micro-decision at env.active_node,
    not a whole-graph broadcast."""

    def __init__(self, max_size: int = 50_000, seed: int = None):
        self.max_size = max_size
        self.buffer = [] # List[Dict[str, Any]]
        self.pos = 0
        # Replay sampling affects which transitions shape the learned weights,
        # hence metrics.json. Give the buffer its OWN generator seeded from the
        # master seed so training is bit-reproducible; seed=None keeps the prior
        # nondeterministic behavior (module-global random).
        self._rng = random.Random(seed) if seed is not None else random

    def add(self, s, a, active_idx, r, s_, next_active_idx, next_mask_row,
            terminated, gamma_eff):
        entry = {"s": s, "a": int(a), "ai": int(active_idx), "r": float(r),
                 "s_": s_, "nai": int(next_active_idx),
                 "m_": next_mask_row, "d": bool(terminated),
                 "g": float(gamma_eff)}
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size: int) -> List[Dict]:
        return self._rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.pos = 0