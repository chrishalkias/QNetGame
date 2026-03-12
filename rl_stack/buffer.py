"""Simple replay buffer storing transitions as dicts of numpy arrays."""

from __future__ import annotations
import random
from typing import List, Dict


class ReplayBuffer:
    """Fixed-size ring buffer for (s, a, r, s', done, mask') transitions."""

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self.buffer = [] # List[Dict[str, Any]]
        self.pos = 0

    def add(self, state, actions, reward, next_state, done, next_mask):
        entry = {"s": state, 
                 "a": actions, 
                 "r": reward,
                 "s_": next_state, 
                 "d": done, 
                 "m_": next_mask
                 }
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.pos = 0