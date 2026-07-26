"""
--------------------------------------------------------------------------------
The heuristic baselines and the winnability oracle.

Merged from strategies.py + winnability.py (2026-07-26): both are small,
env-facing, and consumed by the same callers. The PBRS potential deliberately
stayed in potential.py, which imports nothing from the project; merging it here
would create env_wrapper -> policies -> env_wrapper.

Nothing here imports torch, so numpy-only consumers (cluster jobs running the
heuristics) still load when torch_geometric is broken.

  swap_asap          heuristic baseline
  purify_then_swap   heuristic baseline
  random_policy      heuristic baseline (needs an RNG independent of env.rng)
  WinnabilityCache   purify-then-swap pilot oracle for curriculum pruning

Each heuristic takes a QRNEnv and returns a SCALAR action (NOOP/SWAP/PURIFY)
for env.active_node -- the one node deciding this micro-step under the
serialized sweep. All of them respect that node's action mask.

Entanglement is handled automatically by the environment step.
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np

from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP, PURIFY


# --- heuristic baselines ------------------------------------------------------

def swap_asap(env: QRNEnv) -> int:
    """Swap at env.active_node if it can, immediately.

    The engine's swap function handles same-tick contention gracefully
    (a beaten swap simply fails: its qubits are no longer available once an
    earlier micro-step this tick has consumed them).
    """
    r = env.active_node
    return SWAP if env.action_mask(r)[SWAP] else NOOP


def purify_then_swap(env: QRNEnv) -> int:
    """Purify env.active_node if possible, otherwise swap if possible, else noop."""
    r = env.active_node
    m = env.action_mask(r)
    return PURIFY if m[PURIFY] else (SWAP if m[SWAP] else NOOP)


def random_policy(env: QRNEnv, rng: np.random.Generator) -> int:
    """Uniformly random valid action for env.active_node.

    IMPORTANT: *rng* must be independent of env.rng, otherwise drawing
    action choices perturbs the environment's own random stream
    (link generation, swap outcomes) and invalidates the comparison.
    """
    r = env.active_node
    valid = np.flatnonzero(env.action_mask(r))
    return int(rng.choice(valid)) if len(valid) else NOOP


# --- winnability oracle -------------------------------------------------------

def _bin(p_gen, p_swap, cutoff, n_repeaters, n_ch):
    """Coarse bin: rates to 0.1, cutoff to buckets of 5, N and n_ch exact."""
    return (round(float(p_gen), 1), round(float(p_swap), 1),
            int(cutoff) // 5, int(n_repeaters), int(n_ch))


class WinnabilityCache:
    """Lazy purify-then-swap winnability oracle.

    A cell `(p_gen, p_swap, cutoff, N, n_ch)` is *winnable* if purify-then-swap
    delivers end-to-end at least `min_deliveries` times over `n_pilots` rollouts
    at a generous `probe_steps` horizon, **under the same (idealized) physics as
    training**, otherwise "unwinnable" would conflate physics with a regime the
    agent never trains in. Results are cached per coarse bin so each region of
    the parameter space is probed only once; pruning the training distribution
    then costs essentially nothing after warmup.

    purify-then-swap is the oracle (user decision 2026-07-12): under the
    repaired cutoff physics purification extends link lifetimes, so it dominates
    swap-asap as a feasibility oracle; its n_ch=4 livelock risk is bounded by
    `probe_steps` and a livelocked pilot just marks the cell unwinnable
    (conservative).
    """

    def __init__(self, n_pilots=5, probe_steps=400, min_deliveries=1, seed=0,
                 channel_loss=0.0, F0=1.0):
        self.n_pilots = n_pilots
        self.probe_steps = probe_steps
        self.min_deliveries = min_deliveries
        self.rng = np.random.default_rng(seed)
        # physics must match the training regime (idealized by default)
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
                max_steps=self.probe_steps,
                channel_loss=self.channel_loss, F0=self.F0,
                rng=np.random.default_rng(self.rng.integers(2**32)))
            env.reset()
            info = {}
            while not env.done and env.steps < self.probe_steps:
                _, _, done, info = env.step(purify_then_swap(env))
                if done:
                    break
            if info.get("terminated"):   # delivered (a win, not a timeout)
                deliveries += 1
            if deliveries >= self.min_deliveries:
                break
        result = deliveries >= self.min_deliveries
        self._cache[key] = result
        return result
