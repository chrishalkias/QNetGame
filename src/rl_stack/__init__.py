"""
--------------------------------------------------------------------------------
RL module for quantum repeater network routing.
--------------------------------------------------------------------------------
"""
from .env_wrapper import QRNEnv, N_ACTIONS, NOOP, SWAP, PURIFY
from .buffer import ReplayBuffer
from . import strategies

# torch-dependent imports guarded. Catch any Exception, not just ImportError:
# the import can fail with a non-ImportError even when torch is installed and
# importable. On ALICE the `Python/3.11.3` module was patch-bumped to 3.11.13
# while the venv was built for 3.11.3, so `torch_geometric` raises TypeError
# (MetadataPathFinder.invalidate_caches signature) at import -- torch itself is
# fine. torch-free consumers (env_wrapper, strategies -- the heuristic
# baselines) must still load regardless.
try:
    from .model import QNetwork
    from .agent import QRNAgent
except Exception:
    QNetwork = None
    QRNAgent = None