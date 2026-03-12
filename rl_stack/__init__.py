"""RL module for quantum repeater network routing."""
from .env_wrapper import QRNEnv, N_ACTIONS, NOOP, SWAP, PURIFY
from .buffer import ReplayBuffer
from . import strategies

# torch-dependent imports guarded
try:
    from .model import QNetwork
    from .agent import QRNAgent
except ImportError:
    QNetwork = None
    QRNAgent = None