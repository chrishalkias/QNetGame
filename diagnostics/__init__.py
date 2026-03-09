"""RL module for quantum repeater network routing."""
from rl_stack.env_wrapper import QRNEnv, N_ACTIONS, NOOP, ENTANGLE, SWAP, PURIFY
from rl_stack.buffer import ReplayBuffer
from rl_stack import strategies

# torch-dependent imports guarded
try:
    from rl_stack.model import QNetwork
    from rl_stack.agent import QRNAgent
    from . import policy_interpretation

except ImportError:
    QNetwork = None
    QRNAgent = None
    diagnostics = None