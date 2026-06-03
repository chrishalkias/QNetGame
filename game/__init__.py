"""game: curriculum trainer orchestration over the RL stack."""
from .phases import PhaseConfig, PHASE1
from .runner import run_phase

__all__ = ["PhaseConfig", "PHASE1", "run_phase"]
