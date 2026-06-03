"""game: curriculum trainer orchestration over the RL stack."""
from .phases import PhaseConfig, PHASE1
from .runner import run_phase
from .compare_optimal import compare_to_optimal, load_optimal_pickle
from .report import format_report

__all__ = [
    "PhaseConfig", "PHASE1", "run_phase",
    "compare_to_optimal", "load_optimal_pickle", "format_report",
]
