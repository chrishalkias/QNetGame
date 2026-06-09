"""game: exact-optimal-policy benchmarking for the RL agent."""
from .compare_optimal import compare_to_optimal, load_optimal_pickle
from .report import format_report

__all__ = [
    "compare_to_optimal", "load_optimal_pickle", "format_report",
]
