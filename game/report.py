"""Pure gap math + human-readable formatting for the optimal-comparison report."""
from __future__ import annotations
import math
from typing import Dict, Optional


def gaps(N: int, in_distribution: bool,
         T_opt: Optional[float], T_swap: float, T_agent: float) -> Dict:
    """One comparison row. `T_opt=None` -> optimal unavailable (gap is NaN)."""
    if T_opt is None or (isinstance(T_opt, float) and math.isnan(T_opt)) or T_opt == 0:
        gap_opt = float("nan")
    else:
        gap_opt = 100.0 * (T_agent - T_opt) / T_opt
    avs = 100.0 * (T_swap - T_agent) / T_swap if T_swap else float("nan")
    return {
        "N": N,
        "in_distribution": in_distribution,
        "T_opt": T_opt,
        "T_swap": T_swap,
        "T_agent": T_agent,
        "gap_to_optimal_pct": gap_opt,
        "agent_vs_swap_pct": avs,
    }


def _fmt(x) -> str:
    if x is None:
        return "  --  "
    if isinstance(x, float) and math.isnan(x):
        return "  n/a "
    return f"{x:6.2f}"


def format_report(report: Dict) -> str:
    """Render a report dict (keys 'config', 'rows') as a fixed-width table."""
    cfg = report.get("config", {})
    lines = []
    lines.append(
        f"Optimal-comparison report  "
        f"[n_ch={cfg.get('n_ch')} cutoff={cfg.get('cutoff')} "
        f"p_gen={cfg.get('p_gen')} p_swap={cfg.get('p_swap')} "
        f"H={cfg.get('horizon')}]"
    )
    hdr = (f"{'N':>3} {'in_dist':>7} {'T_opt':>6} {'T_swap':>6} {'T_agent':>7} "
           f"{'gap_to_optimal%':>16} {'agent_vs_swap%':>15}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in report.get("rows", []):
        lines.append(
            f"N={r['N']:<1} {str(r['in_distribution']):>7} "
            f"{_fmt(r['T_opt'])} {_fmt(r['T_swap'])} {_fmt(r['T_agent']):>7} "
            f"{_fmt(r['gap_to_optimal_pct']):>16} {_fmt(r['agent_vs_swap_pct']):>15}"
        )
    return "\n".join(lines)
