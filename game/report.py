"""
Pure gap math + human-readable formatting for the optimal-comparison report.

The DP baseline is optimal only among purify-free {NOOP, SWAP} policies, so it is
reported as `T_opt_swaponly`. The agent is evaluated twice: with its full action
set (`T_agent`, may beat the swap-only optimum via purification) and with PURIFY
masked off (`T_agent_swaponly`, the apples-to-apples test of whether it learned
optimal swap scheduling).
"""
from __future__ import annotations
import math
from typing import Dict, Optional


def _pct(num: float, den: Optional[float]) -> float:
    if den is None or (isinstance(den, float) and math.isnan(den)) or den == 0:
        return float("nan")
    return 100.0 * num / den


def gaps(N: int, in_distribution: bool,
         T_opt_swaponly: Optional[float], T_swap: float,
         T_agent: float, T_agent_swaponly: Optional[float] = None) -> Dict:
    """One comparison row.

    `T_opt_swaponly=None` -> optimal unavailable (its gaps are NaN).
    `T_agent_swaponly=None` -> purify-masked agent not evaluated (its gap NaN).

    Gaps (positive = agent slower than the swap-only optimum):
      - gap_full_pct        : full agent (with purify) vs T_opt_swaponly
      - scheduling_gap_pct  : purify-masked agent vs T_opt_swaponly (≈0 ⇒ learned
                              optimal swap scheduling)
      - agent_vs_swap_pct   : full agent vs swap-asap (positive = agent faster)
    """
    return {
        "N": N,
        "in_distribution": in_distribution,
        "T_opt_swaponly": T_opt_swaponly,
        "T_swap": T_swap,
        "T_agent": T_agent,
        "T_agent_swaponly": T_agent_swaponly,
        "gap_full_pct": _pct((T_agent - T_opt_swaponly) if T_opt_swaponly else 0.0,
                             T_opt_swaponly),
        "scheduling_gap_pct": (
            _pct((T_agent_swaponly - T_opt_swaponly) if (T_opt_swaponly and
                 T_agent_swaponly is not None) else 0.0, T_opt_swaponly)
            if T_agent_swaponly is not None else float("nan")),
        "agent_vs_swap_pct": _pct(T_swap - T_agent, T_swap),
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
        f"H={cfg.get('horizon')}]  (T_opt = optimal SWAP-ONLY policy)"
    )
    hdr = (f"{'N':>3} {'in_dist':>7} {'T_opt_so':>8} {'T_swap':>6} "
           f"{'T_agent':>7} {'T_ag_so':>7} "
           f"{'gap_full%':>10} {'sched_gap%':>11} {'ag_vs_swap%':>12}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in report.get("rows", []):
        lines.append(
            f"N={r['N']:<1} {str(r['in_distribution']):>7} "
            f"{_fmt(r['T_opt_swaponly']):>8} {_fmt(r['T_swap'])} "
            f"{_fmt(r['T_agent']):>7} {_fmt(r.get('T_agent_swaponly')):>7} "
            f"{_fmt(r['gap_full_pct']):>10} {_fmt(r['scheduling_gap_pct']):>11} "
            f"{_fmt(r['agent_vs_swap_pct']):>12}"
        )
    return "\n".join(lines)
