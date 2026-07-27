"""
--------------------------------------------------------------------------------
Batch validation: agent vs swap-ASAP across parameter sweeps.

Produces two figures:
  1. FacetGrid of 4 heatmaps (N = 4, 10, 12, 15) showing relative
     delivery-time improvement of the agent over swap-ASAP as a
     function of p_gen and p_swap.  An inset zooms into the
     high-interest region p_gen ~ 0.1, p_swap ~ 0.9.
  2. Heatmap of relative improvement vs p_gen and cutoff (p_swap = 1).

Both metrics are:
    Δ% = (T_swap - T_agent) / T_swap * 100
    positive -> agent is faster;  negative -> swap-ASAP is faster.

Usage
-----
    python experiments/training/batch_validate.py \
        --model checkpoints/sota/policy.pth \
        --episodes 200 \
        --save_dir results/batch_validate
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -- project imports ----------------------------------------------
from rl_stack.env_wrapper import QRNEnv
from rl_stack.policies import swap_asap, purify_then_swap
from experiments.mc_eval import make_agent_fn, mc_eval_stats

# ----------------------------------------------------------------
#  Constants: baselines and sweep grids
# ----------------------------------------------------------------

# baseline the agent is compared against (pilots always use swap-ASAP so the
# adaptive cutoff/max_steps stay identical across baselines)
BASELINES = {"swap_asap": swap_asap, "purify_swap": purify_then_swap}
BASELINE_LABELS = {"swap_asap": "Swap-ASAP", "purify_swap": "Purify-then-swap"}

# sweep 1: p_gen x p_swap, one panel per chain length
COARSE_GRID = np.round(np.arange(0.1, 1.01, 0.1), 2)
INSET_P_GEN = np.round(np.arange(0.05, 0.25, 0.03), 2)
INSET_P_SWAP = np.round(np.arange(0.75, 1.01, 0.03), 2)
NODE_COUNTS = [4, 10, 12, 15]

# sweeps 2 and 3: fixed chain length, cutoff as an axis or pinned
CUTOFF_GRID = [4, 6, 8, 10, 15, 20, 30, 50, 80]
PGEN_GRID_SWEEP2 = np.round(np.arange(0.1, 1.01, 0.1), 2)
SWEEP2_N = 8

# horizon cap for the swap-ASAP pilots that size each cell
PILOT_CAP = 800
PILOT_EPISODES = 15


# ----------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch validation: agent vs heuristic baselines over parameter sweeps")
    p.add_argument("--model", type=str, required=True, help="path to policy.pth checkpoint")
    p.add_argument("--episodes", type=int, default=200, help="episodes per (strategy, parameter) pair")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--save_dir", type=str, default="results/batch_validate", help="directory for output plots and CSVs")
    p.add_argument("--sweep", type=str, default="both", choices=["both", "pgen_pswap", "pgen_cutoff", "pgen_pswap_fixed_cutoff"], help="which sweep(s) to run; 'pgen_pswap_fixed_cutoff' is the old clean_check (p_gen x p_swap at fixed cutoff(s))")
    p.add_argument("--sweep2_nodes", type=int, default=SWEEP2_N, help="chain length for the pgen_cutoff and pgen_pswap_fixed_cutoff sweeps")
    p.add_argument("--cutoffs", type=str, default="20,80", help="comma-separated fixed cutoffs for the pgen_pswap_fixed_cutoff sweep")
    p.add_argument("--node_counts", type=int, nargs="+", default=NODE_COUNTS, help="chain lengths for the pgen_pswap sweep (one heatmap panel each)")
    p.add_argument("--baseline", type=str, default="swap_asap", choices=sorted(BASELINES), help="heuristic the agent is compared against in the pgen_pswap sweep")
    p.add_argument("--fixed_cutoff", type=int, default=None, help="pin the cutoff for the pgen_pswap sweep instead of adapting it per cell (e.g. 1000000000 ~ infinite memory: no expiry, no decoherence); the horizon is still pilot-estimated per cell")
    p.add_argument("--resume", action="store_true", help="skip work already present in the sweep CSV and append (applies to pgen_cutoff and pgen_pswap_fixed_cutoff; replaces the old partial_validate.py)")
    return p.parse_args()


# ----------------------------------------------------------------
#  Adaptive parameter estimation
# ----------------------------------------------------------------

def estimate_params(
    n_nodes: int,
    p_gen: float,
    p_swap: float,
    *,
    n_ch: int = 4,
    pilot_episodes: int = PILOT_EPISODES,
    success_target: float = 0.70,
    step_cap: int = PILOT_CAP,
    cutoff_floor: int = 4,
    cutoff_ceil: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Return (max_steps, cutoff) so that at least *success_target* of episodes
    reach end-to-end entanglement under the swap-ASAP strategy.

    Strategy
    --------
    1. Estimate a *generous* cutoff from the expected link-generation
       time so that links almost never expire before being used.
    2. Run a short pilot with swap-ASAP to measure actual delivery
       times; set max_steps to cover the *success_target* quantile
       with headroom.
    3. If the pilot success rate is too low, double cutoff and retry
       (up to cutoff_ceil).
    """
    rng = rng or np.random.default_rng()
    hops = n_nodes - 1

    # Heuristic seed: expected steps to generate a single link ~ 1/p_gen.
    # A swap needs two links from distinct neighbours, so the bottleneck
    # repeater waits ~2/p_gen steps.  Cutoff must exceed that by a margin
    # proportional to chain length so interior links survive while the
    # edges are still being built.
    est_gen_time = max(1.0 / max(p_gen, 0.01), 1.0)
    cutoff = int(np.clip(3 * est_gen_time * hops, cutoff_floor, cutoff_ceil))

    for _attempt in range(4):
        # generous initial max_steps, capped to keep pilots fast
        max_steps = int(np.clip(
            6 * est_gen_time * hops / max(p_swap, 0.05),
            40,
            step_cap,
        ))

        delivery_times = _pilot_swap_asap(
            n_nodes, p_gen, p_swap, cutoff,
            max_steps=max_steps,
            n_ch=n_ch,
            n_episodes=pilot_episodes,
            rng=rng,
        )
        successes = [t for t in delivery_times if t < max_steps]

        if len(successes) / max(len(delivery_times), 1) >= success_target:
            # Set max_steps at the 95th-percentile of successful runs + 20 %
            q95 = int(np.percentile(successes, 95))
            max_steps = min(int(q95 * 1.2) + 5, step_cap)
            return max_steps, cutoff

        # Not enough successes, relax the cutoff
        cutoff = min(cutoff * 2, cutoff_ceil)

    # Fallback: generous defaults
    return step_cap, cutoff_ceil


def _pilot_swap_asap(
    n_nodes: int,
    p_gen: float,
    p_swap: float,
    cutoff: int,
    *,
    max_steps: int,
    n_ch: int,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[int]:
    """Run swap-ASAP episodes and return delivery times (max_steps if failed)."""
    times: list[int] = []
    for _ in range(n_episodes):
        env = QRNEnv(
            n_repeaters=n_nodes,
            n_ch=n_ch,
            p_gen=p_gen,
            p_swap=p_swap,
            cutoff=cutoff,
            max_steps=max_steps,
            F0=1.0,
            channel_loss=0.0,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        env.reset()
        info: dict = {}
        while not env.done and env.steps < max_steps:
            _, _, done, info = env.step(swap_asap(env))
            if done:
                break
        fid = info.get("fidelity", 0.0)
        delivered = bool(info.get("terminated")) and fid > 0
        times.append(info["ticks"] if delivered else max_steps)
    return times


def _pilot_max_steps(
    n_nodes: int,
    p_gen: float,
    p_swap: float,
    cutoff: int,
    rng: np.random.Generator,
) -> int:
    """Horizon for one cell whose cutoff is already pinned.

    Sizes max_steps from a swap-ASAP pilot at PILOT_CAP: the 95th percentile of
    the delivering pilot episodes plus 20 %, or the cap when the pilot barely
    delivers.
    """
    times = _pilot_swap_asap(n_nodes, p_gen, p_swap, cutoff,
                             max_steps=PILOT_CAP, n_ch=4,
                             n_episodes=PILOT_EPISODES, rng=rng)
    successes = [t for t in times if t < PILOT_CAP]
    if len(successes) >= 4:
        return min(int(np.percentile(successes, 95) * 1.2) + 5, PILOT_CAP)
    return PILOT_CAP


# ----------------------------------------------------------------
#  Episode runner
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    n_nodes: int
    p_gen: float
    p_swap: float
    cutoff: int
    max_steps: int
    n_ch: int = 4


def run_comparison(
    agent_fn,
    cfg: RunConfig,
    n_episodes: int,
    rng: np.random.Generator,
    baseline: str = "swap_asap",
) -> dict[str, float]:
    """Run agent and the chosen baseline for *n_episodes* through mc_eval.

    Returns dict with keys: "agent", "swap_asap" (mean delivery time T,
    censored at cfg.max_steps) and "agent_succ", "swap_asap_succ" (delivery
    counts). The "swap_asap" key is generic, it holds whichever *baseline*
    was run.

    Both policies are evaluated on the SAME episode seed (drawn from *rng*),
    so the comparison is paired. The hand-rolled loop this replaced drew the
    baseline's seeds AFTER the agent's, comparing the two policies on
    different episodes; it was proved bit-identical to mc_eval episode for
    episode before deletion (2026-07-27), so only the pairing changed.
    """
    baseline_fn = BASELINES[baseline]
    seed = int(rng.integers(2**32))
    fns = {"agent": agent_fn,
           "swap_asap": lambda env, obs: baseline_fn(env)}

    out: dict[str, float] = {}
    for label, fn in fns.items():
        s = mc_eval_stats(fn, cfg.n_nodes, cfg.n_ch, cfg.p_gen, cfg.p_swap,
                          cfg.cutoff, cfg.max_steps, n_episodes, seed=seed)
        out[label] = s["T"]
        out[f"{label}_succ"] = int(round(s["conn_rate"] * n_episodes))
    return out


def relative_improvement(T_agent: float, T_swap: float) -> float:
    """Δ% = (T_swap - T_agent) / T_swap * 100.  Positive -> agent faster."""
    if T_swap == 0:
        return 0.0
    return (T_swap - T_agent) / T_swap * 100.0


# ----------------------------------------------------------------
#  Sweep 1: p_gen x p_swap  for N = 4, 10, 12, 15
# ----------------------------------------------------------------

def _cell_params(
    n_nodes: int, p_gen: float, p_swap: float,
    rng: np.random.Generator, fixed_cutoff: int | None,
) -> tuple[int, int]:
    """(max_steps, cutoff) for one sweep cell.

    fixed_cutoff=None -> fully adaptive (estimate_params). Otherwise the
    cutoff is pinned (e.g. 10**9 = infinite memory) and only the horizon is
    estimated from a swap-ASAP pilot, sweep-3 style.
    """
    if fixed_cutoff is None:
        return estimate_params(n_nodes, p_gen, p_swap, rng=rng)
    return _pilot_max_steps(n_nodes, p_gen, p_swap, fixed_cutoff, rng), fixed_cutoff


def _sweep1_cell(
    agent_fn,
    n_nodes: int,
    p_gen: float,
    p_swap: float,
    region: str,
    n_episodes: int,
    rng: np.random.Generator,
    baseline: str,
    fixed_cutoff: int | None,
) -> dict:
    """Evaluate one (N, p_gen, p_swap) cell of sweep 1 and return its row."""
    max_steps, cutoff = _cell_params(n_nodes, p_gen, p_swap, rng, fixed_cutoff)
    cfg = RunConfig(n_nodes, p_gen, p_swap, cutoff, max_steps)
    res = run_comparison(agent_fn, cfg, n_episodes, rng, baseline=baseline)
    return {
        "N": n_nodes,
        "p_gen": p_gen,
        "p_swap": p_swap,
        "delta_pct": relative_improvement(res["agent"], res["swap_asap"]),
        "both_fail": res["agent_succ"] == 0 and res["swap_asap_succ"] == 0,
        "region": region,
        "cutoff": cutoff,
        "max_steps": max_steps,
        "baseline": baseline,
    }


def sweep_pgen_pswap(
    agent_fn,
    n_episodes: int,
    rng: np.random.Generator,
    save_dir: str = ".",
    node_counts: Sequence[int] = NODE_COUNTS,
    baseline: str = "swap_asap",
    fixed_cutoff: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [N, p_gen, p_swap, delta_pct, ...].

    Runs a coarse (p_gen, p_swap) grid plus a fine inset around
    p_gen ~ 0.1, p_swap ~ 0.9 for every N. The CSV is rewritten after every
    cell so a cluster walltime kill loses nothing.
    """
    csv_path = os.path.join(save_dir, "sweep_pgen_pswap.csv")
    regions = (("coarse", list(itertools.product(COARSE_GRID, COARSE_GRID))),
               ("inset", list(itertools.product(INSET_P_GEN, INSET_P_SWAP))))
    total = len(node_counts) * sum(len(cells) for _, cells in regions)
    rows: list[dict] = []
    done_count = 0

    for n_nodes in node_counts:
        for region, cells in regions:
            for p_gen, p_swap in cells:
                done_count += 1
                _log_progress("sweep1", done_count, total, n_nodes, p_gen, p_swap)
                rows.append(_sweep1_cell(agent_fn, n_nodes, p_gen, p_swap, region,
                                         n_episodes, rng, baseline, fixed_cutoff))
                pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\n[checkpoint] saved sweep1 through N={n_nodes}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------
#  Sweep 2: p_gen x cutoff   (p_swap = 1 fixed)
# ----------------------------------------------------------------

def _concat_rows(df_existing: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Append *rows* to *df_existing*, skipping empty frames.

    The non-resume path seeds df_existing with a column-only DataFrame, whose
    columns are dtype object. Concatenating it in would make the whole result
    object-dtyped, and seaborn then rejects the pivot with "Image data of
    dtype object cannot be converted to float" at plot time.
    """
    frames = [df for df in (df_existing, pd.DataFrame(rows)) if not df.empty]
    if not frames:
        return df_existing
    return pd.concat(frames, ignore_index=True)


def sweep_pgen_cutoff(
    agent_fn,
    n_episodes: int,
    rng: np.random.Generator,
    n_nodes: int = SWEEP2_N,
    save_dir: str = ".",
    resume: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns [p_gen, cutoff, delta_pct].

    Saves incrementally after each p_gen row so partial results
    survive job timeouts.  With *resume* = True, reads any existing
    sweep_pgen_cutoff.csv and computes only the p_gen values still
    missing, appending to it (replaces the old partial_validate.py).
    """
    csv_path = os.path.join(save_dir, "sweep_pgen_cutoff.csv")
    if resume and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        pgens_done = set(df_existing["p_gen"].unique())
        print(f"[resume] {len(df_existing)} existing rows; "
              f"p_gen done: {sorted(pgens_done)}")
    else:
        df_existing = pd.DataFrame(columns=["p_gen", "cutoff", "delta_pct"])
        pgens_done = set()

    pgens_todo = [p for p in PGEN_GRID_SWEEP2 if p not in pgens_done]
    rows: list[dict] = []
    total = len(pgens_todo) * len(CUTOFF_GRID)
    done_count = 0

    if total == 0:
        print("[resume] sweep2 already complete.")
        return df_existing

    for p_gen in pgens_todo:
        for cutoff in CUTOFF_GRID:
            done_count += 1
            _log_progress("sweep2", done_count, total, n_nodes, p_gen, 1.0)

            # cutoff is the independent variable here, estimate only max_steps
            max_steps = _pilot_max_steps(n_nodes, p_gen, 1.0, cutoff, rng)
            cfg = RunConfig(n_nodes, p_gen, p_swap=1.0, cutoff=cutoff,
                            max_steps=max_steps)
            res = run_comparison(agent_fn, cfg, n_episodes, rng)
            rows.append({
                "p_gen": p_gen,
                "cutoff": cutoff,
                "delta_pct": relative_improvement(res["agent"], res["swap_asap"]),
            })

        # -- incremental save after each p_gen --
        df_partial = _concat_rows(df_existing, rows)
        df_partial.to_csv(csv_path, index=False)
        print(f"\n[checkpoint] saved sweep2 through p_gen={p_gen} "
              f"({len(df_partial)} rows)")

    return _concat_rows(df_existing, rows)


# ----------------------------------------------------------------
#  Sweep 3: p_gen x p_swap at FIXED cutoff(s)   (the old clean_check)
# ----------------------------------------------------------------
#
# Disentangles p_swap from cutoff: sweep 1 auto-estimates a cutoff per cell
# (confounding the two), whereas this holds the memory cutoff FIXED and sweeps
# p_gen x p_swap, one panel per cutoff value.  Tests the literature prediction
# (Inesta & Wehner, npj QI 2023) that the agent's advantage over swap-ASAP
# grows as p_swap drops.

def sweep_pgen_pswap_fixed_cutoff(
    agent_fn,
    n_episodes: int,
    rng: np.random.Generator,
    cutoffs: Sequence[int],
    n_nodes: int = SWEEP2_N,
    save_dir: str = ".",
    resume: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns [cutoff, p_gen, p_swap, delta_pct].

    Sweeps p_gen x p_swap (COARSE_GRID) at each fixed cutoff.  Saves
    incrementally after each (cutoff, p_gen) column; with *resume* = True,
    skips columns already present in sweep_fixed_cutoff.csv.
    """
    csv_path = os.path.join(save_dir, "sweep_fixed_cutoff.csv")
    if resume and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        done = {(int(c), float(pg))
                for c, pg in zip(df_existing["cutoff"], df_existing["p_gen"])}
        print(f"[resume] {len(df_existing)} existing rows.")
    else:
        df_existing = pd.DataFrame(
            columns=["cutoff", "p_gen", "p_swap", "delta_pct"])
        done = set()

    todo = [(c, float(pg)) for c in cutoffs for pg in COARSE_GRID
            if (c, float(pg)) not in done]
    rows: list[dict] = []
    total = len(todo) * len(COARSE_GRID)
    done_count = 0

    if total == 0:
        print("[resume] all (cutoff, p_gen) columns already complete.")
        return df_existing

    for cutoff, p_gen in todo:
        for p_swap in COARSE_GRID:
            done_count += 1
            _log_progress(f"fixed(c={cutoff})", done_count, total,
                          n_nodes, p_gen, p_swap)

            # cutoff is fixed; estimate only max_steps from a swap-ASAP pilot
            max_steps = _pilot_max_steps(n_nodes, p_gen, p_swap, cutoff, rng)
            cfg = RunConfig(n_nodes, float(p_gen), float(p_swap),
                            cutoff, max_steps)
            res = run_comparison(agent_fn, cfg, n_episodes, rng)
            rows.append({
                "cutoff": cutoff,
                "p_gen": float(p_gen),
                "p_swap": float(p_swap),
                "delta_pct": relative_improvement(
                    res["agent"], res["swap_asap"]),
            })

        # incremental save after each (cutoff, p_gen) column
        df_partial = _concat_rows(df_existing, rows)
        df_partial.to_csv(csv_path, index=False)
        print(f"\n[checkpoint] saved fixed-cutoff through cutoff={cutoff}, "
              f"p_gen={p_gen} ({len(df_partial)} rows)")

    return _concat_rows(df_existing, rows)


# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------

def _build_fail_mask(sub: pd.DataFrame, index_col: str, columns_col: str) -> pd.DataFrame:
    """Build a pivot of booleans: True where both strategies fail.

    Only marks cells as failed when the 'both_fail' column is explicitly
    present (from runs that track success counts).  Old CSVs without this
    column produce an all-False mask (no grey cells).
    """
    if "both_fail" in sub.columns:
        return sub.pivot_table(
            index=index_col, columns=columns_col,
            values="both_fail", aggfunc="any",
        )
    # No both_fail column, cannot distinguish ties from mutual failure
    pivot = sub.pivot_table(
        index=index_col, columns=columns_col,
        values="delta_pct", aggfunc="mean",
    )
    return pd.DataFrame(False, index=pivot.index, columns=pivot.columns)


def _build_annot(pivot: pd.DataFrame, fail_mask: pd.DataFrame) -> np.ndarray:
    """Build annotation array: 'N/A' for failed cells, formatted number otherwise."""
    annot = np.empty(pivot.shape, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            idx = pivot.index[i]
            col = pivot.columns[j]
            is_fail = (fail_mask.loc[idx, col]
                       if idx in fail_mask.index and col in fail_mask.columns
                       else False)
            if is_fail:
                annot[i, j] = "N/A"
            else:
                annot[i, j] = f"{pivot.iloc[i, j]:.1f}"
    return annot


def _draw_heatmap(
    ax: plt.Axes,
    sub: pd.DataFrame,
    abs_bound: float,
    index_col: str = "p_swap",
    columns_col: str = "p_gen",
    fontsize: int = 7,
) -> None:
    """Draw a single heatmap with N/A greyed-out cells."""
    pivot = sub.pivot_table(
        index=index_col, columns=columns_col,
        values="delta_pct", aggfunc="mean",
    )
    pivot = pivot.sort_index(ascending=False)

    fail_mask = _build_fail_mask(sub, index_col, columns_col)
    fail_mask = fail_mask.reindex_like(pivot).fillna(False)

    annot = _build_annot(pivot, fail_mask)

    # Replace failed cells with NaN so they render as grey
    plot_data = pivot.copy()
    plot_data[fail_mask] = np.nan

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-abs_bound,
        vmax=abs_bound,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": fontsize},
        cbar_kws={"label": "Δ% delivery time", "shrink": 0.8},
        linewidths=0.4,
        linecolor="white",
        mask=fail_mask,
    )
    # Overlay grey for N/A cells
    if fail_mask.any().any():
        grey_data = pivot.copy()
        grey_data[:] = 0
        sns.heatmap(
            grey_data,
            ax=ax,
            cmap=["#d9d9d9"],
            annot=annot,
            fmt="",
            annot_kws={"fontsize": fontsize, "color": "#666666",
                       "fontweight": "bold"},
            cbar=False,
            linewidths=0.4,
            linecolor="white",
            mask=~fail_mask,
        )


def plot_pgen_pswap(df: pd.DataFrame, save_dir: str,
                    node_counts: Sequence[int] = NODE_COUNTS,
                    baseline: str = "swap_asap") -> None:
    """Two separate figures:
    1. FacetGrid of coarse heatmaps, one panel per N in *node_counts*.
    2. FacetGrid of zoomed-in heatmaps (low p_gen, high p_swap).

    Cells where both strategies fail to reach e2e are greyed out with 'N/A'.
    """
    blabel = BASELINE_LABELS[baseline]
    coarse = df[df["region"] == "coarse"]
    inset_df = df[df["region"] == "inset"]

    vmin = df["delta_pct"].quantile(0.02)
    vmax = df["delta_pct"].quantile(0.98)
    abs_bound = max(abs(vmin), abs(vmax))

    ncols = min(2, len(node_counts))
    nrows = -(-len(node_counts) // ncols)   # ceil division

    # -- Figure 1: coarse grid --
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, n_nodes in enumerate(node_counts):
        ax = axes_flat[idx]
        sub = coarse[coarse["N"] == n_nodes]
        _draw_heatmap(ax, sub, abs_bound)
        ax.set_title(f"N = {n_nodes}", fontsize=12, fontweight="bold")
        ax.set_xlabel("$p_{gen}$")
        ax.set_ylabel("$p_{swap}$")

    fig.suptitle(
        f"Agent vs {blabel}: relative delivery-time improvement (%)\n"
        f"positive (blue) = agent faster, negative (red) = {blabel} faster, "
        "grey = both fail",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "heatmap_pgen_pswap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")

    # -- Figure 2: zoomed region (low p_gen, high p_swap) --
    if inset_df.empty:
        return

    fig_z, axes_z = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes_z_flat = np.atleast_1d(axes_z).flatten()

    for idx, n_nodes in enumerate(node_counts):
        ax = axes_z_flat[idx]
        inset_sub = inset_df[inset_df["N"] == n_nodes]
        if inset_sub.empty:
            ax.set_visible(False)
            continue
        _draw_heatmap(ax, inset_sub, abs_bound)
        ax.set_title(f"N = {n_nodes}", fontsize=12, fontweight="bold")
        ax.set_xlabel("$p_{gen}$")
        ax.set_ylabel("$p_{swap}$")

    fig_z.suptitle(
        "Zoom: low $p_{gen}$, high $p_{swap}$ region\n"
        f"positive (blue) = agent faster, negative (red) = {blabel} faster, "
        "grey = both fail",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path_z = os.path.join(save_dir, "heatmap_pgen_pswap_zoom.png")
    fig_z.savefig(path_z, dpi=200, bbox_inches="tight")
    plt.close(fig_z)
    print(f"[plot] saved {path_z}")


def plot_pgen_cutoff(df: pd.DataFrame, save_dir: str,
                     n_nodes: int = SWEEP2_N) -> None:
    """Single heatmap: p_gen vs cutoff (p_swap = 1 fixed)."""
    pivot = df.pivot_table(
        index="cutoff", columns="p_gen", values="delta_pct", aggfunc="mean",
    )
    pivot = pivot.sort_index(ascending=False)

    abs_bound = max(abs(df["delta_pct"].quantile(0.02)),
                    abs(df["delta_pct"].quantile(0.98)))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-abs_bound,
        vmax=abs_bound,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Δ% delivery time"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(
        f"Agent vs Swap-ASAP  (N = {n_nodes}, $p_{{swap}}$ = 1)\n"
        "positive (blue) = agent faster",
        fontsize=12,
    )
    ax.set_xlabel("$p_{gen}$")
    ax.set_ylabel("cutoff")

    path = os.path.join(save_dir, "heatmap_pgen_cutoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def plot_fixed_cutoff(
    df: pd.DataFrame,
    cutoffs: Sequence[int],
    n_nodes: int,
    save_dir: str,
) -> None:
    """One panel per cutoff: p_gen (x) vs p_swap (y) heatmap of delta_pct."""
    abs_bound = max(abs(df["delta_pct"].quantile(0.02)),
                    abs(df["delta_pct"].quantile(0.98)))
    fig, axes = plt.subplots(
        1, len(cutoffs), figsize=(7 * len(cutoffs), 6), squeeze=False,
    )
    for idx, c in enumerate(cutoffs):
        ax = axes[0][idx]
        sub = df[df["cutoff"] == c]
        if sub.empty:
            ax.set_visible(False)
            continue
        _draw_heatmap(ax, sub, abs_bound,
                      index_col="p_swap", columns_col="p_gen")
        ax.set_title(f"cutoff = {c}", fontsize=12, fontweight="bold")
        ax.set_xlabel("$p_{gen}$")
        ax.set_ylabel("$p_{swap}$")
    fig.suptitle(
        f"Agent vs Swap-ASAP at FIXED cutoff (N = {n_nodes})\n"
        "positive (blue) = agent faster; p_swap decreases top→bottom",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "heatmap_pgen_pswap_fixed_cutoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


# ----------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------

def _log_progress(
    sweep: str, done: int, total: int,
    n: int, pg: float, ps: float,
) -> None:
    pct = done / max(total, 1) * 100
    print(
        f"\r[{sweep}] {done}/{total} ({pct:5.1f}%) "
        f"N={n} p_gen={pg:.2f} p_swap={ps:.2f}",
        end="", flush=True,
    )
    if done == total:
        print()


# ----------------------------------------------------------------
#  Entry point
# ----------------------------------------------------------------

def _run_sweep1(args, agent_fn, rng) -> dict:
    """Run sweep 1, write its CSV and figures, return its results.json block."""
    print(f"\n== Sweep 1: p_gen x p_swap (N = {args.node_counts}, "
          f"baseline = {args.baseline}) ==")
    df = sweep_pgen_pswap(agent_fn, args.episodes, rng, save_dir=args.save_dir,
                          node_counts=args.node_counts, baseline=args.baseline,
                          fixed_cutoff=args.fixed_cutoff)
    csv_path = os.path.join(args.save_dir, "sweep_pgen_pswap.csv")
    df.to_csv(csv_path, index=False)
    print(f"[data] saved {csv_path}")
    plot_pgen_pswap(df, args.save_dir, node_counts=args.node_counts,
                    baseline=args.baseline)
    return {
        "description": f"p_gen x p_swap sweep for N = {args.node_counts}, "
                       f"agent vs {args.baseline}",
        "node_counts": args.node_counts,
        "baseline": args.baseline,
        "fixed_cutoff": args.fixed_cutoff,
        "results": df.to_dict(orient="records"),
    }


def _run_sweep2(args, agent_fn, rng) -> dict:
    """Run sweep 2, write its CSV and figure, return its results.json block."""
    n_nodes = args.sweep2_nodes
    print(f"\n== Sweep 2: p_gen x cutoff (N = {n_nodes}, p_swap = 1) ==")
    df = sweep_pgen_cutoff(agent_fn, args.episodes, rng, n_nodes=n_nodes,
                           save_dir=args.save_dir, resume=args.resume)
    csv_path = os.path.join(args.save_dir, "sweep_pgen_cutoff.csv")
    df.to_csv(csv_path, index=False)
    print(f"[data] saved {csv_path}")
    plot_pgen_cutoff(df, args.save_dir, n_nodes=n_nodes)
    return {
        "description": f"p_gen x cutoff sweep (N = {n_nodes}, p_swap = 1)",
        "n_nodes": n_nodes,
        "results": df.to_dict(orient="records"),
    }


def _run_sweep3(args, agent_fn, rng) -> dict:
    """Run sweep 3, write its CSV and figure, return its results.json block."""
    cutoffs = [int(c) for c in args.cutoffs.split(",") if c.strip()]
    n_nodes = args.sweep2_nodes
    print(f"\n== Sweep 3: p_gen x p_swap at FIXED cutoff(s) {cutoffs} "
          f"(N = {n_nodes}) ==")
    df = sweep_pgen_pswap_fixed_cutoff(
        agent_fn, args.episodes, rng, cutoffs, n_nodes=n_nodes,
        save_dir=args.save_dir, resume=args.resume)
    csv_path = os.path.join(args.save_dir, "sweep_fixed_cutoff.csv")
    df.to_csv(csv_path, index=False)
    print(f"[data] saved {csv_path}")
    plot_fixed_cutoff(df, cutoffs, n_nodes, args.save_dir)
    return {
        "description": f"p_gen x p_swap at fixed cutoff(s) {cutoffs} "
                       f"(N = {n_nodes})",
        "cutoffs": cutoffs,
        "n_nodes": n_nodes,
        "results": df.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading model from {args.model}")
    agent_fn = make_agent_fn(args.model)

    results_json: dict = {
        "metadata": {
            "model": args.model,
            "episodes_per_point": args.episodes,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "sweeps": {},
    }

    if args.sweep in ("both", "pgen_pswap"):
        results_json["sweeps"]["pgen_pswap"] = _run_sweep1(args, agent_fn, rng)
    if args.sweep in ("both", "pgen_cutoff"):
        results_json["sweeps"]["pgen_cutoff"] = _run_sweep2(args, agent_fn, rng)
    if args.sweep == "pgen_pswap_fixed_cutoff":
        results_json["sweeps"]["pgen_pswap_fixed_cutoff"] = _run_sweep3(
            args, agent_fn, rng)

    json_path = os.path.join(args.save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[data] saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
