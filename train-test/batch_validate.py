"""
Batch validation: agent vs swap-ASAP across parameter sweeps.

Produces two figures:
  1. FacetGrid of 4 heatmaps (N = 4, 10, 12, 15) showing relative
     delivery-time improvement of the agent over swap-ASAP as a
     function of p_gen and p_swap.  An inset zooms into the
     high-interest region p_gen ~ 0.1, p_swap ~ 0.9.
  2. Heatmap of relative improvement vs p_gen and cutoff (p_swap = 1).

Both metrics are:
    Δ% = (T_swap - T_agent) / T_swap * 100
    positive → agent is faster;  negative → swap-ASAP is faster.

Usage
-----
    python train-test/batch_validate.py \
        --model checkpoints/cluster_004/policy.pth \
        --episodes 200 \
        --save_dir results/batch_validate
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── project imports ──────────────────────────────────────────────
from rl_stack.agent import QRNAgent, _obs_to_data
from rl_stack.env_wrapper import QRNEnv, NOOP, SWAP
from rl_stack.strategies import swap_asap


# ═══════════════════════════════════════════════════════════════════
#  Adaptive parameter estimation
# ═══════════════════════════════════════════════════════════════════

def estimate_params(
    n_nodes: int,
    p_gen: float,
    p_swap: float,
    *,
    n_ch: int = 4,
    pilot_episodes: int = 15,
    success_target: float = 0.70,
    step_cap: int = 800,
    cutoff_floor: int = 4,
    cutoff_ceil: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Return (max_steps, cutoff) so that ≥ *success_target* of episodes
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

    # Heuristic seed: expected steps to generate a single link ≈ 1/p_gen.
    # A swap needs two links from distinct neighbours, so the bottleneck
    # repeater waits ~2/p_gen steps.  Cutoff must exceed that by a margin
    # proportional to chain length so interior links survive while the
    # edges are still being built.
    est_gen_time = max(1.0 / max(p_gen, 0.01), 1.0)
    cutoff = int(np.clip(
        3 * est_gen_time * hops,
        cutoff_floor,
        cutoff_ceil,
    ))

    for _attempt in range(4):
        # generous initial max_steps — capped to keep pilots fast
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

        # Not enough successes – relax cutoff
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
            dt_seconds=0.0,
            topology="chain",
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        obs = env.reset()
        for step in range(max_steps):
            actions = swap_asap(env)
            obs, _, done, info = env.step(actions)
            if done:
                break
        fid = info.get("fidelity", 0.0)
        times.append(step + 1 if done and fid > 0 else max_steps)
    return times


# ═══════════════════════════════════════════════════════════════════
#  Episode runner
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RunConfig:
    n_nodes: int
    p_gen: float
    p_swap: float
    cutoff: int
    max_steps: int
    n_ch: int = 4


def run_comparison(
    agent: QRNAgent,
    cfg: RunConfig,
    n_episodes: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run agent and swap-ASAP for *n_episodes*.

    Returns dict with keys: "agent", "swap_asap" (delivery-time arrays),
    and "agent_succ", "swap_asap_succ" (success counts).
    """
    result: dict[str, list[int]] = {"agent": [], "swap_asap": []}
    successes: dict[str, int] = {"agent_succ": 0, "swap_asap_succ": 0}

    for label, use_agent in [("agent", True), ("swap_asap", False)]:
        for _ in range(n_episodes):
            env = QRNEnv(
                n_repeaters=cfg.n_nodes,
                n_ch=cfg.n_ch,
                p_gen=cfg.p_gen,
                p_swap=cfg.p_swap,
                cutoff=cfg.cutoff,
                max_steps=cfg.max_steps,
                F0=1.0,
                channel_loss=0.0,
                dt_seconds=0.0,
                topology="chain",
                rng=np.random.default_rng(rng.integers(2**32)),
            )
            obs = env.reset()

            for step in range(cfg.max_steps):
                if use_agent:
                    mask = env.get_action_mask()
                    actions = agent.select_actions(obs, mask, training=False)
                else:
                    actions = swap_asap(env)
                obs, _, done, info = env.step(actions)
                if done:
                    break

            fid = info.get("fidelity", 0.0)
            delivery = step + 1 if done and fid > 0 else cfg.max_steps
            result[label].append(delivery)
            if fid > 0:
                successes[f"{label}_succ"] += 1

    out = {k: np.array(v) for k, v in result.items()}
    out.update({k: np.array(v) for k, v in successes.items()})
    return out


def relative_improvement(agent_times: np.ndarray, swap_times: np.ndarray) -> float:
    """Δ% = (T_swap - T_agent) / T_swap * 100.  Positive → agent faster."""
    mean_swap = np.mean(swap_times)
    mean_agent = np.mean(agent_times)
    if mean_swap == 0:
        return 0.0
    return (mean_swap - mean_agent) / mean_swap * 100.0


# ═══════════════════════════════════════════════════════════════════
#  Sweep 1: p_gen × p_swap  for N = 4, 10, 12, 15
# ═══════════════════════════════════════════════════════════════════

COARSE_GRID = np.round(np.arange(0.1, 1.01, 0.1), 2)
INSET_P_GEN = np.round(np.arange(0.05, 0.25, 0.03), 2)
INSET_P_SWAP = np.round(np.arange(0.75, 1.01, 0.03), 2)
NODE_COUNTS = [4, 10, 12, 15]


def sweep_pgen_pswap(
    agent: QRNAgent,
    n_episodes: int,
    rng: np.random.Generator,
    save_dir: str = ".",
) -> pd.DataFrame:
    """Return a DataFrame with columns [N, p_gen, p_swap, delta_pct].

    Saves incrementally after each N value so partial results survive
    job timeouts.
    """
    rows: list[dict] = []
    total = len(NODE_COUNTS) * (
        len(COARSE_GRID) ** 2 + len(INSET_P_GEN) * len(INSET_P_SWAP)
    )
    done_count = 0

    for n_nodes in NODE_COUNTS:
        # ── coarse grid ──
        for p_gen, p_swap in itertools.product(COARSE_GRID, COARSE_GRID):
            done_count += 1
            _log_progress("sweep1", done_count, total, n_nodes, p_gen, p_swap)

            max_steps, cutoff = estimate_params(
                n_nodes, p_gen, p_swap, rng=rng,
            )
            cfg = RunConfig(n_nodes, p_gen, p_swap, cutoff, max_steps)
            res = run_comparison(agent, cfg, n_episodes, rng)
            both_fail = (res["agent_succ"] == 0 and res["swap_asap_succ"] == 0)
            rows.append({
                "N": n_nodes,
                "p_gen": p_gen,
                "p_swap": p_swap,
                "delta_pct": relative_improvement(res["agent"], res["swap_asap"]),
                "both_fail": both_fail,
                "region": "coarse",
            })

        # ── inset (fine grid around p_gen ~ 0.1, p_swap ~ 0.9) ──
        for p_gen, p_swap in itertools.product(INSET_P_GEN, INSET_P_SWAP):
            done_count += 1
            _log_progress("sweep1", done_count, total, n_nodes, p_gen, p_swap)

            max_steps, cutoff = estimate_params(
                n_nodes, p_gen, p_swap, rng=rng,
            )
            cfg = RunConfig(n_nodes, p_gen, p_swap, cutoff, max_steps)
            res = run_comparison(agent, cfg, n_episodes, rng)
            both_fail = (res["agent_succ"] == 0 and res["swap_asap_succ"] == 0)
            rows.append({
                "N": n_nodes,
                "p_gen": p_gen,
                "p_swap": p_swap,
                "delta_pct": relative_improvement(res["agent"], res["swap_asap"]),
                "both_fail": both_fail,
                "region": "inset",
            })

        # ── incremental save after each N ──
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(
            os.path.join(save_dir, "sweep_pgen_pswap.csv"), index=False,
        )
        print(f"\n[checkpoint] saved sweep1 through N={n_nodes}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Sweep 2: p_gen × cutoff   (p_swap = 1 fixed)
# ═══════════════════════════════════════════════════════════════════

CUTOFF_GRID = [4, 6, 8, 10, 15, 20, 30, 50, 80]
PGEN_GRID_SWEEP2 = np.round(np.arange(0.1, 1.01, 0.1), 2)
SWEEP2_N = 8  # fixed chain length for sweep 2


def sweep_pgen_cutoff(
    agent: QRNAgent,
    n_episodes: int,
    rng: np.random.Generator,
    n_nodes: int = SWEEP2_N,
    save_dir: str = ".",
) -> pd.DataFrame:
    """Return a DataFrame with columns [p_gen, cutoff, delta_pct].

    Saves incrementally after each p_gen row so partial results
    survive job timeouts.
    """
    rows: list[dict] = []
    total = len(PGEN_GRID_SWEEP2) * len(CUTOFF_GRID)
    done_count = 0

    for p_gen in PGEN_GRID_SWEEP2:
        for cutoff in CUTOFF_GRID:
            done_count += 1
            _log_progress("sweep2", done_count, total, n_nodes, p_gen, 1.0)

            # Estimate max_steps only (cutoff is the independent variable here)
            pilot_cap = 800
            pilot_times = _pilot_swap_asap(
                n_nodes, p_gen, p_swap=1.0, cutoff=cutoff,
                max_steps=pilot_cap, n_ch=4, n_episodes=15, rng=rng,
            )
            successes = [t for t in pilot_times if t < pilot_cap]
            if len(successes) >= 4:
                max_steps = min(int(np.percentile(successes, 95) * 1.2) + 5, pilot_cap)
            else:
                max_steps = pilot_cap

            cfg = RunConfig(n_nodes, p_gen, p_swap=1.0, cutoff=cutoff, max_steps=max_steps)
            res = run_comparison(agent, cfg, n_episodes, rng)
            rows.append({
                "p_gen": p_gen,
                "cutoff": cutoff,
                "delta_pct": relative_improvement(res["agent"], res["swap_asap"]),
            })

        # ── incremental save after each p_gen ──
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(
            os.path.join(save_dir, "sweep_pgen_cutoff.csv"), index=False,
        )
        print(f"\n[checkpoint] saved sweep2 through p_gen={p_gen}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

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
    # No both_fail column — cannot distinguish ties from mutual failure
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


def plot_pgen_pswap(df: pd.DataFrame, save_dir: str) -> None:
    """Two separate figures:
    1. FacetGrid of 4 coarse heatmaps (N = 4, 10, 12, 15).
    2. FacetGrid of 4 zoomed-in heatmaps (low p_gen, high p_swap).

    Cells where both strategies fail to reach e2e are greyed out with 'N/A'.
    """
    coarse = df[df["region"] == "coarse"]
    inset_df = df[df["region"] == "inset"]

    vmin = df["delta_pct"].quantile(0.02)
    vmax = df["delta_pct"].quantile(0.98)
    abs_bound = max(abs(vmin), abs(vmax))

    # ── Figure 1: coarse grid ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for idx, n_nodes in enumerate(NODE_COUNTS):
        ax = axes_flat[idx]
        sub = coarse[coarse["N"] == n_nodes]
        _draw_heatmap(ax, sub, abs_bound)
        ax.set_title(f"N = {n_nodes}", fontsize=12, fontweight="bold")
        ax.set_xlabel("$p_{gen}$")
        ax.set_ylabel("$p_{swap}$")

    fig.suptitle(
        "Agent vs Swap-ASAP: relative delivery-time improvement (%)\n"
        "positive (blue) = agent faster, negative (red) = swap-ASAP faster, "
        "grey = both fail",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "heatmap_pgen_pswap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")

    # ── Figure 2: zoomed region (low p_gen, high p_swap) ──
    if inset_df.empty:
        return

    fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 12))
    axes_z_flat = axes_z.flatten()

    for idx, n_nodes in enumerate(NODE_COUNTS):
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
        "positive (blue) = agent faster, negative (red) = swap-ASAP faster, "
        "grey = both fail",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path_z = os.path.join(save_dir, "heatmap_pgen_pswap_zoom.png")
    fig_z.savefig(path_z, dpi=200, bbox_inches="tight")
    plt.close(fig_z)
    print(f"[plot] saved {path_z}")


def plot_pgen_cutoff(df: pd.DataFrame, save_dir: str) -> None:
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
        f"Agent vs Swap-ASAP  (N = {SWEEP2_N}, $p_{{swap}}$ = 1)\n"
        "positive (blue) = agent faster",
        fontsize=12,
    )
    ax.set_xlabel("$p_{gen}$")
    ax.set_ylabel("cutoff")

    path = os.path.join(save_dir, "heatmap_pgen_cutoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

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


def load_agent(model_path: str) -> QRNAgent:
    agent = QRNAgent()
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device, weights_only=True)
    )
    agent.policy_net.eval()
    agent.epsilon = 0.0
    return agent


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch validation: agent vs swap-ASAP parameter sweeps",
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to policy.pth checkpoint",
    )
    p.add_argument(
        "--episodes", type=int, default=200,
        help="Episodes per (strategy, parameter) pair",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility",
    )
    p.add_argument(
        "--save_dir", type=str, default="results/batch_validate",
        help="Directory for output plots and CSVs",
    )
    p.add_argument(
        "--sweep", type=str, default="both",
        choices=["both", "pgen_pswap", "pgen_cutoff"],
        help="Which sweep(s) to run",
    )
    p.add_argument(
        "--sweep2_nodes", type=int, default=SWEEP2_N,
        help="Chain length for the p_gen × cutoff sweep",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading model from {args.model}")
    agent = load_agent(args.model)

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
        print("\n══ Sweep 1: p_gen × p_swap (N = 4, 10, 12, 15) ══")
        df1 = sweep_pgen_pswap(agent, args.episodes, rng, save_dir=args.save_dir)
        csv1 = os.path.join(args.save_dir, "sweep_pgen_pswap.csv")
        df1.to_csv(csv1, index=False)
        print(f"[data] saved {csv1}")
        plot_pgen_pswap(df1, args.save_dir)

        results_json["sweeps"]["pgen_pswap"] = {
            "description": "p_gen x p_swap sweep for N = 4, 10, 12, 15",
            "node_counts": NODE_COUNTS,
            "results": df1.to_dict(orient="records"),
        }

    if args.sweep in ("both", "pgen_cutoff"):
        global SWEEP2_N
        SWEEP2_N = args.sweep2_nodes
        print(f"\n══ Sweep 2: p_gen × cutoff (N = {SWEEP2_N}, p_swap = 1) ══")
        df2 = sweep_pgen_cutoff(agent, args.episodes, rng, n_nodes=SWEEP2_N, save_dir=args.save_dir)
        csv2 = os.path.join(args.save_dir, "sweep_pgen_cutoff.csv")
        df2.to_csv(csv2, index=False)
        print(f"[data] saved {csv2}")
        plot_pgen_cutoff(df2, args.save_dir)

        results_json["sweeps"]["pgen_cutoff"] = {
            "description": f"p_gen x cutoff sweep (N = {SWEEP2_N}, p_swap = 1)",
            "n_nodes": SWEEP2_N,
            "results": df2.to_dict(orient="records"),
        }

    json_path = os.path.join(args.save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[data] saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
