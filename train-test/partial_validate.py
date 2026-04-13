"""
Partial validation: complete only the missing p_gen values from sweep 2.

Reads the existing sweep_pgen_cutoff.csv, determines which p_gen values
are missing, runs only those, and appends the results to the CSV.
Sweep 1 is already complete and is skipped entirely.

Usage
-----
    python train-test/partial_validate.py \
        --model checkpoints/cluster_004/policy.pth \
        --episodes 200 \
        --save_dir results/batch_validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────
from batch_validate import (
    CUTOFF_GRID,
    PGEN_GRID_SWEEP2,
    RunConfig,
    _log_progress,
    _pilot_swap_asap,
    load_agent,
    plot_pgen_cutoff,
    plot_pgen_pswap,
    relative_improvement,
    run_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete missing sweep 2 points and regenerate all plots",
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="results/batch_validate")
    p.add_argument("--sweep2_nodes", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_nodes = args.sweep2_nodes

    # ── Load existing sweep 2 results ──
    csv_path = os.path.join(args.save_dir, "sweep_pgen_cutoff.csv")
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        pgens_done = set(df_existing["p_gen"].unique())
        print(f"Found {len(df_existing)} existing rows, "
              f"p_gen done: {sorted(pgens_done)}")
    else:
        df_existing = pd.DataFrame(columns=["p_gen", "cutoff", "delta_pct"])
        pgens_done = set()
        print("No existing CSV found, running full sweep 2")

    pgens_missing = [p for p in PGEN_GRID_SWEEP2 if p not in pgens_done]
    total_missing = len(pgens_missing) * len(CUTOFF_GRID)

    if total_missing == 0:
        print("Sweep 2 is already complete!")
    else:
        print(f"Missing p_gen values: {pgens_missing}")
        print(f"Points to compute: {total_missing}")

        print(f"\nLoading model from {args.model}")
        agent = load_agent(args.model)

        rows: list[dict] = []
        done_count = 0

        for p_gen in pgens_missing:
            for cutoff in CUTOFF_GRID:
                done_count += 1
                _log_progress("sweep2", done_count, total_missing,
                              n_nodes, p_gen, 1.0)

                pilot_cap = 800
                pilot_times = _pilot_swap_asap(
                    n_nodes, p_gen, p_swap=1.0, cutoff=cutoff,
                    max_steps=pilot_cap, n_ch=4, n_episodes=15, rng=rng,
                )
                successes = [t for t in pilot_times if t < pilot_cap]
                if len(successes) >= 4:
                    max_steps = min(
                        int(np.percentile(successes, 95) * 1.2) + 5,
                        pilot_cap,
                    )
                else:
                    max_steps = pilot_cap

                cfg = RunConfig(n_nodes, p_gen, p_swap=1.0,
                                cutoff=cutoff, max_steps=max_steps)
                res = run_comparison(agent, cfg, args.episodes, rng)
                rows.append({
                    "p_gen": p_gen,
                    "cutoff": cutoff,
                    "delta_pct": relative_improvement(
                        res["agent"], res["swap_asap"]),
                })

            # Incremental save after each p_gen
            df_new = pd.concat(
                [df_existing, pd.DataFrame(rows)], ignore_index=True,
            )
            df_new.to_csv(csv_path, index=False)
            print(f"\n[checkpoint] saved through p_gen={p_gen} "
                  f"({len(df_new)} total rows)")

    # ── Regenerate all plots ──
    print("\nRegenerating plots...")

    df_cutoff = pd.read_csv(csv_path)
    plot_pgen_cutoff(df_cutoff, args.save_dir)

    csv1_path = os.path.join(args.save_dir, "sweep_pgen_pswap.csv")
    if os.path.exists(csv1_path):
        df_pswap = pd.read_csv(csv1_path)
        plot_pgen_pswap(df_pswap, args.save_dir)

    # ── Save combined JSON ──
    results_json = {
        "metadata": {
            "model": args.model,
            "episodes_per_point": args.episodes,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "completed by partial_validate.py",
        },
        "sweeps": {},
    }
    if os.path.exists(csv1_path):
        results_json["sweeps"]["pgen_pswap"] = {
            "description": "p_gen x p_swap sweep for N = 4, 10, 12, 15",
            "results": pd.read_csv(csv1_path).to_dict(orient="records"),
        }
    results_json["sweeps"]["pgen_cutoff"] = {
        "description": f"p_gen x cutoff sweep (N = {n_nodes}, p_swap = 1)",
        "n_nodes": n_nodes,
        "results": df_cutoff.to_dict(orient="records"),
    }
    json_path = os.path.join(args.save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[data] saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
