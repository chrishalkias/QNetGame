"""Regenerate training plots from a saved metrics.json — no retraining needed.

train() writes <save_path>/metrics.json; this reloads it and re-runs the
plotting, so stylistic tweaks to the figures are a few-second turnaround.

  PYTHONPATH=$(pwd) python train-test/replot.py --dir checkpoints/compare_optimal
"""
from __future__ import annotations
import argparse
import json
import os

from rl_stack.agent import QRNAgent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True,
                   help="checkpoint dir containing metrics.json (also the output dir)")
    p.add_argument("--window", type=int, default=None,
                   help="rolling-mean window for all smoothed curves "
                        "(default: adaptive per panel). Larger = smoother.")
    args = p.parse_args(argv)

    mpath = os.path.join(args.dir, "metrics.json")
    if not os.path.isfile(mpath):
        raise FileNotFoundError(
            f"{mpath} not found — was the run trained after metrics-saving was added?")
    with open(mpath) as f:
        metrics = json.load(f)

    QRNAgent._plot_training(metrics, args.dir, window=args.window)
    out = ["training_metrics.png"]
    if metrics.get("cmp_agent"):
        out.append("training_compare.png")
    print(f"[replotted] {', '.join(out)} -> {args.dir}/")


if __name__ == "__main__":
    main()
