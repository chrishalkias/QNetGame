"""Permutation feature importance over greedy rollouts.

For each observation feature, permute its values across all interior-node decisions
collected from greedy rollouts, re-feed the (otherwise unchanged) states, and
measure the fraction of decisions whose greedy action changes. A larger value means
the policy relies more on that feature. Distribution-grounded and model-agnostic.

Note: can_swap / can_purify are permuted as observation features while the action
mask is held at its true value, so this isolates reliance on the feature beyond the
mask that already gates those actions.

  compute: PYTHONPATH=. python diagnostics/policy_probes/feature_importance.py --ckpt <path>
  plot:    PYTHONPATH=. python diagnostics/policy_probes/feature_importance.py --plot
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from diagnostics.policy_probes import _collect as C

LABELS = {
    "occ": "Occupancy", "fidelity": "Fidelity", "is_target": "Target",
    "avail": "Availability", "can_swap": "Can swap", "can_purify": "Can purify",
    "p_gen": r"$p_{\mathrm{gen}}$", "p_swap": r"$p_{\mathrm{swap}}$",
    "urgency": "Urgency",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="render from existing json")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def compute(a, out):
    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed)
    model, states, idx, base = d["model"], d["states"], d["idx"], d["A"]
    rng = np.random.default_rng(a.seed)
    flip = {}
    for j, name in enumerate(C.FEATURE_NAMES):
        orig = d["X"][:, j].copy()
        perm = rng.permutation(orig)
        for k, (si, node) in enumerate(idx):
            states[si]["x"][node, j] = perm[k]
        new = C.greedy_actions_for_states(model, states, d["device"])
        new_flat = np.array([new[si][node] for (si, node) in idx])
        flip[name] = float(np.mean(new_flat != base))
        for k, (si, node) in enumerate(idx):
            states[si]["x"][node, j] = orig[k]
        print(f"  {name:<11} {flip[name]:.3f}", flush=True)
    json.dump(flip, open(os.path.join(out, "feature_importance.json"), "w"), indent=2)
    return flip


def render(flip, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(usetex):
        plt.rcParams.update({
            "text.usetex": usetex, "font.family": "serif",
            "mathtext.fontset": "cm", "font.size": 12,
            "axes.labelsize": 13, "axes.titlesize": 14, "figure.dpi": 150,
        })
        names = sorted(flip, key=flip.get)
        vals = [flip[n] for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        bars = ax.barh([LABELS[n] for n in names], vals, color="#4C72B0")
        for b, v in zip(bars, vals):
            ax.text(v + 0.002, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=9)
        ax.set_xlabel("Fraction of decisions altered under feature permutation")
        ax.set_title("Permutation feature importance")
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(alpha=0.3, axis="x")
        return fig

    stem = os.path.join(out, "feature_importance")
    try:
        fig = draw(True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[usetex unavailable ({e}); falling back to mathtext]")
        plt.close("all")
        fig = draw(False)
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"saved -> {stem}.png / .pdf")


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)
    if a.plot:
        flip = json.load(open(os.path.join(out, "feature_importance.json")))
    else:
        flip = compute(a, out)
    render(flip, out)


if __name__ == "__main__":
    main()
