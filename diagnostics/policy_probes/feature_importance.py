"""Permutation feature importance over real greedy rollouts.

For each of the 9 observation features, permute its values across all interior-node
decisions, re-feed the (otherwise unchanged) states, and measure the fraction of
decisions whose greedy action flips. Higher flip-rate => the policy relies more on
that feature. Distribution-grounded (real visited states) and model-agnostic.

Note: can_swap/can_purify are permuted as *observation features* while the action
mask is held at its true value, so this measures reliance on the feature beyond the
mask that already gates those actions.

  PYTHONPATH=. python diagnostics/policy_probes/feature_importance.py --ckpt <path>
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from diagnostics.policy_probes import _collect as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None,
                    help="default: <ckpt dir>/diagnostics")
    return ap.parse_args()


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)

    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed)
    model, states, idx, base = d["model"], d["states"], d["idx"], d["A"]
    rng = np.random.default_rng(a.seed)

    flip = np.zeros(len(C.FEATURE_NAMES))
    for j in range(len(C.FEATURE_NAMES)):
        orig = d["X"][:, j].copy()
        perm = rng.permutation(orig)
        for k, (si, node) in enumerate(idx):
            states[si]["x"][node, j] = perm[k]
        new = C.greedy_actions_for_states(model, states, d["device"])
        new_flat = np.array([new[si][node] for (si, node) in idx])
        flip[j] = float(np.mean(new_flat != base))
        for k, (si, node) in enumerate(idx):           # restore
            states[si]["x"][node, j] = orig[k]
        print(f"  {C.FEATURE_NAMES[j]:<11} flip-rate={flip[j]:.3f}", flush=True)

    order = np.argsort(flip)
    names = [C.FEATURE_NAMES[i] for i in order]
    json.dump({C.FEATURE_NAMES[i]: float(flip[i]) for i in range(len(flip))},
              open(os.path.join(out, "feature_importance.json"), "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.barh(names, flip[order], color="tab:purple")
    ax.set_xlabel("action flip-rate when feature is permuted")
    ax.set_title(f"Permutation feature importance ({len(base)} real decisions)")
    ax.grid(alpha=0.3, axis="x")
    stem = os.path.join(out, "feature_importance")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"saved -> {stem}.png / .pdf")


if __name__ == "__main__":
    main()
