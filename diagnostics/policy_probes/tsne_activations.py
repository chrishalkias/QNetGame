"""t-SNE of conv3 node embeddings from real greedy rollouts.

Collects the post-conv3 (64-d) embedding for every interior-node decision the agent
makes over real rollouts, projects to 2-D with t-SNE, and colours points by (left)
the chosen action and (right) the Q-value margin (decisiveness). Reveals whether the
learned representation organises state space into coherent decision regions — the
canonical "what has the DQN learned" view (cf. Mnih et al. 2015).

  PYTHONPATH=. python diagnostics/policy_probes/tsne_activations.py --ckpt <path>
"""
from __future__ import annotations
import argparse, os
import numpy as np
from diagnostics.policy_probes import _collect as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_points", type=int, default=5000,
                    help="subsample this many decisions for t-SNE")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)

    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed)
    H, A, margin = d["H"], d["A"], d["margin"]

    rng = np.random.default_rng(a.seed)
    if len(H) > a.max_points:
        keep = rng.choice(len(H), a.max_points, replace=False)
        H, A, margin = H[keep], A[keep], margin[keep]
    print(f"t-SNE on {len(H)} points (perplexity={a.perplexity}) ...", flush=True)

    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
               random_state=a.seed).fit_transform(H)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    for act in (0, 1, 2):
        m = A == act
        ax1.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                    color=C.ACTION_COLORS[act], label=C.ACTION_NAMES[act])
    ax1.set_title("conv3 embedding, coloured by chosen action")
    ax1.legend(frameon=False, markerscale=2)
    ax1.set_xticks([]); ax1.set_yticks([])

    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.6, c=margin, cmap="viridis")
    ax2.set_title("coloured by Q-value margin (decisiveness)")
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(sc, ax=ax2, shrink=0.85, label=r"$Q_\mathrm{best}-Q_\mathrm{2nd}$")

    fig.suptitle(f"t-SNE of conv3 embeddings over real rollouts ({len(H)} decisions)")
    stem = os.path.join(out, "tsne_activations")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    print(f"saved -> {stem}.png / .pdf")


if __name__ == "__main__":
    main()
