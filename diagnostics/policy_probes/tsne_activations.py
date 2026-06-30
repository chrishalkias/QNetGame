"""t-SNE of conv3 node embeddings from greedy rollouts.

Collects the post-conv3 (64-d) embedding for every interior-node decision over
greedy rollouts, projects to 2-D with t-SNE, and colours points by (left) the
chosen action and (right) the Q-value margin (decisiveness). Reveals whether the
representation organises state space into coherent decision regions (cf. Mnih et
al. 2015). The margin panel clips colour limits to robust percentiles so a single
outlier does not wash out the scale.

  compute: PYTHONPATH=. python diagnostics/policy_probes/tsne_activations.py --ckpt <path>
  plot:    PYTHONPATH=. python diagnostics/policy_probes/tsne_activations.py --plot
"""
from __future__ import annotations
import argparse, os
import numpy as np
from diagnostics.policy_probes import _collect as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="render from cached npz")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max_points", type=int, default=8000)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


def compute(a, npz):
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
    np.savez(npz, emb=emb, A=A, margin=margin)
    return emb, A, margin


def render(emb, A, margin, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(margin, [2, 98])          # robust scale (ignore outliers)

    def draw(usetex):
        plt.rcParams.update({
            "text.usetex": usetex, "font.family": "serif",
            "mathtext.fontset": "cm", "font.size": 12,
            "axes.titlesize": 14, "figure.dpi": 150,
        })
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        for act in (0, 1, 2):                        # noop under, swap/purify on top
            m = A == act
            ax1.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.6, linewidths=0,
                        color=C.ACTION_COLORS[act], label=C.ACTION_NAMES[act].title())
        ax1.set_title("Coloured by chosen action")
        leg = ax1.legend(frameon=False, markerscale=2.5, loc="best")
        for h in leg.legend_handles:
            h.set_alpha(1.0)
        ax1.set_xticks([]); ax1.set_yticks([])
        sc = ax2.scatter(emb[:, 0], emb[:, 1], s=9, alpha=0.7, linewidths=0,
                         c=np.clip(margin, lo, hi), cmap="viridis", vmin=lo, vmax=hi)
        ax2.set_title("Coloured by Q-value margin")
        ax2.set_xticks([]); ax2.set_yticks([])
        fig.colorbar(sc, ax=ax2, shrink=0.85,
                     label=r"$Q_{\mathrm{best}}-Q_{\mathrm{2nd}}$ (2--98\%)" if usetex
                     else r"$Q_{best}-Q_{2nd}$ (2-98%)")
        fig.suptitle(r"t-SNE of conv$_3$ embeddings", fontsize=15)
        return fig

    stem = os.path.join(out, "tsne_activations")
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
    npz = os.path.join(out, "tsne_activations.npz")
    if a.plot:
        z = np.load(npz)
        emb, A, margin = z["emb"], z["A"], z["margin"]
    else:
        emb, A, margin = compute(a, npz)
    render(emb, A, margin, out)


if __name__ == "__main__":
    main()
