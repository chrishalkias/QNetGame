"""Shared 2-D embedding (t-SNE / UMAP) of conv3 activations, with multi-panel
colouring. Renders six views of the *same* embedding: chosen action, Q-value
margin, and four state features (occupancy, fidelity, link-urgency, can-swap).

A blob under action-colouring but clean gradients under the feature-colourings
means the representation is organised by physical state, not by discrete action.
"""
from __future__ import annotations
import os
import numpy as np
from diagnostics.policy_probes import _collect as C

# continuous/near-continuous feature panels: (obs index, label)
_FEATS = [(0, "Occupancy"), (1, "Fidelity"), (8, "Link urgency"), (4, "Can swap")]


def embed(method, H, seed, perplexity=40.0, n_neighbors=30, min_dist=0.1):
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=seed).fit_transform(H)
    if method == "umap":
        import umap
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed).fit_transform(H)
    raise ValueError(f"unknown method {method}")


def collect_embed(a, method, npz):
    d = C.collect(a.ckpt, episodes=a.episodes, seed=a.seed)
    H, A, margin, X = d["H"], d["A"], d["margin"], d["X"]
    rng = np.random.default_rng(a.seed)
    if len(H) > a.max_points:
        keep = rng.choice(len(H), a.max_points, replace=False)
        H, A, margin, X = H[keep], A[keep], margin[keep], X[keep]
    print(f"{method} on {len(H)} points ...", flush=True)
    emb = embed(method, H, a.seed, perplexity=getattr(a, "perplexity", 40.0),
                n_neighbors=getattr(a, "n_neighbors", 30),
                min_dist=getattr(a, "min_dist", 0.1))
    np.savez(npz, emb=emb, A=A, margin=margin, X=X)
    return emb, A, margin, X


def render(emb, A, margin, X, out, stem, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(margin, [2, 98])       # robust scale for the margin panel

    def draw(usetex):
        plt.rcParams.update({"text.usetex": usetex, "font.family": "serif",
                             "mathtext.fontset": "cm", "font.size": 11,
                             "axes.titlesize": 13, "figure.dpi": 150})
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)
        ax = axes.ravel()
        for act in (0, 1, 2):                     # panel 0: chosen action
            m = A == act
            ax[0].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.55, linewidths=0,
                          color=C.ACTION_COLORS[act], label=C.ACTION_NAMES[act].title())
        ax[0].set_title("Chosen action")
        leg = ax[0].legend(frameon=False, markerscale=3, loc="best")
        for h in leg.legend_handles:
            h.set_alpha(1.0)
        sc = ax[1].scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, linewidths=0,
                           c=np.clip(margin, lo, hi), cmap="viridis", vmin=lo, vmax=hi)
        ax[1].set_title("Q-value margin")
        fig.colorbar(sc, ax=ax[1], shrink=0.8,
                     label=r"$Q_{\mathrm{best}}-Q_{\mathrm{2nd}}$")
        for k, (j, label) in enumerate(_FEATS, start=2):   # feature panels
            sc = ax[k].scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, linewidths=0,
                               c=X[:, j], cmap="viridis")
            ax[k].set_title(label)
            fig.colorbar(sc, ax=ax[k], shrink=0.8)
        for a_ in ax:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(suptitle, fontsize=16)
        return fig

    path = os.path.join(out, stem)
    try:
        fig = draw(True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[usetex unavailable ({e}); falling back to mathtext]")
        plt.close("all")
        fig = draw(False)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", bbox_inches="tight")
    print(f"saved -> {path}.png / .pdf")
