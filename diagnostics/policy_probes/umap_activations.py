"""UMAP of conv3 node embeddings from greedy rollouts, six colourings.

Same as tsne_activations but with a UMAP projection (better preserves global
structure than t-SNE). Renders the embedding coloured by chosen action, Q-value
margin, and the occupancy / fidelity / link-urgency / can-swap features (_embed).

  compute: PYTHONPATH=. python diagnostics/policy_probes/umap_activations.py --ckpt <path>
  plot:    PYTHONPATH=. python diagnostics/policy_probes/umap_activations.py --plot
"""
from __future__ import annotations
import argparse, os
import numpy as np
from diagnostics.policy_probes import _embed


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="render from cached npz")
    ap.add_argument("--ckpt", default="checkpoints/omni_nopen_3k/policy.pth")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max_points", type=int, default=8000)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--layer", choices=["head", "conv3"], default="head",
                    help="head = penultimate decision layer; conv3 = graph encoder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    return ap.parse_args()


_LNAME = {"conv3": r"conv$_3$ (graph encoder)", "head": "head (penultimate, pre-Q)"}


def main():
    a = parse_args()
    out = a.save_dir or os.path.join(os.path.dirname(a.ckpt), "diagnostics")
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, "umap_activations.npz")
    if a.plot:
        z = np.load(npz)
        emb, A, margin, X = z["emb"], z["A"], z["margin"], z["X"]
    else:
        emb, A, margin, X = _embed.collect_embed(a, "umap", npz)
    _embed.render(emb, A, margin, X, out, "umap_activations",
                  rf"UMAP of {_LNAME[a.layer]} embeddings")


if __name__ == "__main__":
    main()
