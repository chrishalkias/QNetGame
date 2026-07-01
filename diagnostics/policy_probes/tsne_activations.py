"""t-SNE of conv3 node embeddings from greedy rollouts, six colourings.

Projects the post-conv3 (64-d) embedding of every interior-node decision to 2-D
with t-SNE and renders it coloured by chosen action, Q-value margin, and the
occupancy / fidelity / link-urgency / can-swap features (see _embed). If the
action panel is a blob but the feature panels show clean gradients, the
representation is organised by physical state rather than by discrete action.

  compute: PYTHONPATH=. python diagnostics/policy_probes/tsne_activations.py --ckpt <path>
  plot:    PYTHONPATH=. python diagnostics/policy_probes/tsne_activations.py --plot
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
    ap.add_argument("--perplexity", type=float, default=40.0)
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
    npz = os.path.join(out, "tsne_activations.npz")
    if a.plot:
        z = np.load(npz)
        emb, A, margin, X = z["emb"], z["A"], z["margin"], z["X"]
    else:
        emb, A, margin, X = _embed.collect_embed(a, "tsne", npz)
    _embed.render(emb, A, margin, X, out, "tsne_activations",
                  rf"t-SNE of {_LNAME[a.layer]} embeddings")


if __name__ == "__main__":
    main()
