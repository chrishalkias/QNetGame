"""
--------------------------------------------------------------------------------
Fit a state-conditioned purify probability P(purify | state) on the
both-legal subset of the purify_map probe data, and export it as plain
coefficients so it can run numpy-only on the cluster (no sklearn at eval
time).

Context: `hybrid_policy.make_hybrid_fn` already tests whether a single
constant q recovers the trained agent's edge over purify_then_swap in the
both-legal branch. This script asks the harder question first: how much
signal is actually IN the state at the moment both PURIFY and SWAP are
legal? `experiments/policy_probes/purify_map.py::model_fits` already found
a logistic ceiling of ~0.63 (s1) / ~0.71 (s3) test AUC over ALL purify-legal
decisions (both-legal and purify-only mixed); this script isolates the
both-legal subset (`can_swap == 1.0`, i.e. SWAP was ALSO legal, so the
decision is genuinely a choice) and fits a fresh model on exactly that
subset, so a later task can graft `q(state)` into the hybrid policy in
place of a constant q.

Input: results/probes/omni_v3_20k_{tag}/purify_map.npz, produced by
experiments/policy_probes/purify_map.py (X: (n,35) float32, columns: (35,)
str, label: (n,) int8 in {0,1} = chose PURIFY, episode: (n,) int32, cell:
(n,) str in {"train","testbed"}).

Split: episode-level, mirroring
experiments/policy_probes/purify_map.py::_episode_split EXACTLY (75/25 by
unique episode id, shuffled with a seeded RNG) so the resulting test AUC is
directly comparable to the 0.63/0.71 ceilings recorded there. (The
`_episode_split` name says "the split used for episode-level holdout"; the
75/25 ratio, not 80/20, is what the function actually does, and that is
what this script mirrors, on purpose, over the wording.)

Output: experiments/q_heuristic/q_conditional_{tag}.json (NOT
results/probes/), because the cluster upload script (scripts/sync/
upload.sh) syncs code under experiments/, not results/; a later task that
runs this policy on the cluster needs the JSON to already be there.

Runtime contract (numpy-only, no sklearn needed to USE the model):
    z = coef . ((x - mu) / sigma) + intercept
    q = 1 / (1 + exp(-z))
`x` is the 34-feature row in `columns` order (all purify_map columns except
`can_swap`, which is constant 1.0 in the both-legal subset and therefore
carries no information as a feature). `mu`/`sigma` are the TRAIN-split
per-feature mean/std (std==0 clamped to 1.0 so the affine map is always
well-defined).

Usage (from repo root, PYTHONPATH=src:.):
    python experiments/q_heuristic/fit_q_conditional.py
    python experiments/q_heuristic/fit_q_conditional.py --tag s1
    python experiments/q_heuristic/fit_q_conditional.py --verify
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", nargs="+", default=["s1", "s3"],
                    help="run tags; each maps to "
                         "results/probes/omni_v3_20k_<tag>/purify_map.npz")
    ap.add_argument("--seed", type=int, default=0,
                    help="episode-split seed, must match purify_map.py's "
                         "_episode_split call (seed 0 there)")
    ap.add_argument("--max_iter", type=int, default=2000,
                    help="LogisticRegression max_iter")
    ap.add_argument("--npz_dir", default="results/probes",
                    help="parent of results/probes/omni_v3_20k_<tag>/purify_map.npz")
    ap.add_argument("--out_dir", default="experiments/q_heuristic",
                    help="where to write q_conditional_<tag>.json")
    ap.add_argument("--verify", action="store_true",
                    help="after fitting, reload the JSON and recompute test AUC "
                         "with pure numpy; assert it matches sklearn to 1e-6")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------
# episode split (mirrors experiments/policy_probes/purify_map.py::_episode_split
# EXACTLY: 0.75 cut, same RNG call sequence, so results are seed-comparable)
# ----------------------------------------------------------------------------
def _episode_split(ep, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    cut = int(0.75 * len(uniq))
    tr = np.isin(ep, uniq[:cut])
    return tr, ~tr


# ----------------------------------------------------------------------------
# numpy-only inference contract (must match the JSON writer's docstring)
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_q(x, mu, sigma, coef, intercept):
    """x: (n, d) raw features, in the same column order as the JSON's
    `columns`. Returns (n,) probabilities. Pure numpy, no sklearn."""
    xs = (x - mu) / sigma
    z = xs @ coef + intercept
    return sigmoid(z)


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def fit_one(tag, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import sklearn

    npz_path = os.path.join(args.npz_dir, f"omni_v3_20k_{tag}", "purify_map.npz")
    d = np.load(npz_path, allow_pickle=True)
    X_all, columns, y_all, ep_all, cell_all = (
        d["X"], list(d["columns"]), d["label"], d["episode"], d["cell"])

    can_swap_idx = columns.index("can_swap")
    both_legal = X_all[:, can_swap_idx] == 1.0

    X = X_all[both_legal]
    y = y_all[both_legal]
    ep = ep_all[both_legal]
    cell = cell_all[both_legal]
    feat_columns = [c for c in columns if c != "can_swap"]
    keep_idx = [i for i, c in enumerate(columns) if c != "can_swap"]
    X = X[:, keep_idx].astype(np.float64)

    n_rows_both_legal = int(X.shape[0])
    base_rate = float(y.mean())

    tr, te = _episode_split(ep, args.seed)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    Xs_tr = (X_tr - mu) / sigma
    Xs_te = (X_te - mu) / sigma

    clf = LogisticRegression(max_iter=args.max_iter).fit(Xs_tr, y_tr)
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])

    p_tr = clf.predict_proba(Xs_tr)[:, 1]
    p_te = clf.predict_proba(Xs_te)[:, 1]
    train_auc = float(roc_auc_score(y_tr, p_tr))
    test_auc = float(roc_auc_score(y_te, p_te))

    testbed_mask = (cell != "train")
    mean_pred_q_testbed = (float(predict_q(X[testbed_mask], mu, sigma, coef,
                                            intercept).mean())
                            if bool(testbed_mask.any()) else None)

    # expected ceiling band from the recorded all-purify-legal AUCs
    expected = {"s1": 0.63, "s3": 0.71}.get(tag)
    if expected is not None and abs(test_auc - expected) > 0.03:
        print(f"WARNING: tag={tag} test_auc={test_auc:.4f} is more than 0.03 "
              f"away from the recorded ceiling {expected} (purify_map.py, "
              f"all purify-legal decisions, not just both-legal; some drift "
              f"is expected but a large one may signal a split/feature bug)")

    order = np.argsort(-np.abs(coef))[:10]
    top_features = [
        {"feature": feat_columns[i], "coef": float(coef[i]),
         "sign": "+" if coef[i] >= 0 else "-"}
        for i in order]

    print(f"\n=== tag={tag} ===")
    print(f"npz: {npz_path}")
    print(f"n_rows_both_legal: {n_rows_both_legal}  "
          f"(train={int(tr.sum())}, test={int(te.sum())})")
    print(f"base_rate P(purify | both-legal): {base_rate:.4f}")
    print(f"train_auc: {train_auc:.4f}   test_auc: {test_auc:.4f}")
    print(f"mean_pred_q_testbed: {mean_pred_q_testbed}")
    print("top-10 |coefficient| features:")
    for r in top_features:
        print(f"  {r['sign']} {r['feature']:16s} {r['coef']:+.4f}")

    data = {
        "tag": tag,
        "columns": feat_columns,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "base_rate_both_legal": base_rate,
        "mean_pred_q_testbed": mean_pred_q_testbed,
        "n_rows_both_legal": n_rows_both_legal,
        "top_features": top_features,
        "meta": {
            "npz_path": npz_path,
            "sklearn_version": sklearn.__version__,
            "seed": args.seed,
            "date": datetime.date.today().isoformat(),
            "runtime_contract":
                "q = sigmoid(coef . ((x - mu) / sigma) + intercept), "
                "x in `columns` order, pure numpy, no sklearn required at eval time",
        },
    }

    out_path = os.path.join(args.out_dir, f"q_conditional_{tag}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"saved -> {out_path}")

    if args.verify:
        _verify(out_path, X_te, y_te, test_auc)

    return data


def _verify(json_path, X_te, y_te, sklearn_test_auc):
    """Reload the JSON, recompute test AUC with pure numpy from mu/sigma/coef/
    intercept alone, and assert it matches sklearn's predict_proba AUC to
    1e-6. This is the proof that the exported contract is faithful."""
    from sklearn.metrics import roc_auc_score
    with open(json_path) as f:
        d = json.load(f)
    mu = np.asarray(d["mu"])
    sigma = np.asarray(d["sigma"])
    coef = np.asarray(d["coef"])
    intercept = d["intercept"]
    q_np = predict_q(X_te, mu, sigma, coef, intercept)
    auc_np = float(roc_auc_score(y_te, q_np))
    diff = abs(auc_np - sklearn_test_auc)
    status = "OK" if diff < 1e-6 else "MISMATCH"
    print(f"VERIFY[{d['tag']}]: numpy_test_auc={auc_np:.9f}  "
          f"sklearn_test_auc={sklearn_test_auc:.9f}  diff={diff:.2e}  {status}")
    assert diff < 1e-6, (
        f"numpy-contract verification failed for {json_path}: "
        f"|{auc_np} - {sklearn_test_auc}| = {diff} >= 1e-6")


def main(argv=None):
    args = parse_args(argv)
    for tag in args.tag:
        fit_one(tag, args)


if __name__ == "__main__":
    main()
