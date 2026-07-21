"""
--------------------------------------------------------------------------------
When does the trained agent choose PURIFY among its purify opportunities?

For every interior-node decision where PURIFY is a legal action, roll out the
GREEDY full agent and record (a) whether it purified, (b) the 9-feature obs, a
1-hop neighbour-mean context, and hand-built engine features about the candidate
BBPSSW pair, (c) episode/cell context. Then search for a compact predictive rule
(base rates, univariate AUCs, logistic ceiling, shallow trees, hypothesis checks)
and render a two-panel figure. This is the data foundation for later grafting a
purify-selectivity rule into a heuristic.

Dual-mode so the figure re-renders without recomputing:
  compute: PYTHONPATH=src:. python experiments/policy_probes/purify_map.py --ckpt <path>
  plot:    PYTHONPATH=src:. python experiments/policy_probes/purify_map.py --plot --out_dir <dir>
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="re-render from the JSON")
    ap.add_argument("--ckpt", default="checkpoints/omni_v3_20k_s1/policy.pth")
    ap.add_argument("--episodes_train", type=int, default=300)
    ap.add_argument("--episodes_testbed", type=int, default=150,
                    help="per testbed N (N in {13,15})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None,
                    help="default: results/probes/<ckpt parent dir name>/")
    ap.add_argument("--smoke", action="store_true",
                    help="20 train episodes at N=6 only; base rates, no fits/plot")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# feature schema
# ----------------------------------------------------------------------------
from experiments.policy_probes._collect import FEATURE_NAMES, greedy  # noqa: E402
from simulator.repeater import NO_PARTNER  # noqa: E402

OBS_COLS = [f"obs_{n}" for n in FEATURE_NAMES]
NBR_COLS = [f"nbr_{n}" for n in FEATURE_NAMES]
ENG_COLS = ["free_slots", "n_avail", "n_pairs_cand", "n_spare",
            "F_hi", "F_lo", "age_frac_hi", "age_frac_lo", "mean_age_frac"]
CTX_COLS = ["can_swap", "pos_frac", "N", "n_ch", "p_gen", "p_swap",
            "cutoff", "t_frac"]
COLUMNS = OBS_COLS + NBR_COLS + ENG_COLS + CTX_COLS


def candidate_partner(ns):
    """Replicate QRNEnv._exec_purify partner selection EXACTLY: among partners
    with >=2 available (occupied & unlocked) qubits, pick the one with the most
    such qubits; ties break to the lowest partner id (np.unique returns sorted
    ids and max() keeps the first maximal). Returns partner id or None."""
    avail = ns.occupied & (~ns.locked)
    if int(avail.sum()) < 2:
        return None
    partners = ns.partner_node[avail]
    unique, counts = np.unique(partners[partners != NO_PARTNER], return_counts=True)
    valid = [(int(p), c) for p, c in zip(unique, counts) if c >= 2]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])[0]


def node_row(env, obs, mask, acts, i, ctx):
    """Build one feature row (list matching COLUMNS) + label + chosen_action for
    interior node i, which is known to have PURIFY masked-legal. Returns None if
    the candidate partner cannot be resolved (defensive; should not happen)."""
    ns = env.net.node_state(i)
    best_nb = candidate_partner(ns)
    if best_nb is None:
        return None
    n_ch = ns.n_ch
    occ, lock = ns.occupied, ns.locked
    avail = occ & (~lock)
    cand = avail & (ns.partner_node == best_nb)
    F = ns.fidelity[cand]
    lc = np.maximum(ns.link_cutoff[cand].astype(np.float64), 1.0)
    age_frac = ns.age[cand].astype(np.float64) / lc
    hi, lo = int(np.argmax(F)), int(np.argmin(F))
    lc_av = np.maximum(ns.link_cutoff[avail].astype(np.float64), 1.0)
    mean_age_frac = float((ns.age[avail].astype(np.float64) / lc_av).mean()) \
        if bool(avail.any()) else 0.0

    ei = obs["edge_index"]
    nbrs = ei[1][ei[0] == i]
    nbr_mean = (obs["x"][nbrs].mean(axis=0) if nbrs.size
                else np.zeros(len(FEATURE_NAMES), np.float64))

    n_pairs_cand = int(cand.sum())
    eng = [n_ch - int(occ.sum()), int(avail.sum()), n_pairs_cand,
           max(0, n_pairs_cand - 2), float(F[hi]), float(F[lo]),
           float(age_frac[hi]), float(age_frac[lo]), mean_age_frac]
    ctx_row = [float(mask[i, 1]), i / (env.N - 1), env.N, n_ch,
               ctx["p_gen"], ctx["p_swap"], ctx["cutoff"], env.steps / env.max_steps]
    row = list(obs["x"][i]) + list(nbr_mean) + eng + ctx_row
    return row, int(acts[i] == 2), int(acts[i])


def rollout(model, env, ctx, ep_id, rows, labels, chosen, episodes, cells):
    # episodes/cells are the shared accumulator lists
    """One episode; append rows for every interior PURIFY-legal decision."""
    obs = env.reset()
    for _ in range(env.max_steps):
        mask = env.get_action_mask()
        acts, _ = greedy(model, obs["x"], obs["edge_index"], mask, "cpu")
        for i in range(env.N):
            if i in (env.source, env.dest) or not mask[i, 2]:
                continue
            r = node_row(env, obs, mask, acts, i, ctx)
            if r is None:
                continue
            row, lab, act = r
            rows.append(row); labels.append(lab); chosen.append(act)
            episodes.append(ep_id); cells.append(ctx["cell"])
        obs, _, done, _ = env.step(acts)
        if done:
            break


def collect(a, model):
    """Roll out train + testbed cells; return matrix + label/aux arrays."""
    from rl_stack.env_wrapper import QRNEnv
    master = np.random.default_rng(a.seed)
    rows, labels, chosen, episodes, cells = [], [], [], [], []
    ep_id = 0

    def env_rng():
        return np.random.default_rng(int(master.integers(2**31)))

    common = dict(p_gen_std=0.15, p_swap_std=0.15, F0=1.0,
                  channel_loss=0.0, dt_seconds=0.0, topology="chain")

    # --- train cell (the _collect.collect training distribution) -------------
    if a.smoke:
        n_train = 20
    else:
        n_train = a.episodes_train
    for _ in range(n_train):
        n = 6 if a.smoke else int(master.choice(range(4, 13)))
        n_ch = int(master.choice((2, 3, 4)))
        p_gen = float(master.uniform(0.4, 0.9))
        p_swap = float(master.uniform(0.4, 0.9))
        cutoff = int(master.integers(10, 41))
        env = QRNEnv(n_repeaters=n, n_ch=n_ch, p_gen=p_gen, p_swap=p_swap,
                     cutoff=cutoff, max_steps=200, rng=env_rng(), **common)
        ctx = dict(p_gen=p_gen, p_swap=p_swap, cutoff=cutoff, cell="train")
        rollout(model, env, ctx, ep_id, rows, labels, chosen, episodes, cells)
        ep_id += 1

    # --- testbed cell (the paper's edge regime) ------------------------------
    if not a.smoke:
        for N in (13, 15):
            for _ in range(a.episodes_testbed):
                env = QRNEnv(n_repeaters=N, n_ch=4, p_gen=0.4, p_swap=0.8,
                             cutoff=30, max_steps=2000, rng=env_rng(), **common)
                ctx = dict(p_gen=0.4, p_swap=0.8, cutoff=30, cell="testbed")
                rollout(model, env, ctx, ep_id, rows, labels, chosen,
                        episodes, cells)
                ep_id += 1

    X = np.asarray(rows, np.float32)
    y = np.asarray(labels, np.int8)
    act = np.asarray(chosen, np.int8)
    ep = np.asarray(episodes, np.int32)
    cell = np.asarray(cells)
    thin = 1
    if X.shape[0] > 1_500_000:
        thin = int(np.ceil(X.shape[0] / 1_500_000))
        X, y, act, ep, cell = X[::thin], y[::thin], act[::thin], ep[::thin], cell[::thin]
    return X, y, act, ep, cell, thin


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def _rate(y, sel):
    n = int(sel.sum())
    return (float(y[sel].mean()) if n else None), n


def base_rates(X, y, cell):
    cs = X[:, COLUMNS.index("can_swap")]
    out = {"overall": _rate(y, np.ones(len(y), bool)),
           "can_swap=0": _rate(y, cs == 0), "can_swap=1": _rate(y, cs == 1),
           "cell=train": _rate(y, cell == "train"),
           "cell=testbed": _rate(y, cell == "testbed")}
    return out


def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None, 0
    a = roc_auc_score(y, score)
    return (a, 1) if a >= 0.5 else (1.0 - a, -1)


def univariate_auc(X, y):
    """Per-feature AUC split by can_swap; orientation flips AUC to >=0.5."""
    cs = X[:, COLUMNS.index("can_swap")]
    res = []
    for j, name in enumerate(COLUMNS):
        rec = {"feature": name}
        for tag, sel in (("cs0", cs == 0), ("cs1", cs == 1)):
            a, orient = _auc(y[sel], X[sel, j]) if int(sel.sum()) else (None, 0)
            rec[f"auc_{tag}"], rec[f"orient_{tag}"] = a, orient
        vals = [v for v in (rec["auc_cs0"], rec["auc_cs1"]) if v is not None]
        rec["auc_mean"] = float(np.mean(vals)) if vals else 0.5
        res.append(rec)
    res.sort(key=lambda r: r["auc_mean"], reverse=True)
    return res


def _episode_split(ep, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    rng.shuffle(uniq)
    cut = int(0.75 * len(uniq))
    tr = np.isin(ep, uniq[:cut])
    return tr, ~tr


def model_fits(X, y, ep, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import roc_auc_score
    tr, te = _episode_split(ep, seed)
    out = {}
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return {"note": "single-class split; fits skipped"}
    Xs = StandardScaler().fit(X[tr]).transform
    lr = LogisticRegression(max_iter=1000).fit(Xs(X[tr]), y[tr])
    out["logistic_test_auc"] = float(roc_auc_score(y[te], lr.predict_proba(Xs(X[te]))[:, 1]))
    out["trees"] = {}
    for d in (1, 2, 3):
        dt = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X[tr], y[tr])
        auc = float(roc_auc_score(y[te], dt.predict_proba(X[te])[:, 1]))
        out["trees"][str(d)] = {"test_auc": auc,
                                "rules": export_text(dt, feature_names=list(COLUMNS))}
    return out


def hypotheses(X, y):
    fs = X[:, COLUMNS.index("free_slots")]
    npc = X[:, COLUMNS.index("n_pairs_cand")]
    cs = X[:, COLUMNS.index("can_swap")]
    r = lambda sel: {"rate": _rate(y, sel)[0], "n": _rate(y, sel)[1]}
    return {
        "memory_pressure": {"free_slots==0": r(fs == 0), "free_slots>0": r(fs > 0)},
        "redundancy": {"n_pairs_cand>2": r(npc > 2), "n_pairs_cand==2": r(npc == 2)},
        "conflict": {"can_swap==1": r(cs == 1), "can_swap==0": r(cs == 0)},
    }


def heatmap_2d(X, y, feat_x, feat_y, nbin=12):
    jx, jy = COLUMNS.index(feat_x), COLUMNS.index(feat_y)
    xv, yv = X[:, jx], X[:, jy]
    xe = np.linspace(xv.min(), xv.max() + 1e-9, nbin + 1)
    ye = np.linspace(yv.min(), yv.max() + 1e-9, nbin + 1)
    xi = np.clip(np.digitize(xv, xe) - 1, 0, nbin - 1)
    yi = np.clip(np.digitize(yv, ye) - 1, 0, nbin - 1)
    grid = np.full((nbin, nbin), np.nan)
    cnt = np.zeros((nbin, nbin), int)
    for bx in range(nbin):
        for by in range(nbin):
            sel = (xi == bx) & (yi == by)
            n = int(sel.sum())
            cnt[by, bx] = n
            if n >= 5:
                grid[by, bx] = float(y[sel].mean())
    return {"feat_x": feat_x, "feat_y": feat_y,
            "x_edges": xe.tolist(), "y_edges": ye.tolist(),
            "grid": [[None if np.isnan(v) else round(float(v), 4) for v in row]
                     for row in grid],
            "counts": cnt.tolist()}


# ----------------------------------------------------------------------------
# compute / plot / main
# ----------------------------------------------------------------------------
def run_compute(a, json_path, npz_path):
    from rl_stack.model import load_qnet
    model = load_qnet(a.ckpt, "cpu")
    model.eval()
    X, y, act, ep, cell, thin = collect(a, model)
    br = base_rates(X, y, cell)
    print(f"rows={X.shape[0]}  episodes={len(np.unique(ep))}  thin={thin}")
    print(f"P(PURIFY|can_purify): overall={br['overall'][0]}  "
          f"can_swap=0={br['can_swap=0'][0]}  can_swap=1={br['can_swap=1'][0]}")
    np.savez_compressed(npz_path, X=X, columns=np.asarray(COLUMNS),
                        label=y, chosen_action=act, episode=ep, cell=cell)
    data = {"ckpt": a.ckpt, "seed": a.seed, "smoke": a.smoke, "thin": thin,
            "n_rows": int(X.shape[0]), "n_episodes": int(len(np.unique(ep))),
            "columns": list(COLUMNS), "base_rates": br}
    if not a.smoke:
        uni = univariate_auc(X, y)
        data["univariate_auc"] = uni
        data["model_fits"] = model_fits(X, y, ep, a.seed)
        data["hypotheses"] = hypotheses(X, y)
        top = [r["feature"] for r in uni][:2]
        data["plot"] = {
            "auc_bars": uni[:10],
            "heatmap": heatmap_2d(X, y, top[0], top[1])}
        print("\ntop univariate AUCs (mean over can_swap splits):")
        for r in uni[:8]:
            print(f"  {r['feature']:16s} cs0={r['auc_cs0']}  cs1={r['auc_cs1']}")
        print("\nlogistic test AUC:", data["model_fits"].get("logistic_test_auc"))
        for d, t in data["model_fits"].get("trees", {}).items():
            print(f"tree depth {d}: test AUC {t['test_auc']:.3f}")
        print("\nhypotheses:", json.dumps(data["hypotheses"], indent=1))
    json.dump(data, open(json_path, "w"), indent=1)
    print(f"saved -> {json_path}")
    return data


def run_plot(data, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = data.get("plot")
    if p is None:
        print("no plot data in JSON (smoke run); nothing to render")
        return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    bars = p["auc_bars"]
    names = [b["feature"] for b in bars][::-1]
    cs0 = [b["auc_cs0"] or 0.5 for b in bars][::-1]
    cs1 = [b["auc_cs1"] or 0.5 for b in bars][::-1]
    yy = np.arange(len(names))
    ax0.barh(yy - 0.2, cs0, height=0.4, color="#d98c1f", label="can_swap=0")
    ax0.barh(yy + 0.2, cs1, height=0.4, color="#1f63d6", label="can_swap=1")
    ax0.set_yticks(yy); ax0.set_yticklabels(names)
    ax0.axvline(0.5, color="0.4", lw=0.8, ls="--")
    ax0.set_xlim(0.5, max(1.0, *cs0, *cs1))
    ax0.set_xlabel("univariate AUC vs P(PURIFY)")
    ax0.set_title("(A) top-10 feature AUCs"); ax0.legend(fontsize=8)

    h = p["heatmap"]
    grid = np.array([[np.nan if v is None else v for v in row] for row in h["grid"]])
    cmap = plt.get_cmap("Greens").copy(); cmap.set_bad("0.88")
    pm = ax1.pcolormesh(h["x_edges"], h["y_edges"], np.ma.masked_invalid(grid),
                        cmap=cmap, vmin=0.0, vmax=1.0, edgecolors="0.6", linewidth=0.4)
    fig.colorbar(pm, ax=ax1, fraction=0.046, pad=0.03, label="P(PURIFY)")
    ax1.set_xlabel(h["feat_x"]); ax1.set_ylabel(h["feat_y"])
    ax1.set_title(f"(B) P(PURIFY) over {h['feat_x']} x {h['feat_y']}")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved -> {stem}.pdf")


def main():
    a = parse_args()
    out = a.out_dir or os.path.join("results", "probes",
                                    os.path.basename(os.path.dirname(a.ckpt)))
    os.makedirs(out, exist_ok=True)
    json_path = os.path.join(out, "purify_map.json")
    npz_path = os.path.join(out, "purify_map.npz")
    stem = os.path.join(out, "purify_map")
    if a.plot:
        run_plot(json.load(open(json_path)), stem)
        return
    data = run_compute(a, json_path, npz_path)
    if not a.smoke:
        run_plot(data, stem)


if __name__ == "__main__":
    main()
