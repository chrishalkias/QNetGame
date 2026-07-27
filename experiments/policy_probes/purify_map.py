"""
--------------------------------------------------------------------------------
Collect the data on when the trained agent chooses PURIFY among its purify
opportunities.

For every micro-step decision where PURIFY is legal at env.active_node, roll out
the GREEDY full agent (serialized sweep) and record (a) whether it purified, (b)
the 7 interpretability obs features, a 1-hop neighbour-mean context, and
hand-built engine features about the candidate BBPSSW pair, (c) episode/cell
context. The rows land in purify_map.npz, keyed by COLUMNS.

This module is a COLLECTOR only. The model-search half it used to carry (base
rates, univariate AUCs, a logistic ceiling, shallow trees, hypothesis checks and
a two-panel figure) was removed once the question it served closed: a fitted
state-conditioned purify probability q(state) does NOT beat a rate-matched
constant q, so the purify-selectivity story is one-parameter, not
state-dependent (see experiments/q_heuristic/).

COLUMNS is a PERSISTED SCHEMA, not a label list: the npz and every fitted
artifact downstream are keyed by these names, so changing the column set is a
schema migration and invalidates fits on disk.

  PYTHONPATH=src:. python experiments/policy_probes/purify_map.py --ckpt <path>
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/sota/policy.pth")
    ap.add_argument("--episodes_train", type=int, default=300)
    ap.add_argument("--episodes_testbed", type=int, default=150,
                    help="per testbed N (N in {13,15})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None,
                    help="default: results/probes/<ckpt parent dir name>/")
    ap.add_argument("--smoke", action="store_true",
                    help="20 train episodes at N=6 only, no testbed cells")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# feature schema
# ----------------------------------------------------------------------------
from experiments.policy_probes._collect import FEATURE_NAMES, greedy  # noqa: E402
from simulator.repeater import NO_PARTNER, QUBIT_OCCUPIED, werner_to_fidelity  # noqa: E402

OBS_COLS = [f"obs_{n}" for n in FEATURE_NAMES]
NBR_COLS = [f"nbr_{n}" for n in FEATURE_NAMES]
ENG_COLS = ["free_slots", "n_avail", "n_pairs_cand", "n_spare",
            "F_hi", "F_lo", "age_frac_hi", "age_frac_lo", "mean_age_frac"]
CTX_COLS = ["can_swap", "pos_frac", "N", "n_ch", "p_gen", "p_swap",
            "cutoff", "t_frac"]
COLUMNS = OBS_COLS + NBR_COLS + ENG_COLS + CTX_COLS


def candidate_partner(rep):
    """Pick the SINGLE partner this probe reports on: among partners with >=2
    available qubits, the one with the most such qubits; ties break to the
    lowest partner id (np.unique returns sorted ids and max() keeps the first
    maximal). Returns partner id or None.

    NOTE this is a probe-side simplification, not a replication: the real
    QRNEnv._exec_purify cascades over EVERY valid partner, not just the
    largest. The probe records one row per decision, so it names the dominant
    partner; do not read these rows as "the only partner purified"."""
    avail = rep.status == QUBIT_OCCUPIED
    if int(avail.sum()) < 2:
        return None
    partners = rep.partner_repeater[avail]
    unique, counts = np.unique(partners[partners != NO_PARTNER], return_counts=True)
    valid = [(int(p), c) for p, c in zip(unique, counts) if c >= 2]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])[0]


def node_row(env, obs, mask, acts, i, ctx):
    """Build one feature row (list matching COLUMNS) + label + chosen_action for
    interior node i, which is known to have PURIFY masked-legal. Returns None if
    the candidate partner cannot be resolved (defensive; should not happen)."""
    rep = env.net.node(i)
    best_nb = candidate_partner(rep)
    if best_nb is None:
        return None
    n_ch = rep.n_ch
    occ = rep.status == QUBIT_OCCUPIED
    avail = occ
    cand = avail & (rep.partner_repeater == best_nb)
    F = werner_to_fidelity(rep.werner_param[cand])
    lc = np.maximum(rep.link_cutoff[cand].astype(np.float64), 1.0)
    age_frac = rep.age[cand].astype(np.float64) / lc
    hi, lo = int(np.argmax(F)), int(np.argmin(F))
    lc_av = np.maximum(rep.link_cutoff[avail].astype(np.float64), 1.0)
    mean_age_frac = float((rep.age[avail].astype(np.float64) / lc_av).mean()) \
        if bool(avail.any()) else 0.0

    ei = obs["edge_index"]
    nbrs = ei[1][ei[0] == i]
    nbr_mean = (obs["x"][nbrs].mean(axis=0) if nbrs.size
                else np.zeros(len(FEATURE_NAMES), np.float64))

    n_pairs_cand = int(cand.sum())
    eng = [int(occ.size) - int(occ.sum()), int(avail.sum()), n_pairs_cand,
           max(0, n_pairs_cand - 2), float(F[hi]), float(F[lo]),
           float(age_frac[hi]), float(age_frac[lo]), mean_age_frac]
    ctx_row = [float(mask[i, 1]), i / (env.N - 1), env.N, n_ch,
               ctx["p_gen"], ctx["p_swap"], ctx["cutoff"], env.steps / env.max_steps]
    # obs/nbr contribute only the 7 interpretability features (FEATURE_NAMES,
    # which now includes relative_position); col 7 (is_active) is the only
    # sweep scaffolding left and is excluded here so the row width matches
    # COLUMNS. is_active is constant 1.0 anyway.
    nf = len(FEATURE_NAMES)
    row = list(obs["x"][i][:nf]) + list(nbr_mean[:nf]) + eng + ctx_row
    return row, int(acts[i] == 2), int(acts[i])


def rollout(model, env, ctx, ep_id, rows, labels, chosen, episodes, cells):
    # episodes/cells are the shared accumulator lists
    """One episode (serialized sweep); append a row for every micro-step whose
    active node has PURIFY legal."""
    obs = env.reset()
    while not env.done:
        i = env.active_node
        mask = env.get_action_mask()
        acts, _ = greedy(model, obs["x"], obs["edge_index"], mask, "cpu")
        if mask[i, 2]:
            r = node_row(env, obs, mask, acts, i, ctx)
            if r is not None:
                row, lab, act = r
                rows.append(row); labels.append(lab); chosen.append(act)
                episodes.append(ep_id); cells.append(ctx["cell"])
        obs, _, done, _ = env.step(int(acts[i]))
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
                  channel_loss=0.0)

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
# compute / main
# ----------------------------------------------------------------------------
def run_compute(a, json_path, npz_path):
    from rl_stack.model import load_qnet
    model = load_qnet(a.ckpt, "cpu")
    model.eval()
    X, y, act, ep, cell, thin = collect(a, model)
    print(f"rows={X.shape[0]}  cols={X.shape[1]}  "
          f"episodes={len(np.unique(ep))}  thin={thin}")
    np.savez_compressed(npz_path, X=X, columns=np.asarray(COLUMNS),
                        label=y, chosen_action=act, episode=ep, cell=cell)
    data = {"ckpt": a.ckpt, "seed": a.seed, "smoke": a.smoke, "thin": thin,
            "n_rows": int(X.shape[0]), "n_episodes": int(len(np.unique(ep))),
            "columns": list(COLUMNS)}
    json.dump(data, open(json_path, "w"), indent=1)
    print(f"saved -> {npz_path}")
    print(f"saved -> {json_path}")
    return data


def main():
    a = parse_args()
    out = a.out_dir or os.path.join("results", "probes",
                                    os.path.basename(os.path.dirname(a.ckpt)))
    os.makedirs(out, exist_ok=True)
    json_path = os.path.join(out, "purify_map.json")
    npz_path = os.path.join(out, "purify_map.npz")
    run_compute(a, json_path, npz_path)


if __name__ == "__main__":
    main()
