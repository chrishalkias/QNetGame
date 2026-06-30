"""Aggregated, multi-node, multi-size interpretability probe for the GraphSAGE
Q-network used in the quantum-repeater RL paper.

Motivation
----------
The paper previously claimed, from a single node (n=1), that the learned node
representation is "compact and more compressed at larger network size, which
explains generalization". That is (a) under-powered (one node) and (b) an
overstated causal claim. This script produces *aggregated* evidence pooled over
ALL interior nodes and many states/timesteps, for several chain sizes, so the
claim can be restated rigorously and softened to "is consistent with".

For each chain size N in {5, 8, 10, 12, 15} we:
  1. Roll out the greedy (eps=0) policy in QRNEnv (chain topology) and collect a
     large pool (>= 5000) of conv3 node-embeddings (the 64-dim final
     message-passing representation, i.e. the input to .head), pooled over all
     interior nodes and many timesteps/states.
  2. Run PCA on the standardized pool and report:
       - explained-variance ratio of PC1,
       - cumulative variance at 3 PCs,
       - number of PCs needed for >= 90% variance.
  3. Fit a cross-validated linear (ridge) probe from the 64-dim embedding to
     (a) node mean_fidelity (feature col 1) and (b) frac_occupied (feature col
     0); report CV R^2.
  4. Compute linear CKA between the N=5 embedding matrix and each larger-N
     embedding matrix (subsampled to equal counts).

Outputs:
  docs/paper/figs/interpretability_aggregate.png   (2-panel figure)
  docs/paper/figs/interpretability_stats.txt        (human-readable table)

Run:
  PYTHONPATH=. .venv/bin/python diagnostics/policy_probes/interpretability_aggregate.py
"""

from __future__ import annotations

import os
import argparse
import numpy as np

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from rl_stack.model import QNetwork, load_qnet
from rl_stack.env_wrapper import QRNEnv, N_ACTIONS, NOOP
from rl_stack.agent import _obs_to_data

                                    
                                                                           
#  ▄▄▄▄▄▄▄               ▄▄                       ▄▄▄                     ▄▄ 
# ███▀▀▀▀▀              ██  ▀▀             ▄      ███                     ██ 
# ███      ▄███▄ ████▄ ▀██▀ ██  ▄████      █      ███      ▄███▄  ▀▀█▄ ▄████ 
# ███      ██ ██ ██ ██  ██  ██  ██ ██   ▀▀▀█▀▀▀   ███      ██ ██ ▄█▀██ ██ ██ 
# ▀███████ ▀███▀ ██ ██  ██  ██▄ ▀████      █      ████████ ▀███▀ ▀█▄██ ▀████ 
#                                  ██                                        
#                                ▀▀▀                                         

SIZES = [5, 8, 10, 12, 15]
TARGET_EMB_PER_N = 5000          # minimum pooled interior-node embeddings per N
MAX_STEPS = 50
PARAM_POINT = dict(              # representative parameter point
    p_gen=0.5,
    p_swap=0.7,
    cutoff=15,
    dt_seconds=0,
    topology="chain",
    n_ch=4,
    F0=0.95,
    channel_loss=0.02,
)
SEED = 12345
HIDDEN = 64
FID_COL = 1                      # mean_fidelity feature index
OCC_COL = 0                      # frac_occupied feature index
VAR_THRESHOLD = 0.90             # variance target for "#PCs for >=90%"
N_PCS_CUM = 3                    # cumulative variance reported at this many PCs

DEFAULT_CKPT = "checkpoints/cluster/cluster_004/policy.pth"
FIG_DIR = "docs/paper/figs"
FIG_PATH = os.path.join(FIG_DIR, "interpretability_aggregate.png")
STATS_PATH = os.path.join(FIG_DIR, "interpretability_stats.txt")


# --------------------------------------------------------------------------- #
# conv3 embedding extraction (forward hook)
# --------------------------------------------------------------------------- #

class Conv3Hook:
    """Capture the post-conv3 (pre-ReLU) output via a forward hook.

    The QNetwork forward applies F.relu(self.conv3(x, ei)) and then the head.
    The "64-dim final message-passing embedding / input to .head" is therefore
    the ReLU of the conv3 output. We capture the conv3 module output and apply
    the same ReLU so the embedding matches exactly what the head receives.
    """

    def __init__(self, model: QNetwork):
        self.value = None
        self._handle = model.conv3.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # output is the raw conv3 message-passing result (pre-ReLU in forward).
        self.value = torch.relu(output).detach().cpu().numpy()

    def remove(self):
        self._handle.remove()


                                                   
#  ▄▄▄▄▄▄▄       ▄▄ ▄▄                               
# ███▀▀▀▀▀       ██ ██              ██               
# ███      ▄███▄ ██ ██ ▄█▀█▄ ▄████ ▀██▀▀ ▄███▄ ████▄ 
# ███      ██ ██ ██ ██ ██▄█▀ ██     ██   ██ ██ ██ ▀▀ 
# ▀███████ ▀███▀ ██ ██ ▀█▄▄▄ ▀████  ██   ▀███▀ ██    
                                                                                         
# --------------------------------------------------------------------------- #
# Roll out greedy policy, collect interior-node conv3 embeddings + targets
# --------------------------------------------------------------------------- #
def collect_embeddings(model, N, target_count, device, rng_seed):
    """Roll out the greedy policy on N-node chains, pooling conv3 embeddings
    over ALL interior nodes and many timesteps. Returns:
        emb   (M, 64)  conv3 embeddings (ReLU'd)
        fid   (M,)     per-node mean_fidelity (feature col 1)
        occ   (M,)     per-node frac_occupied (feature col 0)
    """
    hook = Conv3Hook(model)
    embs, fids, occs = [], [], []
    seed_rng = np.random.default_rng(rng_seed)
    ep = 0
    try:
        while sum(e.shape[0] for e in embs) < target_count:
            env = QRNEnv(
                n_repeaters=N,
                max_steps=MAX_STEPS,
                rng=np.random.default_rng(int(seed_rng.integers(0, 2**32))),
                **PARAM_POINT,
            )
            obs = env.reset()

            # Interior nodes only (exclude source=0 and dest=N-1).
            interior = [i for i in range(env.N) if not env.is_target(i)]

            for _ in range(MAX_STEPS):
                mask = env.get_action_mask()
                data = _obs_to_data(obs, device)
                with torch.no_grad():
                    q = model(data)            # triggers the conv3 hook
                emb = hook.value               # (N, 64)
                feats = obs["x"]               # (N, 8)

                # Record interior nodes for this state.
                for i in interior:
                    embs.append(emb[i:i + 1])
                    fids.append(feats[i, FID_COL])
                    occs.append(feats[i, OCC_COL])

                # Greedy action selection (eps=0), masking invalid actions.
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                q_masked = q.clone()
                q_masked[~mask_t] = -float("inf")
                actions = q_masked.argmax(dim=1).cpu().numpy().astype(np.int32)

                obs, _, done, _ = env.step(actions)
                if done:
                    break
            ep += 1
            # Safety cap to avoid pathological infinite loops.
            if ep > 100000:
                break
    finally:
        hook.remove()

    emb = np.concatenate(embs, axis=0).astype(np.float64)
    fid = np.asarray(fids, dtype=np.float64)
    occ = np.asarray(occs, dtype=np.float64)
    return emb, fid, occ, ep


                                              
#   ▄▄▄▄               ▄▄                       
# ▄██▀▀██▄             ██             ▀▀        
# ███  ███ ████▄  ▀▀█▄ ██ ██ ██ ▄█▀▀▀ ██  ▄█▀▀▀ 
# ███▀▀███ ██ ██ ▄█▀██ ██ ██▄██ ▀███▄ ██  ▀███▄ 
# ███  ███ ██ ██ ▀█▄██ ██  ▀██▀ ▄▄▄█▀ ██▄ ▄▄▄█▀ 
#                           ██                  
#                         ▀▀▀                   
def pca_stats(emb):
    """PCA on standardized embeddings. Returns (pc1_var, cum3, n_pcs_90)."""
    scaler = StandardScaler()
    z = scaler.fit_transform(emb)
    pca = PCA()
    pca.fit(z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pc1 = float(evr[0])
    cum3 = float(cum[min(N_PCS_CUM, len(cum)) - 1])
    n90 = int(np.searchsorted(cum, VAR_THRESHOLD) + 1)
    return pc1, cum3, n90, evr


def probe_r2(emb, target, n_splits=5, seed=0):
    """Cross-validated R^2 of a ridge regression from emb -> target.

    Embeddings are standardized inside the CV via a single global scaler
    (acceptable here: features are not the prediction target and we report
    mean CV R^2 as a representational-encoding statistic, not a deployment
    score). Returns mean CV R^2.
    """
    # Guard: degenerate target (no variance) -> R^2 undefined; return nan.
    if np.std(target) < 1e-9:
        return float("nan")
    z = StandardScaler().fit_transform(emb)
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, z, target, cv=n_splits, scoring="r2")
    return float(np.mean(scores))


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment between two embedding matrices with the
    same number of rows. Both are mean-centered over samples.
    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    xty = Xc.T @ Yc
    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    num = np.sum(xty ** 2)
    den = np.sqrt(np.sum(xtx ** 2)) * np.sqrt(np.sum(yty ** 2))
    if den < 1e-12:
        return float("nan")
    return float(num / den)


                             
# ▄▄▄      ▄▄▄                 
# ████▄  ▄████       ▀▀        
# ███▀████▀███  ▀▀█▄ ██  ████▄ 
# ███  ▀▀  ███ ▄█▀██ ██  ██ ██ 
# ███      ███ ▀█▄██ ██▄ ██ ██                           
                             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target", type=int, default=TARGET_EMB_PER_N)
    parser.add_argument("--save_dir", default=None,
                        help="output dir (default: <ckpt dir>/diagnostics)")
    args = parser.parse_args()

    global FIG_DIR, FIG_PATH, STATS_PATH
    FIG_DIR = args.save_dir or os.path.join(os.path.dirname(args.ckpt), "diagnostics")
    FIG_PATH = os.path.join(FIG_DIR, "interpretability_aggregate.png")
    STATS_PATH = os.path.join(FIG_DIR, "interpretability_stats.txt")
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading model from {args.ckpt}")
    model = load_qnet(args.ckpt, args.device)

    results = {}
    embeddings = {}
    for k, N in enumerate(SIZES):
        print(f"\n=== N={N} : collecting >= {args.target} interior-node "
              f"embeddings ===")
        emb, fid, occ, n_eps = collect_embeddings(
            model, N, args.target, args.device, rng_seed=SEED + 1000 * k)
        print(f"  collected {emb.shape[0]} embeddings over {n_eps} episodes "
              f"(dim={emb.shape[1]})")

        pc1, cum3, n90, evr = pca_stats(emb)
        r2_fid = probe_r2(emb, fid, seed=SEED)
        r2_occ = probe_r2(emb, occ, seed=SEED)

        results[N] = dict(
            n_emb=int(emb.shape[0]), n_eps=int(n_eps),
            pc1=pc1, cum3=cum3, n90=n90,
            r2_fid=r2_fid, r2_occ=r2_occ,
            fid_std=float(np.std(fid)), occ_std=float(np.std(occ)),
        )
        embeddings[N] = emb
        print(f"  PC1={pc1*100:.1f}%  cum3={cum3*100:.1f}%  #PCs@90%={n90}  "
              f"R2_fid={r2_fid:.3f}  R2_occ={r2_occ:.3f}")

    # CKA to N=5 (subsample to equal counts).
    base_N = SIZES[0]
    base_emb = embeddings[base_N]
    cka_rng = np.random.default_rng(SEED)
    for N in SIZES:
        emb = embeddings[N]
        m = min(base_emb.shape[0], emb.shape[0])
        idx_base = cka_rng.choice(base_emb.shape[0], size=m, replace=False)
        idx_n = cka_rng.choice(emb.shape[0], size=m, replace=False)
        cka = linear_cka(base_emb[idx_base], emb[idx_n])
        results[N]["cka_n5"] = cka
        print(f"  CKA(N={base_N}, N={N}) = {cka:.3f}  (m={m})")

    write_stats(results, args.ckpt)
    make_figure(results)
    print(f"\nSaved figure -> {FIG_PATH}")
    print(f"Saved stats  -> {STATS_PATH}")


def write_stats(results, ckpt):
    lines = []
    lines.append("Aggregated multi-node, multi-size interpretability statistics")
    lines.append("=" * 70)
    lines.append(f"Checkpoint : {ckpt}")
    lines.append(f"Param point: p_gen={PARAM_POINT['p_gen']}, "
                 f"p_swap={PARAM_POINT['p_swap']}, "
                 f"cutoff={PARAM_POINT['cutoff']}, "
                 f"dt_seconds={PARAM_POINT['dt_seconds']}, "
                 f"topology={PARAM_POINT['topology']}")
    lines.append(f"Embedding  : conv3 output (ReLU), dim={HIDDEN}, "
                 f"pooled over ALL interior nodes and all timesteps")
    lines.append(f"PCA        : standardized; PC1 var, cum var @ {N_PCS_CUM} PCs, "
                 f"#PCs for >= {int(VAR_THRESHOLD*100)}% var")
    lines.append(f"Probe      : 5-fold CV ridge (alpha=1.0) R^2, "
                 f"emb -> mean_fidelity / frac_occupied")
    lines.append(f"CKA        : linear CKA vs N={SIZES[0]} (subsampled to equal n)")
    lines.append("")
    header = (f"{'N':>4} | {'n_emb':>7} | {'n_eps':>6} | {'PC1%':>6} | "
              f"{'cum3%':>6} | {'#PCs@90':>7} | {'R2_fid':>7} | "
              f"{'R2_occ':>7} | {'CKA_N5':>7} | {'fid_std':>7} | {'occ_std':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for N in SIZES:
        r = results[N]
        lines.append(
            f"{N:>4} | {r['n_emb']:>7} | {r['n_eps']:>6} | "
            f"{r['pc1']*100:>6.1f} | {r['cum3']*100:>6.1f} | {r['n90']:>7} | "
            f"{r['r2_fid']:>7.3f} | {r['r2_occ']:>7.3f} | "
            f"{r.get('cka_n5', float('nan')):>7.3f} | "
            f"{r['fid_std']:>7.3f} | {r['occ_std']:>7.3f}")
    lines.append("")
    lines.append("Column key:")
    lines.append("  PC1%      explained-variance ratio of the 1st principal component")
    lines.append(f"  cum3%     cumulative explained variance at {N_PCS_CUM} PCs")
    lines.append(f"  #PCs@90   number of PCs needed for >= {int(VAR_THRESHOLD*100)}% variance")
    lines.append("  R2_fid    CV R^2 of linear probe predicting node mean_fidelity")
    lines.append("  R2_occ    CV R^2 of linear probe predicting node frac_occupied")
    lines.append(f"  CKA_N5    linear CKA of the N-embedding vs the N={SIZES[0]} embedding")
    lines.append("  fid_std   std of mean_fidelity target (R^2 is nan if ~0)")
    lines.append("  occ_std   std of frac_occupied target (R^2 is nan if ~0)")
    with open(STATS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

                         
# ▄▄▄▄▄▄▄   ▄▄             
# ███▀▀███▄ ██        ██   
# ███▄▄███▀ ██ ▄███▄ ▀██▀▀ 
# ███▀▀▀▀   ██ ██ ██  ██   
# ███       ██ ▀███▀  ██   
                         
                         
def make_figure(results):
    Ns = SIZES
    pc1 = [results[N]["pc1"] * 100 for N in Ns]
    n90 = [results[N]["n90"] for N in Ns]
    r2_fid = [results[N]["r2_fid"] for N in Ns]
    r2_occ = [results[N]["r2_occ"] for N in Ns]
    cka = [results[N].get("cka_n5", np.nan) for N in Ns]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: PC1 variance % (left axis) + #PCs for 90% (right axis).
    color1 = "#1f77b4"
    axL.plot(Ns, pc1, "o-", color=color1, label="PC1 variance %")
    axL.set_xlabel("Chain size N")
    axL.set_ylabel("PC1 explained variance (%)", color=color1)
    axL.tick_params(axis="y", labelcolor=color1)
    axL.set_xticks(Ns)
    axL.grid(alpha=0.3)

    axL2 = axL.twinx()
    color2 = "#d62728"
    axL2.plot(Ns, n90, "s--", color=color2, label="#PCs for >=90% var")
    axL2.set_ylabel("# PCs for >= 90% variance", color=color2)
    axL2.tick_params(axis="y", labelcolor=color2)
    n90_max = max(n90) if n90 else 1
    axL2.set_ylim(0, n90_max + 2)

    axL.set_title("Dimensionality of conv3 representation vs N\n"
                  "(pooled over all interior nodes & timesteps)")
    lines1, labels1 = axL.get_legend_handles_labels()
    lines2, labels2 = axL2.get_legend_handles_labels()
    axL.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    # Right panel: probe R^2 (fid, occ) + CKA-to-N5.
    axR.plot(Ns, r2_fid, "o-", color="#2ca02c", label=r"$R^2$ probe: mean_fidelity")
    axR.plot(Ns, r2_occ, "^-", color="#9467bd", label=r"$R^2$ probe: frac_occupied")
    axR.plot(Ns, cka, "d:", color="#ff7f0e", label="CKA vs N=5")
    axR.set_xlabel("Chain size N")
    axR.set_ylabel(r"$R^2$  /  CKA")
    axR.set_xticks(Ns)
    axR.set_ylim(-0.05, 1.05)
    axR.grid(alpha=0.3)
    axR.axhline(0.0, color="grey", ls=":", lw=0.6)
    axR.set_title("Linear decodability & cross-size similarity vs N")
    axR.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
