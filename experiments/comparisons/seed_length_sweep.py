"""Delivery-time-vs-N sweep over training length, with seed error bars.

Reads every delivery_vs_N JSON in a directory named ds_ep<K>[_s<seed>].json,
groups them by training length <K> (e.g. 5k, 15k), and plots one line per length
= MEAN T_agent over that length's seeds, with a +/-1 std band. This disentangles
seed variance from training length (the single-seed-per-length sweep conflated
them). Heuristic lines (swap-ASAP, purify-then-swap) are ckpt-independent, so
they are read from whichever JSON carries them and drawn once.

  PYTHONPATH=. python experiments/comparisons/seed_length_sweep.py \
      --dir results/comparisons/delivery_vs_N_different_seeds \
      --fig results/figures/delivery_vs_N_seed_length_sweep
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/comparisons/delivery_vs_N_different_seeds")
    ap.add_argument("--n_train_max", type=int, default=12)
    ap.add_argument("--no_swap_asap", action="store_true",
                    help="drop the swap-ASAP line (it dwarfs the agents and "
                         "compresses the y-axis)")
    ap.add_argument("--logy", action="store_true",
                    help="log-scale the y-axis to resolve the closely-spaced "
                         "agent-length curves")
    ap.add_argument("--logx", action="store_true",
                    help="log-scale the x-axis (chain size N)")
    ap.add_argument("--delta", action="store_true",
                    help="plot %% delivery-time reduction vs purify-then-swap "
                         "(per length, mean±std over seeds) instead of raw T")
    ap.add_argument("--fig", default="results/figures/delivery_vs_N_seed_length_sweep")
    return ap.parse_args()


# ds_ep15k.json / ds_ep15k_s7.json -> length "15k" (episodes 15000), optional seed
_PAT = re.compile(r"ds_ep(\d+)k(?:_s(\d+))?\.json$")


def _episodes(k: str) -> int:
    return int(k[:-1]) * 1000   # "15k" -> 15000


def load_groups(d):
    """{length_label: list of {N: row}} grouped over seeds, sorted by episodes."""
    groups: dict[str, list] = {}
    heur = None
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = _PAT.search(os.path.basename(path))
        if not m:
            continue
        label = f"{m.group(1)}k"
        rows = json.load(open(path))
        groups.setdefault(label, []).append({r["N"]: r for r in rows})
        if heur is None and rows and "T_swap_asap" in rows[0]:
            heur = {r["N"]: r for r in rows}
    ordered = sorted(groups, key=_episodes)
    return ordered, groups, heur


def main():
    a = parse_args()
    labels, groups, heur = load_groups(a.dir)
    if not labels:
        raise SystemExit(f"no ds_ep*.json found in {a.dir}")
    Ns = sorted(groups[labels[0]][0].keys())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)

    if a.delta and heur is None:
        raise SystemExit("--delta needs a JSON carrying T_purify_swap")

    for i, lab in enumerate(labels):
        seeds = groups[lab]
        # stack T_agent across seeds, aligned by N (skip N missing in any seed)
        Ns_common = [N for N in Ns if all(N in s for s in seeds)]
        M = np.array([[s[N]["T_agent"] for N in Ns_common] for s in seeds])  # (n_seed, n_N)
        if a.delta:
            # per-seed % reduction vs purify-then-swap (positive => agent faster)
            pur = np.array([heur[N]["T_purify_swap"] for N in Ns_common])
            M = 100.0 * (pur - M) / pur
        mean, std = M.mean(0), M.std(0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1])
        color = cmap(i / max(len(labels) - 1, 1))
        ax.plot(Ns_common, mean, marker="o", lw=1.8, ms=4, color=color,
                label=f"{lab} (n={M.shape[0]})")
        if M.shape[0] > 1:
            ax.fill_between(Ns_common, mean - std, mean + std, color=color, alpha=0.18, lw=0)

    if a.delta:
        ax.axhline(0, color="tab:red", ls="--", lw=1.4, label="purify-then-swap")
    elif heur is not None:
        Nh = sorted(heur)
        if not a.no_swap_asap:
            ax.plot(Nh, [heur[N]["T_swap_asap"] for N in Nh], ls="--", lw=1.4,
                    color="tab:orange", label="swap-ASAP")
        ax.plot(Nh, [heur[N]["T_purify_swap"] for N in Nh], ls="--", lw=1.4,
                color="tab:red", label="purify-then-swap")

    if a.logy:
        ax.set_yscale("log")
    if a.logx:
        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())   # plain N labels, not 10^x
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvline(a.n_train_max, color="grey", ls=":", lw=1.2)
    ax.text(a.n_train_max + 0.06, ax.get_ylim()[1], " out-of-distribution →",
            color="grey", fontsize=8, va="top")
    ax.set_xlabel("chain size $N$")
    if a.delta:
        ax.set_ylabel("delivery-time reduction vs purify-then-swap (%)")
        ax.set_title("Agent vs purify-then-swap — training-length sweep (mean ± std over seeds)\n"
                     r"($p_\mathrm{gen}=0.4$, $p_\mathrm{swap}=0.8$, $n_\mathrm{ch}=4$, cutoff $=20$)")
    else:
        ax.set_ylabel("delivery time $T$ (avg steps to termination)")
        ax.set_title("Delivery time vs chain size — training-length sweep (mean ± std over seeds)\n"
                     r"($p_\mathrm{gen}=0.4$, $p_\mathrm{swap}=0.8$, $n_\mathrm{ch}=4$, cutoff $=20$)")
    ax.set_xticks(Ns); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8, ncol=2)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(f"{a.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {a.fig}.pdf  ({len(labels)} lengths, "
          f"{ {l: len(groups[l]) for l in labels} } seeds)")


if __name__ == "__main__":
    main()
