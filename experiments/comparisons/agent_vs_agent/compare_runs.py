"""
--------------------------------------------------------------------------------
Overlay agent delivery-time (T vs N) curves from several runs + the heuristics.

  PYTHONPATH=src:. python experiments/comparisons/agent_vs_agent/compare_runs.py \
      --runs 3k=results/comparisons/delivery_vs_N.json \
             15k=results/comparisons/delivery_vs_N_omni_nopen_15k.json \
             35k=results/comparisons/delivery_vs_N_omni_nopen_35k.json
--------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="label=path.json pairs")
    p.add_argument("--n_train_max", type=int, default=12)
    p.add_argument("--fig", default="results/figures/delivery_vs_N_compare")
    return p.parse_args()


def main():
    a = parse_args()
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.runs]
    data = [(lab, {r["N"]: r for r in json.load(open(p))}) for lab, p in runs]
    Ns = sorted(data[0][1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for i, (lab, d) in enumerate(data):
        ax.plot(Ns, [d[N]["T_agent"] for N in Ns], marker="o", lw=1.8, ms=4,
                color=cmap(i / max(len(data) - 1, 1)), label=f"agent {lab}")
    d0 = data[0][1]
    ax.plot(Ns, [d0[N]["T_swap_asap"] for N in Ns], ls="--", lw=1.4,
            color="tab:orange", label="swap-ASAP")
    ax.plot(Ns, [d0[N]["T_purify_swap"] for N in Ns], ls="--", lw=1.4,
            color="tab:red", label="purify-then-swap")
    ax.axvline(a.n_train_max, color="grey", ls=":", lw=1.2)
    ax.text(a.n_train_max + 0.06, ax.get_ylim()[1], " out-of-distribution →",
            color="grey", fontsize=8, va="top")
    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title("Delivery time vs chain size — training-length comparison\n"
                 r"($p_\mathrm{gen}=0.4$, $p_\mathrm{swap}=0.8$, $n_\mathrm{ch}=4$, cutoff $=20$)")
    ax.set_xticks(Ns); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(f"{a.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {a.fig}.pdf")


if __name__ == "__main__":
    main()
