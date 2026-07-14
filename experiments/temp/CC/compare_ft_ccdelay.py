"""Overlay: does CC-delay training help? Zero-shot 15k agent (trained without CC
delays) vs the from-scratch CC-trained agent, both on CC-delay chains
(1 step/hop), with swap-ASAP and purify-then-swap as (checkpoint-independent)
reference lines.

  --base : delivery_vs_N_ccdelay.json produced with omni_nopen_15k (agent + heuristics)
  --ft   : delivery_vs_N_ccdelay_*.json produced with the CC-trained agent (--agent_only)

  PYTHONPATH=. python experiments/temp/compare_ft_ccdelay.py
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="results/comparisons/delivery_vs_N_ccdelay.json")
    ap.add_argument("--ft", default="results/comparisons/delivery_vs_N_ccdelay_ft.json")
    ap.add_argument("--fig", default="results/figures/temp/delivery_vs_N_ccdelay_ft_vs_base")
    return ap.parse_args()


def series(rows, key):
    rows = sorted(rows, key=lambda r: r["N"])
    Ns = [r["N"] for r in rows]
    T = np.array([r[f"T_{key}"] for r in rows], float)
    se = np.array([r.get(f"se_{key}", 0.0) for r in rows], float)
    return Ns, T, se


def main():
    a = parse_args()
    base = json.load(open(a.base))
    ft = json.load(open(a.ft))
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)

    for key, label, color in [("swap_asap", "Swap-ASAP", "tab:orange"),
                              ("purify_swap", "Purify-then-swap", "tab:green")]:
        if f"T_{key}" not in base[0]:
            continue                      # heuristic was skipped in this run
        Ns, T, se = series(base, key)
        ax.plot(Ns, T, marker="o", ms=4, lw=1.3, color=color, alpha=0.6, label=label)
        ax.fill_between(Ns, T - se, T + se, color=color, alpha=0.10, lw=0)

    Ns, T, se = series(base, "agent")
    ax.plot(Ns, T, marker="s", ms=4, lw=1.8, ls="--", color="tab:blue",
            label="Agent (zero-shot 15k)")
    ax.fill_between(Ns, T - se, T + se, color="tab:blue", alpha=0.12, lw=0)

    Ns2, T2, se2 = series(ft, "agent")
    ax.plot(Ns2, T2, marker="o", ms=5, lw=2.2, ls="-", color="tab:purple",
            label="Agent (trained on CC)")
    ax.fill_between(Ns2, T2 - se2, T2 + se2, color="tab:purple", alpha=0.20, lw=0)

    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("delivery time $T$ (avg steps to termination)")
    ax.set_title("CC delays: CC-trained vs zero-shot 15k")
    ax.set_xticks(Ns); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(f"{a.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {a.fig}.pdf")

    d = {r["N"]: r for r in ft}
    print("N   base_agent  ft_agent   delta%")
    for r in sorted(base, key=lambda r: r["N"]):
        n = r["N"]
        if n in d:
            b, f = r["T_agent"], d[n]["T_agent"]
            print(f"{n:>2}  {b:9.1f}  {f:8.1f}  {100*(f-b)/b:+6.1f}")


if __name__ == "__main__":
    main()
