"""Extended-abstract figure: delivery time vs N, agent + distilled student + heuristics.

Stitches three JSONs (same config: p_gen=0.4, p_swap=0.8, n_ch=4, cutoff=20, H=300):
  N=4-12  from results/policy-distillation/delivery_vs_N_student_1000ep.json
  N=13-15 agent/heuristics from results/comparisons/delivery_vs_N_omni_nopen_15k.json,
          student from results/policy-distillation/delivery_vs_N_student_1000ep_N13_15.json

  PYTHONPATH=. python experiments/temp/plot_EA_delivery_vs_N.py
"""
from __future__ import annotations
import argparse, json, os


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student_lo", default="results/policy-distillation/delivery_vs_N_student_1000ep.json")
    ap.add_argument("--student_hi", default="results/policy-distillation/delivery_vs_N_student_1000ep_N13_15.json")
    ap.add_argument("--agent_hi", default="results/comparisons/delivery_vs_N_omni_nopen_15k.json")
    ap.add_argument("--n_train_max", type=int, default=12)
    ap.add_argument("--fig", default="results/figures/delivery_vs_N_EA.pdf")
    return ap.parse_args()


def main():
    a = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = json.load(open(a.student_lo))                      # N=4-12, T_student/T_teacher/heuristics
    hi_s = json.load(open(a.student_hi))                    # N=13-15, student
    hi_a = [r for r in json.load(open(a.agent_hi)) if r["N"] > a.n_train_max]

    # per-series (N, T, se) stitched across the two N ranges
    def series(key):
        rows = lo + (hi_s if key == "student" else [])
        if key != "student":
            k15 = "agent" if key == "teacher" else key      # 15k json calls the teacher "agent"
            rows = lo + [dict(N=r["N"], **{f"T_{key}": r[f"T_{k15}"], f"se_{key}": r[f"se_{k15}"]})
                         for r in hi_a]
        rows = sorted(rows, key=lambda r: r["N"])
        return ([r["N"] for r in rows], [r[f"T_{key}"] for r in rows],
                [r[f"se_{key}"] for r in rows])

    style = [("teacher", "Agent (teacher)", "tab:blue", "o", "-"),
             ("student", "Student (distilled)", "tab:purple", "D", "-"),
             ("swap_asap", "Swap-ASAP", "tab:orange", "s", "--"),
             ("purify_swap", "Purify-then-swap", "tab:green", "^", "--")]

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "figure.dpi": 150,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for key, label, color, mk, ls in style:
        Ns, T, se = series(key)
        ax.errorbar(Ns, T, yerr=se, color=color, ls=ls, marker=mk, ms=4.5,
                    lw=1.6, capsize=2, label=label)

    ax.axvline(a.n_train_max, color="grey", ls=":", lw=1.3)
    x1 = ax.get_xlim()[1]
    ax.axvspan(a.n_train_max, x1, color="0.92", zorder=0)
    ax.set_xlim(right=x1)
    y0, y1 = ax.get_ylim()
    ax.text(x1 - 0.1, y1 - 0.06 * (y1 - y0), "out-of-distribution",
            color="black", fontsize=13, va="top", ha="right")

    ax.set_xlabel("chain size $N$")
    ax.set_ylabel("Delivery time $T$ (avg steps)")
    ax.set_xticks(sorted({r["N"] for r in lo + hi_a}))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, bbox_inches="tight")
    print(f"saved -> {a.fig}")


if __name__ == "__main__":
    main()
