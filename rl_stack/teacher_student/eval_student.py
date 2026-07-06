"""CLI: delivery-time-vs-N for the distilled student vs teacher vs heuristics.

Answers the research question: does the 1-hop / 3-feature student keep the
teacher's edge over swap-asap and purify-then-swap? Reuses the canonical
delivery-time evaluator (optimal_baseline.mc_eval) so numbers match the rest of
the suite.

  PYTHONPATH=. python rl_stack/teacher_student/eval_student.py \
      --student checkpoints/teacher_student/student_h16/policy.pth
  PYTHONPATH=. python rl_stack/teacher_student/eval_student.py --plot
"""
from __future__ import annotations
import argparse, json, os

import numpy as np

from rl_stack import strategies
from rl_stack.teacher_student.student_model import load_student
from rl_stack.teacher_student.distill import student_policy_fn
from experiments.heatmap.optimal_baseline import mc_eval, make_agent_fn


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student",
                    default="checkpoints/teacher_student/student_h16/policy.pth")
    ap.add_argument("--teacher",
                    default="checkpoints/omni_initial/omni_nopen_15k/policy.pth")
    ap.add_argument("--n_lo", type=int, default=4)
    ap.add_argument("--n_hi", type=int, default=15)
    ap.add_argument("--Ns", type=int, nargs="+", default=None,
                    help="explicit chain sizes (overrides --n_lo/--n_hi), "
                         "e.g. --Ns 5 8 10")
    ap.add_argument("--n_ch", type=int, default=4)
    ap.add_argument("--p_gen", type=float, default=0.4)
    ap.add_argument("--p_swap", type=float, default=0.8)
    ap.add_argument("--cutoff", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--mc_eps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/comparisons/delivery_vs_N_student.json")
    ap.add_argument("--fig", default="results/figures/delivery_vs_N_student")
    ap.add_argument("--plot", action="store_true", help="render PDF from existing JSON")
    return ap.parse_args()


def run_eval(a):
    student = load_student(a.student)
    policies = {
        "student":     student_policy_fn(student),
        "teacher":     make_agent_fn(a.teacher),
        "swap_asap":   lambda env, obs: strategies.swap_asap(env),
        "purify_swap": lambda env, obs: strategies.purify_then_swap(env),
    }
    rows = []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Ns = a.Ns if a.Ns else list(range(a.n_lo, a.n_hi + 1))
    for N in Ns:
        row = {"N": N}
        for name, fn in policies.items():
            T, seT = mc_eval(fn, N, a.n_ch, a.p_gen, a.p_swap, a.cutoff,
                             a.horizon, a.mc_eps, seed=a.seed)
            row[f"T_{name}"] = T
            row[f"se_{name}"] = seT / np.sqrt(a.mc_eps)
        print(f"N={N:2d}  " + "  ".join(f"{k[2:]}={row[k]:.1f}"
                                        for k in row if k.startswith("T_")))
        rows.append(row)
        json.dump(rows, open(a.out, "w"), indent=2)     # incremental
    print(f"saved -> {a.out}")


def run_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(a.out))
    Ns = [r["N"] for r in rows]
    style = {"student": ("tab:purple", "-"), "teacher": ("tab:blue", "-"),
             "swap_asap": ("tab:orange", "--"), "purify_swap": ("tab:red", "--")}
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for name, (c, ls) in style.items():
        if f"T_{name}" not in rows[0]:
            continue
        ax.errorbar(Ns, [r[f"T_{name}"] for r in rows],
                    yerr=[r.get(f"se_{name}", 0) for r in rows],
                    color=c, ls=ls, marker="o", ms=4, lw=1.8,
                    label=name.replace("_", "-"), capsize=2)
    ax.set_xlabel("chain size $N$"); ax.set_ylabel("delivery time $T$ (avg steps)")
    ax.set_title("Distilled student vs teacher vs heuristics\n"
                 r"($p_\mathrm{gen}=0.4$, $p_\mathrm{swap}=0.8$, $n_\mathrm{ch}=4$, cutoff $=20$)")
    ax.set_xticks(Ns); ax.grid(alpha=0.3); ax.legend(frameon=False)
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(f"{a.fig}.pdf", bbox_inches="tight")
    print(f"saved -> {a.fig}.pdf")


def main():
    a = parse_args()
    run_plot(a) if a.plot else run_eval(a)


if __name__ == "__main__":
    main()
