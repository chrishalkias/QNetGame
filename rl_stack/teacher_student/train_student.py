"""CLI: distill the SOTA teacher into a tiny 1-hop / 3-feature student.

  PYTHONPATH=. python rl_stack/teacher_student/train_student.py \
      --teacher checkpoints/omni_initial/omni_nopen_15k/policy.pth \
      --episodes 400 --epochs 30 --out checkpoints/teacher_student/student_h16
"""
from __future__ import annotations
import argparse, json, os, pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rl_stack.model import load_qnet
from rl_stack.teacher_student.distill import collect_teacher_dataset, distill_student


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher",
                    default="checkpoints/omni_initial/omni_nopen_15k/policy.pth")
    ap.add_argument("--episodes", type=int, default=400,
                    help="teacher rollout episodes for the frozen dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/teacher_student/student_h16")
    ap.add_argument("--dataset",
                    default="results/policy-distillation/teacher_dataset.pkl",
                    help="cached teacher-labeled dataset; loaded if present "
                         "(reused across student runs), else collected + saved here")
    ap.add_argument("--refresh_dataset", action="store_true",
                    help="force re-collection even if the cache exists")
    return ap.parse_args()


def get_dataset(a, teacher):
    """Load the cached teacher dataset, or collect it and cache it. The cache is
    tied to the teacher + collection distribution, so it is reused verbatim by
    every subsequent student run (--episodes is ignored when the cache is hit)."""
    if a.dataset and os.path.exists(a.dataset) and not a.refresh_dataset:
        data = pickle.load(open(a.dataset, "rb"))
        print(f"loaded cached dataset {a.dataset}: {len(data)} states "
              f"(ignoring --episodes; pass --refresh_dataset to re-collect)")
        return data
    print(f"collecting {a.episodes} episodes ...")
    data = collect_teacher_dataset(teacher, episodes=a.episodes, seed=a.seed,
                                   max_steps=a.max_steps)
    if a.dataset:
        os.makedirs(os.path.dirname(a.dataset) or ".", exist_ok=True)
        pickle.dump(data, open(a.dataset, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"cached dataset -> {a.dataset}")
    return data


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    teacher = load_qnet(a.teacher)
    print(f"teacher {a.teacher} loaded")
    dataset = get_dataset(a, teacher)

    student, best_state, best_val, hist = distill_student(
        teacher, dataset, hidden=a.hidden, epochs=a.epochs, lr=a.lr,
        batch=a.batch, seed=a.seed)

    torch.save(best_state, os.path.join(a.out, "policy.pth"))          # best val-MSE
    torch.save(student.state_dict(), os.path.join(a.out, "policy_final.pth"))
    json.dump({"best_val_mse": best_val, "config": vars(a), **hist},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    ep = range(1, len(hist["train_mse"]) + 1)
    ax.plot(ep, hist["train_mse"], label="train"); ax.plot(ep, hist["val_mse"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE (student Q vs teacher Q)")
    ax.set_yscale("log"); ax.legend(frameon=False)
    ax.set_title(f"distillation h={a.hidden}  best val {best_val:.5f}")
    fig.savefig(os.path.join(a.out, "distill_curve.png"), dpi=150)
    print(f"saved -> {a.out}  (best val_mse {best_val:.6f})")


if __name__ == "__main__":
    main()
