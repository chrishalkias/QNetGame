#!/bin/bash -l
#SBATCH --job-name=distill_student
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Teacher-student distillation, cluster edition:
#   1) collect a BIG offline teacher-labeled dataset (cached, reusable)
#   2) distill the student for many epochs
#   3) validate delivery time vs N over the full 4-12 range (student|teacher|heuristics)
#
#   sbatch submit_distill_student.sh [episodes] [epochs] [hidden]
# defaults: 3000 episodes, 1000 epochs, hidden 32

set -euo pipefail
EP="${1:-3000}"; EPOCHS="${2:-1000}"; HID="${3:-32}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/policy-distillation

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  episodes=$EP epochs=$EPOCHS hidden=$HID  $(date)"

TEACHER="checkpoints/sota/policy.pth"
DATASET="results/policy-distillation/teacher_dataset_big.pkl"
OUT="checkpoints/teacher_student/student_h${HID}_${EPOCHS}ep"
JSON="results/policy-distillation/delivery_vs_N_student_${EPOCHS}ep.json"
FIG="results/policy-distillation/student_vs_teacher_delivery_${EPOCHS}ep"

# 1) + 2) collect (cached) then distill
python -u rl_stack/teacher_student/train_student.py \
    --teacher "$TEACHER" --dataset "$DATASET" \
    --episodes "$EP" --epochs "$EPOCHS" --hidden "$HID" \
    --out "$OUT"

# 3) validate over the full 4-12 range, then render the PDF
python -u rl_stack/teacher_student/eval_student.py \
    --student "$OUT/policy.pth" --teacher "$TEACHER" \
    --n_lo 4 --n_hi 12 --mc_eps 1000 --horizon 300 \
    --out "$JSON" --fig "$FIG"
python -u rl_stack/teacher_student/eval_student.py --plot --out "$JSON" --fig "$FIG"

echo "done $(date)"
