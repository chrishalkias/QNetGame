#!/bin/bash -l
#SBATCH --job-name=distill_v3_s3
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# Teacher-student distillation on the POST-FIX SOTA agent omni_v3_20k_s3:
#   1) collect a BIG offline teacher-labeled dataset (8k episodes, cached, reusable)
#   2) distill a 5-feature 1-hop student (urgency/avail/occ/can_swap/can_purify) 2500 epochs
#   3) validate delivery time vs N over 4-12 (student|teacher|swap-asap|purify-then-swap)
#
#   sbatch submit_distill_student_v3_s3.sh [episodes] [epochs] [hidden]
# defaults: 8000 episodes, 2500 epochs, hidden 32

set -euo pipefail
EP="${1:-8000}"; EPOCHS="${2:-2500}"; HID="${3:-32}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/policy-distillation

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)  episodes=$EP epochs=$EPOCHS hidden=$HID  $(date)"

TEACHER="checkpoints/omni_v3_20k_s3/policy.pth"
DATASET="results/policy-distillation/teacher_dataset_v3_s3_8k.pkl"
OUT="checkpoints/teacher_student/student_v3_s3_h${HID}_${EPOCHS}ep"
JSON="results/policy-distillation/delivery_vs_N_student_v3_s3_${EPOCHS}ep.json"
FIG="results/policy-distillation/student_vs_teacher_delivery_v3_s3_${EPOCHS}ep"

# 1) + 2) collect (cached) then distill
python -u rl_stack/teacher_student/train_student.py \
    --teacher "$TEACHER" --dataset "$DATASET" \
    --episodes "$EP" --epochs "$EPOCHS" --hidden "$HID" \
    --out "$OUT"

# 3) validate over the 4-12 training range (H=2000: cutoff=20/H=300 saturates), render PDF
python -u rl_stack/teacher_student/eval_student.py \
    --student "$OUT/policy.pth" --teacher "$TEACHER" \
    --n_lo 4 --n_hi 12 --n_ch 4 --p_gen 0.4 --p_swap 0.8 --cutoff 20 \
    --mc_eps 1000 --horizon 2000 \
    --out "$JSON" --fig "$FIG"
python -u rl_stack/teacher_student/eval_student.py --plot --out "$JSON" --fig "$FIG"

echo "done $(date)"
