#!/bin/bash -l
#SBATCH --job-name=ft_ccdelay
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu-zen4

# TEMP: fine-tune omni_nopen_15k under CC delays (curriculum + inhomogeneities).
#   sbatch experiments/temp/submit_finetune_ccdelay.sh [extra args -> python]
# Writes checkpoints/ft_ccdelay_2k/{policy.pth,policy_final.pth,metrics.json}.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs checkpoints

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
echo "Node: $(hostname)"

python -u experiments/temp/finetune_ccdelay.py \
    --ckpt checkpoints/sota/policy.pth \
    --run_id ft_ccdelay_2k --episodes 2000 "$@"

echo "done $(date)"
