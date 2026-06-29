#!/bin/bash -l
#SBATCH --job-name=qrn_profile
#SBATCH --output=slurm_logs/profile_%j.out
#SBATCH --error=slurm_logs/profile_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

# Profile the training inner loop (env / inference / update split) on a real
# gpu-short node, to replace the laptop-CPU proxy in
# .local/perf/training-speedup-2026-06-29.md with a measured GPU number.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

echo "Job $SLURM_JOB_ID on $(hostname)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"

for N in 12 24; do
  echo "############## N=$N ##############"
  python -u diagnostics/profile_training.py --n_repeaters "$N" --steps 400 --updates 300 --topn 8
done
echo "done at $(date)"
