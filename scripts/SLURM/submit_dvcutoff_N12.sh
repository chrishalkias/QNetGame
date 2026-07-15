#!/bin/bash -l
#SBATCH --job-name=qrn_dvcut_N12
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-zen4
#SBATCH --array=0-10

# Delivery time & delivered fidelity vs memory coherence time tau, for the new
# SOTA agent omni_v3_20k_s3: N=12, coherence up to 200, horizon H=2000. One
# cutoff per array task (the low-cutoff cells run near the full horizon and are
# the expensive ones). Merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/comparisons/dvcut_N12/chunk*.json' \
#     -o results/comparisons/delivery_vs_cutoff_N12.json
# then plot with delivery_vs_cutoff.py --plot --out <that json> --fig <fig>.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/dvcut_N12

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CUTOFFS=(10 20 30 40 50 75 100 125 150 175 200)
CKPT="checkpoints/omni_v3_20k_s3/policy.pth"
i=$SLURM_ARRAY_TASK_ID
CT=${CUTOFFS[$i]}
echo "Node: $(hostname)  task $i -> cutoff=$CT"

python -u experiments/comparisons/delivery_vs_cutoff.py \
    --ckpt "$CKPT" --N 12 --p_gen 0.5 --p_swap 0.5 --n_ch 4 \
    --cutoffs "$CT" --horizon 2000 --mc_eps 2000 \
    --out "results/comparisons/dvcut_N12/chunk${i}.json"

echo "done $(date)"
