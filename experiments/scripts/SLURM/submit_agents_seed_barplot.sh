#!/bin/bash -l
#SBATCH --job-name=qrn_seedbar
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu-short
#SBATCH --array=0-9

# Paired evaluation of the three SOTA seeds (omni_v3_20k_s{1,2,3}) over random
# training-domain episodes, H=2000. 10 chunks x 300 episodes = 3000 paired
# episodes; each chunk is an independent master seed so episodes don't repeat.
# Merge locally with
#   python experiments/comparisons/merge_json.py \
#     'results/comparisons/seedbar/chunk*.json' \
#     -o results/comparisons/agents_seed_barplot.json
# then plot:
#   python experiments/comparisons/agents_seed_barplot.py --plot \
#     --out results/comparisons/agents_seed_barplot.json \
#     --fig results/figures/agents_seed_barplot

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/comparisons/seedbar

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
source "$HOME/.venvs/qnetgame/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

i=$SLURM_ARRAY_TASK_ID
echo "Node: $(hostname)  chunk $i (seed $((100 + i)))"

python -u experiments/comparisons/agents_seed_barplot.py \
    --episodes 300 --horizon 2000 --seed $((100 + i)) \
    --out "results/comparisons/seedbar/chunk${i}.json"

echo "done $(date)"
