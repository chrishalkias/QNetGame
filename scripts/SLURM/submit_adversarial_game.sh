#!/bin/bash -l
#SBATCH --job-name=qrn_adv_game
#SBATCH --output=slurm_logs/adversarial_game_%j.out
#SBATCH --error=slurm_logs/adversarial_game_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#
# Stage III adversarial training.
#
# Submit with defaults:
#   sbatch --account=liacs scripts/SLURM/submit_adversarial_game.sh
#
# Override experiment settings through exported environment variables:
#   FLAVOR=gate_daemon EPISODES=10000 SEED=1 \
#     sbatch --account=liacs --export=ALL \
#     scripts/SLURM/submit_adversarial_game.sh
#
# Common overrides:
#   DEFENDER_CHECKPOINT, FLAVOR, EPISODES, MAX_STEPS, N_RANGE, N_CH, K,
#   P_GEN, P_SWAP, CUTOFF, DEFENDER_LR, ADVERSARY_LR, BATCH_SIZE,
#   BUFFER_SIZE, SEED, OUTPUT_DIR, NO_PLOT.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DEFENDER_CHECKPOINT=${DEFENDER_CHECKPOINT:-checkpoints/inhomo_001/policy_final.pth}
FLAVOR=${FLAVOR:-photon_eater}
EPISODES=${EPISODES:-3000}
MAX_STEPS=${MAX_STEPS:-50}
N_RANGE=${N_RANGE:-"4 5 6 7"}
N_CH=${N_CH:-4}
K=${K:-1}
P_GEN=${P_GEN:-0.8}
P_SWAP=${P_SWAP:-0.7}
CUTOFF=${CUTOFF:-30}
DEFENDER_LR=${DEFENDER_LR:-3e-4}
ADVERSARY_LR=${ADVERSARY_LR:-3e-4}
BATCH_SIZE=${BATCH_SIZE:-64}
BUFFER_SIZE=${BUFFER_SIZE:-80000}
SEED=${SEED:-0}
NO_PLOT=${NO_PLOT:-0}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/cluster/adversarial_game/${FLAVOR}_${SLURM_JOB_ID}}

case "$FLAVOR" in
    photon_eater|gate_daemon) ;;
    *)
        echo "Invalid FLAVOR=$FLAVOR; choose photon_eater or gate_daemon" >&2
        exit 2
        ;;
esac

if [[ "$K" != "1" ]]; then
    echo "Stage III currently requires K=1; received K=$K" >&2
    exit 2
fi

if [[ ! -f "$DEFENDER_CHECKPOINT" ]]; then
    echo "Defender checkpoint not found: $DEFENDER_CHECKPOINT" >&2
    exit 2
fi

read -r -a N_RANGE_VALUES <<< "$N_RANGE"
if [[ ${#N_RANGE_VALUES[@]} -eq 0 ]]; then
    echo "N_RANGE must contain at least one chain size" >&2
    exit 2
fi

mkdir -p slurm_logs "$OUTPUT_DIR"

eval "$(/usr/bin/modulecmd bash purge)" 2>/dev/null || true
eval "$(/usr/bin/modulecmd bash load ALICE/default)"
eval "$(/usr/bin/modulecmd bash load Python/3.11.3-GCCcore-12.3.0)"
eval "$(/usr/bin/modulecmd bash load CUDA/12.4.0)"
source "$HOME/.venvs/qnetgame/bin/activate"

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qnetgame-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Flavor:              $FLAVOR"
echo "Defender checkpoint: $DEFENDER_CHECKPOINT"
echo "Episodes:            $EPISODES"
echo "Chain sizes:         ${N_RANGE_VALUES[*]}"
echo "n_ch / K:            $N_CH / $K"
echo "Seed:                $SEED"
echo "Output directory:    $OUTPUT_DIR"

python - <<'PY'
import sys

try:
    import numpy
    import torch
    import torch_geometric
except ImportError as exc:
    sys.exit(f"[ADVERSARIAL GAME ABORT] remote venv missing dependency: {exc}")

print(
    f"torch {torch.__version__} | numpy {numpy.__version__} | "
    f"torch-geometric {torch_geometric.__version__} | "
    f"cuda {torch.cuda.is_available()}"
)
PY

TRAIN_ARGS=(
    --defender-checkpoint "$DEFENDER_CHECKPOINT"
    --flavor "$FLAVOR"
    --episodes "$EPISODES"
    --max-steps "$MAX_STEPS"
    --n-range "${N_RANGE_VALUES[@]}"
    --n-ch "$N_CH"
    --k "$K"
    --p-gen "$P_GEN"
    --p-swap "$P_SWAP"
    --cutoff "$CUTOFF"
    --defender-lr "$DEFENDER_LR"
    --adversary-lr "$ADVERSARY_LR"
    --batch-size "$BATCH_SIZE"
    --buffer-size "$BUFFER_SIZE"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
)

if [[ "$NO_PLOT" == "1" ]]; then
    TRAIN_ARGS+=(--no-plot)
fi

python -u -m rl_stack.adversarial_game.train "${TRAIN_ARGS[@]}"

echo "Job completed at $(date)"
echo "Outputs: $OUTPUT_DIR"
