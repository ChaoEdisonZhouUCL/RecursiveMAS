#!/bin/bash
#
# submit.sh — Submit RecursiveMAS outer-link training to SLURM clusters or run locally.
#
# Platforms:
#   - JUWELS (booster partition)
#   - JURECA (dc-hwai partition)
#   - CISPA  (xe8545 partition, enroot container)
#   - Local execution
#
# Run  ./submit.sh --help  for the full option reference.
#
# Environment: uses requirements.txt in the project root.
# Training script: train_outerlinks_math500.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Load modules
# =============================================================================

source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/slurm.sh"

# =============================================================================
# CLI Parsing
# =============================================================================

show_help() {
    cat <<'EOF'
Usage: ./submit.sh [OPTIONS]

Training options:
  --n_rounds N          Number of recursion rounds       (default: 3)
  --latent_steps N      Latent rollout steps per agent   (default: 8)
  --steps N             Training steps                   (default: 100)
  --lr LR               Learning rate                    (default: 1e-4)
  --dtype DTYPE         bfloat16 | float16 | float32     (default: bfloat16)

Infrastructure:
  --gpus N              GPUs per SLURM job               (default: 3)
  --nodes N             Number of SLURM nodes            (default: 1)
  --partition NAME      Override SLURM partition
  --container IMG       Enroot image (CISPA only)
  --slurm_time HH:MM:SS Override time limit

Examples:
  # Local single-GPU
  ./submit.sh --gpus 1 --n_rounds 3 --steps 50

  # CISPA (auto-detected), 3 GPUs pipeline
  ./submit.sh --n_rounds 3 --steps 100

  # JUWELS
  ./submit.sh --n_rounds 3 --steps 200 --slurm_time 02:00:00

  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --n_rounds)    N_ROUNDS="$2";   shift 2 ;;
        --latent_steps) LATENT_STEPS="$2"; shift 2 ;;
        --steps)       STEPS="$2";      shift 2 ;;
        --lr)          LR="$2";         shift 2 ;;
        --dtype)       DTYPE="$2";      shift 2 ;;
        --gpus)        NUM_GPUS="$2";   shift 2 ;;
        --nodes)       NUM_NODES="$2";  shift 2 ;;
        --partition)   PARTITION="$2";  shift 2 ;;
        --container)   CONTAINER="$2";  shift 2 ;;
        --slurm_time)
            SLURM_TIME[julich]="$2"
            SLURM_TIME[jureca]="$2"
            SLURM_TIME[cispa]="$2"
            shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

PLATFORM=$(detect_platform)
PROJ_DIR="$(dirname "${SCRIPT_DIR}")"
TRAIN_SCRIPT="${PROJ_DIR}/train_outerlinks_math500.py"
SLURM_JOB_DIR="${PROJ_DIR}/outputs/slurm_jobs"
SLURM_LOG_DIR="${PROJ_DIR}/outputs/slurm_logs"
mkdir -p "${SLURM_JOB_DIR}" "${SLURM_LOG_DIR}"

# Build the train command (relative to PROJ_DIR; torchrun prepends the path)
TRAIN_CMD="${TRAIN_SCRIPT} \
    --n_rounds ${N_ROUNDS} \
    --latent_steps ${LATENT_STEPS} \
    --steps ${STEPS} \
    --lr ${LR} \
    --dtype ${DTYPE}"

IS_HPC=false
[[ "${PLATFORM}" == "cispa" || "${PLATFORM}" == "julich" || "${PLATFORM}" == "jureca" ]] && IS_HPC=true

# =============================================================================
# Plan
# =============================================================================

echo "============================================================"
echo "RecursiveMAS Outer-Link Training  [platform: ${PLATFORM}]"
$IS_HPC && echo "Mode: HPC — SLURM job" || echo "Mode: local"
echo "------------------------------------------------------------"
echo "n_rounds:     ${N_ROUNDS}"
echo "latent_steps: ${LATENT_STEPS}"
echo "steps:        ${STEPS}"
echo "lr:           ${LR}   dtype: ${DTYPE}"
echo "GPUs:         ${NUM_GPUS}   nodes: ${NUM_NODES}"
echo "Script:       ${TRAIN_SCRIPT}"
echo "============================================================"

# =============================================================================
# Dispatch
# =============================================================================

s="${SLURM_JOB_DIR}/${TIMESTAMP}_recursivemas_r${N_ROUNDS}_s${STEPS}.sh"

if $IS_HPC; then
    case "${PLATFORM}" in
        cispa)
            _write_slurm_cispa "${s}"
            sbatch "${s}"
            echo "Submitted to ${CLUSTER_LABEL[cispa]}"
            echo "  Status:  squeue -u \$USER"
            echo "  Logs:    ${SLURM_LOG_DIR}/"
            echo "  Script:  ${s}"
            ;;
        julich)
            _write_slurm_julich "${s}"
            sbatch "${s}"
            echo "Submitted to ${CLUSTER_LABEL[julich]}"
            echo "  Status:  squeue -u \$USER"
            echo "  Logs:    ${SLURM_LOG_DIR}/"
            echo "  Script:  ${s}"
            ;;
        jureca)
            _write_slurm_jureca "${s}"
            sbatch "${s}"
            echo "Submitted to ${CLUSTER_LABEL[jureca]}"
            echo "  Status:  squeue -u \$USER"
            echo "  Logs:    ${SLURM_LOG_DIR}/"
            echo "  Script:  ${s}"
            ;;
    esac
else
    # Local: run directly with torchrun
    MASTER_PORT=$((10000 + RANDOM % 20000))
    export HF_TOKEN="${HF_TOKEN}"
    export OMP_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false

    cd "${PROJ_DIR}"

    if [[ ${NUM_GPUS} -gt 1 ]]; then
        echo "Launching torchrun with ${NUM_GPUS} GPUs (pipeline parallelism)..."
        torchrun \
            --standalone \
            --nproc_per_node=${NUM_GPUS} \
            ${TRAIN_CMD}
    else
        echo "Launching single-GPU..."
        python ${TRAIN_CMD}
    fi
fi
