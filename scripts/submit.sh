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

Eval options (--eval enables eval mode; all training flags are ignored):
  --eval                  Run evaluation instead of training
  --ckpt_path PATH        Checkpoint to evaluate (required with --eval):
                            "released_weights"  → baseline (no trained adapters)
                            path with _original    → sequential_light_trained style
                            path with _shared_roae → sequential_light_shared_roae style
  --eval_dataset NAME     Dataset for eval: math500 | medqa | gpqa | mbppplus  (default: math500)
  --eval_batch_size N     Batch size for eval                                   (default: 32)
  --eval_seed N           Random seed for eval                                  (default: 42)

Training options:
  --n_rounds N            Number of recursion rounds              (default: 3)
  --latent_steps N        Latent rollout steps per agent          (default: 8)
  --batch_size N          Training batch size                     (default: 4)
  --steps N               Training steps                          (default: 100)
  --lr LR                 Learning rate                           (default: 1e-4)
  --dtype DTYPE           bfloat16 | float16 | float32            (default: bfloat16)
  --mode MODE             original | shared_roae | compare | compare_roundskip  (default: original)
  --n_experts N           MoE experts (shared_roae/compare only)  (default: 4)
  --expert_dim_divisor N  Expert inner dim divisor                (default: 4)
  --no_kv_cache           Disable KV cache in latent rollout      (required for latent_steps>=20)
  --no_round_skip         Disable round-skip gate (beta=0, frozen); shared_roae mode only
  --dataset NAME          math500 | s1k | m1k | s1k+m1k (pooled)  (default: math500)
  --max_seq_len N         Max combined Q+A length in tokens (0=off) (default: 0)
  --n_samples N           Training problems per epoch (0=full dataset) (default: 500)
  --n_ckpt N              Checkpoints saved during training, evenly spaced (default: 1)
  --resume_ckpt PATH      step_N checkpoint dir to resume from (default: empty = fresh start)
  --grad_checkpoint       Enable gradient checkpointing (saves activation memory, ~33% slower)
  --grad_accum N          Gradient accumulation steps; effective batch = batch_size * N (default: 1)

Infrastructure:
  --gpus N              GPUs per SLURM job               (default: 3)
  --nodes N             Number of SLURM nodes            (default: 1)
  --partition NAME      Override SLURM partition
  --container IMG       Enroot image (CISPA only)
  --slurm_time HH:MM:SS Override time limit

Examples:
  # Eval: released weights baseline
  ./submit.sh --eval --ckpt_path released_weights

  # Eval: trained original adapters
  ./submit.sh --eval --ckpt_path /path/to/ckpt_original_r5

  # Eval: trained shared_roae adapters
  ./submit.sh --eval --ckpt_path /path/to/ckpt_shared_roae_r5

  # Local single-GPU, original mode
  ./submit.sh --gpus 1 --n_rounds 3 --steps 50

  # CISPA, compare both methods side-by-side
  ./submit.sh --n_rounds 3 --steps 100 --mode compare

  # Ablation: SharedLink with vs without round-skip
  ./submit.sh --n_rounds 3 --steps 100 --mode compare_roundskip

  # CISPA, SharedLink-RoAE only with larger MoE
  ./submit.sh --n_rounds 3 --steps 100 --mode shared_roae --n_experts 8

  # JUWELS
  ./submit.sh --n_rounds 3 --steps 200 --slurm_time 02:00:00

  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval)               EVAL=true;               shift ;;
        --ckpt_path)          CKPT_PATH="$2";          shift 2 ;;
        --eval_dataset)       EVAL_DATASET="$2";       shift 2 ;;
        --eval_batch_size)    EVAL_BATCH_SIZE="$2";    shift 2 ;;
        --eval_seed)          EVAL_SEED="$2";          shift 2 ;;
        --n_rounds)           N_ROUNDS="$2";           shift 2 ;;
        --latent_steps)       LATENT_STEPS="$2";       shift 2 ;;
        --batch_size)         BATCH_SIZE="$2";         shift 2 ;;
        --steps)              STEPS="$2";              shift 2 ;;
        --lr)                 LR="$2";                 shift 2 ;;
        --dtype)              DTYPE="$2";              shift 2 ;;
        --mode)               MODE="$2";               shift 2 ;;
        --n_experts)          N_EXPERTS="$2";          shift 2 ;;
        --expert_dim_divisor) EXPERT_DIM_DIVISOR="$2"; shift 2 ;;
        --no_kv_cache)        NO_KV_CACHE=true;        shift ;;
        --no_round_skip)      USE_ROUND_SKIP=false;    shift ;;
        --n_ckpt)             N_CKPT="$2";             shift 2 ;;
        --resume_ckpt)        RESUME_CKPT="$2";        shift 2 ;;
        --grad_checkpoint)    GRAD_CHECKPOINT=true;    shift ;;
        --grad_accum)         GRAD_ACCUM="$2";         shift 2 ;;
        --dataset)            DATASET="$2";            shift 2 ;;
        --max_seq_len)        MAX_SEQ_LEN="$2";        shift 2 ;;
        --n_samples)          N_SAMPLES="$2";          shift 2 ;;
        --gpus)               NUM_GPUS="$2";           shift 2 ;;
        --nodes)              NUM_NODES="$2";          shift 2 ;;
        --partition)          PARTITION="$2";          shift 2 ;;
        --container)          CONTAINER="$2";          shift 2 ;;
        --slurm_time)
            SLURM_TIME[julich]="$2"
            SLURM_TIME[jureca]="$2"
            SLURM_TIME[cispa]="$2"
            shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Normalize booleans to lowercase so comparisons work regardless of how
# config.sh or the user set them (true/True/TRUE all accepted).
EVAL="${EVAL,,}"
NO_KV_CACHE="${NO_KV_CACHE,,}"
USE_ROUND_SKIP="${USE_ROUND_SKIP,,}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT,,}"

# =============================================================================
# Setup
# =============================================================================

PLATFORM=$(detect_platform)
PROJ_DIR="$(dirname "${SCRIPT_DIR}")"
TRAIN_SCRIPT="${PROJ_DIR}/train_outerlinks_math500.py"
EVAL_SCRIPT="${PROJ_DIR}/run.py"
SLURM_JOB_DIR="${PROJ_DIR}/outputs/slurm_jobs"
SLURM_LOG_DIR="${PROJ_DIR}/outputs/slurm_logs"
mkdir -p "${SLURM_JOB_DIR}" "${SLURM_LOG_DIR}"

IS_HPC=false
[[ "${PLATFORM}" == "cispa" || "${PLATFORM}" == "julich" || "${PLATFORM}" == "jureca" ]] && IS_HPC=true

# =============================================================================
# Build command
# =============================================================================

if [[ "${EVAL}" == "true" ]]; then
    # ── Eval mode ────────────────────────────────────────────────────────────
    if [[ -z "${CKPT_PATH}" ]]; then
        echo "Error: --ckpt_path is required with --eval" >&2
        exit 1
    fi

    _EVAL_COMMON="--batch_size ${EVAL_BATCH_SIZE} --temperature 0.6 --top_p 0.95 \
    --dataset ${EVAL_DATASET} --seed ${EVAL_SEED} --trust_remote_code 1 --device cuda"

    if [[ "${CKPT_PATH}" == "released_weights" ]]; then
        EVAL_STYLE="sequential_light"
        TRAIN_CMD="${EVAL_SCRIPT} --style ${EVAL_STYLE} ${_EVAL_COMMON}"
    elif [[ "${CKPT_PATH}" == *"_original"* ]]; then
        EVAL_STYLE="sequential_light_trained"
        TRAIN_CMD="${EVAL_SCRIPT} --style ${EVAL_STYLE} --ckpt_dir ${CKPT_PATH} ${_EVAL_COMMON}"
    elif [[ "${CKPT_PATH}" == *"_shared_roae"* ]]; then
        EVAL_STYLE="sequential_light_shared_roae"
        TRAIN_CMD="${EVAL_SCRIPT} --style ${EVAL_STYLE} --ckpt_dir ${CKPT_PATH} ${_EVAL_COMMON}"
    else
        echo "Error: cannot infer eval style from --ckpt_path '${CKPT_PATH}'." >&2
        echo "  Path must be 'released_weights', or contain '_original' or '_shared_roae'." >&2
        exit 1
    fi
else
    # ── Training mode ─────────────────────────────────────────────────────────
    TRAIN_CMD="${TRAIN_SCRIPT} \
    --n_rounds ${N_ROUNDS} \
    --latent_steps ${LATENT_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --lr ${LR} \
    --dtype ${DTYPE} \
    --mode ${MODE} \
    --n_experts ${N_EXPERTS} \
    --expert_dim_divisor ${EXPERT_DIM_DIVISOR} \
    --dataset ${DATASET} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --n_samples ${N_SAMPLES} \
    --n_ckpt ${N_CKPT}"
    [[ "${NO_KV_CACHE}" == "true" ]]      && TRAIN_CMD="${TRAIN_CMD} --no_kv_cache"
    [[ "${USE_ROUND_SKIP}" == "false" ]]  && TRAIN_CMD="${TRAIN_CMD} --no_round_skip"
    [[ -n "${RESUME_CKPT}" ]]             && TRAIN_CMD="${TRAIN_CMD} --resume_ckpt ${RESUME_CKPT}"
    [[ "${GRAD_CHECKPOINT}" == "true" ]]  && TRAIN_CMD="${TRAIN_CMD} --grad_checkpoint"
    [[ "${GRAD_ACCUM}" -gt 1 ]] 2>/dev/null && TRAIN_CMD="${TRAIN_CMD} --grad_accum ${GRAD_ACCUM}"
fi

# =============================================================================
# Plan
# =============================================================================

echo "============================================================"
if [[ "${EVAL}" == "true" ]]; then
    echo "RecursiveMAS Evaluation  [platform: ${PLATFORM}]"
    $IS_HPC && echo "Mode: HPC — SLURM job" || echo "Mode: local"
    echo "------------------------------------------------------------"
    echo "ckpt_path:    ${CKPT_PATH}"
    echo "eval_style:   ${EVAL_STYLE}"
    echo "dataset:      ${EVAL_DATASET}   batch_size: ${EVAL_BATCH_SIZE}   seed: ${EVAL_SEED}"
    echo "Script:       ${EVAL_SCRIPT}"
else
    echo "RecursiveMAS Outer-Link Training  [platform: ${PLATFORM}]"
    $IS_HPC && echo "Mode: HPC — SLURM job" || echo "Mode: local"
    echo "------------------------------------------------------------"
    echo "n_rounds:     ${N_ROUNDS}"
    echo "latent_steps: ${LATENT_STEPS}"
    echo "batch_size:   ${BATCH_SIZE}"
    echo "steps:        ${STEPS}"
    echo "lr:           ${LR}   dtype: ${DTYPE}"
    echo "mode:         ${MODE}  (n_experts=${N_EXPERTS}  expert_dim_divisor=${EXPERT_DIM_DIVISOR}  round_skip=${USE_ROUND_SKIP})"
    echo "dataset:      ${DATASET}   max_seq_len: ${MAX_SEQ_LEN}   n_samples: ${N_SAMPLES}   n_ckpt: ${N_CKPT}"
    [[ -n "${RESUME_CKPT}" ]] && echo "resume_ckpt:  ${RESUME_CKPT}"
    echo "grad_checkpoint: ${GRAD_CHECKPOINT}   grad_accum: ${GRAD_ACCUM}"
    echo "GPUs:         ${NUM_GPUS}   nodes: ${NUM_NODES}"
    echo "Script:       ${TRAIN_SCRIPT}"
fi
echo "============================================================"

# =============================================================================
# Dispatch
# =============================================================================

if [[ "${EVAL}" == "true" ]]; then
    _job_tag="${TIMESTAMP}_eval_${EVAL_STYLE}"
else
    _job_tag="${TIMESTAMP}_recursivemas_r${N_ROUNDS}_s${STEPS}"
fi
s="${SLURM_JOB_DIR}/${_job_tag}.sh"

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
    export HF_TOKEN="${HF_TOKEN}"
    export OMP_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false

    cd "${PROJ_DIR}"

    if [[ "${EVAL}" == "true" ]]; then
        echo "Launching eval (single process)..."
        python ${TRAIN_CMD}
    elif [[ ${NUM_GPUS} -gt 1 ]]; then
        echo "Launching torchrun with ${NUM_GPUS} GPUs (pipeline parallelism)..."
        MASTER_PORT=$((10000 + RANDOM % 20000))
        torchrun \
            --standalone \
            --nproc_per_node=${NUM_GPUS} \
            ${TRAIN_CMD}
    else
        echo "Launching single-GPU..."
        python ${TRAIN_CMD}
    fi
fi
