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
# Fallback for a config.sh predating the setting, so an old config still submits.
: "${LINK_DTYPE:=float32}"
source "${SCRIPT_DIR}/slurm.sh"

# =============================================================================
# CLI Parsing
# =============================================================================

show_help() {
    cat <<'EOF'
Usage: ./submit.sh [OPTIONS]

Eval options (--eval enables eval mode; all training flags are ignored):
  --eval                  Run evaluation instead of training
  --ckpt_path PATH[,..]   Checkpoint(s) to evaluate (required with --eval).
                          Comma-separated list allowed; each entry is:
                            "released_weights"  → baseline (no trained adapters)
                            path with _original    → sequential_light_trained style
                            path with _shared_roae → sequential_light_shared_roae style
                            path with _shared_state → same style; link variant auto-detected
  --eval_dataset NAME[,..] Dataset(s): math500 | medqa | gpqa | mbppplus | livecodebench (default: math500)
                          One job is submitted per (ckpt, dataset) combination.
  --eval_batch_size N     Batch size for eval                                   (default: 32)
  --eval_seed N           Random seed for eval                                  (default: 42)
  --eval_latent_steps N   Latent rollout steps for eval (default: LATENT_STEPS)
                            0 = sweep 16/32/48; "protocol" = per-dataset value
                            from eval_protocol.yaml (paper setting)
  --eval_n_rounds N       Recursion rounds for eval                             (default: N_ROUNDS)
  --eval_greedy           Greedy decoding for eval (reproducible; ignores temperature/top_p)

Training options:
  --n_rounds N            Number of recursion rounds              (default: 3)
  --latent_steps N        Latent rollout steps per agent          (default: 8)
  --batch_size N          Training batch size                     (default: 4)
  --steps N               Training steps                          (default: 100)
  --lr LR                 Learning rate                           (default: 1e-4)
  --dtype DTYPE           bfloat16 | float16 | float32            (default: bfloat16)
  --link_dtype DTYPE      dtype of the trained link only; backbones stay at --dtype
                          float32 | bfloat16 | float16 | same     (default: float32)
  --mode MODE             original | shared_roae | shared_state | shared_tied | compare | compare_roundskip  (default: original)
                            shared_state: shared_roae + residual recursive state z
                            bypassing each full round (z' = z + gamma*F(z))
                            shared_tied: shared_roae with tied decoders O_i = S_i^T
                            (semi-orthogonal round trips at init)
  --n_experts N           MoE experts (shared_roae/compare only)  (default: 4)
  --expert_dim_divisor N  Expert inner dim divisor                (default: 4)
  --gamma_init F          Initial recursive-state write gate      (default: 1e-3)
  --no_kv_cache           Disable KV cache in latent rollout      (required for latent_steps>=20)
  --no_round_skip         Disable round-skip gate (beta=0, frozen); shared_roae mode only
  --dataset NAME          math500 | s1k | m1k | s1k+m1k (pooled)  (default: math500)
  --max_seq_len N         Max combined Q+A length in tokens (0=off) (default: 0)
  --n_samples N           Training problems per epoch (0=full dataset) (default: 500)
  --n_ckpt N              Checkpoints saved during training, evenly spaced (default: 1)
  --resume_ckpt PATH      step_N checkpoint dir to resume from (default: empty = fresh start)
  --grad_checkpoint       Enable gradient checkpointing (saves activation memory, ~33% slower)
  --grad_accum N          Gradient accumulation steps; effective batch = batch_size * N (default: 1)
  --wandb                 Log training dynamics to Weights & Biases (loss, per-link grad
                          norms, gate values, hyperparameters); logged from the planner
                          rank only, one run per (mode, n_rounds) combination
  --wandb_project NAME    W&B project name                       (default: recursivemas-outerlinks)
  --wandb_entity NAME     W&B entity (team/user)                  (default: your default entity)
  --wandb_run_name NAME   W&B run name                            (default: auto from mode/n_rounds)
  --wandb_mode MODE       online | offline | disabled             (default: offline; compute nodes
                          here have no internet — sync later with `wandb sync`)

Infrastructure:
  --gpus N              GPUs per SLURM job               (default: 3)
  --nodes N             Number of SLURM nodes            (default: 1)
  --partition NAME      Override SLURM partition
  --container IMG       Enroot image (CISPA only)
  --slurm_time HH:MM:SS Override time limit
  --dependency SPEC     sbatch --dependency=SPEC (e.g. afterok:12345[:12346]).
                        Chains eval jobs behind their training job.

Examples:
  # Eval: released weights baseline
  ./submit.sh --eval --ckpt_path released_weights

  # Eval: trained original adapters
  ./submit.sh --eval --ckpt_path /path/to/ckpt_original_r5

  # Eval: trained shared_roae adapters
  ./submit.sh --eval --ckpt_path /path/to/ckpt_shared_roae_r5

  # Eval sweep: 2 ckpts x 3 datasets = 6 jobs
  ./submit.sh --eval --ckpt_path /path/A_original/step_4000,/path/B_shared_roae/step_4000 \
              --eval_dataset math500,medqa,livecodebench

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
        --ckpt_path)          unset CKPT_PATH;    CKPT_PATH="$2";    shift 2 ;;
        --eval_dataset)       unset EVAL_DATASET; EVAL_DATASET="$2"; shift 2 ;;
        --eval_batch_size)    EVAL_BATCH_SIZE="$2";    shift 2 ;;
        --eval_seed)          EVAL_SEED="$2";          shift 2 ;;
        --eval_latent_steps)  EVAL_LATENT_STEPS="$2";  shift 2 ;;
        --eval_n_rounds)      EVAL_N_ROUNDS="$2";      shift 2 ;;
        --eval_greedy)        EVAL_GREEDY=true;        shift ;;
        --n_rounds)           N_ROUNDS="$2";           shift 2 ;;
        --latent_steps)       LATENT_STEPS="$2";       shift 2 ;;
        --batch_size)         BATCH_SIZE="$2";         shift 2 ;;
        --steps)              STEPS="$2";              shift 2 ;;
        --lr)                 LR="$2";                 shift 2 ;;
        --dtype)              DTYPE="$2";              shift 2 ;;
        --link_dtype)         LINK_DTYPE="$2";         shift 2 ;;
        --mode)               MODE="$2";               shift 2 ;;
        --n_experts)          N_EXPERTS="$2";          shift 2 ;;
        --expert_dim_divisor) EXPERT_DIM_DIVISOR="$2"; shift 2 ;;
        --gamma_init)         GAMMA_INIT="$2";         shift 2 ;;
        --no_kv_cache)        NO_KV_CACHE=true;        shift ;;
        --no_round_skip)      USE_ROUND_SKIP=false;    shift ;;
        --n_ckpt)             N_CKPT="$2";             shift 2 ;;
        --resume_ckpt)        RESUME_CKPT="$2";        shift 2 ;;
        --grad_checkpoint)    GRAD_CHECKPOINT=true;    shift ;;
        --grad_accum)         GRAD_ACCUM="$2";         shift 2 ;;
        --wandb)              WANDB=true;              shift ;;
        --wandb_project)      WANDB_PROJECT="$2";      shift 2 ;;
        --wandb_entity)       WANDB_ENTITY="$2";       shift 2 ;;
        --wandb_run_name)     WANDB_RUN_NAME="$2";     shift 2 ;;
        --wandb_mode)         WANDB_MODE="$2";         shift 2 ;;
        --dataset)            DATASET="$2";            shift 2 ;;
        --max_seq_len)        MAX_SEQ_LEN="$2";        shift 2 ;;
        --n_samples)          N_SAMPLES="$2";          shift 2 ;;
        --gpus)               NUM_GPUS="$2";           shift 2 ;;
        --nodes)              NUM_NODES="$2";          shift 2 ;;
        --partition)          PARTITION="$2";          shift 2 ;;
        --container)          CONTAINER="$2";          shift 2 ;;
        --dependency)         DEPENDENCY="$2";         shift 2 ;;
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
EVAL_GREEDY="${EVAL_GREEDY,,}"
NO_KV_CACHE="${NO_KV_CACHE,,}"
USE_ROUND_SKIP="${USE_ROUND_SKIP,,}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT,,}"
WANDB="${WANDB,,}"

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

    # Default eval latent steps and rounds to the training values so they stay in sync.
    EVAL_LATENT_STEPS="${EVAL_LATENT_STEPS:-${LATENT_STEPS}}"
    EVAL_N_ROUNDS="${EVAL_N_ROUNDS:-${N_ROUNDS}}"

    # CKPT_PATH and EVAL_DATASET accept a bash array or a comma-separated
    # string; one job is submitted per (checkpoint, dataset) combination.
    _as_list() {   # $1 = output array name, $2 = input variable name
        local _joined
        if declare -p "$2" 2>/dev/null | grep -q '^declare -a'; then
            local -n _in="$2"
            _joined="$(IFS=,; echo "${_in[*]}")"
        else
            _joined="${!2}"
        fi
        local -n _out="$1"
        IFS=',' read -r -a _out <<< "${_joined}"
        local i
        for i in "${!_out[@]}"; do _out[$i]="${_out[$i]//[[:space:]]/}"; done
    }
    _as_list EVAL_CKPTS    CKPT_PATH
    _as_list EVAL_DATASETS EVAL_DATASET

    infer_eval_style() {
        case "$1" in
            released_weights) echo "sequential_light" ;;
            *_original*)      echo "sequential_light_trained" ;;
            # shared_state/shared_tied ckpts reuse the shared_roae eval style;
            # run.py auto-detects the link variant from shared_roae_config.json.
            *_shared_roae*|*_shared_state*|*_shared_tied*) echo "sequential_light_shared_roae" ;;
            *) return 1 ;;
        esac
    }

    # Sets TRAIN_CMD and EVAL_STYLE for one (checkpoint, dataset) combination.
    build_eval_cmd() {
        local ckpt="$1" ds="$2"
        EVAL_STYLE="$(infer_eval_style "${ckpt}")"
        TRAIN_CMD="${EVAL_SCRIPT} --style ${EVAL_STYLE}"
        [[ "${ckpt}" != "released_weights" ]] && TRAIN_CMD="${TRAIN_CMD} --ckpt_dir ${ckpt}"
        # temperature/top_p/max_new_tokens/num_rollouts come from
        # eval_protocol.yaml per dataset (see EVAL_PROTOCOL.md).
        # EVAL_LATENT_STEPS="protocol" omits --latent_length so the yaml's
        # per-dataset latent_length applies too.
        local latent_flag=""
        [[ "${EVAL_LATENT_STEPS,,}" != "protocol" ]] && latent_flag="--latent_length ${EVAL_LATENT_STEPS}"
        TRAIN_CMD="${TRAIN_CMD} --batch_size ${EVAL_BATCH_SIZE} \
    --dataset ${ds} --seed ${EVAL_SEED} ${latent_flag} \
    --num_recursive_rounds ${EVAL_N_ROUNDS} \
    --trust_remote_code 1 --device cuda"
        [[ "${EVAL_GREEDY}" == "true" ]] && TRAIN_CMD="${TRAIN_CMD} --greedy"
        # Compute nodes have no internet, so eval writes its metric next to the
        # checkpoint; sync_eval_to_wandb.py attaches it to the training W&B run
        # from the login node afterwards.
        if [[ "${ckpt}" != "released_weights" ]]; then
            TRAIN_CMD="${TRAIN_CMD} --result_json ${ckpt}/eval_${ds}.json"
        fi
        return 0
    }

    # Validate every checkpoint up front so nothing is submitted on a bad list.
    for _ckpt in "${EVAL_CKPTS[@]}"; do
        if ! infer_eval_style "${_ckpt}" >/dev/null; then
            echo "Error: cannot infer eval style from --ckpt_path '${_ckpt}'." >&2
            echo "  Path must be 'released_weights', or contain '_original' or '_shared_roae'." >&2
            exit 1
        fi
    done
else
    # ── Training mode ─────────────────────────────────────────────────────────
    TRAIN_CMD="${TRAIN_SCRIPT} \
    --n_rounds ${N_ROUNDS} \
    --latent_steps ${LATENT_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --lr ${LR} \
    --dtype ${DTYPE} \
    --link_dtype ${LINK_DTYPE} \
    --mode ${MODE} \
    --n_experts ${N_EXPERTS} \
    --expert_dim_divisor ${EXPERT_DIM_DIVISOR} \
    --gamma_init ${GAMMA_INIT} \
    --dataset ${DATASET} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --n_samples ${N_SAMPLES} \
    --n_ckpt ${N_CKPT}"
    [[ "${NO_KV_CACHE}" == "true" ]]      && TRAIN_CMD="${TRAIN_CMD} --no_kv_cache"
    [[ "${USE_ROUND_SKIP}" == "false" ]]  && TRAIN_CMD="${TRAIN_CMD} --no_round_skip"
    [[ -n "${RESUME_CKPT}" ]]             && TRAIN_CMD="${TRAIN_CMD} --resume_ckpt ${RESUME_CKPT}"
    [[ "${GRAD_CHECKPOINT}" == "true" ]]  && TRAIN_CMD="${TRAIN_CMD} --grad_checkpoint"
    [[ "${GRAD_ACCUM}" -gt 1 ]] 2>/dev/null && TRAIN_CMD="${TRAIN_CMD} --grad_accum ${GRAD_ACCUM}"
    if [[ "${WANDB}" == "true" ]]; then
        TRAIN_CMD="${TRAIN_CMD} --wandb --wandb_project ${WANDB_PROJECT} --wandb_mode ${WANDB_MODE}"
        [[ -n "${WANDB_ENTITY}" ]]   && TRAIN_CMD="${TRAIN_CMD} --wandb_entity ${WANDB_ENTITY}"
        [[ -n "${WANDB_RUN_NAME}" ]] && TRAIN_CMD="${TRAIN_CMD} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
fi

# =============================================================================
# Plan
# =============================================================================

echo "============================================================"
if [[ "${EVAL}" == "true" ]]; then
    echo "RecursiveMAS Evaluation  [platform: ${PLATFORM}]"
    $IS_HPC && echo "Mode: HPC — SLURM job" || echo "Mode: local"
    echo "------------------------------------------------------------"
    echo "ckpt_paths:"
    for _ckpt in "${EVAL_CKPTS[@]}"; do echo "  - ${_ckpt}  [$(infer_eval_style "${_ckpt}")]"; done
    echo "datasets:     ${EVAL_DATASETS[*]}"
    echo "jobs:         $(( ${#EVAL_CKPTS[@]} * ${#EVAL_DATASETS[@]} ))  (one per ckpt x dataset)"
    echo "batch_size:   ${EVAL_BATCH_SIZE}   seed: ${EVAL_SEED}   latent_steps: ${EVAL_LATENT_STEPS}   n_rounds: ${EVAL_N_ROUNDS}   greedy: ${EVAL_GREEDY}"
    echo "Script:       ${EVAL_SCRIPT}"
else
    echo "RecursiveMAS Outer-Link Training  [platform: ${PLATFORM}]"
    $IS_HPC && echo "Mode: HPC — SLURM job" || echo "Mode: local"
    echo "------------------------------------------------------------"
    echo "n_rounds:     ${N_ROUNDS}"
    echo "latent_steps: ${LATENT_STEPS}"
    echo "batch_size:   ${BATCH_SIZE}"
    echo "steps:        ${STEPS}"
    echo "lr:           ${LR}   dtype: ${DTYPE}   link_dtype: ${LINK_DTYPE}"
    echo "mode:         ${MODE}  (n_experts=${N_EXPERTS}  expert_dim_divisor=${EXPERT_DIM_DIVISOR}  round_skip=${USE_ROUND_SKIP})"
    echo "dataset:      ${DATASET}   max_seq_len: ${MAX_SEQ_LEN}   n_samples: ${N_SAMPLES}   n_ckpt: ${N_CKPT}"
    [[ -n "${RESUME_CKPT}" ]] && echo "resume_ckpt:  ${RESUME_CKPT}"
    echo "grad_checkpoint: ${GRAD_CHECKPOINT}   grad_accum: ${GRAD_ACCUM}"
    if [[ "${WANDB}" == "true" ]]; then
        echo "wandb:        project=${WANDB_PROJECT}  mode=${WANDB_MODE}${WANDB_ENTITY:+  entity=${WANDB_ENTITY}}${WANDB_RUN_NAME:+  run_name=${WANDB_RUN_NAME}}"
    else
        echo "wandb:        disabled"
    fi
    echo "GPUs:         ${NUM_GPUS}   nodes: ${NUM_NODES}"
    echo "Script:       ${TRAIN_SCRIPT}"
fi
echo "============================================================"

# =============================================================================
# Dispatch
# =============================================================================

# Writes the platform-specific job script for the current TRAIN_CMD and sbatches it.
_submit_hpc_one() {   # $1 = job script path
    case "${PLATFORM}" in
        cispa)  _write_slurm_cispa "$1" ;;
        julich) _write_slurm_julich "$1" ;;
        jureca) _write_slurm_jureca "$1" ;;
    esac
    if [[ -n "${DEPENDENCY}" ]]; then
        sbatch --dependency="${DEPENDENCY}" "$1"
    else
        sbatch "$1"
    fi
}

if [[ "${EVAL}" == "true" ]]; then
    _n=0
    for _ckpt in "${EVAL_CKPTS[@]}"; do
        for _ds in "${EVAL_DATASETS[@]}"; do
            build_eval_cmd "${_ckpt}" "${_ds}"
            _n=$((_n + 1))
            # Include the checkpoint's run directory in the tag: two submit.sh
            # invocations in the same second (e.g. chaining eval behind two
            # training jobs) otherwise write the same script path, since _n
            # restarts at 1 each time.
            _ckpt_tag="$(basename "$(dirname "${_ckpt}")")_$(basename "${_ckpt}")"
            _job_tag="${TIMESTAMP}_eval_${EVAL_STYLE}_${_ckpt_tag}_${_ds}_${_n}"
            s="${SLURM_JOB_DIR}/${_job_tag}.sh"
            if $IS_HPC; then
                _submit_hpc_one "${s}"
                echo "[job ${_n}] dataset=${_ds}  ckpt=${_ckpt}"
                echo "         script: ${s}"
            else
                export HF_TOKEN="${HF_TOKEN}"
                export OMP_NUM_THREADS=8
                export TOKENIZERS_PARALLELISM=false
                cd "${PROJ_DIR}"
                echo "[job ${_n}] Launching eval locally: dataset=${_ds}  ckpt=${_ckpt}"
                python ${TRAIN_CMD}
            fi
        done
    done
    if $IS_HPC; then
        echo "Submitted ${_n} eval job(s) to ${CLUSTER_LABEL[${PLATFORM}]}"
        echo "  Status:  squeue -u \$USER"
        echo "  Logs:    ${SLURM_LOG_DIR}/"
    fi
else
    _job_tag="${TIMESTAMP}_recursivemas_r${N_ROUNDS}_s${STEPS}"
    s="${SLURM_JOB_DIR}/${_job_tag}.sh"
    if $IS_HPC; then
        _submit_hpc_one "${s}"
        echo "Submitted to ${CLUSTER_LABEL[${PLATFORM}]}"
        echo "  Status:  squeue -u \$USER"
        echo "  Logs:    ${SLURM_LOG_DIR}/"
        echo "  Script:  ${s}"
    else
        export HF_TOKEN="${HF_TOKEN}"
        export OMP_NUM_THREADS=8
        export TOKENIZERS_PARALLELISM=false

        cd "${PROJ_DIR}"

        if [[ ${NUM_GPUS} -gt 1 ]]; then
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
fi
