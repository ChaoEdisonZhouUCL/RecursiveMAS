# slurm.sh — SLURM script writers for each cluster.
#            Sourced by submit.sh. Requires variables from config.sh and
#            path variables set in submit.sh (PROJ_DIR, SLURM_LOG_DIR, etc.).
#
# Design: each writer uses a mix of heredoc quoting to separate submit-time
# from runtime values — no sed token substitution needed.
#   - double-quoted EOF  : values expand immediately (SLURM header, injected vars)
#   - single-quoted EOF  : values stay literal; expand when the job runs

# =============================================================================
# CISPA (enroot container via --container-image)
# =============================================================================

_write_slurm_cispa() {
    local s=$1
    local partition="${PARTITION:-${SLURM_PART[cispa]}}"

    # ── SLURM header ──────────────────────────────────────────────────────────
    cat > "$s" << EOF
#!/bin/bash
#SBATCH --job-name=recursivemas_outerlinks_r${N_ROUNDS}_s${STEPS}
#SBATCH --output=${SLURM_LOG_DIR}/job-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/job-%j.err
#SBATCH --partition=${partition}
#SBATCH --exclude=xe8545-a100-04
#SBATCH --time=${SLURM_TIME[cispa]}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:${NUM_GPUS}
#SBATCH --cpus-per-task=$((NUM_GPUS * 12))
EOF

    # ── Runtime env setup ─────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'
if [ ! -f ~/.config/enroot/.credentials ]; then
    mkdir -p ~/.config/enroot/
    ln -s ~/CISPA-home/.config/enroot/.credentials ~/.config/enroot/.credentials 2>/dev/null || true
fi
export PYTHONUNBUFFERED=1
SLURM_EOF

    # ── Inject submit-time values ─────────────────────────────────────────────
    cat >> "$s" << EOF
PROJ_DIR="${PROJ_DIR}"
CONTAINER="${CONTAINER}"
NUM_GPUS=${NUM_GPUS}
HF_TOKEN="${HF_TOKEN}"
TRAIN_CMD="${TRAIN_CMD}"
MASTER_PORT=\$((10000 + RANDOM % 20000))
echo "Job: \${SLURM_JOB_ID}  Node: \${SLURM_NODELIST}"
cd "\${PROJ_DIR}"
EOF

    # ── Runtime body ──────────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HOME}/CISPA-projects/pt_network-2024/.huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

srun --unbuffered \
     --container-image="${CONTAINER}" \
     --container-mounts="${PROJ_DIR}":/workspace \
     conda run --no-capture-output -n base bash -c \
     "pip install -r /workspace/requirements.txt -q && \
      export MASTER_PORT=${MASTER_PORT} && \
      export HF_TOKEN=${HF_TOKEN} && \
      torchrun \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=127.0.0.1 \
        --master_port=${MASTER_PORT} \
        ${TRAIN_CMD}"

echo "Job completed."
SLURM_EOF
}

# =============================================================================
# JUWELS (direct torchrun, offline HF cache)
# =============================================================================

_write_slurm_julich() {
    local s=$1
    local partition="${PARTITION:-${SLURM_PART[julich]}}"

    # ── SLURM header ──────────────────────────────────────────────────────────
    cat > "$s" << EOF
#!/bin/bash -x
#SBATCH --account=${SLURM_ACCOUNT[julich]}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$((NUM_GPUS * 12))
#SBATCH --output=${SLURM_LOG_DIR}/job-%j/job-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/job-%j/job-%j.err
#SBATCH --partition=${partition}
#SBATCH --time=${SLURM_TIME[julich]}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --job-name=recursivemas_outerlinks_r${N_ROUNDS}_s${STEPS}
EOF

    # ── Runtime env ───────────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((10000 + RANDOM % 20000))
_TRITON_SCRATCH="/p/scratch/spare-ml/zhou17/.triton/autotune"
if mkdir -p "${_TRITON_SCRATCH}" 2>/dev/null; then
    export TRITON_CACHE_DIR="${_TRITON_SCRATCH}"
else
    export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
    mkdir -p "${TRITON_CACHE_DIR}"
fi
SLURM_EOF

    # ── Inject submit-time values ─────────────────────────────────────────────
    cat >> "$s" << EOF
PROJ_DIR="${PROJ_DIR}"
NUM_GPUS_N=${NUM_GPUS}
HF_TOKEN="${HF_TOKEN}"
TRAIN_CMD="${TRAIN_CMD}"
EOF

    # HF: booster has no internet → offline mode
    if [[ "${partition}" == "booster" ]]; then
        cat >> "$s" << 'EOF'
export HF_HOME="/p/project1/spare-ml/zhou17/pretrain_dynamic_analysis/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
    else
        cat >> "$s" << 'EOF'
export HF_HOME="/p/project1/spare-ml/zhou17/pretrain_dynamic_analysis/hf_cache"
EOF
    fi

    # ── Runtime body ──────────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'

cd "${PROJ_DIR}"

_BASE_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
source "$(dirname "${PROJ_DIR}")/pretrain_dynamic_analysis/lm/scripts/detect_net.sh" "${_BASE_HOST}"
export MASTER_ADDR

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN}"
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE="${NUM_GPUS_N}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}"
echo "Command: torchrun --nproc_per_node=${GPUS_PER_NODE} ${TRAIN_CMD}"
echo "============================================================"

pip install -r "${PROJ_DIR}/requirements.txt" -q

if [[ ${SLURM_NNODES} -gt 1 ]]; then
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME="${NET_IFNAME}"
    export GLOO_SOCKET_IFNAME="${NET_IFNAME}"
    echo "[net] MASTER_ADDR=${MASTER_ADDR} NET_IFNAME=${NET_IFNAME}"
    srun --ntasks-per-node=1 bash -c '
        export NODE_RANK=${SLURM_NODEID}
        echo "HOST=$(hostname) NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
        torchrun \
            --nnodes=${NNODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${TRAIN_CMD}
    ' &
else
    torchrun \
        --nnodes=1 \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_addr=127.0.0.1 \
        --master_port=${MASTER_PORT} \
        ${TRAIN_CMD} &
fi

TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_EXIT=$?
echo "Training exit code: ${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
SLURM_EOF
}

# =============================================================================
# JURECA (direct torchrun, offline HF cache)
# =============================================================================

_write_slurm_jureca() {
    local s=$1
    local platform="jureca"
    local partition="${PARTITION:-${SLURM_PART[$platform]}}"

    # ── SLURM header ──────────────────────────────────────────────────────────
    cat > "$s" << EOF
#!/bin/bash -x
#SBATCH --account=${SLURM_ACCOUNT[$platform]}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$((NUM_GPUS * 12))
#SBATCH --output=${SLURM_LOG_DIR}/job-%j/job-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/job-%j/job-%j.err
#SBATCH --partition=${partition}
#SBATCH --time=${SLURM_TIME[$platform]}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --job-name=recursivemas_outerlinks_r${N_ROUNDS}_s${STEPS}
EOF

    # ── Runtime env ───────────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
SLURM_EOF

    # ── Inject submit-time values ─────────────────────────────────────────────
    cat >> "$s" << EOF
PROJ_DIR="${PROJ_DIR}"
NUM_GPUS=${NUM_GPUS}
HF_TOKEN="${HF_TOKEN}"
TRAIN_CMD="${TRAIN_CMD}"
MASTER_ADDR="\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="\$(getent ahostsv4 "\${MASTER_ADDR}" | awk 'NR==1{print \$1}')"
MASTER_PORT=\$((10000 + RANDOM % 20000))
echo "Job: \${SLURM_JOB_ID}  Node: \${SLURM_NODELIST}"
cd "\${PROJ_DIR}"
EOF

    # ── Runtime body ──────────────────────────────────────────────────────────
    cat >> "$s" << 'SLURM_EOF'
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/p/home/jusers/zhou17/jureca/hai_1129/pretrain_dynamic_analysis/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_TOKEN="${HF_TOKEN}"
export NNODES="${SLURM_NNODES}"
export GPUS_PER_NODE="${NUM_GPUS}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "============================================================"
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}"
echo "Command: torchrun --nproc_per_node=${GPUS_PER_NODE} ${TRAIN_CMD}"
echo "============================================================"

pip install -r "${PROJ_DIR}/requirements.txt" -q

if [[ $SLURM_NNODES -gt 1 ]]; then
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=^lo
    export GLOO_SOCKET_IFNAME=^lo
    srun --ntasks-per-node=1 bash -c \
        'export NODE_RANK=${SLURM_NODEID}
         echo "HOST=$(hostname) NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
         torchrun \
            --nnodes=${NNODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${TRAIN_CMD}'
else
    torchrun \
        --nnodes=1 \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_addr=127.0.0.1 \
        --master_port=${MASTER_PORT} \
        ${TRAIN_CMD}
fi
echo "Job completed."
SLURM_EOF
}
