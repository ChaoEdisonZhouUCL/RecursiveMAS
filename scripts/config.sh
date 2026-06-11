# config.sh — Defaults and platform configuration for RecursiveMAS training.
#             Sourced by submit.sh before CLI parsing.

# =============================================================================
# Platform Detection
# =============================================================================

detect_platform() {
    if   [[ -n "${SLURM_JOB_ID:-}" ]];                                                          then echo "slurm_job"
    elif command -v sbatch &>/dev/null && [[ $(hostname) == *"juwels"* ]];                       then echo "julich"
    elif command -v sbatch &>/dev/null && [[ $(hostname) == *"jureca"* ]];                       then echo "jureca"
    elif command -v sbatch &>/dev/null && [[ $(hostname) == *"cispa"* || -d "/home/c01chzh" ]];  then echo "cispa"
    else                                                                                              echo "local"
    fi
}

# =============================================================================
# Defaults
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Infrastructure ────────────────────────────────────────────────────────────
NUM_GPUS=3
NUM_NODES=1
PARTITION=""
CONTAINER="projects.cispa.saarland:5005#c01chzh/recursivemas_docker:latest"

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ROUNDS=2
LATENT_STEPS=48
BATCH_SIZE=4
STEPS=10000
LR="5e-4"
DTYPE="bfloat16"
MODE="original"        # original | shared_roae | compare
N_EXPERTS=3
EXPERT_DIM_DIVISOR=3
NO_KV_CACHE=false          # set true when latent_steps >= 20 (paper uses 80)
DATASET="s1k+m1k"          # math500 | s1k | m1k | s1k+m1k (pooled)
MAX_SEQ_LEN=4096              # max combined question+answer length in tokens (0 = no truncation)

# ── HF token ─────────────────────────────────────────────────────────────────
# Set HF_TOKEN in your environment before running (e.g. export HF_TOKEN=hf_...)
# HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN=""
# ── Per-platform SLURM metadata ───────────────────────────────────────────────
declare -A SLURM_ACCOUNT=([julich]="hai_1293"     [jureca]="hai_1129")
declare -A SLURM_PART=(   [julich]="booster"      [jureca]="dc-hwai"   [cispa]="xe8545")
declare -A SLURM_TIME=(   [julich]="23:59:59"      [jureca]="23:59:59"  [cispa]="2-1:00:00")
declare -A CLUSTER_LABEL=([julich]="JUWELS"        [jureca]="JURECA"    [cispa]="CISPA")
