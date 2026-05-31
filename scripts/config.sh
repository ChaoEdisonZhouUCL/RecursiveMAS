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
CONTAINER="projects.cispa.saarland:5005#c01chzh/nanoadam:latest"

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ROUNDS=3
LATENT_STEPS=8
STEPS=100
LR="1e-4"
DTYPE="bfloat16"

# ── HF token ─────────────────────────────────────────────────────────────────
# Set HF_TOKEN in your environment before running (e.g. export HF_TOKEN=hf_...)
HF_TOKEN="${HF_TOKEN:-}"

# ── Per-platform SLURM metadata ───────────────────────────────────────────────
declare -A SLURM_ACCOUNT=([julich]="hai_1293"     [jureca]="hai_1129")
declare -A SLURM_PART=(   [julich]="booster"      [jureca]="dc-hwai"   [cispa]="xe8545")
declare -A SLURM_TIME=(   [julich]="23:59:59"      [jureca]="23:59:59"  [cispa]="2-1:00:00")
declare -A CLUSTER_LABEL=([julich]="JUWELS"        [jureca]="JURECA"    [cispa]="CISPA")
