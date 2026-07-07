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

# ── Eval mode ────────────────────────────────────────────────────────────────
EVAL=false
CKPT_PATH="/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_roae_r3_skip_20260702_184340/step_5000"          # "released_weights" | path containing _original | path containing _shared_roae
EVAL_DATASET="math500"
EVAL_BATCH_SIZE=32
EVAL_SEED=42

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ROUNDS=3
LATENT_STEPS=48
BATCH_SIZE=1
STEPS=10000
LR="5e-4"
DTYPE="bfloat16"
MODE="shared_roae"        # original | shared_roae | compare | compare_roundskip
USE_ROUND_SKIP=false             # true | false — set false to freeze beta=0 (ablation; shared_roae only)
N_EXPERTS=3
EXPERT_DIM_DIVISOR=3
NO_KV_CACHE=false          # set true when latent_steps >= 20 (paper uses 80)
DATASET="s1k+m1k+opencodereasoning+arpo_sft"          # math500 | s1k | m1k | opencodereasoning | arpo_sft | s1k+m1k+opencodereasoning+arpo_sft (pooled)
MAX_SEQ_LEN=4096              # max combined question+answer length in tokens (0 = no truncation)
N_SAMPLES=4000                 # training problems per epoch (0 = use full dataset)
N_CKPT=3                      # checkpoints saved during training (evenly spaced; 1 = end only)
RESUME_CKPT=""                # path to a step_N ckpt dir to resume from (empty = start fresh)
GRAD_CHECKPOINT=false         # true = gradient checkpointing (saves activation memory, ~33% slower)
GRAD_ACCUM=4                 # gradient accumulation steps; effective batch = BATCH_SIZE * GRAD_ACCUM

# ── HF token ─────────────────────────────────────────────────────────────────
# Set HF_TOKEN in your environment before running (e.g. export HF_TOKEN=hf_...)
# HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN=""
# ── Per-platform SLURM metadata ───────────────────────────────────────────────
declare -A SLURM_ACCOUNT=([julich]="hai_1293"     [jureca]="hai_1129")
declare -A SLURM_PART=(   [julich]="booster"      [jureca]="dc-hwai"   [cispa]="xe8545")
declare -A SLURM_TIME=(   [julich]="23:59:59"      [jureca]="23:59:59"  [cispa]="2-1:00:00")
declare -A CLUSTER_LABEL=([julich]="JUWELS"        [jureca]="JURECA"    [cispa]="CISPA")
