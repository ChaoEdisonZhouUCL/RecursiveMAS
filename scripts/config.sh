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
DEPENDENCY=""                 # sbatch --dependency=SPEC (e.g. afterok:JOBID); empty = none
CONTAINER="projects.cispa.saarland:5005#c01chzh/recursivemas_docker:latest"

# ── Eval mode ────────────────────────────────────────────────────────────────
# CKPT_PATH and EVAL_DATASET take a single value, a comma-separated string, or
# a bash array, e.g. EVAL_DATASET=("math500" "medqa"). One job is submitted per
# (checkpoint, dataset) combination.
EVAL=false
CKPT_PATH=("/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_tied_r3_noskip_20260721_102613/step_600"
            "/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_tied_r3_noskip_20260721_102613/step_1200"
            "/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_tied_r3_noskip_20260721_102613/step_1800"
            "/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_tied_r3_noskip_20260721_102613/step_2400"
            "/p/project1/hai_1354/RecursiveMAS/outputs/checkpoints/outerlink_grad_shared_tied_r3_noskip_20260721_102613/step_3000"
            )          # "released_weights" | path containing _original | path containing _shared_roae
EVAL_DATASET=("math500" "medqa" "aime2025" "aime2026" "gpqa" "mbppplus" "livecodebench")          # math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench
# EVAL_DATASET=("math500")          # math500 | medqa | aime2025 | aime2026 | gpqa | mbppplus | livecodebench
EVAL_BATCH_SIZE=32
EVAL_SEED=42
EVAL_LATENT_STEPS="48"          # latent steps for eval (empty = match LATENT_STEPS below; 0 = sweep 16/32/48; "protocol" = per-dataset value from eval_protocol.yaml)
EVAL_N_ROUNDS="3"                # recursion rounds for eval (empty = match N_ROUNDS below)
EVAL_GREEDY=false               # true = greedy decoding (reproducible; ignores temperature/top_p)
EVAL_SELF_INJECT=false          # true = each agent re-reads its own previous-round latent thought

# ── Training hyperparameters ──────────────────────────────────────────────────
N_ROUNDS=3
LATENT_STEPS=48
BATCH_SIZE=1
STEPS=3000
LR="1e-4"
DTYPE="bfloat16"
# dtype of the TRAINED link only; frozen backbones stay at DTYPE.  float32 is
# the default because the link sums one gradient contribution per round into a
# single buffer and those span 5-6 orders of magnitude — bfloat16 addition
# (eps 7.8e-3) drops most of the early rounds.  "same" = old behaviour.
# See ROUND_GRADIENT_WEIGHTING.md.
LINK_DTYPE="float32"      # float32 | bfloat16 | float16 | same
MODE="shared_roae"        # original | shared_roae | shared_state | shared_tied | compare | compare_roundskip
                          # shared_state = shared_roae + residual recursive state z bypassing
                          # each full round: z' = z + gamma*F(z), gamma ReZero-init (~identity)
                          # shared_tied  = shared_roae with tied decoders O_i = S_i^T
                          # (semi-orthogonal round trips at init; no independent out_proj)
USE_ROUND_SKIP=false             # true | false — set false to freeze beta=0 (ablation; shared_roae only)
N_EXPERTS=3
EXPERT_DIM_DIVISOR=3
# Initial recursive-state write gate (shared_state only).  z' = z + GAMMA_INIT*f,
# so this sets how strongly rounds after the first write into the state -- and
# the per-stage round1/round2 gradient ratio is 1/GAMMA_INIT.
GAMMA_INIT="1e-3"

# Train with self-injection: each agent re-reads its own previous-round latent
# thought, spliced into its prompt behind a role-specific label.  Round 0 injects
# nothing.  TRAIN_SELF_INJECT_GRAD additionally keeps that block in the autograd
# graph (a gradient short path across the round boundary); it implies the former.
# Checkpoint directory prefix.  The run directory is
# <OUT_PREFIX>_<MODE>_r<N>[_noskip]_<timestamp>, and the timestamp is generated when
# the job starts -- so arms launched together must use distinct prefixes or they can
# land in the same directory.
OUT_PREFIX="outerlink_grad"

TRAIN_SELF_INJECT=false
TRAIN_SELF_INJECT_GRAD=false
NO_KV_CACHE=false          # set true when latent_steps >= 20 (paper uses 80)
DATASET="s1k+m1k+opencodereasoning+arpo_sft"          # math500 | s1k | m1k | opencodereasoning | arpo_sft | s1k+m1k+opencodereasoning+arpo_sft (pooled)
MAX_SEQ_LEN=4096              # max combined question+answer length in tokens (0 = no truncation)
N_SAMPLES=0                 # training problems per epoch (0 = use full dataset)
N_CKPT=5                      # checkpoints saved during training (evenly spaced; 1 = end only)
RESUME_CKPT=""                # path to a step_N ckpt dir to resume from (empty = start fresh)
GRAD_CHECKPOINT=false         # true = gradient checkpointing (saves activation memory, ~33% slower)
GRAD_ACCUM=4                 # gradient accumulation steps; effective batch = BATCH_SIZE * GRAD_ACCUM
WANDB=true                   # true = log training dynamics to Weights & Biases (loss, per-link
                              # grad norms, gate values, hyperparameters); one run per (mode, n_rounds)
WANDB_PROJECT="recursivemas-outerlinks"
WANDB_ENTITY=""                # empty = your default W&B entity
WANDB_RUN_NAME=""              # empty = auto-generated from mode/n_rounds/timestamp
WANDB_MODE="offline"           # online | offline | disabled — compute nodes here have no internet,
                              # so offline (wandb sync later) is the default

# ── HF token ─────────────────────────────────────────────────────────────────
# Set HF_TOKEN in your environment before running (e.g. export HF_TOKEN=hf_...)
# HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN=""
# ── Per-platform SLURM metadata ───────────────────────────────────────────────
declare -A SLURM_ACCOUNT=([julich]="hai_1293"     [jureca]="hai_1354")
declare -A SLURM_PART=(   [julich]="booster"      [jureca]="dc-hwai"   [cispa]="xe8545")
declare -A SLURM_TIME=(   [julich]="23:59:59"      [jureca]="23:59:59"  [cispa]="2-1:00:00")
declare -A CLUSTER_LABEL=([julich]="JUWELS"        [jureca]="JURECA"    [cispa]="CISPA")
