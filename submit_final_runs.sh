#!/bin/bash
# ============================================================
# Submit the full curriculum chain A→B→C→D as dependent SLURM jobs.
# Run on the LOGIN NODE: bash submit_final_runs.sh
#
# Each rung waits for the previous to succeed (--dependency=afterok).
# Internally calls submit_nibi.sh with the chosen variant flags.
#
# Training variants (VARIANT=):
#   rloo4     — RLOO k=4, default (recommended)
#   rloo8     — RLOO k=8; uses lr=7e-5 to prevent late divergence
#   rollout   — plain rollout baseline (bootstrap catch-22 risk at init)
#   simplegat — SimpleGAT encoder ablation vs HetGAT
#   dynamic   — on-demand training: random partial-order visibility per batch
#
# Options:
#   START_RUNG=B   — start chain from rung B (rung A already done)
#   EPOCHS=300     — override default 200 epochs per rung
#   FRESH=1        — wipe the START_RUNG checkpoint before submitting
#   EVAL=1         — auto-submit submit_evaluate.sh after each training rung
#   WARMSTART=1    — warm-start each rung from the previous best_actor.pt
#
# Examples:
#   bash submit_final_runs.sh
#   VARIANT=rloo8 bash submit_final_runs.sh
#   VARIANT=rloo4 START_RUNG=B bash submit_final_runs.sh
#   FRESH=1 EVAL=1 bash submit_final_runs.sh
#   VARIANT=dynamic EPOCHS=300 bash submit_final_runs.sh
# ============================================================

set -euo pipefail

VARIANT=${VARIANT:-rloo4}
START_RUNG=${START_RUNG:-A}
EPOCHS=${EPOCHS:-200}
FRESH=${FRESH:-0}
EVAL=${EVAL:-0}
WARMSTART=${WARMSTART:-0}

# ── Translate variant → sbatch export flags ──────────────────
case "$VARIANT" in
    rloo4)     VARIANT_EXPORT="RLOO_K=4" ;;
    rloo8)     VARIANT_EXPORT="RLOO_K=8,LR=7e-5" ;;
    rollout)   VARIANT_EXPORT="RLOO_K=0" ;;
    simplegat) VARIANT_EXPORT="RLOO_K=0,ARCH=simplegat" ;;
    dynamic)   VARIANT_EXPORT="RLOO_K=4,DYNAMIC=1" ;;
    *)
        echo "Unknown VARIANT=$VARIANT."
        echo "Valid options: rloo4  rloo8  rollout  simplegat  dynamic"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/submit_nibi.sh"
EVAL_SCRIPT="$SCRIPT_DIR/submit_evaluate.sh"

[ -f "$TRAIN_SCRIPT" ] || { echo "ERROR: $TRAIN_SCRIPT not found."; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found — run this on a login node."; exit 1; }

# ── Build active rung list ────────────────────────────────────
ALL_RUNGS=(A B C D)
ACTIVE_RUNGS=()
SEEN=0
for r in "${ALL_RUNGS[@]}"; do
    [ "$r" = "$START_RUNG" ] && SEEN=1
    [ "$SEEN" = "1" ] && ACTIVE_RUNGS+=("$r")
done
if [ ${#ACTIVE_RUNGS[@]} -eq 0 ]; then
    echo "ERROR: START_RUNG=$START_RUNG is not valid. Use A, B, C, or D."; exit 1
fi

echo "========================================================"
echo " CE-PDPTW curriculum chain"
echo " Variant  : $VARIANT  ($VARIANT_EXPORT)"
echo " Rungs    : ${ACTIVE_RUNGS[*]}"
echo " Epochs   : $EPOCHS | Fresh: $FRESH (start rung only) | Warmstart: $WARMSTART"
echo " Eval     : $([ "$EVAL" = "1" ] && echo "yes — submit_evaluate.sh after each rung" || echo no)"
echo "========================================================"
echo ""

PREV_TRAIN_JOB=""
FIRST_RUNG=1

for RUNG in "${ACTIVE_RUNGS[@]}"; do
    # FRESH only applies to the first submitted rung — later rungs have no checkpoint yet
    _FRESH=0
    [ "$FIRST_RUNG" = "1" ] && _FRESH="$FRESH"

    EXPORT="RUNG=$RUNG,EPOCHS=$EPOCHS,FRESH=${_FRESH},WARMSTART=$WARMSTART,${VARIANT_EXPORT}"

    DEP_FLAG=""
    [ -n "$PREV_TRAIN_JOB" ] && DEP_FLAG="--dependency=afterok:$PREV_TRAIN_JOB"

    TRAIN_JOB=$(sbatch $DEP_FLAG --export="$EXPORT" "$TRAIN_SCRIPT" | awk '{print $NF}')
    echo "  Rung $RUNG  train → job $TRAIN_JOB  (dep: ${PREV_TRAIN_JOB:-none})"
    PREV_TRAIN_JOB="$TRAIN_JOB"

    if [ "$EVAL" = "1" ]; then
        EVAL_EXPORT="RUNG=$RUNG,${VARIANT_EXPORT}"
        EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB --export="$EVAL_EXPORT" "$EVAL_SCRIPT" | awk '{print $NF}')
        echo "  Rung $RUNG  eval  → job $EVAL_JOB   (dep: $TRAIN_JOB)"
    fi

    FIRST_RUNG=0
done

echo ""
echo "========================================================"
echo " Monitor : squeue -u \$USER"
echo " Logs    : logs/train-CE-PDPTW-train-<jobid>.out"
[ "$EVAL" = "1" ] && echo " Eval    : logs/eval-CE-PDPTW-eval-<jobid>.out"
echo "========================================================"
