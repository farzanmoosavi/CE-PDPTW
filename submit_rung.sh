#!/bin/bash
# ============================================================
# Submit a single-rung training job with a chosen variant.
# Run on the LOGIN NODE: bash submit_rung.sh <RUNG> [VARIANT]
#
# Arguments:
#   RUNG     : A | B | C | D  (required)
#   VARIANT  : rloo4 | rloo8 | rollout | simplegat | dynamic
#              (default: rloo4)
#
# Options (env vars):
#   EPOCHS=300       override epoch count (default: 200)
#   FRESH=1          wipe existing checkpoint before starting
#   WARMSTART=1      warm-start from previous rung best_actor.pt
#
# Examples:
#   bash submit_rung.sh A
#   bash submit_rung.sh B rloo8
#   bash submit_rung.sh C simplegat
#   bash submit_rung.sh D dynamic
#   bash submit_rung.sh A rollout
#   FRESH=1 bash submit_rung.sh B rloo4
#   EPOCHS=300 bash submit_rung.sh C rloo4
# ============================================================

set -euo pipefail

RUNG=${1:-}
VARIANT=${2:-rloo4}
EPOCHS=${EPOCHS:-200}
FRESH=${FRESH:-0}
WARMSTART=${WARMSTART:-0}

# ── Validate rung ─────────────────────────────────────────────
case "$RUNG" in
    A|B|C|D) ;;
    *)
        echo "Usage: bash submit_rung.sh <RUNG> [VARIANT]"
        echo "  RUNG    : A | B | C | D"
        echo "  VARIANT : rloo4 | rloo8 | rollout | simplegat | dynamic"
        exit 1
        ;;
esac

# ── Translate variant → export flags ─────────────────────────
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

[ -f "$TRAIN_SCRIPT" ] || { echo "ERROR: $TRAIN_SCRIPT not found."; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found — run on a login node."; exit 1; }

EXPORT="RUNG=$RUNG,EPOCHS=$EPOCHS,FRESH=$FRESH,WARMSTART=$WARMSTART,${VARIANT_EXPORT}"

echo "========================================================"
echo " CE-PDPTW single-rung submit"
echo " Rung    : $RUNG"
echo " Variant : $VARIANT  ($VARIANT_EXPORT)"
echo " Epochs  : $EPOCHS | Fresh: $FRESH | Warmstart: $WARMSTART"
echo "========================================================"
echo ""

JOB=$(sbatch --export="$EXPORT" "$TRAIN_SCRIPT" | awk '{print $NF}')
echo "  Job submitted → $JOB"
echo ""
echo "  Monitor : squeue -u \$USER"
echo "  Log     : logs/train-CE-PDPTW-train-${JOB}.out"
echo "  CSV     : logs/training_${RUNG}-${VARIANT}.csv"
