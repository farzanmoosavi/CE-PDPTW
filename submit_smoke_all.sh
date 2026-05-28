#!/bin/bash
# ============================================================
# Submit smoke tests for all rungs (A B C D) in parallel.
# Run on the LOGIN NODE: bash submit_smoke_all.sh
#
# Each rung gets its own SLURM job (independent, safe to run in parallel).
# Smoke test: 5 epochs, tiny data — validates the full pipeline
# (data → encoder → decoder → REINFORCE loss → rollout baseline
#  t-test update → checkpoint → validation cost).
#
# Options:
#   CHAIN=1   — auto-submit full training (submit_final_runs.sh) after
#               ALL smoke tests pass (uses --dependency=afterok:JA:JB:JC:JD)
#   VARIANT=  — passed to submit_final_runs.sh when CHAIN=1 (default: rloo4)
#   RUNGS=    — subset of rungs to test, e.g. RUNGS="A B" (default: A B C D)
#
# Examples:
#   bash submit_smoke_all.sh                  # test all 4 rungs
#   bash submit_smoke_all.sh CHAIN=1          # test + auto-start training if all pass
#   RUNGS="C D" bash submit_smoke_all.sh      # test rungs C and D only
#   CHAIN=1 VARIANT=rloo8 bash submit_smoke_all.sh
# ============================================================

set -euo pipefail

CHAIN=${CHAIN:-0}
VARIANT=${VARIANT:-rloo4}
RUNGS=${RUNGS:-"A B C D"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_SCRIPT="$SCRIPT_DIR/submit_smoke_rl.sh"
TRAIN_SCRIPT="$SCRIPT_DIR/submit_final_runs.sh"

[ -f "$SMOKE_SCRIPT" ] || { echo "ERROR: $SMOKE_SCRIPT not found."; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found — run on a login node."; exit 1; }

echo "========================================================"
echo " CE-PDPTW smoke tests | Rungs: $RUNGS | $(date)"
echo " Chain to full training: $([ "$CHAIN" = "1" ] && echo "YES (variant=$VARIANT)" || echo no)"
echo "========================================================"
echo ""

SMOKE_JOB_IDS=()

for RUNG in $RUNGS; do
    JOB=$(sbatch --export="RUNG=$RUNG" "$SMOKE_SCRIPT" | awk '{print $NF}')
    echo "  Rung $RUNG  smoke → job $JOB"
    SMOKE_JOB_IDS+=("$JOB")
done

echo ""

# ── Optional: chain full training once all smoke tests pass ──
if [ "$CHAIN" = "1" ]; then
    [ -f "$TRAIN_SCRIPT" ] || { echo "ERROR: $TRAIN_SCRIPT not found — cannot chain."; exit 1; }

    # Build afterok dependency string: afterok:JA:JB:JC:JD
    DEP="afterok:$(IFS=:; echo "${SMOKE_JOB_IDS[*]}")"

    echo "  Submitting full training with dependency: $DEP"
    # submit_final_runs.sh is a login-node script that calls sbatch internally,
    # so we wrap it in a minimal job that runs on the login node or a short queue.
    # Simplest: submit a tiny wrapper job that calls bash submit_final_runs.sh
    CHAIN_JOB=$(sbatch \
        --job-name=CE-PDPTW-chain \
        --account=def-bfarooq \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
        --time=00:10:00 \
        --dependency="$DEP" \
        --output=logs/chain-%j.out \
        --error=logs/chain-%j.out \
        --wrap="VARIANT=$VARIANT bash $TRAIN_SCRIPT" \
        | awk '{print $NF}')
    echo "  Chain launcher    → job $CHAIN_JOB  (dep: $DEP)"
    echo ""
    echo "  If all smoke tests pass, full training starts automatically."
    echo "  If any smoke test FAILS, the chain job will not run."
fi

# ── How to check results ──────────────────────────────────────
echo ""
echo "========================================================"
echo " Monitor:    squeue -u \$USER"
echo ""
echo " After jobs complete, check verdicts:"
for RUNG in $RUNGS; do
    echo "   grep VERDICT logs/smoke-rl-*.out | grep -i rung_${RUNG} || \\"
    echo "     tail -20 logs/smoke-rl-<jobid>.out"
done
echo ""
echo " Quick pass/fail scan (once all jobs finish):"
echo "   grep -h 'VERDICT\\|SMOKE TEST' logs/smoke-rl-*.out"
echo ""
echo " If all pass, start training:"
echo "   bash submit_final_runs.sh                   # rloo4, all rungs"
echo "   VARIANT=rloo8 bash submit_final_runs.sh"
echo "========================================================"
