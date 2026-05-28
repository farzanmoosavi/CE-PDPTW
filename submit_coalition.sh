#!/bin/bash
# ============================================================
# Narval — Coalition cost sweep across all curriculum rungs.
# Runs run_coalition.py for each rung (A→D) so the reviewer
# sees coalition benefit across all problem scales (n_req=5..60).
# Greedy sweep runs CPU-only (parallel workers).
# RL comparison uses 1 GPU (model auto-detects CUDA); workers=1 to avoid
# forking the CUDA context.
#
# Usage:
#   sbatch submit_coalition.sh                      # all rungs, greedy only
#   sbatch --export=WITH_RL=1 submit_coalition.sh   # + RL vs Greedy comparison
#   sbatch --export=RUNG=C submit_coalition.sh       # single rung only
# ============================================================

#SBATCH --job-name=CE-PDPTW-coalition
#SBATCH --account=def-farooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/coalition-%j.out
#SBATCH --error=logs/coalition-%j.err

module purge
module load python/3.10 scipy-stack cuda/12.2
source ~/py310_nibi/bin/activate

_PROJ="$HOME/projects/def-farooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-farooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-farooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs eval_results

EPISODES=${EPISODES:-200}
WITH_RL=${WITH_RL:-0}

# ── Helper: run one rung ─────────────────────────────────────
run_rung() {
    local rung=$1
    local rl_model="CE-PDPTW-${rung}-rloo4-HetGAT/best_actor.pt"
    [ -f "$rl_model" ] || rl_model="CE-PDPTW-${rung}-HetGAT/best_actor.pt"
    local out_dir="eval_results/coalition_${rung}"

    echo ""
    echo "=== Rung $rung: greedy sweep (108 scenarios × $EPISODES episodes) ==="
    python run_coalition.py \
        --rung      "$rung" \
        --episodes  "$EPISODES" \
        --delta     10 \
        --seed      9999 \
        --workers   8 \
        --out-dir   "$out_dir"

    if [ "${WITH_RL}" = "1" ]; then
        if [ -f "$rl_model" ]; then
            echo "--- Rung $rung: RL vs Greedy comparison ---"
            python run_coalition.py \
                --rung        "$rung" \
                --episodes    "$EPISODES" \
                --delta       10 \
                --seed        9999 \
                --workers     1 \
                --rl-model    "$rl_model" \
                --out-dir     "$out_dir"
        else
            echo "WARNING: Rung $rung RL model not found ($rl_model) — skipping RL comparison."
        fi
    fi
    echo "Rung $rung done at $(date)"
}

echo "========================================================"
echo " CE-PDPTW Coalition sweep | $(date)"
echo " Episodes: $EPISODES | Scenarios: 108 | Workers: 8"
echo " Seed: 9999 | RL comparison: ${WITH_RL}"
echo "========================================================"

# ── Single-rung mode ─────────────────────────────────────────
if [ -n "${RUNG}" ]; then
    run_rung "$RUNG"
    echo ""
    echo "Results: eval_results/coalition_${RUNG}/"
    exit 0
fi

# ── All-rungs mode ───────────────────────────────────────────
for rung in A B C D; do
    run_rung "$rung"
done

echo ""
echo "========================================================"
echo " All coalition sweeps complete: $(date)"
echo " Results:"
for rung in A B C D; do
    echo "   eval_results/coalition_${rung}/coalition_sweep_greedy.csv"
done
echo "========================================================"
