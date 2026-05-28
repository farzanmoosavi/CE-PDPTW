#!/bin/bash
# ============================================================
# Narval/Nibi — on-demand dispatch evaluation across Δt values.
#
# Runs benchmark_delta_sweep.py for one rung at Δ ∈ {2,5,10,15,20} min
# with: RL (HetGAT), RL (SimpleGAT, if checkpoint exists), ALNS, Greedy.
# For Rungs A/B: also adds Gurobi (pass --export=GUROBI=1).
# For Rung D: greedy+alns+RL only (exact solvers not tractable).
#
# All methods run on identical instances (seed=9999, instance_i = seed+i)
# so comparisons are fully apples-to-apples.
#
# Usage:
#   sbatch --export=RUNG=C submit_delta_sweep.sh          # RL (HetGAT) + ALNS + Greedy
#   sbatch --export=RUNG=B,GUROBI=1 submit_delta_sweep.sh # + Gurobi for small rung
#   sbatch --export=RUNG=D submit_delta_sweep.sh          # ALNS + Greedy only (no exact)
#
# Dependencies: run AFTER training jobs finish (best_actor.pt must exist).
# ============================================================

#SBATCH --job-name=CE-PDPTW-delta-sweep
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/delta-sweep-%j.out
#SBATCH --error=logs/delta-sweep-%j.err

module purge
module load python/3.10 scipy-stack cuda/12.2
source ~/py310_nibi/bin/activate

export MPLBACKEND=Agg

_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs eval_results

RUNG=${RUNG:-C}
GUROBI=${GUROBI:-0}
N_INST=${N_INST:-50}       # instances per (solver, delta) — 50 is a good balance
DELTAS=${DELTAS:-"2 5 10 15 20"}
SEED=9999                  # must match submit_baseline.sh

echo "========================================================"
echo " CE-PDPTW delta sweep  |  Rung: $RUNG  |  $(date)"
echo " Instances: $N_INST  |  Deltas: $DELTAS  |  Seed: $SEED"
echo "========================================================"

# ── HetGAT RL checkpoint ─────────────────────────────────────
RL_CKPT_HETGAT="CE-PDPTW-${RUNG}-HetGAT/best_actor.pt"
# Also sweep RLOO variants if they exist
for _variant in rloo4 rloo8; do
    _ckpt="CE-PDPTW-${RUNG}-${_variant}-HetGAT/best_actor.pt"
    if [ -f "$_ckpt" ]; then
        RL_CKPT_HETGAT="$_ckpt"
        echo "  Using RLOO variant: $_ckpt"
        break
    fi
done

# ── SimpleGAT checkpoint (optional) ──────────────────────────
RL_CKPT_SIMPLEGAT="CE-PDPTW-${RUNG}-simplegat-HetGAT/best_actor.pt"

# ── Gurobi flag ───────────────────────────────────────────────
_GUROBI_FLAG=""
if [ "${GUROBI}" = "1" ] && [ "$RUNG" != "D" ]; then
    _GUROBI_FLAG="--gurobi"
fi

OUT_DIR="eval_results/delta_sweep_${RUNG}"
mkdir -p "$OUT_DIR"

# ── Run 1: HetGAT + ALNS + Greedy (+ optionally Gurobi) ──────
echo ""
echo "=== HetGAT sweep (arch=hetgat) ==="
if [ -f "$RL_CKPT_HETGAT" ]; then
    python benchmark_delta_sweep.py \
        --rl-model    "$RL_CKPT_HETGAT" \
        --arch        hetgat \
        --rung        "$RUNG" \
        --n-instances "$N_INST" \
        --deltas      $DELTAS \
        --seed        "$SEED" \
        --out-dir     "$OUT_DIR/hetgat" \
        $_GUROBI_FLAG
else
    echo "  WARNING: $RL_CKPT_HETGAT not found — running CPU baselines only"
    python benchmark_delta_sweep.py \
        --rung        "$RUNG" \
        --n-instances "$N_INST" \
        --deltas      $DELTAS \
        --seed        "$SEED" \
        --out-dir     "$OUT_DIR/baselines_only" \
        $_GUROBI_FLAG
fi

# ── Run 2: SimpleGAT (if trained, Rungs C/D only) ────────────
# SimpleGAT is only trained for Rungs C and D.
if [ "$RUNG" = "C" ] || [ "$RUNG" = "D" ]; then
    echo ""
    echo "=== SimpleGAT sweep (arch=simplegat) ==="
    if [ -f "$RL_CKPT_SIMPLEGAT" ]; then
        python benchmark_delta_sweep.py \
            --rl-model    "$RL_CKPT_SIMPLEGAT" \
            --arch        simplegat \
            --rung        "$RUNG" \
            --n-instances "$N_INST" \
            --deltas      $DELTAS \
            --seed        "$SEED" \
            --out-dir     "$OUT_DIR/simplegat"
    else
        echo "  SimpleGAT checkpoint not found ($RL_CKPT_SIMPLEGAT) — skipping."
    fi
fi

echo ""
echo "========================================================"
echo " Delta sweep complete: $(date)"
echo " Results: $OUT_DIR/"
echo " Plot:    $OUT_DIR/hetgat/delta_comparison_table.png"
echo "========================================================"
