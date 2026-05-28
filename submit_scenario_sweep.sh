#!/bin/bash
# ============================================================
# Narval — scenario robustness sweep (CPU-only, no GPU needed).
#
# Runs benchmark_rolling_baselines_with_plots.py for one rung
# across 7 controlled scenario variants, 10 instances each.
# Each scenario produces results_<RUNG>/<scenario>/paper_table_<scenario>.tex
# with mean±std across the 10 instances.
#
# Scenario axes and values:
#   baseline   : demand U[1,6], wind U[0,12], TW-slack N(30,5)∈[15,60]
#   low_demand : demand U[0.5,2]
#   high_demand: demand U[4,10]
#   calm_wind  : wind U[0,3]
#   stormy     : wind U[6,12]
#   tight_tw   : TW-slack N(15,3)∈[10,25]
#   loose_tw   : TW-slack N(50,10)∈[30,90]
#
# Usage:
#   sbatch submit_scenario_sweep.sh                  # Rung C (default)
#   sbatch --export=RUNG=B submit_scenario_sweep.sh
#   sbatch --export=RUNG=C,N_INST=20 submit_scenario_sweep.sh
# ============================================================

#SBATCH --job-name=CE-PDPTW-scenarios
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/scenarios-%j.out
#SBATCH --error=logs/scenarios-%j.err

module purge
module load python/3.10 scipy-stack
source ~/py310_nibi/bin/activate

# ── OR-Tools GLPK probe ──────────────────────────────────────
_HAVE_ORT=false
_ORT_SOLVER="GLPK"
if timeout 30 python -c "
from ortools.linear_solver import pywraplp
s = pywraplp.Solver.CreateSolver('GLPK')
assert s is not None
x = s.NumVar(0, 1, 'x'); s.Maximize(x); s.Solve()
" 2>/dev/null; then
    _HAVE_ORT=true
    echo "OR-Tools GLPK OK."
else
    echo "OR-Tools GLPK probe failed — OR-Tools excluded."
fi

# ── Gurobi probe ─────────────────────────────────────────────
module load gurobi/12.0.0
pip install gurobipy==12.0.0 --no-index
_HAVE_GRB=false
if timeout 120 python -c "
import gurobipy as gp
gp.setParam('OutputFlag', 0)
m = gp.Model(); m.dispose()
" 2>/dev/null; then
    _HAVE_GRB=true
    echo "Gurobi license OK."
else
    echo "WARNING: Gurobi license check failed — Gurobi excluded."
fi

_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs

export MPLBACKEND=Agg

RUNG=${RUNG:-C}
N_INST=${N_INST:-10}
SEED=9999

# Fleet config per rung
case "$RUNG" in
    A) N_REQ=5;  N_UAV=2;  N_ADR=2; N_DU=1; N_DA=1; TL=30;  ALNS_S=3;  ALNS_L=8  ;;
    B) N_REQ=10; N_UAV=4;  N_ADR=3; N_DU=1; N_DA=1; TL=60;  ALNS_S=5;  ALNS_L=15 ;;
    C) N_REQ=25; N_UAV=5;  N_ADR=4; N_DU=2; N_DA=2; TL=120; ALNS_S=10; ALNS_L=30 ;;
    D) N_REQ=60; N_UAV=10; N_ADR=8; N_DU=3; N_DA=3; TL=0;   ALNS_S=20; ALNS_L=60 ;;
    *) echo "Unknown RUNG=$RUNG"; exit 1 ;;
esac

# Baselines: exact solvers only for small rungs
# fifo (FIFO dispatch), greedy (cheapest-insertion), alns (rolling-horizon),
# offline_alns (ALNS solved once and executed without re-planning).
BL="fifo,greedy,alns,offline_alns"
$_HAVE_GRB && [ "$RUNG" != "D" ] && BL="$BL,gurobi"
$_HAVE_ORT && [ "$RUNG" = "A" -o "$RUNG" = "B" ] && BL="$BL,ortools"

COMMON="--n-req $N_REQ --n-uav $N_UAV --n-adr $N_ADR \
        --n-depots-uav $N_DU --n-depots-adr $N_DA \
        --num-instances $N_INST --baselines $BL \
        --alns-small-budget $ALNS_S --alns-large-budget $ALNS_L \
        --ortools-solver $_ORT_SOLVER \
        --seed $SEED --workers 1 --make-plots"
[ "$TL" -gt 0 ] && COMMON="$COMMON --exact-time-limit $TL"

OUT_BASE="results_rung${RUNG}_scenarios"
mkdir -p "$OUT_BASE"

echo "========================================================"
echo " CE-PDPTW scenario sweep  |  Rung $RUNG  |  $(date)"
echo " Instances per scenario: $N_INST  |  Baselines: $BL"
echo "========================================================"

# ── Helper ────────────────────────────────────────────────────
_run_scenario() {
    local label=$1; shift   # remaining args are scenario-specific flags
    local out_dir="$OUT_BASE/$label"
    echo ""
    echo "--- Scenario: $label ---"
    python benchmark_rolling_baselines_with_plots.py \
        $COMMON \
        --scenario-label "$label" \
        --output-dir "$out_dir" \
        "$@"
    echo "  done at $(date)  →  $out_dir/paper_table_${label}.tex"
}

# ── 1. Baseline (default distributions) ──────────────────────
_run_scenario baseline

# ── 2. Low demand ─────────────────────────────────────────────
_run_scenario low_demand \
    --demand-low 0.5 --demand-high 2.0

# ── 3. High demand ────────────────────────────────────────────
_run_scenario high_demand \
    --demand-low 4.0 --demand-high 10.0

# ── 4. Calm wind ──────────────────────────────────────────────
_run_scenario calm_wind \
    --wind-speed-low 0.0 --wind-speed-high 3.0

# ── 5. Stormy wind ────────────────────────────────────────────
_run_scenario stormy \
    --wind-speed-low 6.0 --wind-speed-high 12.0

# ── 6. Tight time windows ─────────────────────────────────────
_run_scenario tight_tw \
    --tw-slack-mean 15.0 --tw-slack-std 3.0 \
    --tw-slack-clip-low 10.0 --tw-slack-clip-high 25.0

# ── 7. Loose time windows ─────────────────────────────────────
_run_scenario loose_tw \
    --tw-slack-mean 50.0 --tw-slack-std 10.0 \
    --tw-slack-clip-low 30.0 --tw-slack-clip-high 90.0

echo ""
echo "========================================================"
echo " All scenario sweeps complete at $(date)"
echo " Results in: $OUT_BASE/{baseline,low_demand,high_demand,calm_wind,stormy,tight_tw,loose_tw}/"
echo " LaTeX tables: each subfolder contains paper_table_<scenario>.tex"
echo "========================================================"
