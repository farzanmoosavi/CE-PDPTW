#!/bin/bash
# ============================================================
# Nibi — CPU-only baseline sweep (ALNS, Gurobi, OR-Tools)
# Runs all four curriculum rungs sequentially in one job.
# No GPU required.
#
# Usage:
#   sbatch submit_baseline.sh
#   sbatch --export=RUNG=C submit_baseline.sh   # single rung only
# ============================================================

#SBATCH --job-name=CE-PDPTW-baselines
#SBATCH --account=def-farooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/baselines-%j.out
#SBATCH --error=logs/baselines-%j.err

module purge
module load python/3.10 scipy-stack
source ~/py310_nibi/bin/activate

# ── OR-Tools ─────────────────────────────────────────────────
# SCIP backend segfaults on CC (ABI mismatch).  Use GLPK instead — it is bundled
# inside the ortools wheel and needs no external library.  Probe before enabling.
# OR-Tools is only used for Rungs A/B (small); too slow at n_req>=25.
_HAVE_ORT=false
_ORT_SOLVER="GLPK"
echo "Testing OR-Tools with GLPK backend..."
if timeout 30 python -c "
from ortools.linear_solver import pywraplp
s = pywraplp.Solver.CreateSolver('GLPK')
assert s is not None, 'GLPK solver not available'
x = s.NumVar(0, 1, 'x')
s.Maximize(x)
s.Solve()
" 2>/dev/null; then
    _HAVE_ORT=true
    echo "OR-Tools GLPK OK."
else
    echo "OR-Tools GLPK probe failed — OR-Tools will be EXCLUDED."
fi

# ── Gurobi ───────────────────────────────────────────────────
# Use the Narval site module — floating license, no internet needed.
# gurobipy must match the module version; install from CC wheelhouse.
module load gurobi/12.0.0
pip install gurobipy==12.0.0 --no-index

# ── Gurobi license pre-flight check ─────────────────────────
# CC uses a token server at license1.computecanada.ca (internal).
# First checkout can take 60-90s; use 120s timeout.
echo "Testing Gurobi license via CC token server (120s timeout)..."
_HAVE_GRB=false
if timeout 120 python -c "
import gurobipy as gp
gp.setParam('OutputFlag', 0)
m = gp.Model()
m.dispose()
" 2>/dev/null; then
    _HAVE_GRB=true
    echo "Gurobi license OK."
else
    echo "WARNING: Gurobi license check failed (token server unreachable or timed out)."
    echo "  Gurobi will be EXCLUDED from baselines for this run."
    echo "  The CC token server (license1.computecanada.ca) may be busy or down."
fi

# Build per-rung baselines strings based on what's available.
# Always included:
#   fifo         — first-in-first-out dispatch (sanity-check lower bound)
#   greedy       — cheapest-insertion (standard VRP heuristic)
#   alns         — rolling-horizon ALNS (re-plans every Δ)
#   offline_alns — ALNS solved once at t=0, executed without re-planning
#                  (oracle planner reference; tests value of online re-planning)
BL_AB="fifo,greedy,alns,offline_alns"
BL_C="fifo,greedy,alns,offline_alns"
$_HAVE_GRB && BL_AB="$BL_AB,gurobi" && BL_C="$BL_C,gurobi"
$_HAVE_ORT && BL_AB="$BL_AB,ortools"   # OR-Tools A/B only — too slow at n_req>=25
BL_D="fifo,greedy,alns,offline_alns"   # exact solvers never tractable at n_req=60

_PROJ="$HOME/projects/def-farooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-farooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-farooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs

# Headless matplotlib: compute nodes have no display; Agg writes to PNG without Xvfb.
export MPLBACKEND=Agg

echo "========================================================"
echo " CE-PDPTW Baseline sweep | Node: $(hostname) | $(date)"
echo " CPUs: $SLURM_CPUS_PER_TASK | Mem: $SLURM_MEM_PER_NODE MB"
echo " Gurobi: $(${_HAVE_GRB} && echo enabled || echo DISABLED - no license)"
echo "========================================================"

# Sequential (workers=1) avoids BrokenProcessPool from forkserver worker crashes.
# OR-Tools and ALNS both have C++ internals that crash reliably in forkserver workers
# on Narval; sequential is reliable and fast enough (A: ~5min, B: ~15min).
WORKERS=1
FRESH=${FRESH:-0}    # set FRESH=1 to wipe results_rungX/ before re-running

if [ "${FRESH}" = "1" ]; then
    echo "FRESH=1: removing stale results directories."
    rm -rf results_rungA results_rungB results_rungC results_rungD
fi

# ── Single-rung mode (sbatch --export=RUNG=C) ──────────────
if [ -n "${RUNG}" ]; then
    echo "Single-rung mode: RUNG=$RUNG"
    case "$RUNG" in
        A)
            python benchmark_rolling_baselines_with_plots.py \
                --n-req 5 --n-uav 2 --n-adr 2 \
                --n-depots-uav 1 --n-depots-adr 1 \
                --num-instances 10 --baselines "$BL_AB" \
                --exact-time-limit 30 \
                --alns-small-budget 3 --alns-large-budget 8 \
                --ortools-solver "$_ORT_SOLVER" \
                --scenario-label baseline \
                --output-dir results_rungA \
                --seed 9999 --workers $WORKERS --make-plots
            ;;
        B)
            python benchmark_rolling_baselines_with_plots.py \
                --n-req 10 --n-uav 4 --n-adr 3 \
                --n-depots-uav 1 --n-depots-adr 1 \
                --num-instances 10 --baselines "$BL_AB" \
                --exact-time-limit 60 \
                --alns-small-budget 5 --alns-large-budget 15 \
                --ortools-solver "$_ORT_SOLVER" \
                --scenario-label baseline \
                --output-dir results_rungB \
                --seed 9999 --workers $WORKERS --make-plots
            ;;
        C)
            python benchmark_rolling_baselines_with_plots.py \
                --n-req 25 --n-uav 5 --n-adr 4 \
                --n-depots-uav 2 --n-depots-adr 2 \
                --num-instances 10 --baselines "$BL_C" \
                --exact-time-limit 120 \
                --alns-small-budget 10 --alns-large-budget 30 \
                --scenario-label baseline \
                --output-dir results_rungC \
                --seed 9999 --workers 1 --make-plots
            ;;
        D)
            python benchmark_rolling_baselines_with_plots.py \
                --n-req 60 --n-uav 10 --n-adr 8 \
                --n-depots-uav 3 --n-depots-adr 3 \
                --num-instances 10 --baselines "$BL_D" \
                --alns-small-budget 20 --alns-large-budget 60 \
                --scenario-label baseline \
                --output-dir results_rungD \
                --seed 9999 --workers 1 --make-plots
            ;;
        *)
            echo "Unknown RUNG=$RUNG. Use A, B, C, or D."; exit 1 ;;
    esac
    echo "Done at $(date)"; exit 0
fi

# ── All-rungs mode (default) ────────────────────────────────

echo ""
echo "=== Rung A: $BL_AB (n_req=5, 10 instances) ==="
python benchmark_rolling_baselines_with_plots.py \
    --n-req 5 --n-uav 2 --n-adr 2 \
    --n-depots-uav 1 --n-depots-adr 1 \
    --num-instances 10 --baselines "$BL_AB" \
    --exact-time-limit 30 \
    --alns-small-budget 3 --alns-large-budget 8 \
    --ortools-solver "$_ORT_SOLVER" \
    --scenario-label baseline \
    --output-dir results_rungA \
    --seed 9999 --workers $WORKERS --make-plots
echo "Rung A done at $(date)"

echo ""
echo "=== Rung B: $BL_AB (n_req=10, 10 instances) ==="
python benchmark_rolling_baselines_with_plots.py \
    --n-req 10 --n-uav 4 --n-adr 3 \
    --n-depots-uav 1 --n-depots-adr 1 \
    --num-instances 10 --baselines "$BL_AB" \
    --exact-time-limit 60 \
    --alns-small-budget 5 --alns-large-budget 15 \
    --ortools-solver "$_ORT_SOLVER" \
    --scenario-label baseline \
    --output-dir results_rungB \
    --seed 9999 --workers $WORKERS --make-plots
echo "Rung B done at $(date)"

echo ""
echo "=== Rung C: $BL_C only (n_req=25, 10 instances) ==="
python benchmark_rolling_baselines_with_plots.py \
    --n-req 25 --n-uav 5 --n-adr 4 \
    --n-depots-uav 2 --n-depots-adr 2 \
    --num-instances 10 --baselines "$BL_C" \
    --exact-time-limit 120 \
    --alns-small-budget 10 --alns-large-budget 30 \
    --scenario-label baseline \
    --output-dir results_rungC \
    --seed 9999 --workers 1 --make-plots
echo "Rung C done at $(date)"

echo ""
echo "=== Rung D: greedy + ALNS only (n_req=60, 10 instances) ==="
python benchmark_rolling_baselines_with_plots.py \
    --n-req 60 --n-uav 10 --n-adr 8 \
    --n-depots-uav 3 --n-depots-adr 3 \
    --num-instances 10 --baselines "$BL_D" \
    --alns-small-budget 20 --alns-large-budget 60 \
    --scenario-label baseline \
    --output-dir results_rungD \
    --seed 9999 --workers 1 --make-plots
echo "Rung D done at $(date)"

echo ""
echo "========================================================"
echo " All baseline runs complete at $(date)"
echo " Results in: results_rungA/ results_rungB/ results_rungC/ results_rungD/"
echo " Paper tables: results_rungX/paper_table.tex"
echo "========================================================"
