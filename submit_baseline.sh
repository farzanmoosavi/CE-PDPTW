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
#SBATCH --account=def-bfarooq
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

# ── Install OR-Tools if missing ───────────────────────────────
# OR-Tools is not in CC's curated wheel set, so install from PyPI.
# Login/compute nodes on Narval have outbound HTTPS; if not, pre-install
# on the login node with: pip install ortools
if ! python -c "import ortools" 2>/dev/null; then
    echo "ortools not found — installing from PyPI..."
    pip install ortools --quiet && echo "ortools installed OK." \
        || echo "WARNING: ortools install failed — ortools_vrp and ortools baselines will be EXCLUDED."
else
    echo "ortools already installed."
fi

# ── OR-Tools (MILP linear solver) ────────────────────────────
# SCIP backend segfaults on CC (ABI mismatch).  Use GLPK instead — it is bundled
# inside the ortools wheel and needs no external library.  Probe before enabling.
# OR-Tools MILP is only used for Rungs A/B (small); too slow at n_req>=25.
_HAVE_ORT=false
_ORT_SOLVER="GLPK"
echo "Testing OR-Tools GLPK (linear solver) backend..."
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
    echo "OR-Tools GLPK probe failed — OR-Tools MILP will be EXCLUDED."
fi

# ── OR-Tools CP-SAT (used by ortools_vrp baseline) ───────────────────────────
# The ortools_vrp baseline uses CP-SAT (ortools.sat.python.cp_model), NOT pywrapcp.
# Test CP-SAT in a fresh subprocess (subprocess.run, no pickling) to mirror the
# exact execution context of the worker.  vrpUpdate.py now lazy-imports torch so
# the worker subprocess never loads torch alongside OR-Tools (was causing SIGSEGV
# due to libprotobuf ABI conflict).
#
# Heredoc inside bash if-then-else requires "then" after the terminator which is
# error-prone in SLURM scripts; use a temp file instead to avoid syntax issues.
_HAVE_VRP=false
echo "Testing OR-Tools CP-SAT in fresh subprocess (used by ortools_vrp baseline)..."
_CPSAT_TMP=$(mktemp /tmp/cpsat_probe_XXXXXX.py)
cat > "$_CPSAT_TMP" << 'PROBE_END'
import subprocess, sys
probe_code = (
    "from ortools.sat.python import cp_model\n"
    "model = cp_model.CpModel()\n"
    "arcs = []\n"
    "for n in range(3):\n"
    "    for m in range(3):\n"
    "        if n != m:\n"
    "            v = model.new_bool_var('a{}_{}'.format(n,m))\n"
    "            arcs.append((n, m, v))\n"
    "    lp = model.new_bool_var('lp{}'.format(n))\n"
    "    arcs.append((n, n, lp))\n"
    "model.add_circuit(arcs)\n"
    "solver = cp_model.CpSolver()\n"
    "solver.parameters.max_time_in_seconds = 5\n"
    "status = solver.solve(model)\n"
    "assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)\n"
)
try:
    r = subprocess.run([sys.executable, '-c', probe_code], timeout=30)
    sys.exit(r.returncode)
except Exception as e:
    print('probe error:', e, file=sys.stderr)
    sys.exit(1)
PROBE_END
if timeout 60 python "$_CPSAT_TMP" 2>/dev/null; then
    _HAVE_VRP=true
    echo "OR-Tools CP-SAT (subprocess) OK."
else
    echo "OR-Tools CP-SAT probe failed — ortools_vrp will be EXCLUDED."
fi
rm -f "$_CPSAT_TMP"

# ── Gurobi ───────────────────────────────────────────────────
# Use the Narval site module — gurobipy must match the module version.
module load gurobi/12.0.0
pip install gurobipy==12.0.0 --no-index

# Resolve project directory (used below and for cd).
_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW"

# ── Gurobi license pre-flight check ─────────────────────────
# Use CC token server configured by module load gurobi/12.0.0.
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
fi

# Build per-rung baselines strings based on what's available.
# Always included:
#   fifo         — first-in-first-out dispatch (sanity-check lower bound)
#   greedy       — cheapest-insertion (standard VRP heuristic)
#   alns         — rolling-horizon ALNS (re-plans every Δ)
#   offline_alns — ALNS solved once at t=0, executed without re-planning
#                  (oracle planner reference; tests value of online re-planning)
#   ortools_vrp  — rolling-horizon OR-Tools VRP routing (pywrapcp.RoutingModel,
#                  ~8s/re-plan, GLS metaheuristic; primary RL comparison baseline)
BL_AB="fifo,greedy,alns,offline_alns,cw,offline_cw,regret,offline_regret,vns,offline_vns"
BL_C="fifo,greedy,alns,offline_alns,cw,offline_cw,regret,offline_regret,vns,offline_vns"
$_HAVE_VRP && BL_AB="$BL_AB,ortools_vrp"  # isolated spawn subprocess avoids torch/libpyg.so ABI conflict
$_HAVE_GRB && BL_AB="$BL_AB,gurobi" && BL_C="$BL_C,gurobi"
$_HAVE_ORT && BL_AB="$BL_AB,ortools"   # OR-Tools MILP A/B only — too slow at n_req>=25
BL_D="fifo,greedy,alns,offline_alns,cw,offline_cw,regret,offline_regret,vns,offline_vns"

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
