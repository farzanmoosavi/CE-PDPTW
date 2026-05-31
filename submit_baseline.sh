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
# CP-SAT has its own shared libraries and can crash independently of pywrapcp.
# This probe runs a minimal add_circuit model — the same API the solver uses.
# Important: the probe is run in a fresh subprocess (spawn) to match the exact
# conditions under which the baseline runs.  pywrapcp may pass while CP-SAT fails.
_HAVE_VRP=false
echo "Testing OR-Tools CP-SAT (subprocess spawn) — used by ortools_vrp baseline..."
if timeout 60 python -c "
import multiprocessing, sys, os

def _probe(q):
    try:
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        # Minimal add_circuit model: 3 nodes, 1 vehicle, depot=2
        arcs = []
        lits = []
        for n in range(3):
            for m in range(3):
                if n != m:
                    v = model.new_bool_var(f'a_{n}_{m}')
                    arcs.append((n, m, v))
                    lits.append(v)
            loop = model.new_bool_var(f'lp_{n}')
            arcs.append((n, n, loop))
        model.add_circuit(arcs)
        model.minimize(sum(lits))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f'bad status {status}'
        q.put('ok')
    except Exception as e:
        q.put(f'err:{e}')

ctx = multiprocessing.get_context('spawn')
q = ctx.Queue()
p = ctx.Process(target=_probe, args=(q,))
p.start()
p.join(timeout=30)
if p.is_alive():
    p.terminate(); p.join()
    sys.exit(1)
if p.exitcode != 0:
    sys.exit(1)
result = q.get_nowait()
if result != 'ok':
    sys.exit(1)
" 2>/dev/null; then
    _HAVE_VRP=true
    echo "OR-Tools CP-SAT (spawn subprocess) OK."
else
    echo "OR-Tools CP-SAT probe failed (subprocess exit -11 = SIGSEGV, or other error)."
    echo "  ortools_vrp baseline will be EXCLUDED."
fi

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
