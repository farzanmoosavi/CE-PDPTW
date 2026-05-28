#!/bin/bash
# ============================================================
# Nibi (H100) — post-training evaluation job.
#
# Step 1 (optional, ABLATION=1): evaluate.py — RL ablations,
#         generalization, edge-feature ablation, mode analysis.
#         Heavy: 1280-sample decode + ablation suite.
#         Skipped by default so step 2 always runs promptly.
#
# Step 2: benchmark_rolling_baselines_with_plots.py --baselines rl
#         Tests RL model through RollingHorizonDispatcher with
#         demand revealed incrementally — same protocol as CPU
#         baselines (ALNS / OR-Tools / Gurobi).  GPU, sequential
#         (CUDA context cannot be forked across workers).
#         This is the comparison that goes in the paper table.
#
# Step 3: Merge CPU-baseline aggregate (results_rungX/) + RL
#         aggregate (results_rungX_rl/) → combined paper_table.tex.
#         CPU baselines from submit_baseline.sh may run in parallel
#         or finish first; merge handles both cases gracefully.
#
# Step 4: Training curve plots.
#
# Usage:
#   sbatch submit_evaluate.sh                           # Rung A, RL dispatch only
#   sbatch --export=RUNG=B submit_evaluate.sh           # Rung B, RL dispatch only
#   sbatch --export=RUNG=B,ABLATION=1 submit_evaluate.sh  # + full ablation suite
#   sbatch --export=RUNG=C,N_SAMPLES=1280 submit_evaluate.sh
# ============================================================

#SBATCH --job-name=CE-PDPTW-eval
#SBATCH --account=def-farooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/eval-%x-%j.out
#SBATCH --error=logs/eval-%x-%j.out

module purge
module load python/3.10 scipy-stack cuda/12.2
source ~/py310_nibi/bin/activate

_PROJ="$HOME/projects/def-farooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-farooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-farooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs eval_results

RUNG=${RUNG:-A}
N_TEST=${N_TEST:-2048}
N_SAMPLES=${N_SAMPLES:-1280}
RLOO_K=${RLOO_K:-4}  # must match training default (RLOO k=4)
# Guard: Alliance Canada clusters set ARCH=x86_64 in the environment.
case "${ARCH:-hetgat}" in hetgat|simplegat) ;; *) ARCH=hetgat ;; esac
DYNAMIC=${DYNAMIC:-0}     # set DYNAMIC=1 to evaluate the --dynamic training variant
ABLATION=${ABLATION:-0}   # set ABLATION=1 to run the full ablation suite (step 1)

# Build folder/checkpoint path — must match the naming convention in VRP_Rollout_train.py.
# Priority: rloo > simplegat > dynamic > plain rollout.
if [ "${RLOO_K}" -gt 0 ]; then
    SUFFIX="-rloo${RLOO_K}"
elif [ "$ARCH" = "simplegat" ]; then
    SUFFIX="-simplegat"
elif [ "${DYNAMIC}" = "1" ]; then
    SUFFIX="-dynamic"
else
    SUFFIX=""
fi
FOLDER="CE-PDPTW-${RUNG}${SUFFIX}-HetGAT"
CHECKPOINT="${FOLDER}/best_actor.pt"

# Pass --dynamic to evaluate.py / benchmark when DYNAMIC=1 so instances are
# sub-batched (n_vis ~ U[n_req/4, n_req]) to match training distribution.
_DYNAMIC_FLAG=""
[ "${DYNAMIC}" = "1" ] && _DYNAMIC_FLAG="--dynamic"

echo "========================================================"
echo " CE-PDPTW evaluation  |  Rung: $RUNG  |  $(date)"
echo " Checkpoint: $CHECKPOINT"
echo " Dynamic variant: $([ "${DYNAMIC}" = "1" ] && echo YES || echo no)"
echo " Test instances: $N_TEST  |  Sampling K: $N_SAMPLES"
echo " Ablation suite: $([ "${ABLATION}" = "1" ] && echo ENABLED || echo disabled — use ABLATION=1 to enable)"
echo "========================================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: $CHECKPOINT not found."
    echo "Run training first:  sbatch --export=RUNG=$RUNG submit_cc.sh"
    exit 1
fi

# ── Rung parameters ─────────────────────────────────────────
case "$RUNG" in
    A) N_REQ=5;  N_UAV=2;  N_ADR=2; N_DEP_UAV=1; N_DEP_ADR=1; N_INST=20 ;;
    B) N_REQ=10; N_UAV=4;  N_ADR=3; N_DEP_UAV=1; N_DEP_ADR=1; N_INST=20 ;;
    C) N_REQ=25; N_UAV=5;  N_ADR=4; N_DEP_UAV=2; N_DEP_ADR=2; N_INST=10 ;;
    D) N_REQ=60; N_UAV=10; N_ADR=8; N_DEP_UAV=3; N_DEP_ADR=3; N_INST=10 ;;
    *) echo "Unknown RUNG=$RUNG. Use A, B, C, or D."; exit 1 ;;
esac

RL_OUT="results_rung${RUNG}${SUFFIX}_rl"

# ── Step 1: RL ablation study (optional, heavy) ──────────────
if [ "${ABLATION}" = "1" ]; then
    echo ""
    echo "=== Step 1: evaluate.py — generalization, edge ablation, mode analysis ==="
    python evaluate.py \
        --checkpoint  "$CHECKPOINT" \
        --rung        "$RUNG" \
        --n-test      "$N_TEST" \
        --n-samples   "$N_SAMPLES" \
        --arch        "$ARCH" \
        --generalize \
        --ablate-edges \
        --mode-analysis \
        ${_DYNAMIC_FLAG} \
        --out-dir     eval_results
    echo "Ablation study done at $(date)"
else
    echo ""
    echo "=== Step 1: SKIPPED (ABLATION=0) — run with ABLATION=1 for full suite ==="
    echo "    Quick greedy-only eval for mode-analysis:"
    python evaluate.py \
        --checkpoint  "$CHECKPOINT" \
        --rung        "$RUNG" \
        --n-test      "$N_TEST" \
        --arch        "$ARCH" \
        --mode-analysis \
        ${_DYNAMIC_FLAG} \
        --out-dir     eval_results
    echo "Quick eval done at $(date)"
fi

# ── Step 2: RL in dispatch-sim (GPU, sequential) ────────────
# Demand is revealed incrementally; RL model is used as the
# solver inside RollingHorizonDispatcher — apples-to-apples
# with the CPU baselines in submit_baseline.sh.
# workers=1: GPU CUDA context cannot be shared across fork'd workers.
echo ""
echo "=== Step 2: RL dispatch-sim  |  Rung $RUNG  |  $N_INST instances ==="
python benchmark_rolling_baselines_with_plots.py \
    --n-req         $N_REQ \
    --n-uav         $N_UAV \
    --n-adr         $N_ADR \
    --n-depots-uav  $N_DEP_UAV \
    --n-depots-adr  $N_DEP_ADR \
    --num-instances $N_INST \
    --baselines     rl \
    --rl-model      "$CHECKPOINT" \
    --arch          "$ARCH" \
    --output-dir    "$RL_OUT" \
    --seed          9999 \
    --workers       1 \
    --make-plots
_RL_EXIT=$?
echo "RL dispatch-sim done at $(date) (exit code: $_RL_EXIT)"

if [ $_RL_EXIT -ne 0 ]; then
    echo "ERROR: RL dispatch-sim failed (exit $_RL_EXIT). Check logs above."
    echo "  Common causes: model load error, dispatch_sim import error."
    echo "  Diagnostic: python -c \"from main import build_solver; build_solver('$CHECKPOINT')\""
    exit $_RL_EXIT
fi

# ── Step 3: Merge CPU-baseline + RL → combined paper_table.tex ──
# CPU results (results_rungX/aggregate_results.json) come from
# submit_baseline.sh which can run in parallel.  If they are not
# yet present, the merged table contains RL only and can be
# re-generated later by re-running this script (checkpoints persist).
echo ""
echo "=== Step 3: Merging results → results_rung${RUNG}/paper_table.tex ==="
python - <<PYEOF
import json, pathlib, argparse, sys
sys.path.insert(0, '.')

cpu_agg = pathlib.Path("results_rung${RUNG}/aggregate_results.json")
rl_agg  = pathlib.Path("${RL_OUT}/aggregate_results.json")
out_dir = pathlib.Path("results_rung${RUNG}")
out_dir.mkdir(parents=True, exist_ok=True)

combined = []
if cpu_agg.exists():
    cpu_rows = json.loads(cpu_agg.read_text())
    combined += cpu_rows
    print(f"  CPU baselines: {[r.get('baseline') for r in cpu_rows]} loaded from {cpu_agg}")
else:
    print(f"  NOTE: CPU baselines not found at {cpu_agg}.")
    print(f"        Run submit_baseline.sh, then re-run this job to get the full table.")

if not rl_agg.exists():
    print(f"  ERROR: RL aggregate not found at {rl_agg}"); sys.exit(1)

rl_rows = json.loads(rl_agg.read_text())
combined += rl_rows
print(f"  RL dispatch-sim: {[r.get('baseline') for r in rl_rows]} loaded from {rl_agg}")

try:
    from benchmark_rolling_baselines_with_plots import _print_paper_table
    args = argparse.Namespace(
        output_dir=str(out_dir),
        n_req=${N_REQ},
        n_uav=${N_UAV},
        n_adr=${N_ADR},
        num_instances=${N_INST},
    )
    _print_paper_table(combined, args)
    print(f"  Combined paper_table.tex → {out_dir}/paper_table.tex")
except Exception as exc:
    print(f"  WARNING: _print_paper_table import failed ({exc}); saving RL table only.")
    import shutil
    rl_tex = rl_agg.parent / "paper_table.tex"
    if rl_tex.exists():
        shutil.copy(str(rl_tex), str(out_dir / "paper_table_rl_only.tex"))
        print(f"  RL-only table saved → {out_dir}/paper_table_rl_only.tex")
PYEOF

# ── Step 4: Training curve plots ────────────────────────────
echo ""
echo "=== Step 4: Training curve plots ==="
# Collect all CSVs for this rung (rollout, rloo4, rloo8, simplegat variants)
_CONV_CSVS=""
_CONV_LABELS=""
for _variant in "" "-rloo4" "-rloo8" "-simplegat"; do
    _csv="logs/training_${RUNG}${_variant}.csv"
    if [ -f "$_csv" ]; then
        _CONV_CSVS="$_CONV_CSVS $_csv"
        if [ -z "$_variant" ]; then
            _CONV_LABELS="$_CONV_LABELS HetGAT-rollout"
        else
            _CONV_LABELS="$_CONV_LABELS HetGAT${_variant}"
        fi
    fi
done

if [ -n "$_CONV_CSVS" ]; then
    mkdir -p "eval_results/plots_${RUNG}"
    python plot_convergence.py \
        --csv    $_CONV_CSVS \
        --labels $_CONV_LABELS \
        --out    "eval_results/plots_${RUNG}/convergence_${RUNG}.png" \
        --show-train-reward \
        --smooth 3
else
    echo "  No training CSVs found for rung $RUNG — skipping convergence plot."
fi

echo "========================================================"
echo " Evaluation complete: $(date)"
echo " Checkpoint used:      $CHECKPOINT  (ARCH=$ARCH, RLOO_K=$RLOO_K)"
echo " RL ablation study:    eval_results/  (ABLATION=${ABLATION})"
echo " RL dispatch-sim:      ${RL_OUT}/"
echo " Combined paper table: results_rung${RUNG}/paper_table.tex"
echo "   (CPU baselines + RL rows in one table)"
echo "   If CPU baselines were not ready, re-run this job after"
echo "   submit_baseline.sh finishes — checkpoints are preserved."
echo " Convergence plots:    eval_results/plots_${RUNG}/"
echo "========================================================"
