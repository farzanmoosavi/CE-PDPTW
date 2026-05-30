#!/bin/bash
# ============================================================
# Nibi (H100) — single-rung RL training job.
#
# Usage:
#   sbatch submit_nibi.sh                              # Rung A, 200 epochs
#   sbatch --export=RUNG=B submit_nibi.sh
#   sbatch --export=RUNG=B,RLOO_K=4 submit_nibi.sh    # Rung B with RLOO-k=4
#   sbatch --export=RUNG=C,EPOCHS=300 submit_nibi.sh
#   sbatch --export=RUNG=B,FRESH=1 submit_nibi.sh      # wipe checkpoint, restart clean
#
# One-time setup on login node before first sbatch:
#   module purge
#   module load python/3.10 scipy-stack cuda/12.2
#   virtualenv ~/py310_nibi
#   source ~/py310_nibi/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install torch_geometric
#   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#       -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
#   pip install scipy ortools
#   pip install numpy --no-index
# ============================================================

#SBATCH --job-name=CE-PDPTW-train
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:3
#SBATCH --time=1-00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/train-%x-%j.out
#SBATCH --error=logs/train-%x-%j.err

module purge
module load python/3.10 scipy-stack
source ~/py310_nibi/bin/activate

# Load the CUDA module that matches the installed torch wheel.
# torch.version.cuda tells us the CUDA the wheel was compiled against;
# try that first, then fall back to known-good versions.
_TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
_CUDA_LOADED=0
for _CV in "$_TORCH_CUDA" 13.2 13.1 13.0 12.6 12.2; do
    [ -z "$_CV" ] && continue
    if module load cuda/$_CV 2>/dev/null; then
        echo "[cuda] Loaded cuda/$_CV (torch compiled with CUDA $_TORCH_CUDA)"
        _CUDA_LOADED=1
        break
    fi
done
if [ $_CUDA_LOADED -eq 0 ]; then
    echo "WARNING: could not load any CUDA module — training will run on CPU."
fi

# MPS removed: each torchrun rank owns a separate physical H100 (3 GPUs, 3 ranks).
# MPS is for sharing one GPU across multiple processes — not applicable here and
# Error 805 (MPS client failed to connect) blocks CUDA init on all ranks.

_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs

RUNG=${RUNG:-A}
EPOCHS=${EPOCHS:-200}
FRESH=${FRESH:-0}
RLOO_K=${RLOO_K:-4}        # RLOO k=4 is the default — rollout baseline fails at init (advantage≈0)
WARMSTART=${WARMSTART:-0}  # 1 = warm-start from previous rung best_actor.pt (curriculum chain)
# Guard: Alliance Canada clusters set ARCH=x86_64 in the environment.
case "${ARCH:-hetgat}" in hetgat|simplegat) ;; *) ARCH=hetgat ;; esac
DYNAMIC=${DYNAMIC:-0}      # 1 = on-demand training: random partial-order visibility per batch
ENTROPY_COEF=${ENTROPY_COEF:-0.01}   # 0.01 = new default (was 0.005, raised to prevent plateau)
LR=${LR:-1e-4}             # use 7e-5 for RLOO k>=8 to prevent LR-induced divergence at epoch 15+
LR_DECAY=${LR_DECAY:-0.97} # 0.97 = faster decay than old 0.99; halves LR every ~23 epochs

# Variant suffix keeps different runs in separate folders/CSVs.
if [ "${RLOO_K}" -gt 0 ]; then
    SUFFIX="-rloo${RLOO_K}"
    EXTRA_ARGS="--rloo-k ${RLOO_K}"
elif [ "$ARCH" = "simplegat" ]; then
    SUFFIX="-simplegat"
    EXTRA_ARGS="--arch simplegat"
else
    SUFFIX=""
    EXTRA_ARGS=""
fi

# RLOO + simplegat combo (if ever needed)
if [ "${RLOO_K}" -gt 0 ] && [ "$ARCH" = "simplegat" ]; then
    EXTRA_ARGS="--rloo-k ${RLOO_K} --arch simplegat"
    SUFFIX="-simplegat-rloo${RLOO_K}"
fi

if [ "${DYNAMIC}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --dynamic"
    SUFFIX="${SUFFIX}-dynamic"
fi

FOLDER="CE-PDPTW-${RUNG}${SUFFIX}-HetGAT"
CSV_TAG="${RUNG}${SUFFIX}"
CKPT="$FOLDER/checkpoint.pth"

if [ "${FRESH}" = "1" ]; then
    echo "FRESH=1: removing $FOLDER/ and logs/training_${CSV_TAG}.csv for clean restart."
    rm -rf "$FOLDER"
    rm -f  "logs/training_${CSV_TAG}.csv"
elif [ -f "$CKPT" ]; then
    echo "Found $CKPT — will RESUME from last saved epoch."
    echo "To start fresh instead: sbatch --export=RUNG=${RUNG},RLOO_K=${RLOO_K},FRESH=1 submit_nibi.sh"
fi

echo "========================================================"
echo " CE-PDPTW training (Nibi H100)  |  Rung: $RUNG  |  Epochs: $EPOCHS  |  RLOO_K: $RLOO_K  |  ARCH: $ARCH"
echo " LR: $LR  |  LR_DECAY: $LR_DECAY  |  ENTROPY_COEF: $ENTROPY_COEF"
# ── Detect actual allocated GPU count ──────────────────────────────
# Previously hardcoded --nproc_per_node=3, which crashed with
#   "CUDA error: invalid device ordinal"
# when SLURM allocated fewer than 3 GPUs (e.g. the Nibi GPU partition
# was tight and only 1-2 H100s were available).  Detect at launch time:
#   1) CUDA_VISIBLE_DEVICES (set by SLURM) — count commas + 1
#   2) nvidia-smi -L line count  — fallback if CUDA_VISIBLE_DEVICES unset
#   3) SLURM_GPUS_ON_NODE         — last resort
N_GPUS=0
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if [ "$N_GPUS" -le 0 ] && [ -n "$SLURM_GPUS_ON_NODE" ]; then
    N_GPUS="$SLURM_GPUS_ON_NODE"
fi
if [ "$N_GPUS" -le 0 ]; then
    echo "WARNING: no GPUs detected — falling back to single-rank CPU run."
    N_GPUS=1
fi

echo " Node: $(hostname)  |  $(date)"
echo " GPUs allocated: $N_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo "========================================================"

export BL_EVAL_FREQ=5
# Fail fast when a rank crashes: 120s instead of the default 600s.
# Keeps the NCCL watchdog from wasting 10 min of GPU time on a dead peer.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

# Use a per-job unique rendezvous port so two chained jobs on the same node
# don't collide if scheduled back-to-back (29500 + last 3 digits of JOBID).
RDZV_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

python - <<'PROBE'
import torch, sys
ok = torch.cuda.is_available()
# Use real CUDA count; DirectML can fake device_count=1 even when CUDA unavailable.
n = torch.cuda.device_count() if ok else 0
print(f'[probe] CUDA available={ok}  device_count={n}')
for i in range(n):
    try:
        print(f'[probe]   GPU {i}: {torch.cuda.get_device_name(i)}')
    except Exception as e:
        print(f'[probe]   GPU {i}: <name unavailable: {e}>')
if not ok:
    print('[probe] ERROR: no CUDA GPUs — aborting before torchrun', file=sys.stderr)
    print('[probe] Check: module load cuda/12.2 loaded, torch installed with CUDA support.', file=sys.stderr)
    print('[probe] Verify: python -c "import torch; print(torch.version.cuda)"', file=sys.stderr)
    sys.exit(1)
PROBE
PROBE_EXIT=$?
if [ $PROBE_EXIT -ne 0 ]; then
    echo "ERROR: CUDA probe failed (exit $PROBE_EXIT) — torchrun will NOT be launched."
    echo "Diagnosis steps on a login node:"
    echo "  module purge && module load python/3.10 scipy-stack cuda/12.2"
    echo "  source ~/py310_nibi/bin/activate"
    echo "  python -c \"import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)\""
    echo "If CUDA shows False, reinstall torch with the CUDA 12.1 wheel:"
    echo "  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node=$N_GPUS \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$RDZV_PORT \
    VRP_Rollout_train.py --rung "$RUNG" --epochs "$EPOCHS" \
        ${EXTRA_ARGS} \
        $([ -n "$SUFFIX" ] && echo "--variant ${SUFFIX#-}") \
        $([ "${WARMSTART}" = "1" ] && echo "--warmstart") \
        --entropy-coef "$ENTROPY_COEF" \
        --lr           "$LR" \
        --lr-decay     "$LR_DECAY"

echo "========================================================"
echo " Training complete: $(date)"
echo " Checkpoint: $FOLDER/checkpoint.pth"
echo " Best model: $FOLDER/best_actor.pt"
echo " Next rung:  sbatch --export=RUNG=$(echo ABCD | grep -oP "(?<=${RUNG}).") submit_nibi.sh"
echo " Evaluate:   sbatch --export=RUNG=${RUNG} submit_evaluate.sh"
echo "========================================================"
