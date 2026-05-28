#!/bin/bash
# ============================================================
# Narval — RL smoke test + decoder profile + reward check
#
# Purpose (run this BEFORE the full training job):
#   1. Smoke test: 5 epochs on tiny data — validates the full
#      pipeline (data → forward → REINFORCE loss → rollout
#      baseline t-test update → checkpoint → validation cost).
#   2. Decoder profile: measures encoder vs sequential/parallel
#      decoder timing — tells you if parallel_select will help.
#   3. Reward convergence check: reads the smoke training log
#      and reports whether REINFORCE is learning. If rewards
#      are flat or diverging even on tiny data, prints a PPO
#      recommendation with the specific thresholds to watch.
#
# Usage:
#   sbatch submit_smoke_rl.sh              # Rung A (default)
#   sbatch --export=RUNG=B submit_smoke_rl.sh
# ============================================================

#SBATCH --job-name=CE-PDPTW-smoke-rl
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/smoke-rl-%j.out
#SBATCH --error=logs/smoke-rl-%j.err

module purge
module load python/3.10 scipy-stack cuda/12.2
source ~/py310_nibi/bin/activate

_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW" ] && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW"
cd "$_PROJ"
mkdir -p logs

RUNG=${RUNG:-A}

echo "========================================================"
echo " CE-PDPTW RL smoke test | Rung: $RUNG | $(date)"
echo " Node: $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo "========================================================"

# ── Step 1: Smoke test ───────────────────────────────────────
# 5 epochs, 256 train / 64 val, single GPU.
# Validates: data → encoder → decoder → REINFORCE loss →
#   rollout baseline t-test → checkpoint → val cost.
# Writes: logs/training_${RUNG}.csv (reward/loss/grad per epoch)
#         CE-PDPTW-${RUNG}-HetGAT/costs.txt
#         CE-PDPTW-${RUNG}-HetGAT/checkpoint.pth

echo ""
echo "=== Step 1: Smoke test (5 epochs, tiny data) ==="
python VRP_Rollout_train.py --smoke --rung "$RUNG" --rloo-k 4
SMOKE_EXIT=$?

if [ $SMOKE_EXIT -ne 0 ]; then
    echo "SMOKE TEST FAILED (exit $SMOKE_EXIT)"
    echo "Fix the error above before submitting full training."
    exit $SMOKE_EXIT
fi
echo "Smoke test PASSED."

# ── Step 2: Decoder profile ──────────────────────────────────
# Measures: encoder / sequential decoder / parallel decoder /
#   baseline.eval timing.
# Key question: is the encoder >60% of forward pass time?
#   YES → sequential decoder not the bottleneck, keep as-is
#   NO  → parallel_select saves meaningful time at inference

echo ""
echo "=== Step 2: Encoder vs decoder timing profile ==="
python profile_decoder.py \
    --rung "$RUNG" \
    --batch-size 64 \
    --warmup 10 \
    --repeats 30

# Also profile the next rung for scaling context
declare -A NEXT=([A]=B [B]=C [C]=D [D]=D)
NEXT_RUNG="${NEXT[$RUNG]}"
if [ "$NEXT_RUNG" != "$RUNG" ]; then
    echo ""
    echo "--- Scaling reference: next rung ($NEXT_RUNG) ---"
    python profile_decoder.py \
        --rung "$NEXT_RUNG" \
        --batch-size 32 \
        --warmup 5 \
        --repeats 20
fi

# ── Step 3: Reward convergence check ────────────────────────
# Reads the smoke training CSV and checks:
#   - val_cost trend (should decrease = improving policy)
#   - grad_norm stability (should stay in 0.05–2.0 range)
#   - baseline update frequency (policy must improve to trigger)
#   - loss sign (should oscillate around 0, not blow up)
# Prints a REINFORCE-vs-PPO recommendation.

echo ""
echo "=== Step 3: Reward convergence check ==="
CSV_PATH="logs/training_${RUNG}.csv"

python - "$CSV_PATH" "$RUNG" << 'PYEOF'
import sys, csv, os

csv_path = sys.argv[1]
rung     = sys.argv[2]

if not os.path.exists(csv_path):
    print(f"  No training CSV found at {csv_path}")
    print("  (Smoke test may have written 0 complete epochs — check the log above)")
    sys.exit(0)

rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        rows.append(row)

if not rows:
    print("  CSV is empty — smoke test wrote no epochs.")
    sys.exit(0)

def flt(x):
    try: return float(x)
    except: return float('nan')

epochs       = [int(r['epoch'])              for r in rows]
val_costs    = [flt(r['val_cost'])           for r in rows]
rewards      = [flt(r['train_reward_mean'])  for r in rows]
losses       = [flt(r['train_loss_mean'])    for r in rows]
gnorms       = [flt(r['grad_norm_mean'])     for r in rows]
gnorms_max   = [flt(r['grad_norm_max'])      for r in rows]
bl_updates   = [r.get('baseline_updated','0').strip() for r in rows]

n = len(rows)

W = 58
print('=' * W)
print(f'  REINFORCE convergence check — Rung {rung} ({n} epochs)')
print('=' * W)

# Val cost trend (primary metric — should decrease)
vc_first, vc_last = val_costs[0], val_costs[-1]
vc_delta  = vc_last - vc_first
vc_pct    = 100.0 * vc_delta / max(abs(vc_first), 1e-9)
vc_ok     = vc_delta < 0
print(f'\n  Validation cost:')
print(f'    epoch 0 → {n-1}: {vc_first:.4f} → {vc_last:.4f}  '
      f'({"↓" if vc_ok else "↑"} {abs(vc_pct):.1f}%  '
      f'{"IMPROVING" if vc_ok else "NOT IMPROVING"})')

# Gradient norm
gn_mean = sum(gnorms) / max(len(gnorms), 1)
gn_max  = max(gnorms_max) if gnorms_max else float('nan')
gn_ok   = 0.05 <= gn_mean <= 2.5
print(f'\n  Gradient norm:')
print(f'    mean={gn_mean:.4f}  max={gn_max:.4f}  '
      f'{"OK" if gn_ok else "WARNING: out of 0.05–2.5 range"}')
if gn_mean < 0.05:
    print('    >> Vanishing gradients — learning rate may be too low,')
    print('       or policy is stuck at a local minimum.')
elif gn_mean > 2.5:
    print('    >> Large gradients — try lowering lr or raising max_grad_norm.')

# Baseline updates
n_updates = sum(1 for u in bl_updates if u in ('1','True','true'))
print(f'\n  Rollout baseline updates: {n_updates}/{n} epochs')
if n_updates == 0:
    print('    >> Baseline never updated — policy did not improve enough')
    print('       to pass the t-test threshold (p<0.05).')
    print('       Normal for epoch 0–3 on tiny smoke data; watch for this')
    print('       in the first 10 real epochs.')

# Loss sign
loss_final = losses[-1] if losses else float('nan')
print(f'\n  Actor loss (final epoch): {loss_final:.5f}')
print(f'    (Expected near 0 or small negative — policy improving)')

# ── Verdict ──────────────────────────────────────────────────
print()
print('-' * W)

issues = []
if not vc_ok:
    issues.append('val_cost not decreasing')
if not gn_ok:
    issues.append(f'grad_norm {"too low" if gn_mean < 0.05 else "too high"}')
if n_updates == 0 and n >= 3:
    issues.append('baseline never updated')

if not issues:
    print('  VERDICT: REINFORCE is working correctly on smoke data.')
    print()
    print('  Next steps:')
    print('    1. Submit full Rung A training:  sbatch submit_cc.sh')
    print('    2. Watch val_cost in logs/training_A.csv every 10 epochs.')
    print('    3. If val_cost plateau persists past epoch 30 → consider PPO.')
    print()
    print('  Parallel_select: see Step 2 profile above.')
    print('  If encoder > 60% of fwd pass → sequential decoder is fine.')
    print('  If decoder_seq >> decoder_par → enable BL_EVAL_FREQ=5 first')
    print('  (zero quality cost), then consider parallel_select for training.')
else:
    print(f'  VERDICT: WARNING — issues detected: {", ".join(issues)}')
    print()
    print('  Before switching to PPO, try these fixes in order:')
    print()
    print('  [1] Increase batch size (smoke uses 32; real training uses 256+)')
    print('      High variance from small batches mimics "not learning".')
    print('      sbatch submit_cc.sh  (uses full batch_size from RUNG_CONFIG)')
    print()
    print('  [2] Let it train for 20+ real epochs.')
    print('      REINFORCE on a rollout baseline typically needs 10–20 epochs')
    print('      before the baseline is meaningful enough to reduce variance.')
    print()
    print('  [3] If val_cost is STILL flat at epoch 25–30 of real training:')
    print('      Switch to PPO.')
    print()
    print('  Why PPO would help:')
    print('    - PPO uses a value network (critic) for variance reduction')
    print('      instead of the Monte-Carlo rollout baseline.')
    print('    - Better suited when reward signal is high-variance or sparse.')
    print('    - PPO clip ratio (ε=0.2) prevents destructive policy updates.')
    print('    - Lower sample efficiency but more stable gradient estimates.')
    print()
    print('  PPO switch checklist:')
    print('    - VRP_PPO_Model.py already exists in the repo.')
    print('    - PPO_train.py is the training entry point.')
    print('    - Swap sbatch submit_cc.sh → submit_ppo.sh (create from template).')

print('=' * W)
PYEOF

# ── Step 4: Artifact check ───────────────────────────────────
echo ""
echo "=== Step 4: Artifact verification ==="

COST_FILE="CE-PDPTW-${RUNG}-HetGAT/costs.txt"
CKPT_FILE="CE-PDPTW-${RUNG}-HetGAT/checkpoint.pth"
CSV_FILE="logs/training_${RUNG}.csv"

for f in "$COST_FILE" "$CKPT_FILE" "$CSV_FILE"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f  ($(du -sh "$f" 2>/dev/null | cut -f1))"
    else
        echo "  [MISSING] $f"
    fi
done

if [ -f "$COST_FILE" ]; then
    echo ""
    echo "  costs.txt contents (val cost per epoch):"
    cat "$COST_FILE"
fi

echo ""
echo "========================================================"
echo " Smoke + profile + convergence check complete: $(date)"
echo ""
echo " If all steps passed:"
echo "   sbatch submit_cc.sh                 # full Rung A training"
echo "   sbatch --export=RUNG=B submit_cc.sh # after Rung A completes"
echo ""
echo " Run baselines in parallel (no GPU needed):"
echo "   sbatch submit_baseline.sh"
echo "========================================================"
