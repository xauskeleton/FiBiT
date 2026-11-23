#!/usr/bin/env bash
# =============================================================================
# FiBiT main pipeline (Table 1 and the pruned rows of Table 3) for ONE backbone:
#   stage 1  FP fine-tuning on CIFAR-100
#   stage 2  Fisher-guided structural pruning (global, r_p = 0.5) + recovery
#   stage 3  hybrid binarization + DPLR-guided QAT, r_b in {0.95, 0.99} x A1/A2
#
# USAGE (from the repo root):
#   bash scripts/run_main.sh                                   # Swin-S
#   MODEL=deit_small_patch16_224 bash scripts/run_main.sh
#   MODEL=vit_small_patch16_224  bash scripts/run_main.sh
#   CONFIGS="0.99:2" bash scripts/run_main.sh                  # one QAT cell only
#   QUICK=1 bash scripts/run_main.sh                           # smoke test
#
# Finished stages are skipped (best.pth / best_pruned_model.pt already present).
# Outputs: ckpt_<family>/{ft,pruned,prune_quant_br<rb>_act<a>}/ with train.log,
# best.pth and best_packed.pt (the packed binary format whose size is reported).
# =============================================================================
set -euo pipefail

PY="${PY:-python}"
MODEL="${MODEL:-swin_small_patch4_window7_224}"
DATA_DIR="${DATA_DIR:-}"            # CIFAR-100 is downloaded here by torchvision
SEED="${SEED:-3407}"
BATCH="${BATCH:-64}"
CONFIGS="${CONFIGS:-0.95:2 0.99:2 0.95:1 0.99:1}"   # "r_b:act_bits"
# Activation-scale init: "test" reproduces the tables of the paper (scales set by the
# first test batch, the protocol of the original runs); "train" initializes them on a
# training batch instead (Appendix C.3).
ACT_INIT="${ACT_INIT:-test}"
export PYTHONUTF8=1

OUT="${OUT:-ckpt_${MODEL%%_*}}"
mkdir -p "$OUT"
DBG=""; [ "${QUICK:-0}" = "1" ] && DBG="--debug"
DD=(); [ -n "$DATA_DIR" ] && DD=(--data-dir "$DATA_DIR")

# ---- stage 1: FP fine-tuning ----
FT="$OUT/ft/best.pth"
if [ ! -f "$FT" ]; then
  $PY fibit/finetune.py --dataset cifar100 "${DD[@]}" --model "$MODEL" \
    --epochs 50 --patience 15 --lr 1e-4 --weight-decay 1e-4 --label-smoothing 0.1 \
    --batch-size "$BATCH" --num-workers 2 --seed "$SEED" --out-dir "$OUT/ft" $DBG
fi

# ---- stage 2: Fisher-guided structural pruning + recovery fine-tuning ----
PRUNED="$OUT/pruned/best_pruned_model.pt"
if [ ! -f "$PRUNED" ]; then
  $PY fibit/prune_cifar.py --dataset cifar100 "${DD[@]}" --model "$MODEL" \
    --baseline-ckpt "$FT" --importance fisher --rank-mode global --global-metric per_param \
    --head-ratio 0.5 --mlp-ratio 0.5 --min-heads 1 --mlp-keep-frac 0.05 --fim-batches 10 \
    --epochs 50 --patience 15 --lr 1e-4 --weight-decay 1e-4 --label-smoothing 0.1 \
    --batch-size "$BATCH" --num-workers 4 --seed "$SEED" --out-dir "$OUT/pruned" $DBG
fi

# ---- stage 3: hybrid binarization + DPLR-guided QAT ----
for c in $CONFIGS; do
  BR="${c%:*}"; ACT="${c#*:}"
  Q="$OUT/prune_quant_br${BR}_act${ACT}"
  [ -f "$Q/best.pth" ] && { echo ">> skip $Q (done)"; continue; }
  $PY fibit/qat.py --dataset cifar100 "${DD[@]}" --model "$MODEL" --init-model "$PRUNED" \
    --binary-ratio "$BR" --partition magnitude --act-bits "$ACT" --act-init "$ACT_INIT" \
    --epochs 50 --patience 15 --lr 1e-4 --batch-size "$BATCH" --fim-batches 5 \
    --use-dplr-loss --dplr-lambda 3000 --fim-p1 1.0 --fim-p2 1.0 \
    --num-workers 2 --seed "$SEED" --out-dir "$Q" $DBG
done

grep -H "DONE. Best val top1" "$OUT"/prune_quant_*/train.log 2>/dev/null || true
