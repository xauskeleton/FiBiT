#!/usr/bin/env bash
# =============================================================================
# FiBiT ablations on Swin-S (CIFAR-100). Run scripts/run_main.sh first: the ablations
# reuse its fine-tuned baseline (ckpt_swin/ft) and pruned model (ckpt_swin/pruned).
#
#   PART=prune      Table 2: importance x allocation x r_p, 30 recovery epochs
#   PART=partition  Table 4: partition {fisher, magnitude} x DPLR {on, off} x A1/A2
#   PART=unpruned   Table 3: hybrid binarization of the UNPRUNED Swin-S (r_b 0.95/0.99)
#
# USAGE:
#   PART=prune bash scripts/run_ablations.sh
#   PART=partition bash scripts/run_ablations.sh
#   PART=unpruned BR_LIST="0.99" bash scripts/run_ablations.sh
#   QUICK=1 PART=partition bash scripts/run_ablations.sh      # smoke test
# =============================================================================
set -euo pipefail

PY="${PY:-python}"
MODEL="swin_small_patch4_window7_224"
DATA_DIR="${DATA_DIR:-}"
SEED="${SEED:-3407}"
BATCH="${BATCH:-64}"
ACT_INIT="${ACT_INIT:-test}"       # see scripts/run_main.sh
DBG=""; [ "${QUICK:-0}" = "1" ] && DBG="--debug"   # smoke test
PART="${PART:?set PART=prune, partition or unpruned}"
export PYTHONUTF8=1

FT="ckpt_swin/ft/best.pth"
PRUNED="ckpt_swin/pruned/best_pruned_model.pt"
ABL="ckpt_swin/ablation"
mkdir -p "$ABL"
DD=(); [ -n "$DATA_DIR" ] && DD=(--data-dir "$DATA_DIR")

qat() {  # qat <out-dir> <extra args...>
  local out="$1"; shift
  [ -f "$out/best.pth" ] && { echo ">> skip $out (done)"; return; }
  $PY fibit/qat.py --dataset cifar100 ${DD[@]+"${DD[@]}"} --model "$MODEL" \
    --act-init "$ACT_INIT" --epochs 50 --patience 15 --lr 1e-4 --batch-size "$BATCH" \
    --fim-batches 5 --num-workers 2 --seed "$SEED" --out-dir "$out" "$@" $DBG
}

case "$PART" in
  prune)
    for rp in ${RP_LIST:-0.5 0.7 0.9}; do
      for cell in "magnitude global" "fisher per_layer" "fisher global"; do
        set -- $cell; imp="$1"; mode="$2"
        out="$ABL/prune_${imp}_${mode}_rp${rp}"
        [ -f "$out/best_pruned_model.pt" ] && { echo ">> skip $out (done)"; continue; }
        $PY fibit/prune_cifar.py --dataset cifar100 ${DD[@]+"${DD[@]}"} --model "$MODEL" \
          --baseline-ckpt "$FT" --importance "$imp" --mag-norm l2sq \
          --rank-mode "$mode" --global-metric per_param \
          --head-ratio "$rp" --mlp-ratio "$rp" --min-heads 1 --mlp-keep-frac 0.05 \
          --fim-batches 10 --epochs 30 --patience 15 --lr 1e-4 --weight-decay 1e-4 \
          --label-smoothing 0.1 --batch-size "$BATCH" --num-workers 4 --seed "$SEED" \
          --out-dir "$out" $DBG
      done
    done ;;
  partition)
    for part in fisher magnitude; do
      for dplr in on off; do
        for act in 1 2; do
          extra=(); [ "$dplr" = "on" ] && extra=(--use-dplr-loss --dplr-lambda 3000)
          qat "$ABL/qat_${part}_${dplr}dplr_act${act}" --init-model "$PRUNED" \
            --binary-ratio 0.99 --partition "$part" --act-bits "$act" ${extra[@]+"${extra[@]}"}
        done
      done
    done ;;
  unpruned)
    for br in ${BR_LIST:-0.95 0.99}; do
      qat "$ABL/unpruned_apb_br${br}_act2" --baseline-ckpt "$FT" \
        --binary-ratio "$br" --partition magnitude --act-bits 2 \
        --use-dplr-loss --dplr-lambda 3000
    done ;;
  *) echo "unknown PART=$PART"; exit 1 ;;
esac
