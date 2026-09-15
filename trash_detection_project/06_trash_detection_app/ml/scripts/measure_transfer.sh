#!/bin/bash
# 전이학습 모델 8개의 클래스별 F1 최대 임계값을 같은 test 500장으로 측정한다.
# WSL 에서 실행: wsl bash /mnt/c/.../ml/scripts/measure_transfer.sh
set -u
ML=/mnt/c/Users/USER/Desktop/yolo_trash_detection/ml
OUT=$HOME/litert_export/w_transfer
DATA=$HOME/litert_export/testset.yaml
source "$HOME/litert_export/.venv/bin/activate"
cd "$ML/scripts" || exit 1

MODELS="transfer_bbox_stage1 transfer_bbox_stage2 transfer_bbox_stage3 transfer_bbox_final transfer_seg_stage1 transfer_seg_stage2 transfer_seg_stage3_final transfer_seg_final_rejected"

for m in $MODELS; do
  echo "############ $m ############"
  python eval_inference.py --weights "$OUT/$m.pt" --data "$DATA" --device cpu 2>&1 \
    | sed -n '/3) 클래스별/,$p'
done
