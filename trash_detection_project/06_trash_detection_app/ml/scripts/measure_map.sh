#!/bin/bash
# 전이학습 bbox 모델의 mAP50-95 를 기존 표와 같은 지표로 측정한다.
set -u
OUT=$HOME/litert_export/w_transfer
DATA=$HOME/litert_export/testset.yaml
source "$HOME/litert_export/.venv/bin/activate"
cd /mnt/c/Users/USER/Desktop/yolo_trash_detection/ml/scripts || exit 1
for m in transfer_bbox_stage1 transfer_bbox_stage2 transfer_bbox_stage3 transfer_bbox_final; do
  echo "############ $m ############"
  python eval_inference.py --weights "$OUT/$m.pt" --data "$DATA" --device cpu 2>&1 \
    | sed -n '/== 2)/,/== 3)/p'
done
