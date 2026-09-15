#!/bin/bash
# 전이학습 모델 8개를 안드로이드용 LiteRT(.tflite, FP32, nms=None) 로 변환한다.
# WSL 에서 실행: wsl bash /mnt/c/.../ml/scripts/convert_transfer.sh
set -u
SRC=/mnt/c/Users/USER/Desktop/yolo_trash_detection/ml/weights
OUT=$HOME/litert_export/w_transfer
mkdir -p "$OUT"
cd "$OUT" || exit 1
source "$HOME/litert_export/.venv/bin/activate"

MODELS="transfer_bbox_stage1 transfer_bbox_stage2 transfer_bbox_stage3 transfer_bbox_final transfer_seg_stage1 transfer_seg_stage2 transfer_seg_stage3_final transfer_seg_final_rejected"

for m in $MODELS; do
  echo "=== $m ==="
  if [ ! -f "$SRC/$m.pt" ]; then echo "SKIP $m (원본 없음)"; continue; fi
  cp -f "$SRC/$m.pt" "$OUT/$m.pt"
  yolo export model="$OUT/$m.pt" format=litert nms=None imgsz=640 2>&1 | tail -2
done

echo "=== 결과 ==="
find "$OUT" -name "*.tflite" -printf "%p %s\n" | sort
