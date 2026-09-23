#!/usr/bin/env bash
# (E, 2026-09-23) 운영 5클래스 yolo26 변형 실험 세팅 — 실행은 하지 않았다. 필요할 때 한 줄씩 골라 돌린다.
#   기준: runs/sort5_y26s_r3_20260923 (yolo26s, imgsz 640, batch 16) = yolo26s_best_260923.pt, val 마스크 mAP50 0.995 / 50-95 0.987, 2.8 ms
#   데이터: ~/jhw/data/SL/yolo26_dataset/20260923/holes_all_export_r3/dataset.yaml (5클래스, 구멍 유지)
#   GPU: RTX 4070 Ti 만 (CUDA_DEVICE_ORDER=PCI_BUS_ID 기준 1)
#   비교: scripts/exp/eval_sort5.py <weights...> 로 같은 val 에서 mAP·추론 시간을 나란히 잰다
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=${PY:-$HOME/anaconda3/envs/yolo_mask_reviewer/bin/python}
D=$HOME/jhw/data/SL
DATA=$D/yolo26_dataset/20260923/holes_all_export_r3/dataset.yaml
DEV=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep 4070 | cut -d, -f1)
VAR=${1:-help}
train() {  # name base imgsz batch
  cd $D && $PY - <<PYEOF
from ultralytics import YOLO
YOLO('$2').train(data='$DATA', epochs=40, imgsz=$3, batch=$4, device='$DEV', workers=8, project='$D/runs', name='$1', exist_ok=False,
    optimizer='AdamW', lr0=0.001, cos_lr=True, warmup_epochs=1.0, warmup_bias_lr=0.0, patience=12, close_mosaic=8, fliplr=0.5, flipud=0.0, plots=True)
PYEOF
}
case "$VAR" in
  s768)   train sort5_y26s_768_$(date +%Y%m%d) yolo26s-seg.pt 768 12 ;;   # 경계 정밀도 ↑ (마스크 프로토 192²), 추론 약 1.4배
  m640)   train sort5_y26m_640_$(date +%Y%m%d) yolo26m-seg.pt 640 8 ;;    # 중간 크기, 추론 약 2배
  m768)   train sort5_y26m_768_$(date +%Y%m%d) yolo26m-seg.pt 768 6 ;;
  trt)    # 현장 PC 에서: TensorRT FP16 엔진 (ultralytics ≥8.4, tensorrt 설치 필요). 운영 코드는 YOLO('<name>.engine') 로 그대로 로드 가능
          echo "cd ~/SL_Inspection_Automation/models && yolo export model=yolo26s_best_20260923.pt format=engine half=True imgsz=640 device=0" ;;
  *) echo "usage: $0 {s768|m640|m768|trt}"; exit 1 ;;
esac
