#!/usr/bin/env python3
"""(2번, 2026-09-23) 운영 5클래스 yolo26s 를 증강 강화로 재학습 — 사무실(다른 조명·벨트색)에서의 마스크 불안정 대응.
기준(sort5_y26s_r3_20260923) 대비 바꾼 것: hsv_h 0.015→0.03, hsv_s 0.7→0.8, hsv_v 0.4→0.6, degrees 0→10, translate 0.1→0.15, scale 0.5→0.7,
copy_paste 0→0.1, bgr 0→0.05(채널 순서 바꿈), albumentations 설치 시 Blur/MedianBlur/ToGray/CLAHE p=0.01 자동 적용. 데이터·에폭·옵티마이저는 동일.
  CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/exp/train_sort5_aug.py [device]
"""
import sys
from ultralytics import YOLO
dev = sys.argv[1] if len(sys.argv) > 1 else '1'
YOLO('yolo26s-seg.pt').train(data='/home/dxr-core-desktop/jhw/data/SL/yolo26_dataset/20260923/holes_all_export_r3/dataset.yaml',
    epochs=40, imgsz=640, batch=16, device=dev, workers=8, project='/home/dxr-core-desktop/jhw/data/SL/runs', name='sort5_y26s_aug_20260923', exist_ok=False,
    optimizer='AdamW', lr0=0.001, cos_lr=True, warmup_epochs=1.0, warmup_bias_lr=0.0, patience=15, close_mosaic=8,
    hsv_h=0.03, hsv_s=0.8, hsv_v=0.6, degrees=10.0, translate=0.15, scale=0.7, fliplr=0.5, flipud=0.0, mosaic=1.0, copy_paste=0.1, bgr=0.05,
    plots=True, verbose=True)
