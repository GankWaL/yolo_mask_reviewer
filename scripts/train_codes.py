#!/usr/bin/env python3
"""12자리 제품코드 클래스 YOLO-seg 를 학습한다 (헤드 교체, LH/RH 거울상이므로 좌우반전 증강 금지).

  python scripts/train_codes.py DATA.yaml [--base MODEL.pt] [--name NAME] [--epochs 40] [--device 0]

  DATA.yaml  train/val 이 목록이어도 된다 (record + field 합본). names 는 build_record_yolo.py 의 것.
  --base     시작 가중치. 기본은 운영 모델(도메인 적응된 백본). 클래스 수가 달라 ultralytics 가 헤드를 새로 만든다.
train_round.py 와 달리 클래스 이름 일치 검사를 하지 않는다. fliplr=0 은 바꾸지 않는다.
결과: RUNS/<name>/weights/best.pt
"""
import argparse
import datetime as dt
import os

DEFAULT_BASE = os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo11s_best_20260919.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--runs', default=os.path.expanduser('~/jhw/data/SL/runs'))
    ap.add_argument('--name', default='codes_' + dt.datetime.now().strftime('%Y%m%d_%H%M'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr0', type=float, default=0.001)
    ap.add_argument('--optimizer', default='AdamW')
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--freeze', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    from ultralytics import YOLO
    if a.resume:
        YOLO(os.path.join(a.runs, a.name, 'weights', 'last.pt')).train(resume=True)
        return
    YOLO(a.base).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        project=a.runs, name=a.name, exist_ok=False,
        optimizer=a.optimizer, lr0=a.lr0, cos_lr=True, warmup_epochs=1.0, warmup_bias_lr=0.0,
        freeze=a.freeze or None, patience=a.patience, close_mosaic=8,
        fliplr=0.0, flipud=0.0,  # LH/RH 는 거울상 → 반전 금지
        plots=True, verbose=True,
    )


if __name__ == '__main__':
    main()
