#!/usr/bin/env python3
"""검수 결과(내보내기 폴더)로 운영 YOLO-seg 모델을 한 라운드 파인튜닝한다 (ultralytics, yolo26 환경).

  conda activate yolo26
  python scripts/train_round.py EXPORT_DIR [--base MODEL.pt] [--extra-data A.yaml B.yaml] [--epochs 30] ...

  EXPORT_DIR   mask_reviewer 내보내기 폴더 (dataset.yaml 포함). val 비율 0 으로 내보냈으면 train=val 로 학습한다
               (과적합 감시 불가 — 실제 라운드에서는 val 비율을 두고 내보낼 것).
  --base       기준 가중치. 기본은 운영 코드가 쓰는 SL_Inspection_Automation/models/yolo11s_best_20260326.pt.
  --extra-data 원본 학습 데이터 등 함께 학습할 데이터셋 yaml (여러 개 가능). 검수분만으로 학습하면 다른 제품을 잊을 수
               있으므로 원본 학습셋이 있는 PC 에서는 반드시 같이 준다. 클래스 이름·순서가 기준 모델과 같아야 한다.
  --freeze     앞쪽 N 개 레이어 동결 (기본 10 = 백본). 적은 데이터로 잊음을 줄인다.
  --optimizer  기본 AdamW. ultralytics 기본 'auto' 는 lr0 를 무시(AdamW 0.0011)하므로 lr0 를 지키려면 명시해야 한다.

결과: RUNS/<name>/weights/best.pt 와 그 사본 RUNS/<name>/<base>_round_<YYYYMMDD>.pt.
운영 배치는 사본을 SL_Inspection_Automation/models/ 에 넣고 yolo_base_model 이름을 바꾸는 것으로 한다 (이 스크립트는 복사하지 않는다).
"""
import argparse
import datetime as dt
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.expanduser('~/SL_Inspection_Automation/models/yolo11s_best_20260326.pt')


def load_yaml(p):
    with open(p, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def resolve_split(y, key):
    """dataset.yaml 의 train/val 항목을 절대경로 목록으로."""
    v = y.get(key)
    if v is None:
        return []
    root = y.get('path') or os.path.dirname(os.path.abspath(y['__file__']))
    items = v if isinstance(v, list) else [v]
    return [os.path.normpath(os.path.join(root, it)) if not os.path.isabs(it) else it for it in items]


def names_of(y):
    n = y.get('names')
    if isinstance(n, dict):
        return {int(k): str(v) for k, v in n.items()}
    return {i: str(v) for i, v in enumerate(n or [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('export_dir')
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--extra-data', nargs='*', default=[], help='함께 학습할 데이터셋 yaml')
    ap.add_argument('--runs', default=os.path.join(os.path.dirname(HERE), 'runs'))
    ap.add_argument('--name', default=None, help='run 이름 (기본 round_<YYYYMMDD_HHMM>)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--freeze', type=int, default=10)
    ap.add_argument('--lr0', type=float, default=0.002)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--optimizer', default='AdamW',
                    help="명시 필수: ultralytics 'auto' 는 lr0 를 무시하고 AdamW lr 0.0011 로 바꾼다")
    ap.add_argument('--warmup-epochs', type=float, default=1.0)
    ap.add_argument('--warmup-bias-lr', type=float, default=0.0,
                    help='기본 0.1 은 워밍업 동안 bias lr 을 0.1 로 올려 사전학습 가중치를 흔든다 → 0')
    a = ap.parse_args()

    from ultralytics import YOLO

    exp_yaml = os.path.join(a.export_dir, 'dataset.yaml')
    if not os.path.exists(exp_yaml):
        sys.exit(f'dataset.yaml 이 없습니다: {a.export_dir}')
    base = YOLO(a.base)
    base_names = {int(k): str(v) for k, v in base.names.items()}
    train, val = [], []
    for p in [exp_yaml] + a.extra_data:
        y = load_yaml(p)
        y['__file__'] = p
        if names_of(y) != base_names:
            sys.exit(f'클래스가 기준 모델과 다릅니다: {p}\n  모델 {base_names}\n  데이터 {names_of(y)}')
        train += resolve_split(y, 'train')
        val += resolve_split(y, 'val')
    for d in train + val:
        if not os.path.isdir(d):
            sys.exit(f'이미지 폴더가 없습니다: {d}')
    if set(train) == set(val):
        print('경고: train 과 val 이 같습니다 (val 비율 0 으로 내보낸 폴더). 과적합을 감시할 수 없습니다.')

    name = a.name or 'round_' + dt.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(a.runs, exist_ok=True)
    merged = {'train': train if len(train) > 1 else train[0], 'val': val if len(val) > 1 else val[0],
              'names': base_names}
    merged_yaml = os.path.join(a.runs, name + '_data.yaml')
    with open(merged_yaml, 'w', encoding='utf-8') as f:
        f.write(f'# base={a.base}\n# sources={[exp_yaml] + a.extra_data}\n')
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
    print(f'기준 {a.base}\ntrain {train}\nval {val}\n→ {a.runs}/{name}')

    base.train(data=merged_yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
               project=a.runs, name=name, exist_ok=True, freeze=a.freeze, lr0=a.lr0, workers=a.workers,
               patience=a.patience, pretrained=True, plots=True, verbose=False,
               optimizer=a.optimizer, warmup_epochs=a.warmup_epochs, warmup_bias_lr=a.warmup_bias_lr,
               ## 강건성 기본 증강 (2026-09-23 jhw 규칙, docs/CLAUDE.md "학습 규칙") — 5클래스 대분류라 fliplr 은 기본(0.5) 유지
               hsv_h=0.03, hsv_s=0.8, hsv_v=0.6, degrees=10.0, translate=0.15, scale=0.7, copy_paste=0.1, bgr=0.05)
    best = os.path.join(a.runs, name, 'weights', 'best.pt')
    if not os.path.exists(best):
        sys.exit('best.pt 가 만들어지지 않았습니다')
    tag = os.path.splitext(os.path.basename(a.base))[0] + '_round_' + dt.datetime.now().strftime('%Y%m%d')
    out = os.path.join(a.runs, name, tag + '.pt')
    shutil.copy2(best, out)
    print(f'\n완료: {out}\n다음: python scripts/relabel_with_model.py DATASET --model {out}')


if __name__ == '__main__':
    main()
