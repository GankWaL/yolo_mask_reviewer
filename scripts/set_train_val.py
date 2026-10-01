#!/usr/bin/env python3
"""학습 폴더의 val 을 고정 검증셋으로 바꾼다 — 폴더 안의 val 분할은 train 으로 합치고 dataset.yaml 의 val 은 검증 폴더를 가리키게 (2026-10-01).

  python scripts/set_train_val.py TRAIN --valid VALID

  TRAIN  학습 폴더 (train_class<N>, 루트 = 제품코드, sort5/ = 5클래스). 제자리에서 고친다.
  VALID  고정 검증 폴더 (valid_class<N>, 같은 구조). 학습 폴더와 겹치는 프레임이 있으면 멈춘다.

결과: TRAIN/images/train 에 전 프레임, images/val 은 비움(삭제), dataset.yaml 의 val = VALID/images/val (절대 경로), manifest.csv 의 split 은 train.
학습 중 val 수치가 곧 고정 검증 수치가 되고, 학습에는 검증셋을 뺀 모든 프레임이 쓰인다 (jhw 2026-10-01).
"""
import argparse
import csv
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mask_reviewer.dataset import Dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('train')
    ap.add_argument('--valid', required=True)
    a = ap.parse_args()
    t, v = Dataset(a.train), Dataset(a.valid)
    torig = {s.partition('__')[2] for s in t.stems}
    dup = [s for s in v.stems if s.partition('__')[2] in torig]
    if dup:
        sys.exit(f'검증셋 {len(dup)}장이 학습 폴더에 있습니다 (예 {dup[:3]}). 먼저 rebuild_valid_disjoint.py 로 검증셋을 다시 만드세요.')
    for sub in ('', 'sort5'):
        root = os.path.join(a.train, sub)
        n = 0
        for d in ('images', 'labels'):
            src = os.path.join(root, d, 'val')
            if os.path.isdir(src):
                for fn in os.listdir(src):
                    os.replace(os.path.join(src, fn), os.path.join(root, d, 'train', fn))
                    n += d == 'images'
                os.rmdir(src)
        mp = os.path.join(root, 'manifest.csv')
        if os.path.exists(mp):
            with open(mp, newline='', encoding='utf-8') as f:
                rd = csv.DictReader(f)
                rows, fields = list(rd), rd.fieldnames
            for r in rows:
                r['split'] = 'train'
            with open(mp, 'w', newline='', encoding='utf-8') as f:
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader()
                wr.writerows(rows)
        yp = os.path.join(root, 'dataset.yaml')
        with open(yp, encoding='utf-8') as f:
            lines = f.read().splitlines()
        head = [l for l in lines if l.startswith('#')]
        y = yaml.safe_load('\n'.join(l for l in lines if not l.startswith('#')))
        y['train'] = 'images/train'
        y['val'] = os.path.abspath(os.path.join(a.valid, sub, 'images', 'val'))
        with open(yp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(head) + f'\n# val = 고정 검증셋 {os.path.abspath(os.path.join(a.valid, sub))} (set_train_val.py); 폴더 안의 val 분할 {n}장은 train 으로 합침\n')
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
        print(f'{root}: val {n}장 → train, val = {y["val"]}')
    t2 = Dataset(a.train)
    print(f'train {len(t2.stems)}장 (split {sorted(set(t2._split.values()))}), 검증 {len(v.stems)}장, 겹침 0')


if __name__ == '__main__':
    main()
