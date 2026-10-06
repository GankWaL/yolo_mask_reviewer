#!/usr/bin/env python3
"""라벨 없는 가장자리 조각을 모델로 검출해 가장자리 제품 규칙(가운데 제품 면적의 10% 미만 → none)에 통과시킨다 (2026-10-01).

  python scripts/edge_gate_autolabel.py DATASET [--model M.pt] [--device 0] [--conf 0.25] [--iou 0.3] [--min-area 1500] [--dry]

  DATASET  검수 툴로 여는 split 구조 학습·검증 폴더 (루트 = 제품코드, sort5/ = 5클래스). 두 쪽 라벨을 제자리에서 고친다.
프레임마다 모델 검출 중 기존 라벨과 안 겹치는(IoU < --iou) 마스크를 후보로 두고, 기존 라벨 + 후보에 `codes_to_sort5.edge_none_rule` 을 적용한다.
규칙에서 none 이 된 후보만 none 인스턴스로 추가하고, 10% 이상인 후보는 추가하지 않는다. 단 **이미지 가장자리(--border-px)에 닿은 후보는
크기와 관계없이 none 으로 추가**한다 (jhw 2026-10-06, --no-border-all 로 끔). 결과: DATASET/edge_gate_report.csv (stem, 추가한 none 수, 넘긴 후보 수).
이미 none 라벨이 있는 프레임도 다시 본다 (기존 라벨과 안 겹치는 후보만 추가하므로 중복되지 않는다).
"""
import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codes_to_sort5 import NAMES as SORT5, edge_none_rule, major_map, major_of  # noqa: E402
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402

DEFAULT_MODEL = os.path.expanduser("~/jhw/data/SL/runs/sort5_y26s_aug_r12_20261002/weights/best.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.3)
    ap.add_argument('--min-area', type=float, default=1500.0)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--border-px', type=int, default=3, help='이 안쪽까지 닿으면 가장자리 조각')
    ap.add_argument('--no-border-all', action='store_true', help='가장자리 조각도 비율 게이트만 적용 (기본은 전부 none)')
    a = ap.parse_args()
    from ultralytics import YOLO
    ds = Dataset(a.dataset)
    s5 = Dataset(os.path.join(a.dataset, 'sort5'))
    major = major_map()
    cid = {v: k for k, v in ds.names.items()}
    model = YOLO(a.model)
    rows, tot = [], collections.Counter()
    for i in range(0, len(ds.stems), 16):
        part = ds.stems[i:i + 16]
        res = model.predict([ds.image_path(s) for s in part], imgsz=640, conf=a.conf, retina_masks=True, device=a.device, verbose=False)
        for s, r in zip(part, res):
            h, w = r.orig_shape
            text = ds.label_text(s)
            lab = ds.parse_instances(text, w, h)
            polys = [(ds.names[int(l.split()[0])], ' '.join(l.split()[1:])) for l in text.splitlines() if len(l.split()) >= 7]
            cand, border = [], []
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy() > 0.5:
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if m.sum() >= a.min_area and all(maskops.iou(m, l.mask) < a.iou for l in lab):
                        line = maskops.mask_to_yolo_line(0, m, w, h, 0.7, 16.0, keep_holes=ds.keep_holes)
                        if line:
                            ys, xs = np.nonzero(m)
                            b = a.border_px
                            touch = xs.min() < b or ys.min() < b or xs.max() >= w - b or ys.max() >= h - b
                            cand.append(('__cand__', ' '.join(line.split()[1:])))
                            border.append(touch)
            if not cand or not polys:
                continue
            gated = edge_none_rule(polys + cand)
            add = [('none', p) for (n, p), touch in zip(gated[len(polys):], border) if n == 'none' or (touch and not a.no_border_all)]
            skip = len(cand) - len(add)
            tot['frames'] += 1
            tot['added'] += len(add)
            tot['skipped'] += skip
            rows.append(dict(stem=s, code=ds.code(s), n_label=len(polys), added_none=len(add), skipped_candidates=skip))
            if add and not a.dry:
                new = polys + add
                sp = ds.split(s)
                with open(os.path.join(ds.root, 'labels', sp, s + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(''.join(f'{cid[n]} {p}\n' for n, p in new))
                with open(os.path.join(s5.root, 'labels', sp, s + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(''.join(f'{SORT5.index(major_of(n, major))} {p}\n' for n, p in new))
    with open(os.path.join(a.dataset, 'edge_gate_report.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['stem', 'code', 'n_label', 'added_none', 'skipped_candidates'])
        wr.writeheader()
        wr.writerows(rows)
    print(f'{len(ds.stems)}장 중 후보가 있는 프레임 {tot["frames"]}장: none 추가 {tot["added"]}개 ({sum(1 for r in rows if r["added_none"])}장), '
          f'넘긴 후보(가장자리에 안 닿고 10% 이상) {tot["skipped"]}개{" (dry)" if a.dry else ""} → {a.dataset}/edge_gate_report.csv')


if __name__ == '__main__':
    main()
