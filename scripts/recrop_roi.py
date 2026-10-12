#!/usr/bin/env python3
"""검수 데이터셋의 이미지를 다른 ROI 로 다시 잘라내고 라벨을 그만큼 옮긴다 (2026-10-12).

현장 detector_process_roi 가 605,190,685,685 → 628,191,686,686 (cam_intrinsics/SL/roi_ui.yaml, 09-23) 으로 바뀌었는데
수집 파이프라인(field_autolabel AL_ROI)은 옛 값으로 잘라 왔다. 원본 FHD raw(field/<날짜>/save_pose_debug/<실행>/<idx>_<코드>_raw.png)
에서 새 ROI 로 다시 자르고, labels/·labels_auto/·labels_reviewed/ 의 폴리곤은 마스크로 그린 뒤 옛/새 ROI 차이만큼 옮겨 다시 폴리곤으로 쓴다.
새 ROI 에만 들어오는 띠(오른쪽 23px 등)는 라벨이 없으므로, 보류 프레임의 labels_auto 는 뒤에 relabel_with_model.py 로 다시 추론한다.

  python scripts/recrop_roi.py DS                       # 기본: 605,190,685,685 → 628,191,686,686
  python scripts/recrop_roi.py DS --dry                 # 바꾸지 않고 대상·누락만 센다
  옵션: --old x,y,w,h  --new x,y,w,h  --raw-root ~/jhw/data/SL/field  --stems a,b (일부만)

안전장치: 옛 이미지가 raw 의 옛 ROI 자리와 픽셀 단위로 같을 때만 바꾼다 (다르면 건너뛰고 보고). raw 가 없는 stem 도 건너뛴다.
"""
import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mask_reviewer.dataset import Dataset, Instance  # noqa: E402

STEM_RE = re.compile(r'^(?:.*__)?(\d{8})_(\d{6}_\d{3})_(\d{5}_[0-9A-Z]{12})$')


def raw_path(root, stem):
    m = STEM_RE.match(stem)
    if not m:
        return None
    d, run, rest = m.groups()
    return os.path.join(root, d, 'save_pose_debug', run, rest + '_raw.png')


def shift_mask(mask, old, new):
    """옛 ROI 안 마스크 → 새 ROI 안 마스크 (겹치지 않는 부분은 0)."""
    ox, oy, ow, oh = old
    nx, ny, nw, nh = new
    out = np.zeros((nh, nw), bool)
    x0, x1 = max(ox, nx), min(ox + ow, nx + nw)
    y0, y1 = max(oy, ny), min(oy + oh, ny + nh)
    if x1 > x0 and y1 > y0:
        out[y0 - ny:y1 - ny, x0 - nx:x1 - nx] = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--old', default='605,190,685,685', help='지금 이미지가 잘린 ROI x,y,w,h (FHD px)')
    ap.add_argument('--new', default='628,191,686,686', help='새 ROI x,y,w,h')
    ap.add_argument('--raw-root', default=os.path.expanduser('~/jhw/data/SL/field'))
    ap.add_argument('--stems', default='', help='쉼표 목록 (기본 전부)')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    old = [int(v) for v in a.old.split(',')]
    new = [int(v) for v in a.new.split(',')]
    ds = Dataset(a.dataset)
    stems = [s for s in a.stems.split(',') if s] or ds.stems
    label_dirs = [ds.labels_dir, ds.auto_dir, ds.reviewed_dir]
    n_done = n_drop = 0
    skipped = []   # (stem, 이유)
    for i, stem in enumerate(stems):
        rp = raw_path(a.raw_root, stem)
        if not rp or not os.path.isfile(rp):
            skipped.append((stem, 'raw 없음'))
            continue
        raw = cv2.imread(rp, cv2.IMREAD_COLOR)
        img = cv2.imread(ds.image_path(stem), cv2.IMREAD_COLOR)
        if raw is None or img is None:
            skipped.append((stem, '이미지 읽기 실패'))
            continue
        ox, oy, ow, oh = old
        if img.shape[:2] != (oh, ow) or not np.array_equal(raw[oy:oy + oh, ox:ox + ow], img):
            skipped.append((stem, f'옛 ROI 자리와 다름 (크기 {img.shape[1]}x{img.shape[0]})'))
            continue
        nx, ny, nw, nh = new
        crop = raw[ny:ny + nh, nx:nx + nw]
        if crop.shape[:2] != (nh, nw):
            skipped.append((stem, '새 ROI 가 raw 를 벗어남'))
            continue
        texts = {}
        for d in label_dirs:
            p = os.path.join(d, stem + '.txt')
            if not os.path.isfile(p):
                continue
            with open(p, encoding='utf-8') as f:
                insts = ds.parse_instances(f.read(), ow, oh)
            moved = [Instance(x.cls, shift_mask(x.mask, old, new)) for x in insts]
            text = ds.instances_to_text(moved, nw, nh)
            n_drop += len(insts) - text.count('\n')
            texts[p] = text
        if not a.dry:
            cv2.imwrite(ds.image_path(stem), crop)
            for p, text in texts.items():
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(text)
        n_done += 1
        print(f'\r{i + 1}/{len(stems)} {stem}    ', end='', flush=True)
    print()
    print(f'{"(dry) " if a.dry else ""}재크롭 {n_done}장 · 건너뜀 {len(skipped)}장 · 옮기며 사라진 인스턴스 {n_drop}개  ({a.old} → {a.new})')
    if skipped:
        rep = os.path.join(ds.root, 'recrop_skipped.csv')
        with open(rep, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['stem', 'reason'])
            w.writerows(skipped)
        reasons = {}
        for _, r in skipped:
            reasons[r] = reasons.get(r, 0) + 1
        print(f'  건너뜀 사유: {reasons} → {rep}')
    if not a.dry and n_done:
        p = os.path.join(ds.root, 'dataset.yaml')
        with open(p, encoding='utf-8') as f:
            body = f.read()
        from datetime import date
        note = (f'# {date.today().isoformat()}: ROI {a.old} → {a.new} 로 재크롭 (scripts/recrop_roi.py, 현장 roi_ui.yaml 기준). '
                f'라벨은 같은 양만큼 이동, 건너뛴 {len(skipped)}장은 recrop_skipped.csv.\n')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(note + body)
    return 0


if __name__ == '__main__':
    sys.exit(main())
