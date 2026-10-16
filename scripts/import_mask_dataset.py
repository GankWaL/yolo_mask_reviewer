#!/usr/bin/env python3
"""클래스 인덱스 마스크 PNG 데이터셋(images/ + masks/) → 검수용 데이터셋(images/ labels/ manifest.csv dataset.yaml).

Workspace 라벨링 도구(labeling_tool.py)가 만든 결함 데이터: masks/<stem>.png 의 픽셀값 v 는 클래스 v-1 (0 = 배경).
클래스 이름은 labeling_tool 의 defect.yaml(`classes: {id: name}`) 또는 YOLO dataset.yaml(`names`)에서 읽는다.
같은 클래스 값이라도 떨어진 조각(8-연결 성분)은 각각 한 인스턴스로 만든다 — 검수 툴에서 잘못된 조각만 지울 수 있게.

이미지는 같은 파일시스템이면 하드링크(기본), 아니면 복사한다. 원본 폴더는 바꾸지 않는다.

python scripts/import_mask_dataset.py --out ~/jhw/data/SL_defects/relabeling_20251219_20260106 \
    --names ~/Workspace/AI/labeling_tool/defect.yaml  SRC [SRC ...]      # SRC = images/ masks/ 를 가진 폴더
"""
import argparse
import csv
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mask_reviewer import maskops  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def load_names(path):
    with open(path, encoding='utf-8') as f:
        y = yaml.safe_load(f) or {}
    names = y.get('classes') or y.get('names') or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def labeler_of(stem):
    """251219__pyoJW_0011_07_094030 → pyoJW, 260105_robot0_001_0075_05_162550 → robot0, 260106_001_0012_02_182528 → ''."""
    m = re.match(r'^\d{6}_+([A-Za-z][A-Za-z0-9]*)_', stem)
    return m.group(1) if m else ''


def place_image(src, dst, copy):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_one(job):
    src_img, src_mask, out_img, out_lbl, copy, min_area, eps = job
    m = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, f'mask unreadable: {src_mask}'
    h, w = m.shape[:2]
    lines, classes = [], set()
    for v in np.unique(m):
        if v == 0:
            continue
        cls = int(v) - 1
        binary = (m == v).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            comp = np.zeros_like(binary)
            comp[y:y + bh, x:x + bw] = (lab[y:y + bh, x:x + bw] == i)
            line = maskops.mask_to_yolo_line(cls, comp, w, h, eps=eps, min_area=min_area)
            if line:
                lines.append(line)
                classes.add(cls)
    place_image(src_img, out_img, copy)
    with open(out_lbl, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    return (len(lines), sorted(classes), (w, h)), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+', help='images/ masks/ 를 가진 폴더')
    ap.add_argument('--out', required=True)
    ap.add_argument('--names', required=True, help='defect.yaml(classes) 또는 dataset.yaml(names)')
    ap.add_argument('--copy', action='store_true', help='하드링크 대신 복사')
    ap.add_argument('--min-area', type=float, default=16.0, help='이보다 작은 조각(px²)은 버림')
    ap.add_argument('--eps', type=float, default=0.7, help='폴리곤 단순화(px)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    names = load_names(a.names)
    out = os.path.abspath(os.path.expanduser(a.out))
    img_dir, lbl_dir = os.path.join(out, 'images'), os.path.join(out, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    srcs = [os.path.abspath(os.path.expanduser(s)) for s in a.src]
    common = os.path.commonpath(srcs) if len(srcs) > 1 else os.path.dirname(srcs[0])
    jobs, meta, seen = [], {}, {}
    for src in srcs:
        si, sm = os.path.join(src, 'images'), os.path.join(src, 'masks')
        imgs = {os.path.splitext(f)[0]: f for f in os.listdir(si) if f.lower().endswith(IMG_EXTS)}
        masks = {os.path.splitext(f)[0]: f for f in os.listdir(sm) if f.lower().endswith('.png')}
        for stem in sorted(set(imgs) - set(masks)):
            print(f'[warn] 마스크 없음, 건너뜀: {src}/images/{imgs[stem]}')
        for stem in sorted(set(masks) - set(imgs)):
            print(f'[warn] 이미지 없음, 건너뜀: {src}/masks/{masks[stem]}')
        for stem in sorted(set(imgs) & set(masks)):
            if stem in seen:
                sys.exit(f'stem 중복: {stem} ({seen[stem]} / {src})')
            seen[stem] = src
            ext = os.path.splitext(imgs[stem])[1].lower()
            jobs.append((os.path.join(si, imgs[stem]), os.path.join(sm, masks[stem]),
                         os.path.join(img_dir, stem + ext), os.path.join(lbl_dir, stem + '.txt'),
                         a.copy, a.min_area, a.eps))
            meta[stem] = {'source': os.path.relpath(src, common), 'mask_src': os.path.join(sm, masks[stem])}

    rows, errors, done = [], [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for job, (res, err) in zip(jobs, ex.map(convert_one, jobs)):
            done += 1
            stem = os.path.splitext(os.path.basename(job[1]))[0]
            if err:
                errors.append(err)
                continue
            n_inst, classes, (w, h) = res
            rows.append({'stem': stem, 'source': meta[stem]['source'],
                         'code': '', 'labeler': labeler_of(stem), 'n_inst': n_inst,
                         'classes': ';'.join(names.get(c, str(c)) for c in classes),
                         'width': w, 'height': h, 'mask_src': meta[stem]['mask_src']})
            if done % 500 == 0:
                print(f'  {done}/{len(jobs)}', flush=True)

    fields = ['stem', 'source', 'code', 'labeler', 'n_inst', 'classes', 'width', 'height', 'mask_src']
    with open(os.path.join(out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    y = {'path': out, 'train': 'images', 'val': 'images',
         'names': {i: names[i] for i in sorted(names)}, 'keep_holes': False,
         'source_masks': 'masks/<stem>.png 픽셀값 v = 클래스 v-1 (labeling_tool.py). 조각(8-연결)마다 한 인스턴스.',
         'source_dirs': srcs}
    with open(os.path.join(out, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)

    per_cls, empty = {}, 0
    for r in rows:
        if r['n_inst'] == 0:
            empty += 1
        for c in filter(None, r['classes'].split(';')):
            per_cls[c] = per_cls.get(c, 0) + 1
    print(f'{out}: {len(rows)} 장 (인스턴스 없음 {empty}), 인스턴스 {sum(r["n_inst"] for r in rows)}')
    print('클래스별 프레임 수:', dict(sorted(per_cls.items())))
    for e in errors:
        print('[error]', e)


if __name__ == '__main__':
    main()
