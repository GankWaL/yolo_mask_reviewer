#!/usr/bin/env python3
"""검수 폴더에 있는 제품코드마다 원본 데이터셋의 ★ 대표 프레임을 검수 폴더에 넣고 ★ 로 표시한다 (2026-10-02).

  conda activate yolo_mask_reviewer
  python scripts/add_exemplars_to_review.py REVIEW_DS --from SRC_DS [SRC_DS ...] [--dry-run]

  REVIEW_DS  mask_reviewer 검수 폴더 (images/ labels/ manifest.csv review_state.json). 예: yolo26_dataset/recheck_holes_20261002
  SRC_DS     ★ 대표가 review_state.json 에 표시된 데이터셋 (train_class137, valid_class130 …). 클래스 id 는 이름으로 맞춘다.

대표가 이미 검수 폴더에 있으면 state 에 exemplar 만 켠다 (판정은 그대로). 없으면 이미지(하드링크)·현재 라벨(편집본 > 자동 > 원본)·
manifest 행을 넣고 보류 + ★ 로 둔다. 메모 앞에 '★ 대표(<출처>) 먼저 검수' 를 붙인다 — 검수자가 대표부터 보고 구멍을 확정한 뒤
`자동 → 대표 구멍 전파` 로 나머지 프레임에 구멍을 옮긴다. 대표가 어느 원본에도 없는 코드는 끝에 출력한다 (검수 폴더에서 R 로 지정).
manifest 의 n_holes·small_holes·major·code_mode·code_max·mismatch 열이 있으면 같은 방식으로 채운다 (작은 구멍 = 400px 미만).
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402

SMALL_HOLE_PX = 400


def hole_stats(label_path, w, h):
    """(구멍 수, 작은 구멍 수)."""
    n = small = 0
    for line in open(label_path, encoding='utf-8'):
        r = maskops.parse_yolo_line(line, w, h)
        if r is None:
            continue
        m = maskops.polygon_to_mask(r[1], h, w)
        holes = maskops.fill_holes(m) & ~m
        k, _, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        for i in range(1, k):
            n += 1
            small += int(stats[i, cv2.CC_STAT_AREA] < SMALL_HOLE_PX)
    return n, small


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('review')
    ap.add_argument('--from', dest='sources', nargs='+', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    rds = Dataset(a.review)
    codes = {rds.code(s) for s in rds.stems}
    rname = {v: k for k, v in rds.names.items()}
    mpath = os.path.join(rds.root, 'manifest.csv')
    rows = list(csv.DictReader(open(mpath, newline='', encoding='utf-8'))) if os.path.exists(mpath) else []
    fields = list(rows[0].keys()) if rows else ['stem', 'source', 'code']
    code_rows = {}
    for r in rows:
        code_rows.setdefault(r['code'], r)
    try:
        from codes_to_sort5 import major_map, major_of
        mtab = major_map()
    except Exception:
        mtab = None

    n_mark = n_add = 0
    covered = set()
    for src in a.sources:
        sds = Dataset(src)
        tag = os.path.basename(os.path.normpath(src))
        for s in sds.exemplars():
            code = sds.code(s)
            if code not in codes:
                continue
            covered.add(code)
            e = rds.state.get(s)
            prefix = f'★ 대표({tag}) 먼저 검수'
            if s in rds._img_path:
                if e.get('exemplar'):
                    continue
                note = e.get('note', '')
                if not a.dry_run:
                    rds.state.set(s, exemplar=True, note=(prefix + (' · ' + note if note else '')))
                n_mark += 1
                print(f'★ 표시  {s}')
                continue
            img_src = sds.image_path(s)
            lab_src = sds.effective_label_path(s)
            if not os.path.exists(lab_src):
                print(f'건너뜀(라벨 없음) {s}')
                continue
            img = cv2.imread(img_src)
            if img is None:
                print(f'건너뜀(이미지 없음) {s}')
                continue
            h, w = img.shape[:2]
            lines = []
            for line in open(lab_src, encoding='utf-8'):
                v = line.split()
                if len(v) < 7:
                    continue
                name = sds.names.get(int(float(v[0])))
                if name not in rname:
                    print(f'  클래스 {name} 가 검수 폴더에 없어 인스턴스 제외: {s}')
                    continue
                lines.append(' '.join([str(rname[name])] + v[1:]))
            if not lines:
                print(f'건너뜀(인스턴스 없음) {s}')
                continue
            ext = os.path.splitext(img_src)[1]
            dst_img = os.path.join(rds.images_dir, s + ext)
            dst_lab = os.path.join(rds.labels_dir, s + '.txt')
            row = {k: '' for k in fields}
            row.update(stem=s, source=tag, code=code)
            if 'n_holes' in fields:
                n, small = hole_stats(lab_src, w, h)
                row['n_holes'] = n
                if 'small_holes' in fields:
                    row['small_holes'] = small
                base = code_rows.get(code, {})
                for k in ('code_mode', 'code_max'):
                    if k in fields:
                        row[k] = base.get(k, '')
                if 'mismatch' in fields:
                    row['mismatch'] = int(str(base.get('code_mode', '')) not in ('', str(n)))
            if 'major' in fields and mtab is not None:
                row['major'] = major_of(code, mtab) or code_rows.get(code, {}).get('major', '')
            mm = f'구멍 {row.get("n_holes", "?")}개' + (f' (이 코드 최빈 {row["code_mode"]}개)' if row.get('code_mode') else '')
            if not a.dry_run:
                os.makedirs(rds.labels_dir, exist_ok=True)
                try:
                    os.link(img_src, dst_img)
                except OSError:
                    shutil.copy2(img_src, dst_img)
                with open(dst_lab, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
                rows.append(row)
                rds.state.set(s, status='pending', exemplar=True, note=f'{prefix} · {mm} · 출처 {tag}')
            n_add += 1
            print(f'추가    {s} ({mm})')
    if not a.dry_run and rows:
        with open(mpath, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
    missing = sorted(codes - covered)
    print(f'\n{"(dry-run) " if a.dry_run else ""}제품코드 {len(codes)}종: ★ 표시 {n_mark}장, 추가 {n_add}장, 대표 있는 코드 {len(covered)}종')
    if missing:
        print(f'대표 없는 코드 {len(missing)}종 (검수 폴더에서 R 로 지정): {" ".join(missing)}')


if __name__ == '__main__':
    main()
