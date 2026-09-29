#!/usr/bin/env python3
"""제품코드 클래스 내보내기 폴더를 대분류 5클래스(b_cvr, none, h_cvr, hsg, scalp) YOLO 폴더로 바꾼다 (2026-09-28).

  python scripts/codes_to_sort5.py SRC OUT

  SRC  mask_reviewer 내보내기 폴더 (images/<split>, labels/<split>, export_manifest.csv). 클래스 = 12자리 제품코드.
  OUT  같은 프레임(이미지 하드링크), 라벨 첫 열만 그 프레임 코드의 대분류 id 로 바꾼 폴더.

대분류는 obj_product_name 폴더의 obj 이름에서 얻는다 (relabel_with_model.py --map-codes 와 같은 표).
코드는 export_manifest.csv 의 code 열(재지정 코드 반영). obj 가 없는 코드의 프레임은 건너뛴다.
1번 클래스는 cap 이 아니라 **none**(제품이 아닌 물체: 게이트 조각·상자 등)이다 (2026-09-29). CAP 대분류 제품은 현장에서 검사하지 않아
학습에 넣지 않으므로 cap 코드의 프레임은 건너뛴다. 프레임 코드가 `none`·`gate` 면 none 이다.
"""
import argparse
import collections
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_holes_review import OBJ_DIR  # noqa: E402
from reassign_codes import obj_class  # noqa: E402

NAMES = ['b_cvr', 'none', 'h_cvr', 'hsg', 'scalp']
NONE_CODES = ('none', 'gate')   # 검수에서 제품이 아니라고 표시한 프레임의 코드 (대소문자 무시)


def major_map():
    """제품코드 → 대분류 이름 (obj 이름 기준). cap 제품은 넣지 않는다. `major_of(code, table)` 로 조회한다."""
    table = {}
    for fn in sorted(os.listdir(OBJ_DIR)):
        if fn.lower().endswith('.obj'):
            table.setdefault(fn[:12], obj_class(fn[:-4]))
    return {c: m for c, m in table.items() if m in NAMES and m != 'none'}


def major_of(code, table):
    """코드의 대분류 (none·gate 는 'none', 모르는 코드·cap 제품은 None)."""
    return 'none' if (code or '').lower() in NONE_CODES else table.get(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('out')
    a = ap.parse_args()
    major = major_map()
    with open(os.path.join(a.src, 'export_manifest.csv'), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    cnt = collections.Counter()
    skipped = collections.Counter()
    kept = []
    for r in rows:
        cls = major_of(r['code'], major)
        if cls not in NAMES:
            skipped[r['code']] += 1
            continue
        sp = r['split']
        lab = os.path.join(a.src, 'labels', sp, r['stem'] + '.txt')
        lines = [f'{NAMES.index(cls)} ' + ' '.join(l.split()[1:]) for l in open(lab, encoding='utf-8') if len(l.split()) >= 7]
        if not lines:
            skipped['(라벨 없음)'] += 1
            continue
        img = next(fn for fn in os.listdir(os.path.join(a.src, 'images', sp)) if os.path.splitext(fn)[0] == r['stem']) \
            if not os.path.exists(os.path.join(a.src, 'images', sp, r['stem'] + '.png')) else r['stem'] + '.png'
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(a.out, d, sp), exist_ok=True)
        dst = os.path.join(a.out, 'images', sp, img)
        if not os.path.exists(dst):
            os.link(os.path.join(a.src, 'images', sp, img), dst)
        with open(os.path.join(a.out, 'labels', sp, r['stem'] + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        cnt[(sp, cls)] += len(lines)
        kept.append(r)
    with open(os.path.join(a.src, 'dataset.yaml'), encoding='utf-8') as f:
        src_yaml = yaml.safe_load(f) or {}
    y = {'path': os.path.abspath(a.out), 'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(NAMES))}
    if src_yaml.get('keep_holes'):
        y['keep_holes'] = True
    with open(os.path.join(a.out, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'# {os.path.abspath(a.src)} 의 제품코드 클래스를 대분류 5클래스로 바꿈 (codes_to_sort5.py)\n')
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(a.out, 'export_manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(kept)
    print(f'프레임 {len(kept)} / {len(rows)}, 인스턴스 {dict(sorted(cnt.items()))}, 건너뜀 {dict(skipped)} → {a.out}')


if __name__ == '__main__':
    main()
