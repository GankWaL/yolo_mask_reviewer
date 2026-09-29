#!/usr/bin/env python3
"""고정 검증용 데이터셋을 만든다 — 제품코드(클래스)마다 확정 프레임 1~5장, ★ 대표는 반드시 포함 (2026-09-28).

  python scripts/build_valid_set.py --names CODES.yaml --out-root DIR SRC [SRC ...]

  SRC        검수 데이터셋 (images/ labels*/ review_state.json). 확정(ok) 프레임만 쓴다. 정제 제외(train_exclude)도 쓴다.
  --names    제품코드 클래스 목록 yaml (dropped_names 의 코드는 뺀다)
  --out-root 이 아래에 valid_class<N>/ 을 만든다 (N = 모인 제품코드 수)

코드마다 고르는 순서: ① ★ 대표 → ② 학습에 안 쓴 프레임(val 분할 → 정제 제외) → ③ 학습에 쓴 프레임. 같은 순위 안에서는 seed 고정 무작위.
결과:
  valid_class<N>/images/val, labels/val, dataset.yaml          클래스 = 12자리 제품코드
  valid_class<N>/sort5/images/val, labels/val, dataset.yaml    같은 프레임, 클래스 = 대분류 5클래스
  valid_class<N>/manifest.csv                                   stem, source, code, sort5, exemplar, pick, train_round, train_split, train_exclude
한 프레임의 인스턴스는 전부 그 프레임의 코드로 쓴다 (remap_field_codes.py 와 같음). 이미지는 하드링크.
"""
import argparse
import collections
import csv
import os
import random
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codes_to_sort5 import NAMES as SORT5, major_map, major_of  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402

PICK_LABEL = ('exemplar', 'unused_val', 'unused_excluded', 'trained')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+')
    ap.add_argument('--names', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--max-per-code', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.7)
    ap.add_argument('--min-area', type=float, default=16.0)
    a = ap.parse_args()

    with open(a.names, encoding='utf-8') as f:
        ref = yaml.safe_load(f)
    names = {int(k): str(v) for k, v in ref['names'].items()}
    cid = {v: k for k, v in names.items()}
    major = major_map()

    cand = collections.defaultdict(list)   # code -> [(순위, ds, stem)]
    skipped = collections.Counter()
    for root in a.src:
        ds = Dataset(root)
        for s in ds.stems:
            if ds.state.status(s) != 'ok':
                continue
            code = ds.code(s)
            if code not in cid or major_of(code, major) not in SORT5:
                skipped[code] += 1
                continue
            if not any(len(l.split()) >= 7 for l in ds.label_text(s).splitlines()):
                continue
            e = ds.state.get(s)
            rank = (0 if e.get('exemplar') else 1 if e.get('train_split') == 'val' else 2 if e.get('train_exclude') else 3)
            cand[code].append((rank, ds, s))

    rng = random.Random(a.seed)
    picked = []
    for code in sorted(cand):
        c = sorted(cand[code], key=lambda t: (t[1].root, t[2]))
        rng.shuffle(c)
        c.sort(key=lambda t: t[0])       # 안정 정렬: 순위 안에서는 섞인 순서
        picked += [(code,) + t for t in c[:a.max_per_code]]
    codes = sorted({p[0] for p in picked})
    out = os.path.join(a.out_root, f'valid_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    for sub in ('', 'sort5'):
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(out, sub, d, 'val'))

    rows = []
    for code, rank, ds, s in picked:
        tag = os.path.basename(ds.root)
        src = ds.image_path(s)
        stem = f'{tag}__{s}'
        if ds.is_edited(s):                # 내보내기와 같이 편집본은 마스크 → 폴리곤으로 다시 쓴다
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            text = ds.instances_to_text(ds.parse_instances(ds.label_text(s), w, h), w, h, a.eps, a.min_area)
        else:
            text = ds.label_text(s)
        polys = [' '.join(l.split()[1:]) for l in text.splitlines() if len(l.split()) >= 7]
        for sub, k in (('', cid[code]), ('sort5', SORT5.index(major_of(code, major)))):
            os.link(src, os.path.join(out, sub, 'images', 'val', stem + os.path.splitext(src)[1]))
            with open(os.path.join(out, sub, 'labels', 'val', stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{k} {p}\n' for p in polys))
        e = ds.state.get(s)
        rows.append(dict(stem=stem, source=tag, code=code, sort5=major_of(code, major), exemplar=int(bool(e.get('exemplar'))),
                         pick=PICK_LABEL[rank], n_inst=len(polys), train_round=e.get('train_round', ''),
                         train_split=e.get('train_split', ''), train_exclude=int(bool(e.get('train_exclude')))))
    with open(os.path.join(out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    head = (f'# 고정 검증용 데이터셋 (build_valid_set.py): 제품코드 {len(codes)}종 × 1~{a.max_per_code}장 = {len(rows)}장, ★ 대표 포함\n'
            f'# sources={[os.path.abspath(p) for p in a.src]} names={os.path.abspath(a.names)} seed={a.seed}\n')
    for sub, nm in (('', names), ('sort5', dict(enumerate(SORT5)))):
        with open(os.path.join(out, sub, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(head)
            yaml.safe_dump({'path': os.path.abspath(os.path.join(out, sub)), 'train': 'images/val', 'val': 'images/val',
                            'names': nm, 'keep_holes': True}, f, allow_unicode=True, sort_keys=False)
    per = collections.Counter(r['code'] for r in rows)
    n_ex_all = sum(1 for c in cand.values() for t in c if t[0] == 0)
    print(f'제품코드 {len(codes)}종, {len(rows)}장 (코드당 {min(per.values())}~{max(per.values())}장, '
          f'{dict(sorted(collections.Counter(per.values()).items()))})')
    print(f'선택 구분 {dict(collections.Counter(r["pick"] for r in rows))}, 대표 {sum(r["exemplar"] for r in rows)} / 전체 대표 {n_ex_all}')
    print(f'대분류 {dict(collections.Counter(r["sort5"] for r in rows))}')
    if skipped:
        print(f'클래스 목록에 없거나 제외된 코드는 건너뜀: {dict(skipped)}')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
