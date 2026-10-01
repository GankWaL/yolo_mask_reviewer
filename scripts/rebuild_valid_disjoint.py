#!/usr/bin/env python3
"""고정 검증셋을 학습 데이터와 겹치지 않게 다시 만든다 — 겹치는 프레임을 빼고, 뺀 만큼 비학습 프레임에서 제품코드별로 채운다 (2026-10-01).

  python scripts/rebuild_valid_disjoint.py VALID --train TRAIN --out-root DIR [--cands DS ...] [--sources DS ...] [--pools DIR ...]

  VALID      기존 검증 폴더 (건드리지 않는다). 학습 폴더(TRAIN 의 train·val 전체)에 있는 프레임은 뺀다.
  --cands    비학습 후보 ①: 검수 툴로 여는 split 구조 폴더 (예 상한 전 학습 폴더 train_class139). TRAIN 에 없는 확정 프레임을 쓴다.
  --sources  비학습 후보 ②: 검수 데이터셋 (review_state.json·라벨·manifest.csv). 어느 후보①·TRAIN 에도 없는 확정 프레임 (정제 제외분).
             이미지가 없으면 --pools 에서 stem 으로 찾는다.
코드마다 뺀 만큼(최대 --max-per-code) ① → ② 순으로 seed 고정 무작위로 채운다. 후보가 모자라면 그만큼 적게 남고, 하나도 없으면 그 코드는 빠진다.
새로 넣은 프레임은 review_state.json 에 보류 + 메모로 표시된다 (검수 PC 에서 확인). none 프레임은 그대로 둔다.
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from add_none_frames import find_images  # noqa: E402
from codes_to_sort5 import NAMES as SORT5, drop_same_code_dups, edge_none_rule, major_map, major_of  # noqa: E402
from mask_reviewer.dataset import STATE_FILE, Dataset, is_full_code, load_manifest  # noqa: E402

TAG = {'field_new_20260923': 'field_new_export_20260928', 'field_data_hole_all_20260921': 'holes_all_export_r3'}


def train_key(stem):
    """검증 stem(<검수 데이터셋>__<원본>) → 학습 폴더 stem(<내보내기 태그>__<원본>)."""
    a, sep, b = stem.partition('__')
    return f'{TAG.get(a, a)}__{b}' if sep else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('valid')
    ap.add_argument('--train', required=True)
    ap.add_argument('--cands', nargs='*', default=[])
    ap.add_argument('--sources', nargs='*', default=[])
    ap.add_argument('--pools', nargs='*', default=[])
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--max-per-code', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    v, t = Dataset(a.valid), Dataset(a.train)
    tset = set(t.stems)
    torig = {s.partition('__')[2] for s in t.stems}
    names = dict(v.names)
    cid = {n: k for k, n in names.items()}
    major = major_map()

    rows, dropped = [], collections.Counter()
    for s in v.stems:
        m = dict(v.manifest.get(s, {}))
        if train_key(s) in tset:
            dropped[v.code(s)] += 1
            continue
        polys = [(v.names[int(l.split()[0])], ' '.join(l.split()[1:])) for l in v.label_text(s).splitlines() if len(l.split()) >= 7]
        rows.append(dict(m, stem=s, code=v.code(s), sort5=m.get('sort5') or major_of(v.code(s), major), exemplar=int(v.state.exemplar(s)),
                         reviewed=1, pick='kept', n_inst=len(polys), _img=v.image_path(s), _polys=polys, _new=False))
    have = collections.Counter(r['code'] for r in rows)

    # 비학습 후보 (검증셋에 남는 프레임과 겹치지 않게)
    cand = collections.defaultdict(list)   # code -> [(순위, stem, img, text, source)]
    seen = {r['stem'].partition('__')[2] for r in rows}
    for root in a.cands:
        d = Dataset(root)
        for s in d.stems:
            code = d.code(s)
            if s in tset or d.state.status(s) == 'reject' or not is_full_code(code) or code not in cid:
                continue
            seen.add(s.partition('__')[2])
            cand[code].append((0, s, d.image_path(s), d.label_text(s), os.path.basename(d.root)))
    images = find_images([r for r in a.sources if os.path.isdir(os.path.join(r, 'images'))] + a.pools)
    for root in a.sources:
        tag = os.path.basename(os.path.normpath(root))
        with open(os.path.join(root, STATE_FILE), encoding='utf-8') as f:
            state = json.load(f)
        man = load_manifest(root)
        for s, e in state.items():
            code = e.get('code') or man.get(s, {}).get('code', '')
            if e.get('status') != 'ok' or s in torig or s in seen or not is_full_code(code) or code not in cid or s not in images:
                continue
            text = ''
            for sub in ('labels_reviewed', 'labels_auto', 'labels'):
                p = os.path.join(root, sub, s + '.txt')
                if os.path.exists(p):
                    with open(p, encoding='utf-8') as f:
                        text = f.read()
                    break
            if any(len(l.split()) >= 7 for l in text.splitlines()):
                seen.add(s)
                cand[code].append((1, f'{tag}__{s}', images[s], text, tag))

    rng = random.Random(a.seed)
    filled, short = collections.Counter(), {}
    for code, n_drop in sorted(dropped.items()):
        want = min(a.max_per_code, have[code] + n_drop) - have[code]
        c = sorted(cand.get(code, []), key=lambda x: x[1])
        rng.shuffle(c)
        c.sort(key=lambda x: x[0])
        for rank, s, img, text, src in c[:want]:
            polys = drop_same_code_dups(edge_none_rule([(code, ' '.join(l.split()[1:])) for l in text.splitlines() if len(l.split()) >= 7]), code)
            rows.append(dict(stem=s, source=src, code=code, sort5=major_of(code, major), exemplar=0, reviewed=0,
                             pick='refill_capped' if rank == 0 else 'refill_excluded', n_inst=len(polys), _img=img, _polys=polys, _new=True))
            filled[code] += 1
        if filled[code] < want:
            short[code] = (want, filled[code])
    rows = [r for r in rows if r['sort5'] in SORT5]
    codes = sorted({r['code'] for r in rows})
    out = os.path.join(a.out_root, f'valid_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    for r in rows:
        ext = os.path.splitext(r['_img'])[1]
        for sub in ('', 'sort5'):
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(out, sub, d, 'val'), exist_ok=True)
            os.link(r['_img'], os.path.join(out, sub, 'images', 'val', r['stem'] + ext))
            with open(os.path.join(out, sub, 'labels', 'val', r['stem'] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{cid[n] if sub == "" else SORT5.index(major_of(n, major))} {p}\n' for n, p in r['_polys']))
    fields = ['stem', 'source', 'code', 'sort5', 'exemplar', 'reviewed', 'pick', 'n_inst']
    state = {r['stem']: {'status': 'pending', 'note': f'비학습 후보로 새로 넣음 ({r["pick"]}) — 코드·마스크 확인 필요'} for r in rows if r['_new']}
    state.update({r['stem']: {'status': 'ok', 'exemplar': bool(r['exemplar']), 'note': '대표' if r['exemplar'] else '검수 완료 (이전 검증셋)'}
                  for r in rows if not r['_new']})
    n_new = sum(r['_new'] for r in rows)
    head = (f'# 고정 검증용 데이터셋 (rebuild_valid_disjoint.py): {os.path.abspath(a.valid)} 에서 학습 데이터({os.path.abspath(a.train)})와 겹치는 '
            f'{sum(dropped.values())}장을 빼고 비학습 프레임 {n_new}장으로 채움\n# 클래스 {len(codes)}종, {len(rows)}장 (학습 데이터와 겹침 0)\n')
    for sub, nm in (('', names), ('sort5', dict(enumerate(SORT5)))):
        with open(os.path.join(out, sub, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rows)
        with open(os.path.join(out, sub, STATE_FILE), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
        with open(os.path.join(out, sub, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(head)
            yaml.safe_dump({'path': os.path.abspath(os.path.join(out, sub)).rstrip('/'), 'train': 'images/val', 'val': 'images/val',
                            'names': nm, 'keep_holes': True}, f, allow_unicode=True, sort_keys=False)
    per = collections.Counter(r['code'] for r in rows)
    lost = sorted(c for c in dropped if c not in per)
    print(f'{len(v.stems)}장 → {len(rows)}장: 유지 {len(rows) - n_new}, 겹쳐서 뺌 {sum(dropped.values())}, 새로 채움 {n_new} '
          f'({dict(collections.Counter(r["pick"] for r in rows if r["_new"]))}), 클래스 {len(codes)}종 (코드당 1~{max(per.values())}장)')
    print(f'모자란 코드 {len(short)}종 (원하는/채운): {dict(short)}')
    print(f'후보가 없어 빠진 코드 {len(lost)}종: {lost}')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
