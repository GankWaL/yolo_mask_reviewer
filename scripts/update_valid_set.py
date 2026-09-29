#!/usr/bin/env python3
"""고정 검증용 데이터셋에 학습 폴더의 검수 결과(코드 재지정·편집본·제외·★ 대표)를 반영해 새 valid_class<N>/ 을 만든다 (2026-09-29).

  python scripts/update_valid_set.py VALID --review REVIEWED --map VTAG=RTAG [...] --out-root DIR

  VALID     기존 검증 폴더 (build_valid_set.py 결과, stem = `<VTAG>__<원본 stem>`)
  --review  검수 툴로 검수한 학습 폴더 (review_state.json·labels_reviewed, stem = `<RTAG>__<원본 stem>`)
  --map     검증 폴더의 태그 → 검수 폴더의 태그 (같은 원본 프레임을 잇는다)
  --out-root 이 아래에 valid_class<N>/ 을 새로 만든다 (N = 제품코드 수). VALID 는 건드리지 않는다.

검수 폴더에 같은 프레임이 있으면 그 코드·라벨·대표 여부를 쓰고 제외(reject)면 뺀다. 없는 프레임은 그대로 둔다 (manifest 의 reviewed=0).
코드 재지정으로 한 코드가 --max-per-code(5)장을 넘으면 ★ 대표 → 검수된 프레임 → 나머지 순으로 남긴다.
결과 구조는 build_valid_set.py 와 같다 (루트 = 제품코드 클래스, sort5/ = 대분류 5클래스) + review_state.json(★ 대표).
"""
import argparse
import collections
import csv
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from codes_to_sort5 import NAMES as SORT5, major_map, major_of  # noqa: E402
from mask_reviewer.dataset import STATE_FILE, Dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('valid')
    ap.add_argument('--review', required=True)
    ap.add_argument('--map', nargs='+', required=True, metavar='VTAG=RTAG')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--max-per-code', type=int, default=5)
    a = ap.parse_args()

    v = Dataset(a.valid)
    rv = Dataset(a.review)
    tag = dict(m.split('=') for m in a.map)
    cid = {n: k for k, n in v.names.items()}
    major = major_map()

    rows = []
    stat = collections.Counter()
    for s in v.stems:
        vt, _, orig = s.partition('__')
        r = f'{tag.get(vt, vt)}__{orig}'
        m = dict(v.manifest.get(s, {}))
        old = v.code(s)
        if r in rv._img_path:
            if rv.state.status(r) == 'reject':
                stat['reject'] += 1
                continue
            code, text, ex, rev = rv.code(r), rv.label_text(r), rv.state.exemplar(r), 1
            stat['reviewed'] += 1
            stat['code_changed'] += code != old
            stat['label_edited'] += rv.is_edited(r)
        else:
            code, text, ex, rev = old, v.label_text(s), str(m.get('exemplar', '0')) == '1', 0
            stat['not_reviewed'] += 1
        if code not in cid or major_of(code, major) not in SORT5:
            stat['unknown_code'] += 1
            continue
        polys = [' '.join(l.split()[1:]) for l in text.splitlines() if len(l.split()) >= 7]
        if not polys:
            stat['nolabel'] += 1
            continue
        rows.append(dict(stem=s, source=m.get('source', vt), code=code, prev_code=old if old != code else '', sort5=major_of(code, major),
                         exemplar=int(bool(ex)), reviewed=rev, pick=m.get('pick', ''), n_inst=len(polys),
                         train_round=m.get('train_round', ''), train_split=m.get('train_split', ''),
                         train_exclude=m.get('train_exclude', ''), _polys=polys))

    by_code = collections.defaultdict(list)
    for r in rows:
        by_code[r['code']].append(r)
    kept = []
    for code in sorted(by_code):
        rs = sorted(by_code[code], key=lambda r: (-r['exemplar'], -r['reviewed'], r['stem']))
        stat['over_cap'] += max(0, len(rs) - a.max_per_code)
        kept += rs[:a.max_per_code]
    codes = sorted({r['code'] for r in kept})
    out = os.path.join(a.out_root, f'valid_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    for r in kept:
        src = v.image_path(r['stem'])
        for sub, k in (('', cid[r['code']]), ('sort5', SORT5.index(r['sort5']))):
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(out, sub, d, 'val'), exist_ok=True)
            os.link(src, os.path.join(out, sub, 'images', 'val', os.path.basename(src)))
            with open(os.path.join(out, sub, 'labels', 'val', r['stem'] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{k} {p}\n' for p in r['_polys']))
    fields = [k for k in kept[0] if not k.startswith('_')]
    state = {r['stem']: {'exemplar': True, 'status': 'ok', 'note': '대표'} for r in kept if r['exemplar']}
    head = (f'# 고정 검증용 데이터셋 (update_valid_set.py): {os.path.abspath(a.valid)} 에 {os.path.abspath(a.review)} 의 검수 결과 반영\n'
            f'# 제품코드 {len(codes)}종, {len(kept)}장 (검수 반영 {sum(r["reviewed"] for r in kept)}장, ★ 대표 {len(state)}장)\n')
    for sub, nm in (('', v.names), ('sort5', dict(enumerate(SORT5)))):
        with open(os.path.join(out, sub, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(kept)
        with open(os.path.join(out, sub, STATE_FILE), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
        with open(os.path.join(out, sub, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(head)
            yaml.safe_dump({'path': os.path.abspath(os.path.join(out, sub)).rstrip('/'), 'train': 'images/val', 'val': 'images/val',
                            'names': nm, 'keep_holes': True}, f, allow_unicode=True, sort_keys=False)
    per = collections.Counter(r['code'] for r in kept)
    print(f'{len(v.stems)}장 → {len(kept)}장, 제품코드 {len(codes)}종 (코드당 {min(per.values())}~{max(per.values())}장), {dict(stat)}')
    print(f'★ 대표 {len(state)}장, 검수 반영 {sum(r["reviewed"] for r in kept)} / 미검수 {sum(1 - r["reviewed"] for r in kept)}장, '
          f'대분류 {dict(collections.Counter(r["sort5"] for r in kept))}')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
