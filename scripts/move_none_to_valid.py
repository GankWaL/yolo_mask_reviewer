#!/usr/bin/env python3
"""학습 폴더에서 검수된 none 프레임 일부를 고정 검증셋으로 옮긴다 — 새 valid_class<N>/ 을 만든다 (2026-09-30).

  python scripts/move_none_to_valid.py VALID --train TRAIN --out-root DIR [--n 10] [--device 0]

  VALID    기존 검증 폴더 (건드리지 않는다)
  --train  검수 툴로 검수한 학습 폴더. 코드가 none 이고 확정(ok)인 프레임에서 고른다 (프레임 전체 DINOv2 임베딩으로 외형이 고르게).
결과: valid_class<N>/ (기존 검증 프레임 + none 프레임, 구조는 같음) 와 그 안의 `moved_from_train.txt`(옮긴 stem 목록).
학습 폴더에서는 `cap_per_code.py --exclude-stems valid_class<N>/moved_from_train.txt` 로 뺀다.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from cap_per_code import pick_diverse  # noqa: E402
from codes_to_sort5 import NAMES as SORT5, NONE_CODES  # noqa: E402
from mask_reviewer.dataset import STATE_FILE, Dataset  # noqa: E402
from reassign_codes import dino_crop_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('valid')
    ap.add_argument('--train', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    v, t = Dataset(a.valid), Dataset(a.train)
    names = dict(t.names) if 'none' in t.names.values() else dict(v.names)
    if 'none' not in names.values():
        names[max(names) + 1] = 'none'
    cid = {n: k for k, n in names.items()}
    cand = [s for s in t.stems if t.code(s).lower() in NONE_CODES and t.state.status(s) == 'ok'
            and any(len(l.split()) >= 7 for l in t.label_text(s).splitlines())]
    if len(cand) > a.n:
        os.makedirs(a.out_root, exist_ok=True)
        cache = os.path.join(a.out_root, '.none_valid_dino.npz')
        emb = dino_crop_embed([(s, t.image_path(s), None) for s in cand], f'cuda:{a.device}' if a.device.isdigit() else a.device, cache)
        os.remove(cache)
        cand = [cand[i] for i in pick_diverse(np.stack([emb[s] for s in cand]), a.n, a.seed)]

    rows = []
    for s in v.stems:
        m = dict(v.manifest.get(s, {}))
        polys = [' '.join(l.split()[1:]) for l in v.label_text(s).splitlines() if len(l.split()) >= 7]
        rows.append(dict(m, stem=s, code=v.code(s), n_inst=len(polys), _img=v.image_path(s), _polys=polys, _ex=v.state.exemplar(s)))
    fields = list(v.manifest[v.stems[0]].keys())
    for s in cand:
        polys = [' '.join(l.split()[1:]) for l in t.label_text(s).splitlines() if len(l.split()) >= 7]
        r = dict.fromkeys(fields, '')
        r.update(stem=s, source=t.manifest.get(s, {}).get('source', ''), code='none', sort5='none', exemplar=int(t.state.exemplar(s)),
                 reviewed=1, pick='none', n_inst=len(polys), train_split=t.split(s), _img=t.image_path(s), _polys=polys,
                 _ex=t.state.exemplar(s))
        rows.append(r)
    codes = sorted({r['code'] for r in rows})
    out = os.path.join(a.out_root, f'valid_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    for r in rows:
        ext = os.path.splitext(r['_img'])[1]
        for sub, k in (('', cid[r['code']]), ('sort5', SORT5.index(r['sort5']))):
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(out, sub, d, 'val'), exist_ok=True)
            os.link(r['_img'], os.path.join(out, sub, 'images', 'val', r['stem'] + ext))
            with open(os.path.join(out, sub, 'labels', 'val', r['stem'] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{k} {p}\n' for p in r['_polys']))
    state = {r['stem']: {'exemplar': True, 'status': 'ok', 'note': '대표'} for r in rows if r['_ex']}
    head = (f'# 고정 검증용 데이터셋 (move_none_to_valid.py): {os.path.abspath(a.valid)} + none 프레임 {len(cand)}장 ({os.path.abspath(a.train)} 에서 옮김)\n'
            f'# 클래스 {len(codes)}종(제품코드 {len(codes) - 1} + none), {len(rows)}장\n')
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
    with open(os.path.join(out, 'moved_from_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(cand) + '\n')
    print(f'검증 {len(v.stems)}장 + none {len(cand)}장 = {len(rows)}장, 클래스 {len(codes)}종, '
          f'none 인스턴스 {sum(r["n_inst"] for r in rows if r["code"] == "none")}개, 대분류 {dict(collections.Counter(r["sort5"] for r in rows))}')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
