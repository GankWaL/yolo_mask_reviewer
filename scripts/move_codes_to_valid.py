#!/usr/bin/env python3
"""학습에만 있는 제품코드의 프레임을 고정 검증셋에 넣는다 — 새 valid_class<N>/ 을 만든다 (2026-10-15, jhw).
  python scripts/move_codes_to_valid.py VALID --train TRAIN --out-root DIR [--codes auto|A,B,...] [--n 5] [--dup-max 5] [--device 0]
  VALID    기존 검증 폴더 (건드리지 않는다)
  --train  검수된 학습 폴더(상한 전 전체본, 예 train_class161_review4). 코드가 --codes 이고 확정(ok)인 프레임에서 고른다.
  --codes  auto(기본) = TRAIN 에는 있고 VALID 에는 없는 제품코드 전부. 쉼표 목록으로 지정할 수도 있다.
코드마다:
  · 학습 프레임이 --dup-max(5)장 이하면 전부 검증에도 넣고 학습에서는 빼지 않는다 (**중복 허용**, jhw 2026-10-15 — 프레임이 너무 적은 코드는 양쪽에 다 쓴다).
  · 그보다 많으면 --n(5)장을 학습에서 빼서 옮긴다: ★ 대표는 학습에 남기고(대표 아닌 프레임이 모자랄 때만 대표도 씀), 물체 크롭 DINOv2 임베딩으로 외형이 고르게 고른다.
검증 라벨은 검증셋 규칙대로 만든다 (가장자리 규칙 edge_none_rule → 같은 코드 중복 인스턴스는 가장 넓은 것만, drop_same_code_dups).
결과: valid_class<N>/ (기존 검증 프레임 + 추가 프레임, 구조 같음, manifest pick = moved/dup) 와
  `moved_from_train.txt`(학습에서 뺄 stem — cap_per_code.py --exclude-stems), `dup_in_train.txt`(학습에도 남는 stem — set_train_val.py --allow-overlap).
"""
import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from cap_per_code import label_bbox, pick_diverse  # noqa: E402
from codes_to_sort5 import NAMES as SORT5, drop_same_code_dups, edge_none_rule, major_map, major_of  # noqa: E402
from mask_reviewer.dataset import STATE_FILE, Dataset, is_full_code  # noqa: E402
from reassign_codes import dino_crop_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('valid')
    ap.add_argument('--train', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--codes', default='auto')
    ap.add_argument('--n', type=int, default=5, help='코드당 학습에서 빼서 옮기는 최대 장수')
    ap.add_argument('--dup-max', type=int, default=5, help='학습 프레임이 이 이하인 코드는 빼지 않고 중복으로 넣는다')
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    v, t = Dataset(a.valid), Dataset(a.train)
    names = dict(v.names)
    cid = {n: k for k, n in names.items()}
    major = major_map()
    v_codes = {v.code(s) for s in v.stems}
    by_code = collections.defaultdict(list)
    for s in t.stems:
        c = t.code(s)
        if t.state.status(s) != 'ok' or not is_full_code(c) or c not in cid or major_of(c, major) not in SORT5:
            continue
        if not any(len(l.split()) >= 7 for l in t.label_text(s).splitlines()):
            continue
        by_code[c].append(s)
    codes = sorted(c for c in by_code if c not in v_codes) if a.codes == 'auto' else [c for c in a.codes.split(',') if c]
    moved, dup = {}, {}      # stem -> code
    need_emb = []
    plan = {}
    for c in codes:
        rs = by_code.get(c, [])
        if not rs:
            print(f'  {c}: 학습 프레임 없음, 건너뜀')
            continue
        if len(rs) <= a.dup_max:
            for s in rs:
                dup[s] = c
            plan[c] = ('dup', rs)
        else:
            pool = [s for s in rs if not t.state.exemplar(s)]
            if len(pool) < a.n:                       # 대표 아닌 프레임이 모자라면 대표도 쓴다
                pool = pool + [s for s in rs if t.state.exemplar(s)][:a.n - len(pool)]
            plan[c] = ('move', pool)
            need_emb += pool
    if need_emb:
        os.makedirs(a.out_root, exist_ok=True)
        cache = os.path.join(a.out_root, '.codes_valid_dino.npz')
        items = []
        for s in need_emb:
            img = cv2.imread(t.image_path(s))
            h, w = img.shape[:2]
            items.append((s, t.image_path(s), label_bbox(t.label_text(s), w, h)))
        emb = dino_crop_embed(items, f'cuda:{a.device}' if a.device.isdigit() else a.device, cache)
        os.remove(cache)
    for c, (kind, pool) in plan.items():
        if kind == 'move':
            pick = [pool[i] for i in pick_diverse(np.stack([emb[s] for s in pool]), a.n, a.seed)] if len(pool) > a.n else pool
            for s in pick:
                moved[s] = c
    print(f'대상 코드 {len(plan)}: 옮김 {sum(1 for k, _ in plan.values() if k == "move")}코드 {len(moved)}장, 중복 {sum(1 for k, _ in plan.values() if k == "dup")}코드 {len(dup)}장')

    rows = []
    for s in v.stems:
        m = dict(v.manifest.get(s, {}))
        polys = [(v.names.get(int(l.split()[0]), v.code(s)), ' '.join(l.split()[1:])) for l in v.label_text(s).splitlines() if len(l.split()) >= 7]
        rows.append(dict(m, stem=s, code=v.code(s), n_inst=len(polys), _img=v.image_path(s), _polys=polys, _ex=v.state.exemplar(s)))
    fields = list(v.manifest[v.stems[0]].keys())
    for s, c in list(moved.items()) + list(dup.items()):
        polys = []
        for l in t.label_text(s).splitlines():
            w_ = l.split()
            if len(w_) < 7:
                continue
            nm = t.names.get(int(w_[0]), '')
            if nm == t.orig_code(s) or nm not in cid or major_of(nm, major) not in SORT5:
                nm = c
            polys.append((nm, ' '.join(w_[1:])))
        polys = drop_same_code_dups(edge_none_rule(polys), c)
        r = dict.fromkeys(fields, '')
        r.update(stem=s, source=t.manifest.get(s, {}).get('source', ''), code=c, sort5=major_of(c, major), exemplar=int(bool(t.state.exemplar(s))),
                 reviewed=1, pick='moved' if s in moved else 'dup', n_inst=len(polys), train_split=t.split(s), _img=t.image_path(s), _polys=polys,
                 _ex=t.state.exemplar(s))
        rows.append(r)
    all_codes = sorted({r['code'] for r in rows})
    out = os.path.join(a.out_root, f'valid_class{len(all_codes)}')
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
    state = {}
    for r in rows:
        e = {'status': 'ok', 'exemplar': bool(r['_ex'])}
        if r['stem'] in moved:
            e['note'] = f'학습에서 옮김 (move_codes_to_valid, {r["code"]})'
        elif r['stem'] in dup:
            e['note'] = f'학습과 중복 (프레임 {len(by_code[r["code"]])}장뿐인 코드, move_codes_to_valid)'
        elif r['_ex']:
            e['note'] = '대표'
        state[r['stem']] = e
    head = (f'# 고정 검증용 데이터셋 (move_codes_to_valid.py): {os.path.abspath(a.valid)} + 학습에만 있던 코드 {len(plan)}종의 프레임 '
            f'{len(moved) + len(dup)}장 ({os.path.abspath(a.train)} 에서; 옮김 {len(moved)}·학습과 중복 {len(dup)})\n'
            f'# 클래스 {len(all_codes)}종, {len(rows)}장. 중복 프레임(dup_in_train.txt)은 학습 프레임이 {a.dup_max}장 이하인 코드라 학습에도 남긴다 (jhw 2026-10-15)\n')
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
        f.write('\n'.join(sorted(moved)) + ('\n' if moved else ''))
    with open(os.path.join(out, 'dup_in_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(dup)) + ('\n' if dup else ''))
    with open(os.path.join(out, 'move_report.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['code', 'train_frames', 'mode', 'n_valid', 'train_left'])
        for c, (kind, pool) in sorted(plan.items()):
            k = sum(1 for s, cc in (moved if kind == 'move' else dup).items() if cc == c)
            wr.writerow([c, len(by_code[c]), kind, k, len(by_code[c]) - (k if kind == 'move' else 0)])
    print(f'검증 {len(v.stems)}장 + {len(moved) + len(dup)}장 = {len(rows)}장, 클래스 {len(all_codes)}종 → {out}')
    print(f'학습에서 뺄 stem {len(moved)} (moved_from_train.txt), 학습에도 남는 중복 stem {len(dup)} (dup_in_train.txt), 코드별 내역 move_report.csv')


if __name__ == '__main__':
    main()
