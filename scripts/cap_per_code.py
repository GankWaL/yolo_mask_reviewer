#!/usr/bin/env python3
"""학습 데이터셋의 제품코드별 장수를 상한(기본 60)으로 줄인다 — 클래스 균형용, 외형 다양성을 남기는 쪽으로 고른다 (2026-09-29).

  python scripts/cap_per_code.py SRC --out-root DIR [--cap 60] [--device 0] [--exemplars TAG=DATASET ...]

  SRC        검수 툴로 여는 학습 폴더 (images/<split>, labels/<split>, manifest.csv, 있으면 review_state.json·labels_reviewed).
  --out-root 이 아래에 train_class<N>/ 을 새로 만든다 (N = 남은 제품코드 수). SRC 는 건드리지 않는다.
  --exemplars ★ 대표 표시를 가져올 원본 검수 데이터셋. SRC 의 stem 이 `<TAG>__<원본 stem>` 일 때 TAG 마다 그 데이터셋 폴더
             (review_state.json 만 있으면 된다). SRC 자체의 ★ 도 쓴다.

검수 반영: 제외(reject) 프레임은 빼고, 편집본이 있으면 그 라벨을, 재지정 코드가 있으면 그 코드를 쓴다.
고르는 법 (코드·split 마다): 상한을 split 비율대로 나누고(val 이 있으면 최소 1장), ★ 대표 → 사람이 고친 프레임(편집본·코드 재지정) → 나머지
순으로 채운다. 자리가 모자란 단계에서는 물체 크롭의 DINOv2 임베딩을 k-means 로 남은 자리 수만큼 묶어 묶음마다 중심에 가장 가까운 한 장을
고른다 — 비슷한 프레임이 몰린 곳은 줄고 드문 외형은 남는다. 상한 이하인 코드는 전부 남긴다.
결과: train_class<N>/{images,labels}/<split>, sort5/(같은 프레임 5클래스), dataset.yaml, manifest.csv, cap_report.csv(전 프레임의 keep·사유),
review_state.json(★ 대표 = exemplar·확정 — 검수 툴의 대표 필터·목록 ★ 표시용).
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
from codes_to_sort5 import NAMES as SORT5, major_map, major_of  # noqa: E402
from mask_reviewer.dataset import STATE_FILE, Dataset  # noqa: E402
from reassign_codes import dino_crop_embed  # noqa: E402


def label_bbox(text, w, h):
    """라벨의 가장 큰 폴리곤(bbox 면적 기준)의 픽셀 bbox xyxy. 없으면 None."""
    best = None
    for line in text.splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        p = np.array(v[1:len(v) - (len(v) - 1) % 2], dtype=np.float64).reshape(-1, 2)
        x0, y0, x1, y1 = p[:, 0].min() * w, p[:, 1].min() * h, p[:, 0].max() * w, p[:, 1].max() * h
        a = (x1 - x0) * (y1 - y0)
        if best is None or a > best[0]:
            best = (a, (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))))
    return best[1] if best else None


def pick_diverse(emb, k, seed=0):
    """emb (n, d) L2 정규화 → 고른 행 번호 k 개: k-means 묶음마다 중심에 가장 가까운 한 장."""
    from sklearn.cluster import KMeans
    n = len(emb)
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb)
    out = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        out.append(int(idx[np.argmin(((emb[idx] - km.cluster_centers_[c]) ** 2).sum(1))]))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--cap', type=int, default=60)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exemplars', nargs='*', default=[], metavar='TAG=DATASET')
    ap.add_argument('--emb-cache', default=None, help='임베딩 캐시 npz (기본: 출력 폴더의 .cap_dino.npz, 끝나면 지움)')
    a = ap.parse_args()

    ds = Dataset(a.src)
    cid = {v: k for k, v in ds.names.items()}
    major = major_map()

    ex_state = {}
    for spec in a.exemplars:
        tag, _, root = spec.partition('=')
        with open(os.path.join(os.path.expanduser(root), STATE_FILE), encoding='utf-8') as f:
            ex_state[tag] = json.load(f)

    def is_exemplar(s):
        tag, sep, orig = s.partition('__')
        return ds.state.exemplar(s) or bool(sep and ex_state.get(tag, {}).get(orig, {}).get('exemplar'))

    report = []
    by_code = collections.defaultdict(list)
    for s in ds.stems:
        code = ds.code(s)
        human = bool(ds.is_edited(s) or ds.state.code(s))
        row = dict(stem=s, split=ds.split(s), code=code, status=ds.state.status(s), human=int(human), exemplar=int(is_exemplar(s)),
                   keep=0, reason='')
        report.append(row)
        if ds.state.status(s) == 'reject':
            row['reason'] = 'reject'
        elif code not in cid or major_of(code, major) not in SORT5:
            row['reason'] = 'unknown_code'
        elif not any(len(l.split()) >= 7 for l in ds.label_text(s).splitlines()):
            row['reason'] = 'nolabel'
        else:
            by_code[code].append(row)

    over = {c: r for c, r in by_code.items() if len(r) > a.cap}
    items = []
    for rows in over.values():
        for r in rows:
            img = cv2.imread(ds.image_path(r['stem']))
            h, w = img.shape[:2]
            items.append((r['stem'], ds.image_path(r['stem']), label_bbox(ds.label_text(r['stem']), w, h)))
    os.makedirs(a.out_root, exist_ok=True)
    cache = a.emb_cache or os.path.join(a.out_root, '.cap_dino.npz')
    emb = dino_crop_embed(items, f'cuda:{a.device}' if a.device.isdigit() else a.device, cache) if items else {}
    if not a.emb_cache and os.path.exists(cache):
        os.remove(cache)

    for code, rows in by_code.items():
        if code not in over:
            for r in rows:
                r['keep'], r['reason'] = 1, 'under_cap'
            continue
        splits = collections.defaultdict(list)
        for r in rows:
            splits[r['split']].append(r)
        # 상한을 split 비율대로 (큰 split 이 나머지를 받는다, 작은 split 은 최소 1장)
        order = sorted(splits, key=lambda sp: len(splits[sp]))
        quota, left = {}, a.cap
        for i, sp in enumerate(order):
            q = left if i == len(order) - 1 else max(1, round(a.cap * len(splits[sp]) / len(rows)))
            quota[sp] = min(q, len(splits[sp]))
            left -= quota[sp]
        for sp, rs in splits.items():
            # 우선순위: ★ 대표 → 사람이 고친 프레임 → 나머지. 자리가 모자란 단계에서만 다양성 기준으로 고르고 그 아래 단계는 모두 뺀다
            tiers = [('exemplar', [r for r in rs if r['exemplar']]),
                     ('human', [r for r in rs if r['human'] and not r['exemplar']]),
                     ('diverse', [r for r in rs if not (r['human'] or r['exemplar'])])]
            left_sp = quota[sp]
            for name, tier in tiers:
                if len(tier) <= left_sp:
                    pick = set(range(len(tier)))
                elif left_sp > 0:
                    pick = set(pick_diverse(np.stack([emb[r['stem']] for r in tier]), left_sp, a.seed))
                else:
                    pick = set()
                for i, r in enumerate(tier):
                    r['keep'], r['reason'] = (1, name) if i in pick else (0, 'over_cap')
                left_sp -= len(pick)

    kept = [r for r in report if r['keep']]
    codes = sorted({r['code'] for r in kept})
    out = os.path.join(a.out_root, f'train_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    n = collections.Counter()
    for r in kept:
        s, sp = r['stem'], r['split']
        src = ds.image_path(s)
        polys = [' '.join(l.split()[1:]) for l in ds.label_text(s).splitlines() if len(l.split()) >= 7]
        for sub, k in (('', cid[r['code']]), ('sort5', SORT5.index(major_of(r['code'], major)))):
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(out, sub, d, sp), exist_ok=True)
            os.link(src, os.path.join(out, sub, 'images', sp, os.path.basename(src)))
            with open(os.path.join(out, sub, 'labels', sp, s + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{k} {p}\n' for p in polys))
        r['sort5'] = major_of(r['code'], major)
        n[sp] += 1
    man_src = ds.manifest
    fields = ['stem', 'split', 'code', 'sort5', 'exemplar', 'human', 'reason', 'export', 'source', 'label_src']
    for sub in ('', 'sort5'):
        with open(os.path.join(out, sub, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            for r in kept:
                m = man_src.get(r['stem'], {})
                wr.writerow(dict(r, export=m.get('export', ''), source=m.get('source', ''), label_src=m.get('label_src', '')))
    with open(os.path.join(out, 'cap_report.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['stem', 'split', 'code', 'status', 'exemplar', 'human', 'keep', 'reason'], extrasaction='ignore')
        wr.writeheader()
        wr.writerows(report)
    state = {r['stem']: {'exemplar': True, 'status': 'ok', 'note': '대표 (원본 검수 데이터셋에서 가져옴)'} for r in kept if r['exemplar']}
    for sub in ('', 'sort5'):
        with open(os.path.join(out, sub, STATE_FILE), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
    head = (f'# 학습 데이터셋 (cap_per_code.py): {os.path.abspath(a.src)} 에서 검수 반영 + 제품코드별 상한 {a.cap}장 (외형 다양성 기준 선택)\n'
            f'# 제품코드 {len(codes)}종, ' + ' / '.join(f'{sp} {n[sp]}' for sp in sorted(n)) + f' (원본 {len(ds.stems)}장)\n')
    for sub, nm in (('', ds.names), ('sort5', dict(enumerate(SORT5)))):
        with open(os.path.join(out, sub, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(head)
            yaml.safe_dump({'path': os.path.abspath(os.path.join(out, sub)).rstrip('/'), 'train': 'images/train', 'val': 'images/val',
                            'names': nm, 'keep_holes': True}, f, allow_unicode=True, sort_keys=False)
    per = collections.Counter(r['code'] for r in kept)
    print(f'원본 {len(ds.stems)}장 → {len(kept)}장 ({dict(sorted(n.items()))}), 제품코드 {len(codes)}종, 코드당 {min(per.values())}~{max(per.values())}장')
    print(f'상한 초과 코드 {len(over)}종, 사유 {dict(collections.Counter(r["reason"] for r in report))}')
    print(f'대분류 {dict(collections.Counter(r["sort5"] for r in kept))}')
    n_ex = sum(r['exemplar'] for r in report if r['reason'] != 'reject')
    print(f'★ 대표 {len(state)} / {n_ex} 장 남김, 대표 있는 코드 {len({r["code"] for r in kept if r["exemplar"]})} / {len(codes)}')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
