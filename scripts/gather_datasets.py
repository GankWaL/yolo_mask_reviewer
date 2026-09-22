#!/usr/bin/env python3
"""여러 검수 데이터셋을 하나의 검수 데이터셋으로 모은다 (유효 라벨·재지정 코드·판정·대표 유지).

  python scripts/gather_datasets.py --out OUT SRC [SRC ...] [--prefix] [--names-from SRC]

  SRC   mask_reviewer 데이터셋 (images/ labels/ [labels_reviewed/ labels_auto/ review_state.json manifest.csv aux/]).
        images/<split>/ 구조(YOLO 폴더)도 받는다.
  OUT/images/<stem>.png          하드링크 (같은 stem 이 여러 SRC 에 있으면 앞의 SRC 가 이긴다; --prefix 면 stem 앞에 SRC 이름을 붙여 모두 보존)
  OUT/labels/<stem>.txt          SRC 의 **유효 라벨**(편집본 > 자동 > 원본) 을 원본 라벨로 둔다 → 툴에서 그대로 보이고, 다시 편집·전파 가능
  OUT/aux/<stem>.png             있으면 복사
  OUT/review_state.json          status·code·exemplar·note·auto 메타 유지 (code 는 SRC 의 재지정 코드; 없으면 안 씀)
  OUT/manifest.csv               stem, source(SRC/split), code(최종), status, label_src, has_holes(라벨에 구멍 있음), n_inst, reason 등 SRC manifest 열
  OUT/dataset.yaml               names 는 --names-from(기본 첫 SRC), keep_holes: true
"""
import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_reviewer.dataset import Dataset, load_dataset_yaml  # noqa: E402


def has_holes(text, w, h):
    for line in text.splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = (np.array(v[1:], dtype=np.float64).reshape(-1, 2) * [w, h]).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        if cv2.connectedComponents(1 - m)[0] - 2 > 0:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', action='store_true')
    ap.add_argument('--names-from', default='')
    a = ap.parse_args()
    for d in ('images', 'labels', 'aux'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    names = None
    state = {}
    rows = []
    seen = set()
    extra_cols = set()
    for src in a.src:
        tag = os.path.basename(os.path.normpath(src))
        # YOLO 폴더(images/<split>) 는 split 별로 임시 Dataset 처럼 다룬다
        subs = [src]
        if not any(f.lower().endswith(('.png', '.jpg')) for f in os.listdir(os.path.join(src, 'images'))):
            subs = []
        ds = Dataset(src) if subs else None
        y = load_dataset_yaml(src)
        if names is None and (not a.names_from or os.path.abspath(a.names_from) == os.path.abspath(src)):
            n = y.get('names')
            if n:
                names = {int(k): str(v) for k, v in (n.items() if isinstance(n, dict) else enumerate(n))}
        items = []
        if ds is not None:
            for s in ds.stems:
                items.append((s, ds.image_path(s), ds.label_text(s), ds.label_source(s), ds.code(s), ds.state.get(s), ds.manifest.get(s, {}), tag,
                              ds.aux_mask(s) if ds.has_aux else None))
        else:
            for sp in sorted(os.listdir(os.path.join(src, 'images'))):
                idir = os.path.join(src, 'images', sp)
                for fn in sorted(os.listdir(idir)):
                    s, ext = os.path.splitext(fn)
                    lab = os.path.join(src, 'labels', sp, s + '.txt')
                    text = open(lab, encoding='utf-8').read() if os.path.exists(lab) else ''
                    items.append((s, os.path.join(idir, fn), text, 'original', '', {}, {}, f'{tag}/{sp}', None))
        n_src = 0
        for s, ipath, text, lsrc, code, st, man, source, aux in items:
            stem = f'{tag}__{s}' if a.prefix else s
            if stem in seen:
                continue
            seen.add(stem)
            n_src += 1
            dst = os.path.join(a.out, 'images', stem + os.path.splitext(ipath)[1])
            if not os.path.exists(dst):
                try:
                    os.link(ipath, dst)
                except OSError:
                    shutil.copy2(ipath, dst)
            with open(os.path.join(a.out, 'labels', stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write(text if text.endswith('\n') or not text else text + '\n')
            if aux is not None:
                cv2.imwrite(os.path.join(a.out, 'aux', stem + '.png'), aux)
            img = cv2.imread(ipath)
            h, w = img.shape[:2]
            e = {k: v for k, v in (st or {}).items() if k in ('status', 'exemplar', 'note', 'auto', 'updated', 'cross_iou', 'holes')}
            if code:
                e['code'] = code   # 재지정 여부와 무관하게 최종 코드를 고정 (파일명 코드가 틀린 경우가 있으므로)
            e['source'] = source
            state[stem] = e
            row = dict(stem=stem, source=source, code=code, status=(st or {}).get('status', 'pending'), label_src=lsrc,
                       has_holes=has_holes(text, w, h), n_inst=sum(1 for l in text.splitlines() if len(l.split()) >= 7),
                       exemplar=int(bool((st or {}).get('exemplar'))))
            for k in ('reason', 'n_det', 'top_cls', 'top_conf', 'top_fill', 'cad_status', 'disagree_px'):
                if k in man:
                    row[k] = man[k]
                    extra_cols.add(k)
            rows.append(row)
        print(f'{tag}: {n_src} 장')
    fields = ['stem', 'source', 'code', 'status', 'label_src', 'has_holes', 'n_inst', 'exemplar'] + sorted(extra_cols)
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(a.out, 'review_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'# gather_datasets.py: {", ".join(os.path.basename(os.path.normpath(s)) for s in a.src)} 를 모음 (유효 라벨 = 원본, 재지정 코드 고정)\n')
        yaml.safe_dump({'path': os.path.abspath(a.out), 'train': 'images', 'val': 'images', 'names': names or {},
                        'keep_holes': True, 'aux_dir': 'aux'}, f, sort_keys=False, allow_unicode=True)
    import collections
    print(f'완료 {len(rows)} 장 → {a.out}: 코드 {len({r["code"] for r in rows if r["code"]})}, 구멍 라벨 {sum(r["has_holes"] for r in rows)}, '
          f'상태 {dict(collections.Counter(r["status"] for r in rows))}, 라벨 출처 {dict(collections.Counter(r["label_src"] for r in rows))}')


if __name__ == '__main__':
    main()
