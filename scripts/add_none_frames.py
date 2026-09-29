#!/usr/bin/env python3
"""학습 폴더에 none 클래스(제품이 아닌 물체: 게이트 조각·상자 등) 프레임을 더해 새 train_class<N>/ 을 만든다 (2026-09-29).

  python scripts/add_none_frames.py TRAIN --sources DS [DS ...] --pools DIR [DIR ...] --out-root DIR [--n 300] [--model M.pt]

  TRAIN      기존 학습 폴더 (train_class<N>, 검수 툴로 여는 split 구조). 건드리지 않는다.
  --sources  검수 데이터셋 (review_state.json·라벨·manifest.csv 만 있어도 된다). 검수에서 제품코드를 `gate`·`none` 으로 바꾼 프레임을 쓴다.
  --pools    이미지를 찾을 폴더들 (그 아래 images/ 를 stem 으로 찾는다). sources 에 images/ 가 있으면 그것이 먼저.
  --n        넣을 프레임 수. 후보가 더 많으면 프레임 전체의 DINOv2 임베딩으로 외형이 고르게 남도록 고른다.
  --model    프레임 안의 나머지 물체를 찾는 모델 (기본: 운영 5클래스 모델). 검수 라벨은 물체 일부에만 있는 경우가 많아,
             모델이 찾은 마스크 중 기존 라벨과 안 겹치는 것(IoU < 0.3)도 none 으로 넣는다 — 라벨 없는 물체가 남으면 배경으로 학습된다.

none 프레임은 코드 `none`, stem `none__<원본 stem>`, 10장마다 1장이 val. 제품코드 클래스 목록에는 `none` 을 맨 뒤에 붙이고
5클래스(sort5/)에서는 1번 클래스 none 이다. review_state.json 에 none 프레임은 보류 + 메모로 남겨 검수 툴에서 확인할 수 있다.
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
from cap_per_code import pick_diverse  # noqa: E402
from codes_to_sort5 import NAMES as SORT5, NONE_CODES, major_map, major_of  # noqa: E402
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import IMG_EXTS, STATE_FILE, Dataset, load_manifest  # noqa: E402
from reassign_codes import dino_crop_embed  # noqa: E402

DEFAULT_MODEL = os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo26s_best_20260929_aug2.pt')
NOTE = 'none 자동 라벨 (검수 라벨 + 모델 검출) — 확인 필요'


def find_images(roots):
    out = {}
    for root in roots:
        d = os.path.join(root, 'images')
        for base, _, files in os.walk(d):
            for fn in files:
                stem, ext = os.path.splitext(fn)
                if ext.lower() in IMG_EXTS:
                    out.setdefault(stem, os.path.join(base, fn))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('train')
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--pools', nargs='*', default=[])
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.3, help='기존 라벨과 이보다 많이 겹치는 검출은 버린다')
    ap.add_argument('--min-area', type=float, default=1500.0, help='none 인스턴스 최소 면적 (px²)')
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    tr = Dataset(a.train)
    images = find_images([s for s in a.sources if os.path.isdir(os.path.join(s, 'images'))] + a.pools)
    cand = []
    seen = set()
    for src in a.sources:
        with open(os.path.join(src, STATE_FILE), encoding='utf-8') as f:
            state = json.load(f)
        man = load_manifest(src)
        for s in sorted(state):
            code = (state[s].get('code') or '').lower()
            if code not in NONE_CODES or s not in man or s not in images or s in seen:
                continue
            seen.add(s)
            text = ''
            for sub in ('labels_reviewed', 'labels_auto', 'labels'):
                p = os.path.join(src, sub, s + '.txt')
                if os.path.exists(p):
                    with open(p, encoding='utf-8') as f:
                        text = f.read()
                    break
            cand.append(dict(stem=s, src=os.path.basename(os.path.normpath(src)), code=code, img=images[s], text=text))
    print(f'후보 {len(cand)}장 (코드 {dict(collections.Counter(c["code"] for c in cand))})')

    model = YOLO(a.model)
    stat = collections.Counter()
    ready = []
    for i in range(0, len(cand), 16):
        part = cand[i:i + 16]
        res = model.predict([c['img'] for c in part], imgsz=640, conf=a.conf, retina_masks=True, device=a.device, verbose=False)
        for c, r in zip(part, res):
            h, w = r.orig_shape
            masks = [m for m in (inst.mask for inst in Dataset.parse_instances(c['text'], w, h)) if m.sum() >= a.min_area]
            n_label = len(masks)
            if r.masks is not None:
                for m in r.masks.data.cpu().numpy() > 0.5:
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    if m.sum() >= a.min_area and all(maskops.iou(m, k) < a.iou for k in masks):
                        masks.append(m)
            polys = []
            for m in masks:
                line = maskops.mask_to_yolo_line(0, m, w, h, 0.7, 16.0, keep_holes=tr.keep_holes)
                if line:
                    polys.append(' '.join(line.split()[1:]))
            if not polys:
                stat['no_instance'] += 1
                continue
            c.update(polys=polys, n_label=n_label, n_model=len(masks) - n_label)
            stat['label_inst'] += n_label
            stat['model_inst'] += len(masks) - n_label
            ready.append(c)
    print(f'인스턴스가 있는 후보 {len(ready)}장, {dict(stat)}')

    if len(ready) > a.n:
        os.makedirs(a.out_root, exist_ok=True)
        cache = os.path.join(a.out_root, '.none_dino.npz')
        emb = dino_crop_embed([(c['stem'], c['img'], None) for c in ready], f'cuda:{a.device}' if a.device.isdigit() else a.device, cache)
        os.remove(cache)
        ready = [ready[i] for i in pick_diverse(np.stack([emb[c['stem']] for c in ready]), a.n, a.seed)]
    for i, c in enumerate(ready):
        c['split'] = 'val' if i % a.val_every == a.val_every - 1 else 'train'

    major = major_map()
    names = dict(tr.names)
    if 'none' not in names.values():
        names[max(names) + 1] = 'none'
    cid = {v: k for k, v in names.items()}
    rows = []
    for s in tr.stems:
        code = tr.code(s)
        if tr.state.status(s) == 'reject' or code not in cid or major_of(code, major) not in SORT5:
            continue
        polys = [' '.join(l.split()[1:]) for l in tr.label_text(s).splitlines() if len(l.split()) >= 7]
        if polys:
            m = tr.manifest.get(s, {})
            rows.append(dict(stem=s, split=tr.split(s), code=code, sort5=major_of(code, major), exemplar=int(tr.state.exemplar(s)),
                             human=m.get('human', ''), reason=m.get('reason', ''), export=m.get('export', ''), source=m.get('source', ''),
                             label_src=m.get('label_src', ''), n_inst=len(polys), _img=tr.image_path(s), _polys=polys))
    for c in ready:
        rows.append(dict(stem='none__' + c['stem'], split=c['split'], code='none', sort5='none', exemplar=0, human=0,
                         reason=f'none({c["code"]})', export='', source=c['src'], label_src=f'label {c["n_label"]} + model {c["n_model"]}',
                         n_inst=len(c['polys']), _img=c['img'], _polys=c['polys']))
    codes = sorted({r['code'] for r in rows})
    out = os.path.join(a.out_root, f'train_class{len(codes)}')
    if os.path.exists(out):
        sys.exit(f'이미 있습니다: {out}')
    n = collections.Counter()
    for r in rows:
        ext = os.path.splitext(r['_img'])[1]
        for sub, k in (('', cid[r['code']]), ('sort5', SORT5.index(r['sort5']))):
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(out, sub, d, r['split']), exist_ok=True)
            os.link(r['_img'], os.path.join(out, sub, 'images', r['split'], r['stem'] + ext))
            with open(os.path.join(out, sub, 'labels', r['split'], r['stem'] + '.txt'), 'w', encoding='utf-8') as f:
                f.write(''.join(f'{k} {p}\n' for p in r['_polys']))
        n[(r['split'], r['code'] == 'none')] += 1
    state = {r['stem']: {'exemplar': True, 'status': 'ok', 'note': '대표'} for r in rows if r['exemplar']}
    state.update({r['stem']: {'status': 'pending', 'note': NOTE} for r in rows if r['code'] == 'none'})
    fields = [k for k in rows[0] if not k.startswith('_')]
    n_none = sum(r['code'] == 'none' for r in rows)
    head = (f'# 학습 데이터셋 (add_none_frames.py): {os.path.abspath(a.train)} + none 프레임 {n_none}장 (검수에서 gate·none 으로 제외된 프레임)\n'
            f'# 클래스 {len(codes)}종(제품코드 {len(codes) - 1} + none), train {n[("train", False)] + n[("train", True)]} / '
            f'val {n[("val", False)] + n[("val", True)]}\n')
    for sub, nm in (('', names), ('sort5', dict(enumerate(SORT5)))):
        with open(os.path.join(out, sub, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rows)
        with open(os.path.join(out, sub, STATE_FILE), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
        with open(os.path.join(out, sub, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(head)
            yaml.safe_dump({'path': os.path.abspath(os.path.join(out, sub)).rstrip('/'), 'train': 'images/train', 'val': 'images/val',
                            'names': nm, 'keep_holes': True}, f, allow_unicode=True, sort_keys=False)
    print(f'제품 프레임 {len(rows) - n_none}장 + none {n_none}장 (train {n[("train", True)]} / val {n[("val", True)]}, '
          f'인스턴스 {sum(r["n_inst"] for r in rows if r["code"] == "none")}개), 클래스 {len(codes)}종')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
