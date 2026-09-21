#!/usr/bin/env python3
"""현장 성공 프레임(raw + json)을 기준 모델로 의사 라벨링해 리허설(rehearsal) 학습셋과 보존 게이트셋을 만든다.

원본 학습셋이 없을 때 "지금 모델이 잘하는 것" 을 학습셋에 섞어 잊음을 막는 용도 (§21-bt B 안).

  conda activate pro
  python scripts/build_rehearsal_set.py ~/jhw/data/SL/field/20260918 --out ~/jhw/data/SL/rehearsal_20260918

입력: save_pose_debug/<실행>/<id>_<코드>_raw.png 와 같은 이름의 .json (json 이 있으면 성공 건).
출력 OUT/
  images/{train,val}, labels/{train,val}   리허설 학습셋 (제품코드 층화 분할), dataset.yaml
  gate/images, gate/labels_base            보존 게이트셋 (학습에 쓰지 않음, 기준 모델 라벨 = 참조)
  manifest.csv                             stem, code, set(train/val/gate), large_sort, n_det, top_cls, top_conf, source
의사 라벨 규칙: conf ≥ --min-conf 검출만, 더 높은 conf 인스턴스와 마스크가 크게 겹치면(같은 물체 박스 2개) 버림.
--exclude-codes: 두 톤 실패 제품(§21-bt)은 성공 프레임에도 같은 누락 편향이 있을 수 있어 기본 제외 (6자리 앞머리 비교).
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import sys

import cv2
import numpy as np
import yaml

DEFAULT_BASE = os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo11s_best_20260326.pt')
DEFAULT_EXCLUDE = '10E212,10J212,10M142,20P071,10T031,10T032,20Q021,20Q022,20Y041,10Q051'
LARGE_SORT = {'SCALP': 'scalp', 'H/CVR': 'h_cvr', 'B/CVR': 'b_cvr', 'HSG': 'hsg', 'CAP': 'cap'}


def scan(dirs):
    items = []
    for root in dirs:
        for dp, _dn, files in os.walk(root):
            fs = set(files)
            for f in sorted(files):
                if not f.endswith('_raw.png') or f.startswith('nodet_'):
                    continue
                stem = f[:-8]
                if f'{stem}.json' not in fs:
                    continue
                code = stem.split('_', 1)[1] if '_' in stem else 'unknown'
                run = os.path.basename(dp)
                date = next((c for c in reversed(dp.split(os.sep)) if re.fullmatch(r'\d{8}', c)), 'nodate')
                items.append(dict(code=code, stem=f'{date}_{run}_{stem}', raw=os.path.join(dp, f),
                                  js=os.path.join(dp, f'{stem}.json')))
    return items


def uniform(seq, n):
    if n <= 0 or len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in sorted(set(idx.tolist()))]


def pseudo_labels(r, min_conf, overlap, min_top_conf):
    """ultralytics 결과 -> [(cls, conf, mask(bool), poly(N,2))] conf 내림차순, 겹침 제거.
    top-1 은 min_top_conf 이상이면 남기고(운영은 conf 0.1 의 top 을 씀), 나머지는 min_conf 이상만."""
    if r.masks is None:
        return []
    md = r.masks.data.cpu().numpy().astype(bool)
    out = []
    order = np.argsort(-r.boxes.conf.cpu().numpy())
    for i in order:
        conf = float(r.boxes.conf[i])
        if conf < (min_top_conf if not out else min_conf):
            continue
        m = md[i]
        a = m.sum()
        if a < 50:
            continue
        dup = False
        for _, _, km, _ in out:
            inter = np.logical_and(m, km).sum()
            if inter / min(a, km.sum()) > overlap:
                dup = True
                break
        if dup:
            continue
        poly = r.masks.xy[i]
        if len(poly) < 3:
            continue
        out.append((int(r.boxes.cls[i]), conf, m, poly))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='field/<날짜> 폴더 (save_pose_debug 포함)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--roi', default='605,190,685,685', help='x,y,w,h (FHD px) 운영 detector_process_roi')
    ap.add_argument('--per-code', type=int, default=60, help='리허설 제품별 상한 (시간 균등)')
    ap.add_argument('--gate-per-code', type=int, default=15, help='게이트 제품별 상한 (리허설과 겹치지 않음)')
    ap.add_argument('--exclude-codes', default=DEFAULT_EXCLUDE)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.1, help='추론 conf (운영과 동일)')
    ap.add_argument('--min-conf', type=float, default=0.3, help='top-1 외 인스턴스를 라벨로 남길 최소 conf')
    ap.add_argument('--min-top-conf', type=float, default=0.2, help='top-1 최소 conf — 미만이면 프레임 자체를 제외 (빈 라벨 금지)')
    ap.add_argument('--keep-mismatch', action='store_true', help='json large_sort 와 top-1 클래스가 달라도 포함')
    ap.add_argument('--overlap', type=float, default=0.6)
    ap.add_argument('--device', default='0')
    a = ap.parse_args()

    ex = tuple(c.strip() for c in a.exclude_codes.split(',') if c.strip())
    rx, ry, rw, rh = [int(v) for v in a.roi.split(',')]
    items = scan(a.dirs)
    groups = collections.defaultdict(list)
    n_ex = 0
    for it in items:
        if it['code'].startswith(ex):
            n_ex += 1
            continue
        groups[it['code']].append(it)
    rng = random.Random(a.seed)
    sel = []
    for code in sorted(groups):
        lst = sorted(groups[code], key=lambda t: t['raw'])
        n_train = min(a.per_code, len(lst))
        picked = uniform(lst, a.per_code) if a.per_code > 0 else lst
        picked_set = set(p['stem'] for p in picked)
        rest = [t for t in lst if t['stem'] not in picked_set]
        gate = uniform(rest, a.gate_per_code) if a.gate_per_code > 0 else []
        tv = list(picked)
        rng.shuffle(tv)
        n_val = int(round(len(tv) * a.val))
        n_val = min(n_val, len(tv) - 1) if len(tv) > 1 else 0
        for i, t in enumerate(tv):
            sel.append((('val' if i < n_val else 'train'), t))
        for t in gate:
            sel.append(('gate', t))
    cnt = collections.Counter(s for s, _ in sel)
    print(f'성공 프레임 {len(items)} (제외 {n_ex}, 제품 {len(groups)}) → train {cnt["train"]} val {cnt["val"]} gate {cnt["gate"]}')
    if not sel:
        sys.exit('선택된 프레임이 없습니다')

    from ultralytics import YOLO
    model = YOLO(a.base)
    names = {int(k): str(v) for k, v in model.names.items()}
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val', 'gate/images', 'gate/labels_base'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rows = []
    cls_cnt = collections.Counter()
    skipped = collections.Counter()
    for k, (split, it) in enumerate(sel):
        fr = cv2.imread(it['raw'])
        if fr is None:
            print('skip', it['raw'], file=sys.stderr)
            continue
        crop = fr[ry:ry + rh, rx:rx + rw]
        h, w = crop.shape[:2]
        r = model.predict(crop, imgsz=a.imgsz, conf=a.conf, retina_masks=True, device=a.device, verbose=False)[0]
        dets = pseudo_labels(r, a.min_conf, a.overlap, a.min_top_conf)
        try:
            ls = json.load(open(it['js'])).get('large_sort')
        except Exception:
            ls = None
        if not dets:
            skipped['weak_top'] += 1
            continue
        if ls and not a.keep_mismatch and LARGE_SORT.get(ls) not in (None, names[dets[0][0]]):
            skipped['cls_mismatch'] += 1
            continue
        lines = []
        for cls, conf, m, poly in dets:
            pts = np.clip(poly / np.array([w, h], np.float32), 0, 1)
            lines.append(f'{cls} ' + ' '.join(f'{v:.5f}' for v in pts.reshape(-1)))
            if split != 'gate':
                cls_cnt[names[cls]] += 1
        if split == 'gate':
            ip, lp = os.path.join(a.out, 'gate/images', it['stem'] + '.png'), os.path.join(a.out, 'gate/labels_base', it['stem'] + '.txt')
        else:
            ip, lp = os.path.join(a.out, 'images', split, it['stem'] + '.png'), os.path.join(a.out, 'labels', split, it['stem'] + '.txt')
        cv2.imwrite(ip, crop)
        with open(lp, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        top = dets[0] if dets else None
        rows.append(dict(stem=it['stem'], code=it['code'], set=split, large_sort=ls or '', n_det=len(dets),
                         top_cls=names[top[0]] if top else '', top_conf=round(top[1], 3) if top else '',
                         source=os.path.relpath(it['raw'], a.dirs[0])))
        if (k + 1) % 200 == 0:
            print(f'\r{k + 1}/{len(sel)}', end='', flush=True)
    print()
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as f:
        f.write(f'# rehearsal pseudo labels by {os.path.basename(a.base)} (min_conf {a.min_conf}, overlap {a.overlap}); '
                f'excluded codes {a.exclude_codes}\n')
        yaml.safe_dump({'path': os.path.abspath(a.out), 'train': 'images/train', 'val': 'images/val', 'names': names},
                       f, allow_unicode=True, sort_keys=False)
    fin = collections.Counter(r['set'] for r in rows)
    print(f'완료 {a.out}: train {fin["train"]} val {fin["val"]} gate {fin["gate"]} / 제외 {dict(skipped)} / 인스턴스 {dict(cls_cnt)}')


if __name__ == '__main__':
    main()
