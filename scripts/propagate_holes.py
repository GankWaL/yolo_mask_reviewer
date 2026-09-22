#!/usr/bin/env python3
"""구멍까지 라벨한 대표(★)를 같은 제품의 보류 프레임에 SAM2 로 쌍 전파해 구멍 있는 자동 라벨을 만든다 (§29 B 단계).

  conda activate yolo_mask_reviewer
  python scripts/propagate_holes.py DATASET [--max-exemplars 3] [--limit 0] [--device cuda:0]

mask_reviewer.propagate.Sam2Propagator 를 쓰되, 대표 선택과 게이트를 구멍 라벨에 맞게 바꿨다.
  · 대표 선택: SAM2 객체 점수는 거의 전부 1.0 이라 변별력이 없다 → 전파 결과(채움)와 원본 라벨(채움)의 바깥 IoU 가 가장 큰 대표.
  · 게이트(통과하면 status ok, 아니면 pending 유지 — 둘 다 labels_auto/ 에 쓴다):
      바깥 IoU ≥ --min-iou(0.85)          옆 물체로 번짐·표류 차단 (물체 2개 프레임에서 점묘처럼 흩어지는 실패)
      전경 조각 수 ≤ 3                     점묘 차단
      면적비(전파/대표 같은 클래스) 0.4~2.5   표류 차단 (field_autolabel 과 같은 값)
      구멍 수 ≤ 대표 최대 구멍 수 + 3, 구멍 면적/채운 면적 ≤ 0.6
  · 저장은 dataset.yaml keep_holes 에 따라 다리 폴리곤(구멍 유지). state 에 cross_iou·holes·note 기록.
대상: 보류 상태이고 편집본·대표가 아닌 프레임 (이미 자동 라벨이 있어도 다시 계산).
  --exemplars-from DS ...  다른 데이터셋(예: holes_review_*)의 ★ 대표도 같은 코드의 대표로 쓴다. --holed-only 면 구멍이 있는 대표만
                           (구멍 없이 저장된 대표가 구멍 있는 대표를 밀어내지 않도록). 코드는 각 데이터셋의 재지정 코드 기준.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import Dataset, Instance  # noqa: E402
from mask_reviewer.propagate import Sam2Propagator  # noqa: E402

SIZE_RATIO = (0.4, 2.5)


def n_components(mask):
    return int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)


def n_holes(mask):
    return max(0, int(cv2.connectedComponents((~mask).astype(np.uint8))[0] - 2))


def union(masks, h, w):
    u = np.zeros((h, w), bool)
    for m in masks:
        u |= m
    return u


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--codes', default='')
    ap.add_argument('--max-exemplars', type=int, default=3)
    ap.add_argument('--min-iou', type=float, default=0.85)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--exemplars-from', nargs='*', default=[])
    ap.add_argument('--holed-only', action='store_true')
    a = ap.parse_args()

    ds = Dataset(a.dataset)
    # 대표 풀: (데이터셋, stem) — 자기 데이터셋 + --exemplars-from
    pools = [ds] + [Dataset(d) for d in a.exemplars_from]
    ex_by_code = {}
    for pd_ in pools:
        for s in pd_.exemplars():
            c = pd_.code(s)
            if not c:
                continue
            if a.holed_only:
                _, insts = pd_.load(s)
                if not any(n_holes(i.mask) > 0 for i in insts):
                    continue
            ex_by_code.setdefault(c, []).append((pd_, s))
    if not ds.keep_holes:
        print('경고: dataset.yaml 에 keep_holes: true 가 없어 구멍이 채워져 저장됩니다')
    codes = [c for c in a.codes.split(',') if c] or sorted(ex_by_code)
    jobs = []
    for code in codes:
        exs = ex_by_code.get(code, [])
        if not exs:
            continue
        for s in ds.auto_targets(code, include_auto=True):
            if ds.state.status(s) == 'reject':      # 제외(reject)는 절대 대상이 아니다 (auto_targets 도 보류만 주지만 명시)
                continue
            jobs.append((code, s, exs[:a.max_exemplars] if a.max_exemplars > 0 else exs))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'대상 {len(jobs)} 장 / 코드 {len(codes)} (대표 {sum(len(v) for v in ex_by_code.values())}, 풀 {len(pools)})')
    prop = Sam2Propagator(a.ckpt, a.device)
    ref_cache = {}
    n_ok = n_pend = n_none = 0
    reasons = {}
    t0 = time.time()
    try:
        for i, (code, stem, exs) in enumerate(jobs):
            tgt = cv2.imread(ds.image_path(stem), cv2.IMREAD_COLOR)
            if tgt is None:
                continue
            h, w = tgt.shape[:2]
            _, orig = ds.load(stem)
            orig_filled = maskops.fill_holes(union([o.mask for o in orig], h, w)) if orig else None
            best = None
            for pd_, ex in exs:
                key = (id(pd_), ex)
                if key not in ref_cache:
                    ref_cache[key] = pd_.load(ex)
                rimg, rinst = ref_cache[key]
                res = prop.propagate(rimg, rinst, tgt, ref_key=f'{pd_.root}/{ex}')
                if not res:
                    continue
                filled = maskops.fill_holes(union([m for _, m, _ in res], h, w))
                key = iou(filled, orig_filled) if orig_filled is not None else float(np.mean([r[2] for r in res]))
                if best is None or key > best[0]:
                    best = (key, ex, res, rinst)
            if best is None:
                ds.write_auto(stem, [], w, h, source='sam2-holes', score=0.0, ref=exs[0][1])
                ds.state.set(stem, note='auto-none sam2-holes')
                n_none += 1
                continue
            x, ex, res, rinst = best
            flags = []
            if orig_filled is not None and x < a.min_iou:
                flags.append(f'iou<{a.min_iou}')
            ex_holes = max(n_holes(r.mask) for r in rinst)
            holes = []
            for cls, m, _ in res:
                if n_components(m) > 3:
                    flags.append('speckle')
                fm = maskops.fill_holes(m)
                nh = n_holes(m)
                holes.append(nh)
                if nh > ex_holes + 3:
                    flags.append(f'holes{nh}>{ex_holes}+3')
                if (fm.sum() - m.sum()) / max(fm.sum(), 1) > 0.6:
                    flags.append('hole_area')
                same = [r.area for r in rinst if r.cls == cls]
                if same:
                    ratio = fm.sum() / max(np.mean(same), 1)
                    if not (SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]):
                        flags.append(f'size{ratio:.2f}')
            insts = [Instance(c, m) for c, m, _ in res]
            ds.write_auto(stem, insts, w, h, source='sam2-holes', score=round(x, 4), ref=ex)
            flags = sorted(set(flags))
            if flags:
                ds.state.set(stem, status='pending', cross_iou=round(x, 3), holes=holes,
                             note=f'auto-pending sam2-holes iou={x:.2f} ' + ','.join(flags))
                n_pend += 1
                for f in flags:
                    reasons[f.split('<')[0].split('>')[0].rstrip('0123456789.')] = reasons.get(f.split('<')[0].split('>')[0].rstrip('0123456789.'), 0) + 1
            else:
                ds.state.set(stem, status='ok', cross_iou=round(x, 3), holes=holes,
                             note=f'auto-ok sam2-holes iou={x:.2f} holes={holes}')
                n_ok += 1
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(jobs)} ok {n_ok} pending {n_pend} none {n_none} ({(time.time() - t0) / (i + 1):.1f}s/장)')
    finally:
        prop.close()
    print(f'완료 {len(jobs)} 장: auto-ok {n_ok}, pending {n_pend}, 없음 {n_none}, 보류 사유 {reasons}, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
