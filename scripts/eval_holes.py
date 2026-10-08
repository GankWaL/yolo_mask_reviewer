#!/usr/bin/env python3
"""고정 검증셋에서 관통 구멍 성능을 크기별로 잰다 — 구멍 재현율·정밀도·IoU (2026-10-08).

  CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/eval_holes.py VALID W1.pt [W2.pt ...] [--device 0] [--conf 0.25] [--imgsz 640] [--csv OUT.csv]

  VALID  valid_class<N> 폴더. 모델 클래스가 5개 이하면 sort5/, 아니면 루트를 쓴다 (클래스는 무시하고 마스크만 본다).
프레임마다 정답 인스턴스(none 제외)와 예측 마스크를 IoU 0.5 이상으로 짝짓고, 정답 마스크의 구멍(내부 윤곽, 면적 ≥ 4px)마다
예측 마스크에 그 자리에 구멍이 있는지(정답 구멍 영역의 50% 이상이 예측 마스크에서 비어 있으면 검출) 센다.
  재현율 = 검출된 정답 구멍 / 정답 구멍, 크기 구간은 등가 지름(px, 원본 해상도) 기준.
  정밀도 = 정답 구멍과 겹치는 예측 구멍 / 예측 구멍 (헛구멍 비율의 반대).
  구멍 IoU = 정답 구멍 합집합 vs 예측 구멍 합집합의 IoU (프레임 평균, 정답 구멍이 있는 프레임만).
"""
import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402

BUCKETS = ((0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 10 ** 9))


def holes_of(mask):
    """마스크의 구멍 목록 [(면적, 구멍 bool 마스크)]."""
    m = mask.astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] == -1:
            continue
        a = cv2.contourArea(c)
        if a < 4:
            continue
        h = np.zeros_like(m)
        cv2.drawContours(h, [c], -1, 1, thickness=-1)
        out.append((a, h.astype(bool)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('valid')
    ap.add_argument('weights', nargs='+')
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    from ultralytics import YOLO

    rows = []
    for w in a.weights:
        m = YOLO(w)
        root = os.path.join(a.valid, 'sort5') if len(m.names) <= 5 else a.valid
        ds = Dataset(root)
        name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(w))))
        tot = collections.Counter()
        hit = collections.Counter()
        pred_holes = pred_hit = 0
        ious = []
        n_frames = 0
        stems = ds.stems
        for i in range(0, len(stems), 16):
            part = stems[i:i + 16]
            res = m.predict([ds.image_path(s) for s in part], imgsz=a.imgsz, conf=a.conf, retina_masks=True, device=a.device, verbose=False)
            for s, r in zip(part, res):
                h, w_ = r.orig_shape
                gts = [inst for inst in ds.parse_instances(ds.label_text(s), w_, h) if ds.names.get(inst.cls) != 'none' and inst.area > 0]
                pms = []
                if r.masks is not None:
                    for pm in r.masks.data.cpu().numpy() > 0.5:
                        if pm.shape != (h, w_):
                            pm = cv2.resize(pm.astype(np.uint8), (w_, h), interpolation=cv2.INTER_NEAREST) > 0
                        pms.append(pm)
                gt_union = np.zeros((h, w_), bool)
                pr_union = np.zeros((h, w_), bool)
                any_gt_hole = False
                for g in gts:
                    gh = holes_of(g.mask)
                    best, bj = 0.0, -1
                    for j, pm in enumerate(pms):
                        v = maskops.iou(g.mask, pm)
                        if v > best:
                            best, bj = v, j
                    pm = pms[bj] if bj >= 0 and best >= 0.5 else None
                    for area, hm in gh:
                        any_gt_hole = True
                        d = 2.0 * np.sqrt(area / np.pi)
                        b = next(k for k, (lo, hi) in enumerate(BUCKETS) if lo <= d < hi)
                        tot[b] += 1
                        gt_union |= hm
                        if pm is not None and (~pm[hm]).sum() >= 0.5 * hm.sum():
                            hit[b] += 1
                    if pm is not None:
                        for area, ph in holes_of(pm):
                            pred_holes += 1
                            pr_union |= ph
                            if any(maskops.iou(ph, hm) > 0 for _, hm in gh):
                                pred_hit += 1
                if any_gt_hole:
                    n_frames += 1
                    ious.append(maskops.iou(gt_union, pr_union))
        n_gt = sum(tot.values())
        n_hit = sum(hit.values())
        row = dict(model=name, frames_with_holes=n_frames, gt_holes=n_gt,
                   recall=round(n_hit / max(n_gt, 1), 3), precision=round(pred_hit / max(pred_holes, 1), 3),
                   pred_holes=pred_holes, hole_iou=round(float(np.mean(ious)) if ious else 0.0, 3))
        for k, (lo, hi) in enumerate(BUCKETS):
            row[f'recall_{lo}_{hi if hi < 10 ** 9 else "inf"}'] = f'{hit[k]}/{tot[k]}' + (f' ({100 * hit[k] / tot[k]:.0f}%)' if tot[k] else '')
        rows.append(row)
        print('HOLES ' + ' | '.join(f'{k}={v}' for k, v in row.items()), flush=True)
    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == '__main__':
    main()
