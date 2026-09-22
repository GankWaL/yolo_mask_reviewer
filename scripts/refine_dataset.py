#!/usr/bin/env python3
"""학습 데이터셋 정제 — 제품이 중앙 부근에 온전히(전체 형상의 95% 이상) 단독으로 있는 프레임만 학습에 쓴다 (2026-09-22).

  conda activate yolo_mask_reviewer
  python scripts/refine_dataset.py DATASET [--model <현장 YOLO>] [--device 1] [--apply]

프레임마다 유효 라벨(편집본 > 자동 > 원본)의 가장 큰 인스턴스를 보고 아래를 잰다. 하나라도 걸리면 제외.
  · multi      라벨 인스턴스가 2개 이상 (제품이 여럿 라벨됨)
  · border     마스크가 크롭 경계에 닿음(--border-px 폭 안의 픽셀 ≥ --border-min) → 잘린 형상
  · partial    마스크가 경계 띠(--edge-px 6)에 조금이라도 닿고 면적 / 같은 제품코드의 온전한 프레임 면적 중앙값 < --min-area-ratio(0.95),
               → 일부만 보임 (자세에 따라 면적이 달라지므로 경계에 안 닿으면 비율만으로는 안 뺀다)
  · offcenter  마스크 중심이 화면 중심에서 폭·높이의 --center-frac(0.25) 보다 멀리 있음
  · neighbor   현장 YOLO(--model, conf 0.1)로 재검출했을 때 라벨과 안 겹치는 다른 물체(≥ --neighbor-px), 또는 벨트색(녹색 H 18~75)이 아닌
               전경 덩어리(≥ --blob-px, 라벨 --shadow-px 팽창 영역 밖, 벨트 중앙 띠 --band-frac 안)가 있음 → 주변 노이즈 (가장자리에 잘린 이웃 제품도 잡힘)
결과: DATASET/refine_report.csv (stem, code, status, keep, reasons, 수치). --apply 면 state 에 train_exclude(true/false)·refine_reasons 를 쓴다.
판정(status)은 바꾸지 않는다. 내보내기(export)는 train_exclude 인 프레임을 기본으로 건너뛴다 (--keep-excluded 로 포함).
"""
import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_reviewer.dataset import Dataset  # noqa: E402

DEFAULT_MODEL = os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo11s_best_20260919.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--device', default='1')
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--edge-px', type=int, default=6, help='partial 판정용 경계 띠')
    ap.add_argument('--blob-px', type=int, default=2500, help='벨트색과 다른 전경 덩어리 최소 면적 (이웃 제품)')
    ap.add_argument('--band-frac', type=float, default=0.15, help='이웃 검사에서 제외할 위아래 비율 (레일·조명 띠)')
    ap.add_argument('--shadow-px', type=int, default=15, help='라벨 주변 이 폭은 이웃 검사에서 제외 (제품 그림자)')
    ap.add_argument('--border-px', type=int, default=3)
    ap.add_argument('--border-min', type=int, default=20, help='경계 띠 안의 마스크 픽셀이 이보다 많으면 border')
    ap.add_argument('--min-area-ratio', type=float, default=0.95)
    ap.add_argument('--center-frac', type=float, default=0.25)
    ap.add_argument('--neighbor-px', type=int, default=2000)
    ap.add_argument('--statuses', default='ok,pending', help='검사할 판정 (reject 는 어차피 학습에 안 씀)')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    ds = Dataset(a.dataset)
    sts = set(a.statuses.split(','))
    stems = [s for s in ds.stems if ds.state.status(s) in sts]
    if a.limit:
        stems = stems[:a.limit]
    print(f'검사 {len(stems)} 장 (판정 {sorted(sts)})')

    # 1) 라벨 기반 수치
    info = {}
    for i, s in enumerate(stems):
        img, insts = ds.load(s)
        h, w = img.shape[:2]
        if not insts:
            info[s] = dict(n_inst=0)
            continue
        inst = max(insts, key=lambda x: x.area)
        m = inst.mask
        b = a.border_px
        border = int(m[:b, :].sum() + m[-b:, :].sum() + m[:, :b].sum() + m[:, -b:].sum())
        ys, xs = np.where(m)
        cx, cy = xs.mean() / w, ys.mean() / h
        e = a.edge_px
        edge_touch = int(m[:e, :].sum() + m[-e:, :].sum() + m[:, :e].sum() + m[:, -e:].sum())
        # 벨트색과 다른 전경 덩어리 (라벨 밖, 벨트 중앙 띠 안): 벨트는 녹색(H 18~75, S·V 중간)이고 제품은 검정(V 낮음)·흰/은색(S 낮음)·
        # 다른 색상(H 밖)이다. 조명으로 벨트색이 위치마다 달라 통계 대신 고정 임계값을 쓴다. 제품 그림자는 라벨 주변(--shadow-px)을 빼서 피한다.
        u8 = m.astype(np.uint8)
        near = cv2.dilate(u8, np.ones((2 * a.shadow_px + 1, 2 * a.shadow_px + 1), np.uint8)) > 0
        band = np.zeros((h, w), bool)
        band[int(a.band_frac * h):int((1 - a.band_frac) * h), :] = True
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H_, S_, V_ = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        fg = ((S_ < 30) | (V_ < 45) | (H_ < 18) | (H_ > 75)) & (~near) & band
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        nb, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        blob_px = 0
        n_blob = 0
        for k in range(1, nb):
            if stats[k, cv2.CC_STAT_AREA] >= a.blob_px:
                n_blob += 1
                blob_px = max(blob_px, int(stats[k, cv2.CC_STAT_AREA]))
        info[s] = dict(n_inst=len(insts), area=int(m.sum()), border_px=border, edge_touch=edge_touch, cx=round(float(cx), 3), cy=round(float(cy), 3),
                       center_off=round(float(max(abs(cx - 0.5), abs(cy - 0.5))), 3), mask=m, hw=(h, w), n_blob=n_blob, blob_px=blob_px)
        if (i + 1) % 1000 == 0:
            print(f'  라벨 {i + 1}/{len(stems)}')
    # 같은 코드의 온전한(경계 미접촉) 프레임 면적 중앙값
    full = collections.defaultdict(list)
    for s, d in info.items():
        if d.get('n_inst', 0) >= 1 and d['border_px'] < a.border_min:
            full[ds.code(s)].append(d['area'])
    med = {c: float(np.median(v)) for c, v in full.items() if v}

    # 2) 현장 YOLO 로 주변 물체 검사
    from ultralytics import YOLO
    model = YOLO(a.model)
    B = 16
    for i in range(0, len(stems), B):
        chunk = stems[i:i + B]
        res = model.predict([ds.image_path(s) for s in chunk], imgsz=640, conf=a.conf, retina_masks=True, device=a.device, verbose=False)
        for s, r in zip(chunk, res):
            d = info[s]
            d['n_neighbor'] = 0
            d['neighbor_px'] = 0
            if r.masks is None or d.get('n_inst', 0) == 0:
                continue
            h, w = d['hw']
            gt = d['mask']
            for m in r.masks.data.cpu().numpy():
                m = cv2.resize(m, (w, h)) > 0.5
                area = int(m.sum())
                if area < a.neighbor_px:
                    continue
                inter = (m & gt).sum()
                iou = inter / max((m | gt).sum(), 1)
                if iou < 0.3 and inter / area < 0.5:
                    d['n_neighbor'] += 1
                    d['neighbor_px'] = max(d['neighbor_px'], area)
        if (i // B) % 40 == 0:
            print(f'  검출 {min(i + B, len(stems))}/{len(stems)}')

    # 3) 판정
    rows = []
    reasons_cnt = collections.Counter()
    for s in stems:
        d = info[s]
        code = ds.code(s)
        rs = []
        if d.get('n_inst', 0) == 0:
            rs.append('nolabel')
        else:
            if d['n_inst'] > 1:
                rs.append('multi')
            if d['border_px'] >= a.border_min:
                rs.append('border')
            ratio = d['area'] / med[code] if code in med else None
            d['area_ratio'] = round(ratio, 3) if ratio is not None else ''
            if ratio is not None and d['edge_touch'] > 0 and ratio < a.min_area_ratio:   # 자세에 따라 면적이 달라 경계에 닿을 때만
                rs.append('partial')
            if d['center_off'] > a.center_frac:
                rs.append('offcenter')
            if d.get('n_neighbor', 0) > 0 or d.get('n_blob', 0) > 0:
                rs.append('neighbor')
        for r_ in rs:
            reasons_cnt[r_] += 1
        rows.append(dict(stem=s, code=code, status=ds.state.status(s), keep=int(not rs), reasons='|'.join(rs),
                         n_inst=d.get('n_inst', 0), area=d.get('area', ''), area_ratio=d.get('area_ratio', ''),
                         border_px=d.get('border_px', ''), edge_touch=d.get('edge_touch', ''), center_off=d.get('center_off', ''),
                         n_neighbor=d.get('n_neighbor', ''), neighbor_px=d.get('neighbor_px', ''), n_blob=d.get('n_blob', ''), blob_px=d.get('blob_px', '')))
    rp = os.path.join(ds.root, 'refine_report.csv')
    with open(rp, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    keep = sum(r['keep'] for r in rows)
    by_st = collections.Counter((r['status'], r['keep']) for r in rows)
    print(f'정제: 통과 {keep} / {len(rows)}, 제외 사유 {dict(reasons_cnt)}, 판정별(status, keep) {dict(by_st)} → {rp}')
    if a.apply:
        for r in rows:
            ds.state.data.setdefault(r['stem'], {})['train_exclude'] = not r['keep']
            ds.state.data[r['stem']]['refine_reasons'] = r['reasons']
        ds.state.save()
        print(f'state 반영: train_exclude {len(rows) - keep} 장')


if __name__ == '__main__':
    main()
