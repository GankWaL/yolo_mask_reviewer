"""대표(★)의 관통 구멍을 같은 제품의 보류 프레임에 옮긴다 — 바깥 윤곽은 프레임의 현재 라벨을 그대로 두고 구멍만 바꾼다.

SAM2 영상 전파(propagate.py)는 1024 해상도 저해상 로짓(256) 을 거쳐서 작은 구멍(수십 px)이 뭉개진다. 여기서는
  1. 대표 인스턴스(구멍 채움)를 대상 인스턴스(구멍 채움)에 유사변환(회전·배율·이동)으로 정합한다
     — 마스크 IoU 최대. 전수 각도 탐색(5°) → 각도·이동·배율 미세 조정. 같은 카메라 크롭이라 배율은 거의 1 이다.
  2. 정합된 대표의 구멍을 대상 안으로 옮긴다 (대상 채운 마스크 − 옮긴 구멍). 대상의 기존 구멍은 버린다 (대표가 기준).
  3. (선택) 구멍마다 대상 이미지의 작은 창을 잘라 SAM2 이미지 예측기(박스 + 중심점)로 경계를 다듬는다.
     창을 자르면 SAM2 의 1024 입력에서 작은 구멍도 크게 보인다. 옮긴 구멍과 IoU 가 낮거나 면적이 크게 다르면 옮긴 것을 그대로 둔다.
대상: 보류 상태이고 편집본·대표가 아닌 프레임 (dataset.auto_targets). 결과는 labels_auto/ (keep_holes 면 다리 폴리곤) 에 쓰고
state 에 auto(source 'holes-align'), cross_iou(정합 IoU 최소), holes(인스턴스별 구멍 수), note 를 남긴다. 판정은 보류 그대로.
"""
import math
import time

import cv2
import numpy as np

from . import maskops
from .dataset import Instance

MIN_HOLE_PX = 4          # 이보다 작은 구멍 성분은 버린다
REFINE_MIN_IOU = 0.3     # SAM2 로 다듬은 구멍이 옮긴 구멍과 이보다 덜 겹치면 버린다
REFINE_AREA_RATIO = (0.4, 2.5)
ALIGN_WARN_IOU = 0.85    # 정합 IoU 가 이보다 낮으면 메모에 align-low


def _moments(m):
    mm = cv2.moments(m.astype(np.uint8), binaryImage=True)
    if mm['m00'] <= 0:
        return 0.0, 0.0, 0.0
    return mm['m00'], mm['m10'] / mm['m00'], mm['m01'] / mm['m00']


def _affine(src_c, dst_c, deg, scale, dx=0.0, dy=0.0):
    """src 중심을 축으로 deg 회전·scale 배율 후 dst 중심(+dx, dy)으로 옮기는 2x3 행렬."""
    M = cv2.getRotationMatrix2D((float(src_c[0]), float(src_c[1])), float(deg), float(scale))
    M[0, 2] += dst_c[0] - src_c[0] + dx
    M[1, 2] += dst_c[1] - src_c[1] + dy
    return M


def warp(mask, M, shape):
    """bool 마스크를 M 으로 옮긴다 (선형 보간 후 절반 문턱 — 작은 구멍이 최근접 보간으로 사라지지 않게)."""
    u = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    return u >= 128


def align(src, dst, coarse_step=5):
    """src(bool, 채움) 를 dst(bool, 채움) 에 유사변환으로 정합. (M 2x3, IoU, deg, scale). 비어 있으면 (None, 0, 0, 1)."""
    a_s, cxs, cys = _moments(src)
    a_d, cxd, cyd = _moments(dst)
    if a_s <= 0 or a_d <= 0:
        return None, 0.0, 0.0, 1.0
    s0 = math.sqrt(a_d / a_s)
    sc, dc = (cxs, cys), (cxd, cyd)
    # dst 의 bbox 주변만 비교해 빠르게
    x, y, w, h = cv2.boundingRect(dst.astype(np.uint8))
    pad = int(max(w, h) * 0.3) + 4
    y0, y1 = max(0, y - pad), min(dst.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(dst.shape[1], x + w + pad)
    dst_roi = dst[y0:y1, x0:x1]
    shift = np.array([[0, 0, -x0], [0, 0, -y0]], np.float64)

    def score(deg, scale, dx=0.0, dy=0.0):
        M = _affine(sc, dc, deg, scale, dx, dy) + shift
        wmask = warp(src, M, dst_roi.shape)
        return float((wmask & dst_roi).sum() / max((wmask | dst_roi).sum(), 1))

    # 거친 각도 탐색은 반해상도로 (속도), 미세 조정은 원해상도로
    small_src = cv2.resize(src.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) > 0
    small_dst = cv2.resize(dst_roi.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) > 0
    half = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]])   # 반해상도: 이동 성분만 절반

    def score_small(deg):
        M = (_affine(sc, dc, deg, s0) + shift) * half
        wmask = warp(small_src, M, small_dst.shape)
        return float((wmask & small_dst).sum() / max((wmask | small_dst).sum(), 1))

    cands = sorted(((score_small(d), d) for d in range(0, 360, coarse_step)), reverse=True)[:2]
    best = (-1.0, 0.0, s0, 0.0, 0.0)
    for _, d in cands:
        cur = (score(d, s0), float(d), s0, 0.0, 0.0)
        for step in (1.0, 0.25):
            for _ in range(2):
                _, deg, scale, dx, dy = cur
                for k in range(-4, 5):
                    v = score(deg + k * step, scale, dx, dy)
                    if v > cur[0]:
                        cur = (v, deg + k * step, scale, dx, dy)
                _, deg, scale, dx, dy = cur
                for ox in range(-2, 3):
                    for oy in range(-2, 3):
                        v = score(deg, scale, dx + ox, dy + oy)
                        if v > cur[0]:
                            cur = (v, deg, scale, dx + ox, dy + oy)
                _, deg, scale, dx, dy = cur
                for f in (0.97, 0.985, 1.015, 1.03):
                    v = score(deg, scale * f, dx, dy)
                    if v > cur[0]:
                        cur = (v, deg, scale * f, dx, dy)
        if cur[0] > best[0]:
            best = cur
    v, deg, scale, dx, dy = best
    return _affine(sc, dc, deg, scale, dx, dy), v, deg % 360.0, scale


def holes_of(mask):
    """마스크의 관통 구멍 (채운 마스크 − 마스크)."""
    return maskops.fill_holes(mask) & ~mask


def components(mask, min_px=MIN_HOLE_PX):
    return maskops.split_components(mask, min_area=min_px)


def transfer_holes(ref_mask, tgt_mask, min_px=MIN_HOLE_PX):
    """대표 인스턴스 마스크의 구멍을 대상 인스턴스에 옮긴다.
    반환 (새 대상 마스크, 옮긴 구멍 bool, 정합 IoU, deg). 정합 실패면 (채운 대상, 빈 구멍, 0, 0)."""
    ref_f = maskops.fill_holes(ref_mask)
    tgt_f = maskops.fill_holes(tgt_mask)
    M, al, deg, _ = align(ref_f, tgt_f)
    holes = np.zeros(tgt_mask.shape, bool)
    if M is None:
        return tgt_f, holes, 0.0, 0.0
    moved = warp(holes_of(ref_mask), M, tgt_mask.shape)
    for c in components(moved, min_px):
        inside = c & tgt_f
        if inside.sum() >= min_px and inside.sum() / c.sum() >= 0.5:
            holes |= inside
    return tgt_f & ~holes, holes, al, deg


class HoleRefiner:
    """구멍 하나씩 대상 이미지의 작은 창을 잘라 SAM2 이미지 예측기로 경계를 다듬는다."""

    def __init__(self, ckpt=None, device=None, segmenter=None, pad_frac=1.0, min_pad=12):
        from . import sam2_helper
        self.seg = segmenter or sam2_helper.Sam2Segmenter(ckpt or sam2_helper.find_checkpoint(), device)
        self.pad_frac, self.min_pad = pad_frac, min_pad

    def refine(self, img, hole, inside):
        """hole(bool, 옮긴 구멍 한 성분) → 다듬은 구멍 bool 또는 None(게이트 실패)."""
        H, W = hole.shape
        x, y, w, h = cv2.boundingRect(hole.astype(np.uint8))
        pad = max(self.min_pad, int(max(w, h) * self.pad_frac))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        if crop.size == 0:
            return None
        self.seg.key = None          # set_image 는 id(img) 로 같은 이미지인지 보는데 크롭은 id 가 재사용된다
        self.seg.set_image(crop)
        _, cx, cy = _moments(hole)
        pts, labs = [], []
        if hole[int(round(cy)), int(round(cx))]:
            pts, labs = [(cx - x0, cy - y0)], [1]
        box = (max(0, x - x0 - 1), max(0, y - y0 - 1), min(x1 - x0, x + w - x0 + 1), min(y1 - y0, y + h - y0 + 1))
        m, _ = self.seg.predict(pts, labs, box)
        if m is None:
            return None
        full = np.zeros((H, W), bool)
        full[y0:y1, x0:x1] = m
        full &= inside
        # 옮긴 구멍과 가장 많이 겹치는 성분만
        best = None
        for c in components(full, 1):
            ov = int((c & hole).sum())
            if best is None or ov > best[0]:
                best = (ov, c)
        if best is None or best[0] == 0:
            return None
        c = best[1]
        if maskops.iou(c, hole) < REFINE_MIN_IOU:
            return None
        r = c.sum() / max(hole.sum(), 1)
        if not (REFINE_AREA_RATIO[0] <= r <= REFINE_AREA_RATIO[1]):
            return None
        return c


def apply_holes(img, ref_insts, tgt_insts, refiner=None, min_px=MIN_HOLE_PX):
    """프레임 하나: 대표 인스턴스 목록(여러 대표를 합쳐도 됨) → 대상 인스턴스 목록.
    대상 인스턴스마다 같은 클래스의 대표 인스턴스 중 정합 IoU 가 가장 큰 것을 쓴다. 같은 클래스 대표가 없으면 그대로 둔다.
    반환 (새 인스턴스 목록, info dict: iou_min, ious, holes, n_ref, n_refined, n_kept)."""
    out = []
    ious, holes_n = [], []
    n_refined = n_kept = 0
    for t in tgt_insts:
        refs = [r for r in ref_insts if r.cls == t.cls]
        if not refs:
            out.append(t.copy())
            continue
        best = None
        for r in refs:
            new, holes, al, deg = transfer_holes(r.mask, t.mask, min_px)
            if best is None or al > best[1]:
                best = (new, al, holes, r)
        new, al, holes, r = best
        if refiner is not None and holes.any():
            tgt_f = maskops.fill_holes(t.mask)
            refined = np.zeros_like(holes)
            for c in components(holes, 1):
                rc = refiner.refine(img, c, tgt_f)
                if rc is None:
                    refined |= c
                    n_kept += 1
                else:
                    refined |= rc
                    n_refined += 1
            new = tgt_f & ~refined
        ious.append(round(al, 3))
        holes_n.append(len(components(holes_of(new), 1)))
        out.append(Instance(t.cls, new))
    return out, dict(iou_min=min(ious) if ious else 0.0, ious=ious, holes=holes_n, n_ref=len(ious),
                     n_refined=n_refined, n_kept=n_kept)


def propagate_holes_dataset(ds, codes=None, max_exemplars=3, include_auto=True, refiner=None, refine=True,
                            ckpt=None, max_targets=0, progress=None, should_stop=None, include_empty_edited=False,
                            eps=0.7, min_area=16.0):
    """대표가 있는 제품의 보류 프레임에 대표의 구멍을 옮긴 자동 라벨을 쓴다 (propagate.propagate_dataset 와 같은 꼴).
    refine=True 면 SAM2 로 구멍 경계를 다듬는다 (refiner 를 주면 그것을, 아니면 새로 만든다). 요약 dict 반환."""
    if refine and refiner is None:
        refiner = HoleRefiner(ckpt)
    if codes is None:
        codes = sorted({ds.code(s) for s in ds.exemplars()})
    jobs = []
    for code in codes:
        ex = ds.exemplars(code)
        if not ex:
            continue
        for s in ds.auto_targets(code, include_auto=include_auto, include_empty_edited=include_empty_edited):
            jobs.append((code, s, ex[:max_exemplars] if max_exemplars > 0 else ex))
    if max_targets > 0:
        jobs = jobs[:max_targets]
    n_done = n_empty = n_cleared = n_low = 0
    per_code = {}
    ious = []
    n_refined = n_kept = 0
    t0 = time.time()
    ref_cache = {}
    for i, (code, stem, exs) in enumerate(jobs):
        if should_stop and should_stop():
            break
        refs = []
        for ex in exs:
            if ex not in ref_cache:
                ref_cache[ex] = ds.load(ex)[1]
            refs.extend(ref_cache[ex])
        img, tgt = ds.load(stem)
        h, w = img.shape[:2]
        if include_empty_edited and ds.reviewed_is_empty(stem):
            ds.revert(stem)
            n_cleared += 1
        if not tgt or not refs:
            ds.write_auto(stem, tgt, w, h, source='holes-align', score=0.0, ref=exs[0], eps=eps, min_area=min_area)
            ds.state.set(stem, note='auto-none holes-align (대상 또는 대표 인스턴스 없음)')
            n_empty += 1
            info = '없음'
        else:
            new, r = apply_holes(img, refs, tgt, refiner)
            ds.write_auto(stem, new, w, h, source='holes-align', score=r['iou_min'], ref=exs[0], eps=eps, min_area=min_area)
            flags = []
            if r['n_ref'] and r['iou_min'] < ALIGN_WARN_IOU:
                flags.append('align-low')
                n_low += 1
            if r['n_ref'] == 0:
                flags.append('no-ref-class')
            note = (f'auto holes-align iou={r["iou_min"]:.2f} holes={r["holes"]} refined={r["n_refined"]}/{r["n_refined"] + r["n_kept"]}'
                    + (' ' + ','.join(flags) if flags else ''))
            ds.state.set(stem, cross_iou=r['iou_min'], holes=r['holes'], note=note)
            ious.append(r['iou_min'])
            n_refined += r['n_refined']
            n_kept += r['n_kept']
            info = f'정합 {r["iou_min"]:.2f} 구멍 {r["holes"]}'
            n_done += 1
        per_code[code] = per_code.get(code, 0) + 1
        if progress:
            progress(i, len(jobs), stem, info)
    return dict(n_jobs=len(jobs), n_done=n_done, n_empty=n_empty, n_cleared=n_cleared, n_low=n_low, per_code=per_code,
                iou_mean=float(np.mean(ious)) if ious else 0.0, iou_min=float(np.min(ious)) if ious else 0.0,
                n_refined=n_refined, n_kept=n_kept, sec=time.time() - t0)
