"""대표 구멍 전파(holes.py): 합성 도형으로 정합·구멍 옮기기 검증. 실행: python3 -m pytest tests/test_holes.py -q"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mask_reviewer import holes, maskops  # noqa: E402
from mask_reviewer.dataset import Instance  # noqa: E402


def _shape(deg, cx, cy, scale=1.0, with_holes=True):
    """비대칭 L 자 판 + 구멍 3개(큰 것 1, 작은 것 2) 를 deg 회전·scale 배율로 그린 bool 마스크."""
    m = np.zeros((400, 400), np.uint8)
    body = np.array([[-90, -60], [90, -60], [90, 0], [20, 0], [20, 60], [-90, 60]], np.float64)
    hs = [(np.array([[-60, -40], [-20, -40], [-20, -10], [-60, -10]], np.float64), 1),
          (np.array([[40, -45], [48, -45], [48, -37], [40, -37]], np.float64), 0),     # 8x8 작은 구멍
          (np.array([[-40, 20], [-34, 20], [-34, 26], [-40, 26]], np.float64), 0)]      # 6x6 작은 구멍
    R = cv2.getRotationMatrix2D((0, 0), deg, scale)

    def tf(p):
        q = (R[:, :2] @ p.T).T + np.array([cx, cy])
        return np.round(q).astype(np.int32)
    cv2.fillPoly(m, [tf(body)], 1)
    if with_holes:
        for h, _ in hs:
            cv2.fillPoly(m, [tf(h)], 0)
    return m.astype(bool)


def test_align_recovers_rotation_and_shift():
    src = maskops.fill_holes(_shape(0, 200, 200))
    for deg, cx, cy, sc in ((37, 230, 180, 1.0), (200, 170, 220, 1.03), (-80, 210, 210, 0.97)):
        dst = maskops.fill_holes(_shape(deg, cx, cy, sc))
        M, iou, d, s = holes.align(src, dst)
        assert iou > 0.97, (deg, iou)
        assert abs(((d - deg) + 180) % 360 - 180) < 1.5, (deg, d)
        assert abs(s - sc) < 0.02, (sc, s)


def test_transfer_holes_moves_small_holes():
    ref = _shape(0, 200, 200)
    tgt_filled = _shape(37, 230, 180, with_holes=False)        # 구멍 없는 대상 (바깥 윤곽만)
    truth = _shape(37, 230, 180)
    new, moved, iou, _ = holes.transfer_holes(ref, tgt_filled)
    assert iou > 0.97
    assert len(holes.components(holes.holes_of(new), 1)) == 3           # 작은 구멍 2개도 옮겨짐
    assert maskops.iou(holes.holes_of(new), holes.holes_of(truth)) > 0.7
    assert (maskops.fill_holes(new) == maskops.fill_holes(tgt_filled)).all()   # 바깥 윤곽은 그대로


def test_apply_holes_replaces_existing_holes_and_keeps_other_classes():
    ref = [Instance(0, _shape(0, 200, 200))]
    wrong = _shape(37, 230, 180, with_holes=False)
    wrong[170:176, 250:256] = False                                   # 엉뚱한 구멍
    other = np.zeros((400, 400), bool)
    other[10:30, 10:30] = True
    img = np.zeros((400, 400, 3), np.uint8)
    out, info = holes.apply_holes(img, ref, [Instance(0, wrong), Instance(1, other)])
    assert info['n_ref'] == 1 and info['holes'] == [3] and info['iou_min'] > 0.97
    assert (out[1].mask == other).all()
    assert not holes.holes_of(out[0].mask)[170:176, 250:256].any()
