"""마스크 <-> 폴리곤 변환과 마스크 편집 연산 (numpy / OpenCV 만 사용, Qt 무관).

내부 표현은 인스턴스마다 bool (H, W) 마스크다. YOLO-seg 라벨(정규화 폴리곤 1줄)을
읽을 때 마스크로 채우고, 저장·내보내기 때 외곽선(폴리곤)으로 되돌린다.
"""
import cv2
import numpy as np


# ---------------------------------------------------------------- 폴리곤 <-> 마스크
def parse_yolo_line(line, w, h):
    """'<cls> x y x y ...'(정규화) -> (cls, (N,2) 픽셀 좌표). 유효하지 않으면 None."""
    v = line.split()
    if len(v) < 7:
        return None
    try:
        cls = int(float(v[0]))
        pts = np.array(v[1:], dtype=np.float64)
    except ValueError:
        return None
    if len(pts) % 2:
        pts = pts[:-1]
    pts = pts.reshape(-1, 2) * np.array([w, h], dtype=np.float64)
    return cls, pts


def polygon_to_mask(pts, h, w):
    """(N,2) 픽셀 폴리곤 -> bool 마스크 (경계 픽셀 포함)."""
    m = np.zeros((h, w), np.uint8)
    if pts is not None and len(pts) >= 3:
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def mask_contours(mask, min_area=0.0):
    """외곽선(구멍 무시) 목록, 면적 내림차순. (N,2) float 배열들."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        out.append((a, c.reshape(-1, 2).astype(np.float64)))
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out]


def simplify(poly, eps):
    """Douglas-Peucker 단순화 (eps 픽셀). 결과가 3점 미만이면 원본 유지."""
    if eps <= 0 or len(poly) < 4:
        return poly
    c = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), float(eps), True).reshape(-1, 2)
    return c.astype(np.float64) if len(c) >= 3 else poly


def _min_index(a, b):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(d, axis=None), d.shape)


def merge_multi_segment(segments):
    """떨어진 조각 여러 개를 한 폴리곤으로 잇는다 (ultralytics JSON2YOLO 와 같은 방식:
    가장 가까운 점끼리 얇은 선으로 연결). 입력은 (N,2) 배열 목록."""
    segments = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in segments]
    if len(segments) == 1:
        return segments[0]
    s = []
    idx_list = [[] for _ in range(len(segments))]
    for i in range(1, len(segments)):
        i1, i2 = _min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(i1)
        idx_list[i].append(i2)
    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in (0, len(idx_list) - 1):
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in (0, len(idx_list) - 1):
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return np.concatenate(s, axis=0)


def mask_to_polygon(mask, eps=0.7, min_area=16.0):
    """마스크 -> YOLO 용 단일 폴리곤 (N,2). 조각이 여럿이면 잇는다. 비어 있으면 None."""
    cs = mask_contours(mask, min_area)
    if not cs:
        return None
    cs = [simplify(c, eps) for c in cs]
    return cs[0] if len(cs) == 1 else merge_multi_segment(cs)


def mask_to_polygon_holes(mask, eps=0.7, min_area=16.0):
    """마스크 -> 구멍을 남기는 단일 폴리곤 (N,2). 가장 큰 바깥 윤곽에 그 안의 구멍 윤곽들을 폭 0 의 다리로 잇는다
    (바깥 → 구멍 한 바퀴 → 같은 점으로 복귀 → 바깥 계속). cv2.fillPoly / ultralytics polygon2mask 의 짝홀 채움이
    구멍을 그대로 남긴다. 떨어진 조각은 가장 큰 것만 남는다. 비어 있으면 None."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cs or hier is None:
        return None
    hier = hier[0]
    outers = [i for i, hh in enumerate(hier) if hh[3] < 0 and len(cs[i]) >= 3 and cv2.contourArea(cs[i]) >= min_area]
    if not outers:
        return None
    oi = max(outers, key=lambda i: cv2.contourArea(cs[i]))
    outer = simplify(cs[oi].reshape(-1, 2).astype(np.float64), eps)
    holes = []
    for i, hh in enumerate(hier):
        if hh[3] == oi and len(cs[i]) >= 3 and cv2.contourArea(cs[i]) >= min_area:
            hp = simplify(cs[i].reshape(-1, 2).astype(np.float64), eps)
            if len(hp) >= 3:
                holes.append(hp)
    if not holes:
        return outer
    attach = {}
    for hp in holes:
        i, j = _min_index(outer, hp)
        loop = np.concatenate([hp[j:], hp[:j], hp[j:j + 1]], 0)
        attach.setdefault(int(i), []).append(loop)
    parts = []
    for i, pt in enumerate(outer):
        parts.append(pt[None, :])
        for loop in attach.get(i, []):
            parts.append(loop)
            parts.append(pt[None, :])
    return np.concatenate(parts, 0)


def yolo_line(cls, poly, w, h):
    pts = np.clip(poly / np.array([w, h], dtype=np.float64), 0.0, 1.0)
    return f'{int(cls)} ' + ' '.join(f'{v:.6f}' for v in pts.reshape(-1))


def mask_to_yolo_line(cls, mask, w, h, eps=0.7, min_area=16.0, keep_holes=False):
    """keep_holes=False 면 외곽선만(구멍 채워짐), True 면 다리 폴리곤으로 구멍을 남긴다."""
    poly = mask_to_polygon_holes(mask, eps, min_area) if keep_holes else mask_to_polygon(mask, eps, min_area)
    return None if poly is None else yolo_line(cls, poly, w, h)


# ---------------------------------------------------------------- 편집 연산 (제자리 수정)
def _u8(mask):
    """bool 마스크를 같은 메모리를 보는 uint8 뷰로 (0/1 만 써야 한다)."""
    return mask.view(np.uint8)


def brush(mask, p0, p1, radius, add=True):
    """p0 -> p1 로 굵기 2r 선 + 양 끝 원. 좌표는 (x, y) 정수."""
    m = _u8(mask)
    val = 1 if add else 0
    r = max(1, int(round(radius)))
    p0 = (int(p0[0]), int(p0[1]))
    p1 = (int(p1[0]), int(p1[1]))
    cv2.circle(m, p0, r, val, -1, lineType=cv2.LINE_8)
    if p0 != p1:
        cv2.line(m, p0, p1, val, thickness=2 * r, lineType=cv2.LINE_8)
        cv2.circle(m, p1, r, val, -1, lineType=cv2.LINE_8)


def fill_polygon(mask, pts, add=True):
    if pts is None or len(pts) < 3:
        return
    cv2.fillPoly(_u8(mask), [np.round(np.asarray(pts)).astype(np.int32)], 1 if add else 0)


def wand_region(img_bgr, seed, tol, blur=3):
    """seed 픽셀과 색이 비슷한(각 채널 ±tol, 기준 고정) 연결 영역 -> bool 마스크."""
    h, w = img_bgr.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), bool)
    src = cv2.medianBlur(img_bgr, blur) if blur and blur >= 3 else img_bgr
    ff = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(src.copy(), ff, (x, y), 0, (tol,) * 3, (tol,) * 3, flags)
    return ff[1:-1, 1:-1] > 0


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grabcut_refine(img_bgr, mask, margin=25, band=20, iters=3):
    """현재 마스크를 씨앗으로 GrabCut. 마스크 안쪽 = 확실한 전경, 마스크 = 전경 추정,
    band 픽셀 바깥 띠 = 배경 추정, 그 밖 = 확실한 배경. 새 bool 마스크 반환."""
    bb = bbox(mask)
    if bb is None:
        return mask.copy()
    h, w = mask.shape
    x0, y0, x1, y1 = bb
    x0, y0 = max(0, x0 - margin - band), max(0, y0 - margin - band)
    x1, y1 = min(w, x1 + margin + band), min(h, y1 + margin + band)
    sub = img_bgr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    gc[cv2.dilate(m, k) > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gc[cv2.erode(m, k2) > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(sub, gc, None, bgd, fgd, int(iters), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask.copy()
    out = mask.copy()
    out[y0:y1, x0:x1] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return out


def fill_holes(mask):
    """마스크 내부 구멍 채우기 (외곽선으로 다시 채움)."""
    out = np.zeros_like(mask, dtype=np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs:
        cv2.fillPoly(out, cs, 1)
    return out.astype(bool)


def keep_largest(mask):
    """가장 큰 연결 조각만 남긴다."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.copy()
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def split_components(mask, min_area=16.0):
    """연결 조각별 bool 마스크 목록 (면적 내림차순). min_area 미만 조각은 버린다."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    parts = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    parts.sort(reverse=True)
    return [lab == i for _, i in parts]


def cut_mask(mask, polyline, thickness=3, min_area=16.0):
    """polyline((N,2) 픽셀 좌표)로 마스크를 가른다. 선을 지운 뒤 연결 조각으로 나누고, 지운 선 픽셀은
    가장 가까운 조각에 되돌려 준다. 조각이 2개 이상이면 목록(면적 내림차순), 갈라지지 않으면 [mask]."""
    pts = np.round(np.asarray(polyline, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) < 2:
        return [mask.copy()]
    line = np.zeros(mask.shape, np.uint8)
    cv2.polylines(line, [pts], False, 1, thickness=max(1, int(thickness)), lineType=cv2.LINE_8)
    line = (line > 0) & mask
    rest = mask & ~line
    parts = split_components(rest, min_area)
    if len(parts) < 2:
        return [mask.copy()]
    lab = np.zeros(mask.shape, np.int32)
    for i, p in enumerate(parts, 1):
        lab[p] = i
    # 지운 선 픽셀을 가까운 조각으로: 라벨을 마스크 안에서 조금씩 번지게 한다
    k = np.ones((3, 3), np.uint8)
    todo = line.copy()
    for _ in range(max(2, thickness + 1)):
        if not todo.any():
            break
        grown = cv2.dilate(lab.astype(np.float32), k).astype(np.int32)   # 이웃 중 가장 큰 라벨 (근사)
        fill = todo & (grown > 0)
        lab[fill] = grown[fill]
        todo &= ~fill
    return [lab == i for i in range(1, len(parts) + 1)]


def smooth(mask, k=3):
    """열림+닫힘으로 가장자리 잔털 정리."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return m.astype(bool)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0
