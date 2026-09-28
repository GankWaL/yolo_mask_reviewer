"""인스턴스 클래스를 이미지의 제품코드에 맞추는 연산 (Qt 무관).

제품코드 클래스 데이터셋(클래스 이름 = 12자리 제품코드)에서 검수자가 이미지의 제품코드를 확정한 뒤 쓴다.
  to_code    모든 인스턴스의 클래스를 제품코드 클래스로 바꾼다
  keep_code  제품코드 클래스 마스크가 있으면 그것만, 없으면 화면 중앙의 마스크를 제품코드 클래스로 바꿔 하나만 남긴다
일괄 적용(apply_bulk)은 라벨 줄의 클래스 번호만 고쳐 labels_reviewed/ 에 쓰므로 폴리곤(구멍 포함)은 그대로다.
"""
import os

import numpy as np

from . import maskops

MODES = ('to_code', 'keep_code')
MODE_LABEL = {'to_code': '클래스 → 제품코드', 'keep_code': '제품코드 마스크만 남기기'}
GRID = 256   # 일괄 적용에서 중앙 마스크를 고를 때 폴리곤을 채우는 격자 (이미지를 읽지 않는다)


def class_id(ds, code):
    """제품코드에 해당하는 클래스 번호. 클래스 목록에 없으면 None."""
    for k in sorted(ds.names):
        if ds.names[k] == code:
            return k
    return None


def center_distance(mask):
    """이미지 중심에서 마스크까지의 거리 (px, 중심을 덮으면 0, 빈 마스크는 inf)."""
    h, w = mask.shape[:2]
    if mask[h // 2, w // 2]:
        return 0.0
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return float('inf')
    return float(np.sqrt(((ys - h // 2) ** 2 + (xs - w // 2) ** 2).min()))


def pick_center(masks, candidates=None):
    """candidates(없으면 전부) 중 이미지 중심에 가장 가까운 마스크의 번호. 거리가 같으면 넓은 것. 빈 마스크뿐이면 None."""
    best = None
    for i in (range(len(masks)) if candidates is None else candidates):
        d = center_distance(masks[i])
        if d == float('inf'):
            continue
        key = (d, -int(masks[i].sum()))
        if best is None or key < best[0]:
            best = (key, i)
    return None if best is None else best[1]


def plan_keep(classes, masks, target):
    """keep_code 에서 남길 인스턴스 번호 목록. target 클래스가 있으면 그것들, 없으면 중앙 마스크 하나 (클래스는 호출자가 target 으로)."""
    match = [i for i, c in enumerate(classes) if c == target and masks[i].any()]
    if match:
        return match
    i = pick_center(masks)
    return [] if i is None else [i]


def convert_text(text, target, mode):
    """라벨 텍스트에 mode 를 적용한 텍스트. 인스턴스가 아닌 줄(값 7개 미만)은 버린다."""
    rows = []
    for line in text.splitlines():
        r = maskops.parse_yolo_line(line, GRID, GRID)
        if r is not None:
            rows.append((r[0], line.split(), r[1]))
    if mode == 'keep_code':
        masks = [maskops.polygon_to_mask(pts, GRID, GRID) for _, _, pts in rows]
        rows = [rows[i] for i in plan_keep([c for c, _, _ in rows], masks, target)]
    lines = [' '.join([str(target)] + v[1:]) for _, v, _ in rows]
    return '\n'.join(lines) + ('\n' if lines else '')


def apply_bulk(ds, stems, mode, progress=None):
    """stems 의 유효 라벨에 mode 를 적용해 바뀐 것만 labels_reviewed/ 에 쓴다.
    반환: {'changed': [stem], 'same': n, 'no_class': {code: n}, 'empty': n} (no_class = 제품코드가 클래스 목록에 없어 건너뜀)."""
    r = {'changed': [], 'same': 0, 'no_class': {}, 'empty': 0}
    ids = {}
    for i, s in enumerate(stems):
        if progress:
            progress(i, len(stems), s)
        code = ds.code(s)
        if code not in ids:
            ids[code] = class_id(ds, code)
        if ids[code] is None:
            r['no_class'][code] = r['no_class'].get(code, 0) + 1
            continue
        text = ds.label_text(s)
        if not any(len(line.split()) >= 7 for line in text.splitlines()):
            r['empty'] += 1
            continue
        new = convert_text(text, ids[code], mode)
        if [line.split() for line in new.splitlines()] == [line.split() for line in text.splitlines() if line.split()]:
            r['same'] += 1
            continue
        os.makedirs(ds.reviewed_dir, exist_ok=True)
        with open(ds.reviewed_label_path(s), 'w', encoding='utf-8') as f:
            f.write(new)
        r['changed'].append(s)
    return r
