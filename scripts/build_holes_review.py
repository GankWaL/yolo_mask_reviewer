#!/usr/bin/env python3
"""검수본(사람이 만든 바깥 윤곽 라벨)에 관통 구멍을 자동으로 뚫어 검수용 데이터셋을 만든다.

  conda activate yolo_mask_reviewer
  python scripts/build_holes_review.py SRC [SRC ...] --out OUT --model best.pt [--device 0] [--workers 8]

  SRC     images/{train,val,''} + labels/ 가 있는 YOLO-seg 폴더 (reviewed_* 처럼 대분류 5클래스, 파일명에 12자리 코드).
  OUT     mask_reviewer 로 여는 폴더: images/ labels/ aux/ manifest.csv dataset.yaml (keep_holes: true).

구멍은 두 근거를 교차한다 (§29 "2번 + 3번"):
  ② 모델: 구멍을 남기도록 학습한 YOLO26-seg(--model) 로 크롭을 추론해, 사람 마스크 안에서 모델이 비운 내부 덩어리
     (사람 마스크 경계에 닿지 않는 것) = 모델 구멍.
  ③ CAD: 운영 yaw 매처(core_utils/yaw_match_pool.py, 파라미터는 운영 코드 main() 에서 AST 로 추출 — 런타임 동일)로
     사람 마스크를 실루엣 캐시(cached_fhd → cached)에 정합하고, 정합된 실루엣(구멍 있는 원본)의 구멍 = CAD 구멍.
     매처는 런타임처럼 채운 실루엣으로 점수를 매기고, 구멍은 같은 yaw 의 안 채운 실루엣에서 가져온다.
  확정 구멍 = 모델 구멍 중 (팽창한) CAD 구멍과 절반 이상 겹치는 것. labels/ 에는 사람 마스크 − 확정 구멍을 다리 폴리곤으로 쓴다.
  aux/<stem>.png (uint8): 1 확정 / 2 모델만 / 3 CAD만 — 검수 툴이 색으로 겹쳐 보여 주고 버튼으로 뺄 수 있다.
  CAD 정합이 없거나(cad_status nocache/nomatch) 점수가 운영 fitness 문턱 미만(low)이면 확정 구멍 없이 모델 구멍만 aux 2 로 남긴다.

manifest.csv: stem, source, code, cls, n_inst, match_yaw, match_score, match_iou, cad_status,
              holes_model_px, holes_cad_px, holes_confirmed_px, model_only_px, cad_only_px, disagree_px, n_confirmed, n_model_only, n_cad_only
"""
import argparse
import ast
import csv
import glob
import os
import re
import sys
import types
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mask_reviewer import maskops  # noqa: E402

SLIA = os.path.expanduser('~/jhw/SL_Inspection_Automation')
CORE = os.path.join(SLIA, 'core_utils')
OBJ_DIR = os.path.join(SLIA, 'obj_product_name_260805')
CODE_RE = re.compile(r'_([0-9A-Z]{12})(?=[_.]|$)')
# 검수 툴 클래스 이름 → 운영 대분류 키 (가중치 표·fitness 문턱)
SORT_OF = {'b_cvr': 'B/CVR', 'cap': 'CAP', 'h_cvr': 'H/CVR', 'hsg': 'HSG', 'scalp': 'SCALP'}


def _stub_gl():
    """mask_matching_gl_utils 가 import 하는 GL/메시 모듈을 흉내 낸다 — 여기서는 numpy/cv2 함수만 쓴다."""
    class _Any(type):
        def __getattr__(cls, n):
            return _Any(n, (), {})

    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            return _Any(n, (), {})

    for m in ('moderngl', 'OpenGL', 'OpenGL.GL', 'open3d', 'trimesh'):
        if m not in sys.modules:
            try:
                __import__(m)
            except Exception:
                sys.modules[m] = _Stub(m)


def runtime_params(path=os.path.join(CORE, 'object_pose_estimator_adjusted.py')):
    """운영 코드 __init__ 기본값 + main() 오버라이드를 AST 로 추출 (replay_yaw_matching.load_runtime_params 와 같음)."""
    tree = ast.parse(open(path, encoding='utf-8').read())
    vals = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            a = node.args
            for arg, d in zip(a.args[len(a.args) - len(a.defaults):], a.defaults):
                try:
                    vals[arg.arg] = ast.literal_eval(d)
                except Exception:
                    pass
            break
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            for c in ast.walk(node):
                if isinstance(c, ast.Call) and getattr(c.func, 'id', '') == 'ObjectPoseEstimator':
                    for kw in c.keywords:
                        try:
                            vals[kw.arg] = ast.literal_eval(kw.value)
                        except Exception:
                            pass
    return vals


# ------------------------------------------------------------------ 워커 (CAD 정합)
_YMP = None
_STORES = None
_P = None
_ROI_CAP = None


def _pool_init(cache_dirs, params, roi_cap):
    global _YMP, _STORES, _P, _ROI_CAP
    _stub_gl()
    sys.path.insert(0, CORE)
    import yaw_match_pool as ymp
    _YMP = ymp
    _P = params
    _ROI_CAP = roi_cap
    _STORES = [ymp._Store(d, max_loaded_caches=16) for d in cache_dirs if os.path.isdir(d)]
    cv2.setNumThreads(1)


def _find_cache(code):
    for st in _STORES:
        for key in (code, code[:9]):
            c = st.get_cache(key)
            if c is not None:
                return c
    return None


def _match_cad(mask_u8, code, sort):
    """사람 마스크(uint8 0/255, 크롭 좌표) → (status, yaw, score, iou, cad_sil_aligned uint8 0/1 크롭 좌표 또는 None)."""
    ymp = _YMP
    cache = _find_cache(code)
    if cache is None:
        return 'nocache', None, None, None, None
    roi, (x0, y0, x1, y1) = ymp.extract_mask_roi(mask_u8)
    H, W = roi.shape[:2]
    roi_in = roi
    if _ROI_CAP and max(H, W) > _ROI_CAP:           # 런타임 score_roi_max_dim 과 동일 처리
        f = _ROI_CAP / float(max(H, W))
        roi_in = cv2.resize(roi, (max(1, int(round(W * f))), max(1, int(round(H * f)))), interpolation=cv2.INTER_NEAREST)
    ymp._STORE = _StoreProxy(cache)
    ymp._P = _P
    best = ymp._match(roi_in, code, sort, int(_P['yaw_min']), int(_P['yaw_max']))
    if best is None or best.get('yaw') is None:
        return 'nomatch', None, None, None, None
    tight = cache.decode_tight(best['yaw'])
    if tight is None:
        return 'nomatch', best['yaw'], float(best['score']), float(best['iou']), None
    sil_tight, _ = tight
    sil = ymp.resize_to_match(sil_tight, (H, W))     # 안 채운 실루엣을 사람 ROI 크기로 (런타임과 같은 tight-bbox 정렬)
    out = np.zeros(mask_u8.shape[:2], np.uint8)
    out[y0:y1 + 1, x0:x1 + 1] = (sil > 0).astype(np.uint8)
    return 'ok', int(best['yaw']), float(best['score']), float(best['iou']), out


class _StoreProxy:
    """_match 가 _STORE.get_cache(model_name)/get_area_bbox 를 부르므로 코드 → 캐시를 고정해서 넘긴다."""

    def __init__(self, cache):
        self.cache = cache

    def get_cache(self, key):
        return self.cache

    def get_area_bbox(self, key, yaw):
        return self.cache.get_area(yaw), self.cache.get_bbox(yaw)


def _interior_components(region, outer_mask, min_px):
    """region(bool) 의 연결 성분 중 outer_mask 경계에 닿지 않고 min_px 이상인 것만 남긴 bool 마스크와 성분 수."""
    boundary = (outer_mask.astype(np.uint8) - cv2.erode(outer_mask.astype(np.uint8), np.ones((3, 3), np.uint8))) > 0
    n, lab = cv2.connectedComponents(region.astype(np.uint8))
    out = np.zeros_like(region, dtype=bool)
    k = 0
    for i in range(1, n):
        c = lab == i
        if c.sum() < min_px or (c & boundary).any():
            continue
        out |= c
        k += 1
    return out, k


def _work(job):
    """한 프레임: (stem, src_img, label_lines, code, cls_names, pred_masks(list of bool), out paths, opts) → manifest row."""
    (stem, img_path, lines, code, names, preds, out_lab, out_aux, min_px, fit_thr_tbl, dilate_frac) = job
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    insts = []
    for line in lines:
        r = maskops.parse_yolo_line(line, w, h)
        if r is None:
            continue
        cls, pts = r
        insts.append((cls, maskops.polygon_to_mask(pts, h, w)))
    aux = np.zeros((h, w), np.uint8)
    row = dict(stem=stem, code=code, n_inst=len(insts), cls='|'.join(names.get(c, str(c)) for c, _ in insts),
               match_yaw='', match_score='', match_iou='', cad_status='',
               holes_model_px=0, holes_cad_px=0, holes_confirmed_px=0, model_only_px=0, cad_only_px=0,
               disagree_px=0, n_confirmed=0, n_model_only=0, n_cad_only=0)
    out_lines = []
    statuses = []
    for cls, human in insts:
        human = maskops.fill_holes(human)            # 사람 라벨은 바깥 윤곽 (RETR_EXTERNAL) — 채운 상태에서 시작
        if human.sum() < min_px:
            continue
        # ② 모델 구멍
        pred = None
        best_iou = 0.
        for pm in preds:
            inter = (pm & human).sum()
            iou = inter / max((pm | human).sum(), 1)
            if iou > best_iou:
                best_iou, pred = iou, pm
        model_holes = np.zeros_like(human)
        if pred is not None and best_iou >= 0.5:
            model_holes, _ = _interior_components(human & ~pred, human, min_px)
        # ③ CAD 구멍
        sort = SORT_OF.get(names.get(cls, ''), None)
        status, yaw, score, iou, sil = _match_cad((human.astype(np.uint8) * 255), code, sort)
        cad_holes = np.zeros_like(human)
        if status == 'ok':
            thr = fit_thr_tbl.get(sort, fit_thr_tbl.get('default', 0.5)) if isinstance(fit_thr_tbl, dict) else fit_thr_tbl
            if score < thr:
                status = 'low'
            sil_b = sil > 0
            cad_raw = maskops.fill_holes(sil_b) & ~sil_b & human
            cad_holes, _ = _interior_components(cad_raw, human, min_px)
        statuses.append(status)
        if yaw is not None:
            row.update(match_yaw=yaw, match_score=round(score, 4), match_iou=round(iou, 4))
        # 교차: 모델 구멍 성분별로 팽창한 CAD 구멍과 절반 이상 겹치면 확정
        confirmed = np.zeros_like(human)
        model_only = np.zeros_like(human)
        n_conf = n_monly = 0
        if status == 'ok' and cad_holes.any():
            bw, bh = cv2.boundingRect(human.astype(np.uint8))[2:]
            k = max(3, int(round(dilate_frac * max(bw, bh))))
            cad_dil = cv2.dilate(cad_holes.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            n, lab = cv2.connectedComponents(model_holes.astype(np.uint8))
            for i in range(1, n):
                c = lab == i
                if (c & cad_dil).sum() / c.sum() >= 0.5:
                    confirmed |= c
                    n_conf += 1
                else:
                    model_only |= c
                    n_monly += 1
            mdl_dil = cv2.dilate(model_holes.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            cad_only = cad_holes & ~mdl_dil
            cad_only, n_conly = _interior_components(cad_only, human, min_px)
        else:
            model_only = model_holes
            n_monly = int(cv2.connectedComponents(model_holes.astype(np.uint8))[0] - 1)
            cad_only = np.zeros_like(human)
            n_conly = 0
        aux[cad_only & (aux == 0)] = 3
        aux[model_only & (aux == 0)] = 2
        aux[confirmed] = 1
        row['holes_model_px'] += int(model_holes.sum())
        row['holes_cad_px'] += int(cad_holes.sum())
        row['holes_confirmed_px'] += int(confirmed.sum())
        row['model_only_px'] += int(model_only.sum())
        row['cad_only_px'] += int(cad_only.sum())
        row['n_confirmed'] += n_conf
        row['n_model_only'] += n_monly
        row['n_cad_only'] += n_conly
        final = human & ~confirmed
        line = maskops.mask_to_yolo_line(cls, final, w, h, keep_holes=True)
        if line:
            out_lines.append(line)
    row['disagree_px'] = row['model_only_px'] + row['cad_only_px']
    row['cad_status'] = '|'.join(sorted(set(statuses))) if statuses else 'noinst'
    with open(out_lab, 'w') as f:
        f.write('\n'.join(out_lines) + ('\n' if out_lines else ''))
    cv2.imwrite(out_aux, aux)
    return row


def collect(src_dirs):
    """[(stem, img_path, label_path, source_tag)] — images/<split>/ 와 images/ 평면 둘 다 지원."""
    items = []
    for src in src_dirs:
        tag = os.path.basename(os.path.normpath(src))
        for idir in sorted(glob.glob(os.path.join(src, 'images', '*'))) + [os.path.join(src, 'images')]:
            if not os.path.isdir(idir):
                continue
            sp = os.path.basename(idir) if idir != os.path.join(src, 'images') else ''
            for fn in sorted(os.listdir(idir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                lab = os.path.join(src, 'labels', sp, stem + '.txt')
                if os.path.exists(lab):
                    items.append((stem, os.path.join(idir, fn), lab, f'{tag}/{sp}'.rstrip('/')))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True, help='구멍을 남기도록 학습한 YOLO-seg 가중치')
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--obj-dir', default=OBJ_DIR)
    ap.add_argument('--cache', nargs='*', default=None, help='실루엣 캐시 폴더 우선순위 (기본 obj/cached_fhd, obj/cached)')
    ap.add_argument('--min-px', type=int, default=16, help='구멍 성분 최소 픽셀')
    ap.add_argument('--dilate-frac', type=float, default=0.03, help='CAD 구멍 팽창 = 물체 bbox 긴 변 × 이 값 (정합 오차 허용)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--into', action='store_true',
                    help='SRC(1개, 검수 데이터셋)의 보류 프레임(대표·편집본·SAM2 구멍 전파 결과 제외)에 결과를 자동 라벨(labels_auto/)과 '
                         'aux/ 로 직접 넣는다. 코드는 state 의 재지정 코드. --out 은 무시. 끝나면 `python -m mask_reviewer rescore SRC` 로 diff 갱신')
    a = ap.parse_args()

    ds_into = None
    if a.into:
        from mask_reviewer.dataset import Dataset
        ds_into = Dataset(a.src[0])
        a.out = ds_into.root
        skip = 0
        keep_stems = set()
        for s in ds_into.stems:
            if ds_into.state.status(s) != 'pending' or ds_into.state.exemplar(s) or ds_into.is_edited(s):
                continue
            if (ds_into.state.auto(s) or {}).get('source') == 'sam2-holes' and ds_into.state.status(s) == 'ok':
                continue
            keep_stems.add(s)
        items = [(s, ds_into.image_path(s), ds_into.original_label_path(s), os.path.basename(ds_into.root))
                 for s in ds_into.stems if s in keep_stems and os.path.exists(ds_into.original_label_path(s))]
        print(f'--into: 보류 프레임 {len(items)} 장 (전체 {len(ds_into.stems)})')
    else:
        items = collect(a.src)
    if a.limit:
        items = items[:a.limit]
    if not items:
        sys.exit('입력 프레임이 없습니다')
    names = {}
    for src in a.src:
        y = os.path.join(src, 'dataset.yaml')
        if os.path.exists(y):
            n = (yaml.safe_load(open(y)) or {}).get('names') or {}
            names = {int(k): str(v) for k, v in (n.items() if isinstance(n, dict) else enumerate(n))}
            break
    if not names:
        sys.exit('클래스 이름(dataset.yaml names)을 찾지 못했습니다')

    for d in (('labels_auto', 'aux') if ds_into else ('images', 'labels', 'aux')):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    # ② 모델 추론 (GPU, 순차)
    from ultralytics import YOLO
    model = YOLO(a.model)
    preds = {}
    B = 16
    print(f'모델 추론 {len(items)} 장 ({a.model})')
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        res = model.predict([it[1] for it in chunk], imgsz=a.imgsz, conf=a.conf, retina_masks=True,
                            device=a.device, verbose=False)
        for it, r in zip(chunk, res):
            ms = []
            if r.masks is not None:
                img_h, img_w = r.orig_shape
                for m in r.masks.data.cpu().numpy():
                    if m.shape != (img_h, img_w):
                        m = cv2.resize(m, (img_w, img_h))
                    ms.append(m > 0.5)
            preds[it[0]] = ms
        if (i // B) % 20 == 0:
            print(f'  {min(i + B, len(items))}/{len(items)}')
    del model

    # ③ CAD 정합 + 교차 (CPU 병렬)
    vals = runtime_params()
    _stub_gl()
    sys.path.insert(0, CORE)
    import yaw_match_pool as ymp
    P = {k: vals.get(k) for k in ymp.PARAM_KEYS}
    roi_cap = vals.get('score_roi_max_dim')
    fit_thr = vals.get('fitness_threshold', 0.5)
    cache_dirs = a.cache or [os.path.join(a.obj_dir, 'cached_fhd'), os.path.join(a.obj_dir, 'cached')]
    print(f'CAD 정합: yaw {P["yaw_min"]}~{P["yaw_max"]} step {P["coarse_step"]}, roi cap {roi_cap}, fitness {fit_thr}, 캐시 {cache_dirs}')

    jobs = []
    src_of = {}
    for stem, img_path, lab, tag in items:
        if ds_into:
            code = ds_into.code(stem) or ''
            dst_img = img_path
            lab_dir = 'labels_auto'
        else:
            m = CODE_RE.search(stem)
            code = m.group(1) if m else ''
            dst_img = os.path.join(a.out, 'images', os.path.basename(img_path))
            lab_dir = 'labels'
            if not os.path.exists(dst_img):
                try:
                    os.link(img_path, dst_img)
                except OSError:
                    import shutil
                    shutil.copy2(img_path, dst_img)
        lines = [l for l in open(lab, encoding='utf-8') if l.strip()]
        src_of[stem] = tag
        jobs.append((stem, dst_img, lines, code, names, preds.get(stem, []),
                     os.path.join(a.out, lab_dir, stem + '.txt'), os.path.join(a.out, 'aux', stem + '.png'),
                     a.min_px, fit_thr, a.dilate_frac))
    rows = []
    with Pool(a.workers, initializer=_pool_init, initargs=(cache_dirs, P, roi_cap)) as pool:
        for k, row in enumerate(pool.imap(_work, jobs, chunksize=4)):
            row['source'] = src_of[row['stem']]
            rows.append(row)
            if (k + 1) % 200 == 0:
                print(f'  정합 {k + 1}/{len(jobs)}')

    fields = ['stem', 'source', 'code', 'cls', 'n_inst', 'match_yaw', 'match_score', 'match_iou', 'cad_status',
              'holes_model_px', 'holes_cad_px', 'holes_confirmed_px', 'model_only_px', 'cad_only_px', 'disagree_px',
              'n_confirmed', 'n_model_only', 'n_cad_only']
    if ds_into:
        import time as _t
        for r in rows:
            n = sum(1 for l in open(os.path.join(a.out, 'labels_auto', r['stem'] + '.txt'), encoding='utf-8') if len(l.split()) >= 7)
            ds_into.state.set(r['stem'], auto={'source': 'model-cad-holes', 'at': _t.strftime('%Y-%m-%d %H:%M:%S'), 'n': n,
                                                'score': r['match_score'] if r['match_score'] != '' else 0.0, 'diff': 0.0},
                              note=f"auto model-cad-holes cad={r['cad_status']} conf={r['holes_confirmed_px']} model_only={r['model_only_px']} cad_only={r['cad_only_px']}")
        with open(os.path.join(a.out, 'holes_model_manifest.csv'), 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        # 툴이 정렬·정보에 쓰는 열을 manifest.csv 에도 합친다
        mp_ = os.path.join(a.out, 'manifest.csv')
        if os.path.exists(mp_):
            base = list(csv.DictReader(open(mp_, newline='')))
            by = {r['stem']: r for r in rows}
            extra = ['cad_status', 'holes_confirmed_px', 'model_only_px', 'cad_only_px', 'disagree_px', 'match_yaw', 'match_score', 'match_iou']
            for b in base:
                r = by.get(b['stem'])
                for k in extra:
                    b[k] = r[k] if r else b.get(k, '')
            with open(mp_, 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(base[0].keys()))
                wr.writeheader()
                wr.writerows(base)
        st = {}
        for r in rows:
            st[r['cad_status']] = st.get(r['cad_status'], 0) + 1
        print(f'완료(--into) {len(rows)} 장 → labels_auto/aux: 확정 구멍 있는 프레임 {sum(1 for r in rows if r["holes_confirmed_px"] > 0)}, CAD 상태 {st}')
        return
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    ds = {'path': os.path.abspath(a.out), 'train': 'images', 'val': 'images', 'names': names,
          'keep_holes': True, 'aux_dir': 'aux',
          'aux_legend': {1: 'confirmed (model & CAD)', 2: 'model only', 3: 'CAD only'}}
    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as f:
        f.write(f'# build_holes_review.py: 검수본 {a.src} 에 구멍 자동 적용 (모델 {os.path.basename(a.model)} × CAD 정합 교차)\n'
                f'# labels/ = 사람 마스크 − 확정 구멍 (다리 폴리곤). aux/ = 1 확정 / 2 모델만 / 3 CAD만\n')
        yaml.safe_dump(ds, f, sort_keys=False, allow_unicode=True)
    st = {}
    for r in rows:
        st[r['cad_status']] = st.get(r['cad_status'], 0) + 1
    n_h = sum(1 for r in rows if r['holes_confirmed_px'] > 0)
    n_d = sum(1 for r in rows if r['disagree_px'] > 0)
    print(f'완료 {len(rows)} 장 → {a.out}: 확정 구멍 있는 프레임 {n_h}, 불일치 있는 프레임 {n_d}, CAD 상태 {st}')


if __name__ == '__main__':
    main()
