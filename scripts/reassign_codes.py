#!/usr/bin/env python3
"""검수 데이터셋의 제품 코드를 자동으로 재배열한다 — 사람이 확정한 기준 프레임(DINOv2 외형 유사도) × obj 실루엣 CAD 정합.

  conda activate yolo_mask_reviewer
  python scripts/reassign_codes.py TARGET --anchors DS1 DS2 ... [--pdf-list pdf_products.txt] [--apply] [--device cuda:0]

  TARGET    코드를 재배열할 검수 데이터셋 (images/ labels/ manifest.csv review_state.json).
  --anchors 사람이 코드를 확정한 프레임을 가져올 데이터셋들 (state 에 code 가 있거나 ★ 대표이거나 사람이 ok 한 것.
            --anchor-ok-snapshot 으로 자동 ok 이전의 state 스냅샷을 주면 그 시점의 ok 만 사람 ok 로 본다).
  기본은 **보고서만** 만든다. 검수 후 안 바뀐 코드는 사람이 맞다고 본 것이므로 자동 처리가 코드를 바꾸지 않는다 (jhw 2026-09-22).
  --apply / --apply-report 는 --force 를 함께 줄 때만 동작한다 (사람이 이미 코드를 바꾼 프레임은 그때도 건드리지 않는다).

판정 (프레임마다):
  ① 기준 유사도: DINOv2 ViT-S/14 물체 크롭 임베딩으로 가장 가까운 기준 프레임의 코드 A 와 코사인 sA.
  ② CAD 정합: 라벨 대분류와 같은 obj 전부(+현재 코드, A)를 운영 yaw 매처(캐시 전용, coarse 10°)로 정합해 점수순 상위 3개.
  · UNKNOWN_* 프레임: CAD 상위 3개를 보고하고, 1위 점수가 대분류 fitness 문턱 이상이면 그 코드를 제안(obj 있음), 아니면 obj 없음 의심.
  · 그 외: A == 현재 → 유지(confirmed). A != 현재 이고 sA ≥ --sim(0.6) 이고 CAD 가 A 를 현재보다 못하게 보지 않으면(score(A) ≥ score(cur) − 0.02) → A 로 변경.
    CAD 가 현재를 강하게 지지하면(score(cur) − score(A) > 0.05) 유지 + conflict 표시. 기준 없이 CAD 만 다른 코드를 크게(+0.08) 선호하면 cad-only 제안(자동 반영 안 함).
결과: TARGET/reassign_report.csv (stem, cur, anchor, sim, cad top3, decision, new_code, flags), 요약 출력.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import multiprocessing as mp

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from mask_reviewer import maskops  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402
import build_holes_review as bhr  # noqa: E402  (_stub_gl, runtime_params, _StoreProxy, SORT_OF, OBJ_DIR)

CODE_RE = re.compile(r'_([0-9A-Z]{12})(?=[_.]|$)')


def obj_class(name):
    n = name.upper()
    if 'SCALP' in n:
        return 'scalp'
    if 'CAP' in n:
        return 'cap'
    if 'BASE_COVER' in n:
        return 'b_cvr'
    if ('HOUSING_COVER' in n) or ('HSG' in n and 'CVR' in n) or ('COVER' in n and 'HOUSING' in n) or ('UPPER_COVER' in n) or ('BEZEL' in n):
        return 'h_cvr'
    if 'HOUSING' in n or 'HSG' in n:
        return 'hsg'
    return 'other'


# ---------------------------------------------------------------- DINOv2
_D = {}


def dino_crop_embed(items, device, cache_path):
    """items [(key, img_path, mask_bbox_xyxy|None)] → {key: 384 L2}. 물체 bbox(+8%) 크롭."""
    cache = {}
    if os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = dict(zip(z['keys'].tolist(), z['emb']))
    need = [it for it in items if it[0] not in cache]
    if need:
        import torch
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(device).eval()
        MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

        def prep(img):
            x = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB) / 255.0
            return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()
        with torch.no_grad():
            for i in range(0, len(need), 32):
                b = need[i:i + 32]
                xs = []
                for _, p, bb in b:
                    im = cv2.imread(p)
                    h, w = im.shape[:2]
                    if bb is not None:
                        x0, y0, x1, y1 = bb
                        px, py = int(0.08 * (x1 - x0)), int(0.08 * (y1 - y0))
                        im = im[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
                    xs.append(prep(im))
                e = m(torch.stack(xs).to(device)).cpu().numpy()
                for (k, _, _), v in zip(b, e):
                    cache[k] = v / (np.linalg.norm(v) + 1e-6)
        ks = list(cache)
        np.savez(cache_path, keys=np.array(ks), emb=np.stack([cache[k] for k in ks]))
    return {it[0]: cache[it[0]] for it in items}


# ---------------------------------------------------------------- 소형 실루엣 캐시 (RAM 고갈 방지)
# 운영 캐시(360 yaw × 800² 또는 1200²)를 워커마다 후보 수만큼 올리면 워커당 8~12GB 가 되어 시스템 OOM 이 났다
# (2026-09-21 서버 다운 2회의 원인). 정합은 ROI 를 240px 로 줄여서 하므로, yaw 마다 tight 실루엣을 최대 --mini-px 로
# 줄인 소형 캐시(obj 당 ~3MB)를 한 번 만들어 워커가 그것만 읽는다. 면적·bbox 는 비율로만 쓰이므로 소형 픽셀 기준이면 된다.
MINI_PX = 256


def _build_mini_one(job):
    src, dst = job
    bhr._stub_gl()
    sys.path.insert(0, bhr.CORE)
    import yaw_match_pool as ymp
    c = ymp.PrebuiltSilhouetteCache(src)
    n = len(c.yaws)
    masks = np.zeros((n, MINI_PX, MINI_PX), np.uint8)
    wh = np.zeros((n, 2), np.int32)
    area = np.zeros(n, np.int64)
    for i, y in enumerate(c.yaws.tolist()):
        t = c.decode_tight(y)
        if t is None:
            continue
        sil, _ = t
        h, w = sil.shape[:2]
        f = MINI_PX / float(max(h, w))
        nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        m = cv2.resize(sil, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        masks[i, :nh, :nw] = m
        wh[i] = (nw, nh)
        area[i] = int(m.sum())
    np.savez_compressed(dst, yaws=c.yaws, masks=np.packbits(masks, axis=2), wh=wh, area=area, px=MINI_PX)
    return os.path.basename(dst)


def build_mini_caches(cache_dirs, mini_dir, workers=8):
    os.makedirs(mini_dir, exist_ok=True)
    jobs = []
    seen = set()
    for d in cache_dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.silhouette_cache.npz'):
                continue
            stem = fn[:-len('.silhouette_cache.npz')]
            if stem in seen:
                continue
            seen.add(stem)
            dst = os.path.join(mini_dir, stem + '.mini.npz')
            if not os.path.exists(dst):
                jobs.append((os.path.join(d, fn), dst))
    if jobs:
        print(f'소형 캐시 생성 {len(jobs)} 개 → {mini_dir}')
        with mp.get_context('fork').Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(_build_mini_one, jobs, chunksize=2)):
                if (k + 1) % 50 == 0:
                    print(f'  {k + 1}/{len(jobs)}')
    return mini_dir


class MiniCache:
    """yaw_match_pool.PrebuiltSilhouetteCache 와 같은 인터페이스(decode_tight/get_area/get_bbox)의 소형판."""

    def __init__(self, path):
        z = np.load(path)
        self.yaws = z['yaws']
        self.packed = z['masks']
        self.wh = z['wh']
        self.area = z['area']
        self.px = int(z['px'])
        self.idx = {int(y): i for i, y in enumerate(self.yaws.tolist())}

    def _i(self, yaw):
        return self.idx.get(((int(yaw) + 180) % 360) - 180)

    def decode_tight(self, yaw):
        i = self._i(yaw)
        if i is None or self.wh[i][0] == 0:
            return None
        w, h = self.wh[i]
        m = np.unpackbits(self.packed[i], axis=1)[:h, :w]
        return (m.astype(np.uint8) * 255), (0, 0, int(w), int(h))

    def get_area(self, yaw):
        i = self._i(yaw)
        return int(self.area[i]) if i is not None else None

    def get_bbox(self, yaw):
        i = self._i(yaw)
        return (0, 0, int(self.wh[i][0]), int(self.wh[i][1])) if i is not None else None


class MiniStore:
    def __init__(self, mini_dir, max_loaded=400):
        self.dir = mini_dir
        self.max = max_loaded
        self.loaded = {}
        self.failed = set()

    def get_cache(self, key):
        if key in self.loaded:
            return self.loaded[key]
        if key in self.failed:
            return None
        cands = sorted(glob.glob(os.path.join(self.dir, f'{key}*.mini.npz')))
        if not cands:
            self.failed.add(key)
            return None
        c = MiniCache(cands[0])
        if len(self.loaded) >= self.max:
            self.loaded.pop(next(iter(self.loaded)))
        self.loaded[key] = c
        return c


# ---------------------------------------------------------------- CAD
_G = {}


def _init(mini_dir, P, roi_cap, stems_by_code):
    bhr._stub_gl()
    sys.path.insert(0, bhr.CORE)
    import yaw_match_pool as ymp
    _G['ymp'] = ymp
    _G['stores'] = [MiniStore(mini_dir)]
    _G['P'] = P
    _G['cap'] = roi_cap
    _G['stems'] = stems_by_code
    cv2.setNumThreads(1)


def _match_codes(job):
    stem, mask_path, sort, cands = job
    ymp = _G['ymp']
    m = cv2.imread(mask_path, 0)
    if m is None or not (m > 0).any():
        return stem, []
    roi, _ = ymp.extract_mask_roi(m)
    H, W = roi.shape[:2]
    cap = _G['cap']
    if cap and max(H, W) > cap:
        f = cap / float(max(H, W))
        roi = cv2.resize(roi, (max(1, int(round(W * f))), max(1, int(round(H * f)))), interpolation=cv2.INTER_NEAREST)
    out = []
    for code in cands:
        cache = None
        for st_ in _G['stores']:
            cache = st_.get_cache(_G['stems'].get(code, code))
            if cache is not None:
                break
        if cache is None:
            continue
        ymp._STORE = bhr._StoreProxy(cache)
        ymp._P = _G['P']
        try:
            best = ymp._match(roi, code, sort, int(_G['P']['yaw_min']), int(_G['P']['yaw_max']))
        except Exception:
            best = None
        if best and best.get('yaw') is not None:
            out.append((code, float(best['score']), float(best['iou']), int(best['yaw'])))
    out.sort(key=lambda t: -t[1])
    return stem, out


def largest_instance(ds, stem):
    """(cls_name, bool mask, bbox xyxy) of the largest instance from the effective label, or (None, None, None)."""
    img = cv2.imread(ds.image_path(stem))
    h, w = img.shape[:2]
    insts = ds.parse_instances(ds.label_text(stem), w, h)
    if not insts:
        return None, None, None, (h, w)
    inst = max(insts, key=lambda i: i.area)
    ys, xs = np.where(inst.mask)
    return ds.names.get(inst.cls, str(inst.cls)), maskops.fill_holes(inst.mask), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), (h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target')
    ap.add_argument('--anchors', nargs='+', required=True)
    ap.add_argument('--anchor-ok-snapshot', nargs='*', default=[], help='DS:state.json — 그 스냅샷의 ok 만 사람 ok 로 인정')
    ap.add_argument('--obj-dir', default=bhr.OBJ_DIR)
    ap.add_argument('--pdf-list', default='')
    ap.add_argument('--sim', type=float, default=0.6)
    ap.add_argument('--margin', type=float, default=0.05, help='1위 코드와 2위 코드의 DINOv2 유사도 차이 (기준 LOO 에서 84% 가 0.05 이상)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--mini-dir', default='', help='소형 실루엣 캐시 폴더 (기본 <데이터 루트>/sil_mini_256, 없으면 생성)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--apply-report', action='store_true', help='이미 만든 reassign_report.csv 의 change 판정만 반영하고 끝낸다 (--force 필요)')
    ap.add_argument('--force', action='store_true', help='코드 자동 반영을 허용 (기본 금지)')
    a = ap.parse_args()

    ds = Dataset(a.target)
    if (a.apply or a.apply_report) and not a.force:
        print('코드 자동 반영은 꺼져 있습니다 (검수 후 안 바뀐 코드는 사람이 확인한 것). 보고서만 만들려면 --apply 없이, 정말 반영하려면 --force 를 함께 주세요.')
        return
    if a.apply_report:
        rp = os.path.join(a.target, 'reassign_report.csv')
        n_app = n_skip = 0
        for r in csv.DictReader(open(rp)):
            if r['decision'] == 'change-anchor' and r['new_code'] != r['cur']:
                if ds.state.code(r['stem']):
                    n_skip += 1
                    continue
                ds.set_code(r['stem'], r['new_code'])
                n_app += 1
        print(f'보고서 반영 {n_app} (사람이 이미 바꾼 프레임 건너뜀 {n_skip})')
        return
    stems = ds.stems[:a.limit] if a.limit else ds.stems
    # obj 목록
    objs = sorted(f[:-4] for f in os.listdir(a.obj_dir) if f.lower().endswith('.obj'))
    stem_of = {}
    for o in objs:
        stem_of.setdefault(o[:12], o)
    by_class = {}
    for o in objs:
        by_class.setdefault(obj_class(o), []).append(o[:12])
    pdf = set()
    if a.pdf_list and os.path.exists(a.pdf_list):
        pdf = {l.split('\t')[-1].strip()[:12] for l in open(a.pdf_list) if l.strip()}
    # 기준 프레임
    snap = {}
    for spec in a.anchor_ok_snapshot:
        d, _, p = spec.partition(':')
        snap[os.path.abspath(d)] = json.load(open(p))
    anchors = []
    for root in a.anchors:
        ads = Dataset(root)
        sn = snap.get(os.path.abspath(root))
        for s in ads.stems:
            e = ads.state.get(s)
            human = bool(e.get('code')) or bool(e.get('exemplar')) or ((sn.get(s, {}).get('status') == 'ok') if sn is not None else (e.get('status') == 'ok'))
            if not human:
                continue
            code = ads.code(s)
            if not code or code.startswith('UNKNOWN'):
                continue
            key = f'{os.path.basename(root)}/{s}'
            if os.path.abspath(root) == os.path.abspath(a.target) and s in stems:
                pass
            _, _, bb, _ = largest_instance(ads, s)
            anchors.append((key, ads.image_path(s), bb, code, s, os.path.abspath(root) == os.path.abspath(a.target)))
    print(f'기준 프레임 {len(anchors)} (코드 {len({x[3] for x in anchors})}), 대상 {len(stems)}')
    # 임베딩
    tmp = os.path.join(a.target, '.reassign')
    os.makedirs(tmp, exist_ok=True)
    tgt_items = []
    tgt_info = {}
    for s in stems:
        cls, mask, bb, hw = largest_instance(ds, s)
        tgt_info[s] = (cls, bb)
        mpath = os.path.join(tmp, s + '.png')
        if mask is not None and not os.path.exists(mpath):
            cv2.imwrite(mpath, mask.astype(np.uint8) * 255)
        tgt_items.append((s, ds.image_path(s), bb))
    emb_t = dino_crop_embed(tgt_items, a.device, os.path.join(tmp, 'dino_target.npz'))
    emb_a = dino_crop_embed([(k, p, bb) for k, p, bb, _, _, _ in anchors], a.device, os.path.join(tmp, 'dino_anchor.npz'))
    A = np.stack([emb_a[k] for k, *_ in anchors])
    # CAD
    vals = bhr.runtime_params()
    bhr._stub_gl()
    sys.path.insert(0, bhr.CORE)
    import yaw_match_pool as ymp
    P = {k: vals.get(k) for k in ymp.PARAM_KEYS}
    P['coarse_step'] = 10
    P['topk_iou_coarse'] = 2
    roi_cap = min(vals.get('score_roi_max_dim') or 360, 240)
    fit = vals.get('fitness_threshold', 0.5)
    cache_dirs = [os.path.join(a.obj_dir, 'cached'), os.path.join(a.obj_dir, 'cached_fhd')]   # 800² 우선(빠름), 없으면 1200²
    mini_dir = build_mini_caches(cache_dirs, a.mini_dir or os.path.join(os.path.dirname(os.path.abspath(a.target)), f'sil_mini_{MINI_PX}'), a.workers)
    jobs = []
    cur_of = {}
    near = {}
    for s in stems:
        cls, bb = tgt_info[s]
        cur = ds.code(s) or (CODE_RE.search(s).group(1) if CODE_RE.search(s) else '')
        cur_of[s] = cur
        v = emb_t[s]
        sims = A @ v
        # 코드별 최고 유사도 (자기 자신 제외) → [(code, sim, anchor_key)] 내림차순. 1위와 2위 코드의 차이가 margin.
        best_by_code = {}
        for i in np.argsort(-sims):
            if anchors[i][5] and anchors[i][4] == s:
                continue
            c = anchors[i][3]
            if c not in best_by_code:
                best_by_code[c] = (c, float(sims[i]), anchors[i][0])
            if len(best_by_code) >= 5:
                break
        near[s] = list(best_by_code.values())
        cands = set(by_class.get(cls, []))
        if cur in stem_of:
            cands.add(cur)
        for c, _, _ in near[s][:2]:
            cands.add(c)
        sort = bhr.SORT_OF.get(cls)
        jobs.append((s, os.path.join(tmp, s + '.png'), sort, sorted(cands)))
    print(f'CAD 정합 {len(jobs)} 장 × 평균 후보 {np.mean([len(j[3]) for j in jobs]):.0f}')
    cad = {}
    # spawn: 부모가 CUDA(DINOv2) 를 초기화한 뒤 fork 하면 워커가 멈춘다 (2026-09-21 확인)
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(mini_dir, P, roi_cap, stem_of)) as pool:
        for k, (s, out) in enumerate(pool.imap_unordered(_match_codes, jobs, chunksize=2)):
            cad[s] = out
            if (k + 1) % 200 == 0:
                print(f'  {k + 1}/{len(jobs)}')
    # 판정
    rows = []
    n_change = n_keep = n_conf = n_cadonly = n_unk = 0
    for s in stems:
        cur = cur_of[s]
        cls, _ = tgt_info[s]
        thr = fit.get(bhr.SORT_OF.get(cls), fit.get('default', 0.5)) if isinstance(fit, dict) else fit
        c = cad.get(s, [])
        sc = {code: score for code, score, _, _ in c}
        top3 = c[:3]
        A1, sA, aref = near[s][0] if near[s] else ('', 0.0, '')
        margin = sA - near[s][1][1] if len(near[s]) > 1 else sA
        flags = []
        new = cur
        if cur.startswith('UNKNOWN') or cur not in stem_of:
            n_unk += 1
            if top3 and top3[0][1] >= thr:
                decision, new = 'unknown->cad', top3[0][0]
                flags.append('obj-exists?')
            else:
                decision = 'unknown-no-obj?'
        elif A1 == cur:
            decision, n_conf = 'confirmed', n_conf + 1
            if top3 and top3[0][0] != cur and top3[0][1] - sc.get(cur, -1) > 0.08:
                flags.append(f'cad-prefers:{top3[0][0]}')
        elif A1 and sA >= a.sim and margin >= a.margin and sc.get(A1, -1) >= sc.get(cur, -1) - 0.02:
            decision, new, n_change = 'change-anchor', A1, n_change + 1
        elif A1 and sA >= a.sim and margin >= a.margin:
            decision, n_keep = 'keep-conflict', n_keep + 1
            flags.append(f'anchor:{A1}@{sA:.2f} cad-prefers-cur')
        elif A1 and sA >= a.sim:
            decision, n_keep = 'keep', n_keep + 1
            flags.append(f'anchor-ambiguous:{A1}@{sA:.2f}/margin{margin:.2f}')
        elif top3 and top3[0][0] != cur and top3[0][1] - sc.get(cur, -1) > 0.08 and top3[0][1] >= thr:
            decision, n_cadonly = 'cad-only-suggest', n_cadonly + 1
            flags.append(f'suggest:{top3[0][0]}')
        else:
            decision, n_keep = 'keep', n_keep + 1
        if new and pdf and new not in pdf:
            flags.append('not-in-pdf')
        rows.append(dict(stem=s, cls=cls or '', cur=cur, anchor=A1, sim=round(sA, 3), margin=round(margin, 3), anchor_ref=aref,
                         cad1=top3[0][0] if top3 else '', cad1_score=round(top3[0][1], 3) if top3 else '',
                         cad2=top3[1][0] if len(top3) > 1 else '', cad2_score=round(top3[1][1], 3) if len(top3) > 1 else '',
                         cad3=top3[2][0] if len(top3) > 2 else '', cad3_score=round(top3[2][1], 3) if len(top3) > 2 else '',
                         score_cur=round(sc[cur], 3) if cur in sc else '', decision=decision, new_code=new, flags=','.join(flags)))
    rp = os.path.join(a.target, 'reassign_report.csv')
    with open(rp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'판정: confirmed {n_conf}, change(anchor) {n_change}, keep {n_keep}, cad-only 제안 {n_cadonly}, unknown {n_unk} → {rp}')
    if a.apply:
        n_app = n_skip = 0
        for r in rows:
            if r['decision'] in ('change-anchor',) and r['new_code'] != r['cur']:
                if ds.state.code(r['stem']):
                    n_skip += 1
                    continue
                ds.set_code(r['stem'], r['new_code'])
                n_app += 1
        print(f'반영 {n_app} (사람이 이미 바꾼 프레임은 건너뜀 {n_skip})')


if __name__ == '__main__':
    main()
