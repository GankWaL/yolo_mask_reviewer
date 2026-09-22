#!/usr/bin/env python3
"""train_scaffold 의 COCO 기록 데이터셋(instance_segmentation_record)을 12자리 제품코드 클래스의 YOLO-seg 폴더로 바꾼다.

  python scripts/build_record_yolo.py COCO_DIR --out OUT [--obj-dir OBJ] [--base MODEL.pt] [--device 0]

  COCO_DIR   train/instances_train.json, val/instances_val.json, train/images, val/images (800x600, 프레임당 어노테이션 1개)
  --out      YOLO 폴더: images/{train,val}, labels/{train,val}, dataset.yaml, manifest.csv
  --roi      잘라낼 창 x y w h (기본 223 132 297 389 = segment_ROI.png 의 경계상자). 라벨된 물체 97.5% 가 이 안에 있고
             이웃 물체는 대부분 밖이라 "라벨 없는 물체" 문제를 피한다.
  --obj-dir  SLIA obj 폴더. 파일명 앞 12자리를 클래스 목록으로 삼는다(정렬, 데이터에 없는 코드도 포함 → 이후 라운드에서
             클래스 id 가 안 바뀐다). 데이터에만 있는 코드는 뒤에 덧붙인다.
  --base     운영 YOLO-seg. 크롭 안에 GT 와 겹치지 않는 물체(이웃 제품)가 잡히면 그 프레임은 버린다(라벨 없는 물체 방지).

  --keep-holes  관통 구멍을 채우지 않고 남긴다. YOLO 폴리곤은 구멍을 직접 못 쓰므로 바깥 윤곽에서 구멍 윤곽으로 폭 0 의
             다리를 넣어 한 폴리곤으로 잇는다(ultralytics polygon2mask 의 짝홀 채움이 구멍을 남긴다 — 리샘플·mask_ratio 4 에서 확인).
  --reuse DIR  같은 COCO 를 이미 변환한 폴더. 이미지는 하드링크하고 이웃 물체 검사 결과(manifest 의 drop_foreign)를 그대로 쓴다
             (라벨만 다시 만든다 — 채움/구멍 두 버전을 5.7GB 복제 없이 만들기 위함).

변환 규칙: RLE 마스크 → 크롭 → 관통 구멍 채움(binary_fill_holes, --keep-holes 면 생략) → 외곽 폴리곤(mask_reviewer 내보내기와
같은 0.7px 단순화, 16px² 미만 조각 버림). 마스크가 크롭 밖으로 --min-inside 미만 남으면 버린다. split 은 COCO 의 train/val 그대로.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import yaml
from scipy.ndimage import binary_fill_holes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mask_reviewer.maskops import mask_to_yolo_line  # noqa: E402  (keep_holes=True 면 다리 폴리곤)

DEFAULT_OBJ = os.path.expanduser('~/jhw/SL_Inspection_Automation/obj_product_name_BK9_260805')
DEFAULT_BASE = os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo11s_best_20260919.pt')


def rle_decode(seg):
    """COCO 압축 RLE(문자열) / 비압축(counts 리스트) → HxW uint8. pycocotools 없이."""
    h, w = seg['size']
    s = seg['counts']
    if isinstance(s, list):
        cnts = s
    else:
        s = s.encode() if isinstance(s, str) else s
        cnts, p = [], 0
        while p < len(s):
            x = k = 0
            more = 1
            while more:
                c = s[p] - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
    m = np.zeros(h * w, np.uint8)
    pos, v = 0, 0
    for c in cnts:
        if v:
            m[pos:pos + c] = 1
        pos += c
        v ^= 1
    return m.reshape((w, h)).T


def obj_codes(obj_dir):
    codes = set()
    for f in os.listdir(obj_dir):
        if f.lower().endswith('.obj') and len(f) >= 12:
            codes.add(f[:12])
    return sorted(codes)


def _work(job):
    """한 어노테이션 → 크롭 이미지·라벨 저장. (stem, status, inside_frac, n_holes_filled_px)"""
    img_path, seg, cls, stem, out_img, out_lab, roi, min_inside, keep_holes, reuse_img = job
    x, y, w, h = roi
    m = rle_decode(seg)
    tot = int(m.sum())
    mc = m[y:y + h, x:x + w]
    inside = mc.sum() / max(tot, 1)
    if inside < min_inside:
        return stem, 'drop_outside', round(float(inside), 4), 0
    filled = binary_fill_holes(mc > 0)
    holes = int(filled.sum() - mc.sum())
    if keep_holes:
        line = mask_to_yolo_line(cls, mc > 0, w, h, keep_holes=True)
    else:
        line = mask_to_yolo_line(cls, filled.astype(bool), w, h)
    if line is None:
        return stem, 'drop_empty', round(float(inside), 4), holes
    if reuse_img:
        if not os.path.exists(reuse_img):
            return stem, 'drop_noimage', 0.0, 0
        if not os.path.exists(out_img):
            os.link(reuse_img, out_img)
    else:
        im = cv2.imread(img_path)
        if im is None:
            return stem, 'drop_noimage', 0.0, 0
        cv2.imwrite(out_img, im[y:y + h, x:x + w])
    with open(out_lab, 'w') as f:
        f.write(line + '\n')
    return stem, 'ok', round(float(inside), 4), holes


def foreign_check(model, img_paths, roi_wh, device, foreign_px, batch=32):
    """운영 모델로 크롭을 다시 검출해 GT(라벨 폴리곤)와 안 겹치는 큰 마스크가 있는 프레임 stem 을 돌려준다."""
    w, h = roi_wh
    bad = {}
    for i in range(0, len(img_paths), batch):
        chunk = img_paths[i:i + batch]
        res = model.predict(chunk, imgsz=640, conf=0.25, retina_masks=True, device=device, verbose=False)
        for p, r in zip(chunk, res):
            if r.masks is None:
                continue
            lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            gt = np.zeros((h, w), np.uint8)
            for row in open(lab):
                v = row.split()
                if len(v) < 7:
                    continue
                pts = (np.array(v[1:], dtype=np.float64).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(gt, [pts], 1)
            for m in r.masks.data.cpu().numpy():
                m = cv2.resize(m, (w, h)) > 0.5
                area = int(m.sum())
                if area < foreign_px:
                    continue
                if (m & (gt > 0)).sum() / max(area, 1) > 0.5:
                    continue
                bad[p] = area
                break
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coco_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--roi', type=int, nargs=4, default=[223, 132, 297, 389], metavar=('X', 'Y', 'W', 'H'))
    ap.add_argument('--obj-dir', default=DEFAULT_OBJ)
    ap.add_argument('--base', default=DEFAULT_BASE, help="이웃 물체 검사용 운영 모델 ('' 이면 검사 생략)")
    ap.add_argument('--device', default='0')
    ap.add_argument('--min-inside', type=float, default=0.9)
    ap.add_argument('--foreign-px', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='split 당 최대 어노테이션 수 (시험용)')
    ap.add_argument('--keep-holes', action='store_true', help='관통 구멍을 남긴다 (다리 폴리곤)')
    ap.add_argument('--reuse', default='', help='이미 변환한 폴더: 이미지 하드링크 + 이웃 검사 결과 재사용')
    a = ap.parse_args()

    names = obj_codes(a.obj_dir)
    splits = {}
    for sp in ('train', 'val'):
        with open(os.path.join(a.coco_dir, sp, f'instances_{sp}.json')) as f:
            splits[sp] = json.load(f)
    extra = set()
    for d in splits.values():
        cat = {c['id']: c['name'] for c in d['categories']}
        for an in d['annotations']:
            if cat[an['category_id']] not in names:
                extra.add(cat[an['category_id']])
    if extra:
        print(f'obj 목록에 없는 코드 {len(extra)}개를 뒤에 덧붙임: {sorted(extra)}')
        names += sorted(extra)
    cid = {n: i for i, n in enumerate(names)}

    x, y, w, h = a.roi
    reuse_status = {}
    if a.reuse:
        with open(os.path.join(a.reuse, 'manifest.csv')) as f:
            for r in csv.DictReader(f):
                reuse_status[r['stem']] = r
        print(f'재사용 {a.reuse}: manifest {len(reuse_status)} 건')
    rows = []
    for sp, d in splits.items():
        os.makedirs(os.path.join(a.out, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(a.out, 'labels', sp), exist_ok=True)
        cat = {c['id']: c['name'] for c in d['categories']}
        imgs = {i['id']: i for i in d['images']}
        anns = d['annotations'][:a.limit] if a.limit else d['annotations']
        jobs = []
        for an in anns:
            info = imgs[an['image_id']]
            code = cat[an['category_id']]
            stem = f"{os.path.splitext(info['file_name'])[0]}_{code}"
            if a.reuse:
                pr = reuse_status.get(stem)
                if pr is None or pr['status'] != 'ok':
                    rows.append({'stem': stem, 'code': code, 'split': sp, 'status': (pr['status'] if pr else 'drop_not_in_reuse'),
                                 'inside': pr['inside'] if pr else '', 'holes_px': pr['holes_px'] if pr else '',
                                 'foreign_px': pr.get('foreign_px', '') if pr else ''})
                    continue
            jobs.append((os.path.join(a.coco_dir, sp, 'images', info['file_name']), an['segmentation'], cid[code], stem,
                         os.path.join(a.out, 'images', sp, stem + '.png'),
                         os.path.join(a.out, 'labels', sp, stem + '.txt'), (x, y, w, h), a.min_inside, a.keep_holes,
                         os.path.join(a.reuse, 'images', sp, stem + '.png') if a.reuse else ''))
        print(f'[{sp}] {len(jobs)} 건 변환 시작')
        stat = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for k, (stem, st, inside, holes) in enumerate(ex.map(_work, jobs, chunksize=64)):
                code = stem[-12:]
                rows.append({'stem': stem, 'code': code, 'split': sp, 'status': st, 'inside': inside, 'holes_px': holes})
                stat[st] = stat.get(st, 0) + 1
                if (k + 1) % 5000 == 0:
                    print(f'  {k + 1}/{len(jobs)} {stat}')
        print(f'[{sp}] 변환 완료 {stat}')

    if a.base and not a.reuse:
        from ultralytics import YOLO
        model = YOLO(a.base)
        for sp in splits:
            paths = [os.path.join(a.out, 'images', sp, r['stem'] + '.png') for r in rows if r['split'] == sp and r['status'] == 'ok']
            bad = foreign_check(model, paths, (w, h), a.device, a.foreign_px)
            for r in rows:
                p = os.path.join(a.out, 'images', sp, r['stem'] + '.png')
                if r['split'] == sp and p in bad:
                    r['status'] = 'drop_foreign'
                    r['foreign_px'] = bad[p]
                    os.remove(p)
                    os.remove(os.path.join(a.out, 'labels', sp, r['stem'] + '.txt'))
            print(f'[{sp}] 이웃 물체 검출로 제외 {len(bad)} / {len(paths)}')

    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['stem', 'code', 'split', 'status', 'inside', 'holes_px', 'foreign_px'])
        wr.writeheader()
        wr.writerows(rows)
    ds = {'path': os.path.abspath(a.out), 'train': 'images/train', 'val': 'images/val', 'names': {i: n for i, n in enumerate(names)}}
    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as f:
        f.write(f'# {os.path.abspath(a.coco_dir)} → ROI {a.roi} 크롭, {"구멍 유지(다리 폴리곤)" if a.keep_holes else "구멍 채움"}, '
                f'12자리 제품코드 클래스 (obj 목록 {len(names)}개 순서)\n')
        yaml.safe_dump(ds, f, sort_keys=False, allow_unicode=True)
    ok = [r for r in rows if r['status'] == 'ok']
    codes = {r['code'] for r in ok}
    print(f'완료: ok {len(ok)} / {len(rows)}, 클래스 {len(codes)} (names {len(names)}), → {a.out}/dataset.yaml')


if __name__ == '__main__':
    main()
