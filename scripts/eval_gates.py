#!/usr/bin/env python3
"""파인튜닝 모델의 성능 보존·회복 게이트 (§21-bt) 를 기준 모델과 비교해 마크다운 표로 출력한다.

  conda activate pro
  python scripts/eval_gates.py --new f10=runs/.../best.pt --new f0=runs/.../best.pt \
      --gate ~/jhw/data/SL/rehearsal_20260918/gate --reviewed ~/jhw/data/SL/reviewed_20260918 \
      --rehearsal ~/jhw/data/SL/rehearsal_20260918 --out ~/jhw/data/SL/gate_report.md

게이트 1 보존: 게이트셋(학습에 안 쓴 성공 프레임)에서 새 모델 top-1 마스크가 기준 모델 top-1 과 IoU ≥ 0.95 인 비율(목표 99%)·클래스 일치율.
게이트 2 회복: 검수 val(사람 라벨) 에서 GT 인스턴스별 최고 IoU 평균·IoU ≥ 0.8 비율·클래스 일치 (기준 vs 새 모델).
게이트 3 mAP: ultralytics val — 리허설 val(의사 라벨) 과 검수 val 의 클래스별 mAP50(M) (기준 대비 하락 0.5pt 이내 목표).
"""
import argparse
import glob
import os

import cv2
import numpy as np


def top1(r):
    if r.masks is None or len(r.boxes) == 0:
        return None
    i = int(np.argmax(r.boxes.conf.cpu().numpy()))
    return int(r.boxes.cls[i]), r.masks.data[i].cpu().numpy().astype(bool)


def all_inst(r):
    if r.masks is None:
        return []
    md = r.masks.data.cpu().numpy().astype(bool)
    return [(int(r.boxes.cls[i]), float(r.boxes.conf[i]), md[i]) for i in range(len(r.boxes))]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 1.0


def read_gt(path, w, h):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        v = line.split()
        if len(v) < 7:
            continue
        pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        out.append((int(float(v[0])), m.astype(bool)))
    return out


def predict(model, img, a):
    return model.predict(img, imgsz=a.imgsz, conf=a.conf, retina_masks=True, device=a.device, verbose=False)[0]


def gate_retention(base, new, imgs, a):
    n = hit = cls_ok = both_none = 0
    ious = []
    for p in imgs:
        img = cv2.imread(p)
        tb, tn = top1(predict(base, img, a)), top1(predict(new, img, a))
        n += 1
        if tb is None and tn is None:
            both_none += 1
            hit += 1
            cls_ok += 1
            continue
        if tb is None or tn is None:
            continue
        v = iou(tb[1], tn[1])
        ious.append(v)
        hit += v >= 0.95
        cls_ok += tb[0] == tn[0]
    return dict(n=n, iou95=hit / n, cls=cls_ok / n, mean_iou=float(np.mean(ious)) if ious else 0, none=both_none)


def gate_recovery(model, imgs, lbl_dir, a):
    n = hit = cls_ok = 0
    ious = []
    per = {}
    for p in imgs:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gts = read_gt(os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + '.txt'), w, h)
        preds = all_inst(predict(model, img, a))
        for gc, gm in gts:
            n += 1
            best, bc = 0.0, None
            for pc, _, pm in preds:
                v = iou(gm, pm)
                if v > best:
                    best, bc = v, pc
            ious.append(best)
            hit += best >= 0.8
            cls_ok += (bc == gc) and best >= 0.5
            d = per.setdefault(gc, [0, 0.0, 0])
            d[0] += 1
            d[1] += best
            d[2] += best >= 0.8
    return dict(n=n, mean_iou=float(np.mean(ious)) if ious else 0, iou80=hit / max(n, 1), cls=cls_ok / max(n, 1),
                per={k: (v[0], v[1] / v[0], v[2] / v[0]) for k, v in per.items()})


def gate_map(model, yaml_path, a):
    m = model.val(data=yaml_path, imgsz=a.imgsz, device=a.device, plots=False, verbose=False, batch=16)
    seg = m.seg
    per = {int(c): float(v) for c, v in zip(seg.ap_class_index, seg.ap50)}
    return dict(map50=float(seg.map50), map=float(seg.map), per=per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=os.path.expanduser('~/jhw/SL_Inspection_Automation/models/yolo11s_best_20260326.pt'))
    ap.add_argument('--new', action='append', default=[], help='name=path.pt (여러 번)')
    ap.add_argument('--gate', required=True, help='rehearsal/gate 폴더 (images/)')
    ap.add_argument('--reviewed', required=True, help='검수 데이터셋 폴더 (dataset.yaml, images/val, labels/val)')
    ap.add_argument('--rehearsal', required=True, help='리허설 데이터셋 폴더 (dataset.yaml)')
    ap.add_argument('--out', default='gate_report.md')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit', type=int, default=0, help='게이트 이미지 수 제한 (검증용)')
    a = ap.parse_args()
    from ultralytics import YOLO
    base = YOLO(a.base)
    news = []
    for s in a.new:
        name, path = s.split('=', 1)
        news.append((name, YOLO(path)))
    names = {int(k): str(v) for k, v in base.names.items()}
    gate_imgs = sorted(glob.glob(os.path.join(a.gate, 'images', '*.png')))
    rv_imgs = sorted(glob.glob(os.path.join(a.reviewed, 'images', 'val', '*.png')))
    if a.limit:
        gate_imgs, rv_imgs = gate_imgs[:a.limit], rv_imgs[:a.limit]
    L = [f'# 게이트 보고 ({os.path.basename(a.base)} 대비)', '']
    L += [f'## 게이트 1 보존 — 게이트셋 {len(gate_imgs)}장 (학습 미사용 성공 프레임)', '',
          '| 모델 | top-1 IoU≥0.95 | 클래스 일치 | 평균 IoU |', '|---|---|---|---|']
    for name, m in news:
        r = gate_retention(base, m, gate_imgs, a)
        L.append(f'| {name} | {r["iou95"] * 100:.2f}% | {r["cls"] * 100:.2f}% | {r["mean_iou"]:.4f} |')
        print(L[-1], flush=True)
    L += ['', f'## 게이트 2 회복 — 검수 val {len(rv_imgs)}장 (사람 라벨 기준)', '',
          '| 모델 | GT 인스턴스 | 평균 IoU | IoU≥0.8 | 클래스 일치 | 클래스별 (n, 평균 IoU, ≥0.8) |', '|---|---|---|---|---|---|']
    for name, m in [('base', base)] + news:
        r = gate_recovery(m, rv_imgs, os.path.join(a.reviewed, 'labels', 'val'), a)
        per = ', '.join(f'{names[k]}({v[0]}, {v[1]:.3f}, {v[2] * 100:.0f}%)' for k, v in sorted(r['per'].items()))
        L.append(f'| {name} | {r["n"]} | {r["mean_iou"]:.4f} | {r["iou80"] * 100:.1f}% | {r["cls"] * 100:.1f}% | {per} |')
        print(L[-1], flush=True)
    L += ['', '## 게이트 3 mAP50(M) — ultralytics val', '',
          '| 모델 | 데이터 | mAP50(M) | mAP50-95(M) | ' + ' | '.join(names[k] for k in sorted(names)) + ' |',
          '|---|---|---|---|' + '---|' * len(names)]
    for dname, y in (('리허설 val', os.path.join(a.rehearsal, 'dataset.yaml')), ('검수 val', os.path.join(a.reviewed, 'dataset.yaml'))):
        for name, m in [('base', base)] + news:
            r = gate_map(m, y, a)
            L.append(f'| {name} | {dname} | {r["map50"]:.4f} | {r["map"]:.4f} | '
                     + ' | '.join(f'{r["per"][k]:.3f}' if k in r['per'] else '-' for k in sorted(names)) + ' |')
            print(L[-1], flush=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('→', a.out)


if __name__ == '__main__':
    main()
