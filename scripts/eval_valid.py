#!/usr/bin/env python3
"""고정 검증용 데이터셋(valid_class<N>)으로 모델을 평가한다: 마스크 mAP·P/R, top-1, 미검출, 추론 시간 (2026-09-29).

  CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/eval_valid.py VALID W1.pt [W2.pt ...] [--device 0] [--conf 0.25] [--csv OUT.csv]

  VALID  build_valid_set.py / update_valid_set.py 결과 폴더. 모델 클래스가 5개 이하면 VALID/sort5, 아니면 VALID(제품코드)를 쓴다.
모델의 클래스 목록이 데이터셋과 달라도(옛 328·367 목록) 이름으로 맞춰 임시 폴더에서 평가한다. 모델에 없는 코드의 프레임은 mAP 에서 빠지고
(no_class 열) top-1 에서는 오답으로 센다. top-1 = 프레임에서 conf 가 가장 높은 검출의 클래스가 정답인 비율.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

import yaml


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

    out = []
    for w in a.weights:
        m = YOLO(w)
        mn = {int(k): str(v) for k, v in m.names.items()}
        inv = {v: k for k, v in mn.items()}
        root = os.path.join(a.valid, 'sort5') if len(mn) <= 5 else a.valid
        key = 'sort5' if len(mn) <= 5 else 'code'
        with open(os.path.join(root, 'dataset.yaml'), encoding='utf-8') as f:
            dn = {int(k): str(v) for k, v in yaml.safe_load(f)['names'].items()}
        with open(os.path.join(root, 'manifest.csv'), newline='', encoding='utf-8') as f:
            man = {r['stem']: r for r in csv.DictReader(f)}
        tmp = tempfile.mkdtemp(prefix='eval_valid_')
        try:
            for d in ('images', 'labels'):
                os.makedirs(os.path.join(tmp, d, 'val'))
            imgs = sorted(os.listdir(os.path.join(root, 'images', 'val')))
            no_class = 0
            for fn in imgs:
                stem = os.path.splitext(fn)[0]
                with open(os.path.join(root, 'labels', 'val', stem + '.txt'), encoding='utf-8') as f:
                    rows = [l.split() for l in f if l.split()]
                if any(dn[int(r[0])] not in inv for r in rows):
                    no_class += 1
                    continue
                with open(os.path.join(tmp, 'labels', 'val', stem + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(''.join(f'{inv[dn[int(r[0])]]} ' + ' '.join(r[1:]) + '\n' for r in rows))
                src = os.path.join(root, 'images', 'val', fn)
                try:
                    os.link(src, os.path.join(tmp, 'images', 'val', fn))
                except OSError:
                    shutil.copy2(src, os.path.join(tmp, 'images', 'val', fn))
            with open(os.path.join(tmp, 'd.yaml'), 'w', encoding='utf-8') as f:
                yaml.safe_dump({'path': tmp, 'train': 'images/val', 'val': 'images/val', 'names': mn}, f, allow_unicode=True)
            r = m.val(data=os.path.join(tmp, 'd.yaml'), split='val', imgsz=a.imgsz, batch=16, device=a.device, plots=False,
                      verbose=False, project=tmp, name='v')
            ok = nodet = 0
            for i in range(0, len(imgs), 32):
                part = imgs[i:i + 32]
                res = m.predict([os.path.join(root, 'images', 'val', x) for x in part], imgsz=a.imgsz, conf=a.conf, device=a.device,
                                verbose=False)
                for fn, p in zip(part, res):
                    if len(p.boxes) == 0:
                        nodet += 1
                        continue
                    ok += mn[int(p.boxes.cls[int(p.boxes.conf.argmax())])] == man[os.path.splitext(fn)[0]][key]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(w)))) if os.path.basename(os.path.dirname(w)) == 'weights' \
            else os.path.splitext(os.path.basename(w))[0]
        row = dict(model=name, kind=key, classes=len(mn), n=len(imgs), no_class=no_class, mAP50=round(float(r.seg.map50), 3),
                   mAP50_95=round(float(r.seg.map), 3), P=round(float(r.seg.mp), 3), R=round(float(r.seg.mr), 3), top1=ok,
                   top1_pct=round(100 * ok / len(imgs), 1), nodet=nodet, ms=round(r.speed['inference'], 1))
        out.append(row)
        print('RESULT ' + ' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            wr.writeheader()
            wr.writerows(out)


if __name__ == '__main__':
    main()
