#!/usr/bin/env python3
"""(E) 5클래스 모델 변형을 같은 val 에서 비교: 마스크 mAP50/50-95, 대분류별 mAP, 추론 ms/img.
  CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/exp/eval_sort5.py W1.pt W2.pt ... [--data DATA.yaml] [--device 1] [--imgsz 640]
"""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('weights', nargs='+'); ap.add_argument('--data', default=os.path.expanduser('~/jhw/data/SL/yolo26_dataset/20260923/holes_all_export_r3/dataset.yaml'))
ap.add_argument('--device', default='1'); ap.add_argument('--imgsz', type=int, default=640); a = ap.parse_args()
from ultralytics import YOLO
print(f'{"weights":50s} {"mask mAP50":>10s} {"50-95":>7s} {"ms/img":>7s}  per-class mAP50-95')
for w in a.weights:
    m = YOLO(w)
    r = m.val(data=a.data, imgsz=a.imgsz, batch=16, device=a.device, verbose=False, plots=False, project='/tmp', name='eval_sort5', exist_ok=True)
    pc = ' '.join(f'{m.names[int(c)]}={r.seg.ap[i]:.3f}' for i, c in enumerate(r.seg.ap_class_index))
    print(f'{os.path.basename(os.path.dirname(os.path.dirname(w))) + "/" + os.path.basename(w):50s} {r.seg.map50:10.3f} {r.seg.map:7.3f} {r.speed["inference"]:7.1f}  {pc}')
