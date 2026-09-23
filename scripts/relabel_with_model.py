#!/usr/bin/env python3
"""재학습한 YOLO-seg 모델로 검수 데이터셋의 보류 이미지를 다시 추론해 자동 라벨(labels_auto/)을 갱신한다.

  conda activate yolo_mask_reviewer
  CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/relabel_with_model.py DATASET [--keep-sam2]   # 기본 모델 = retrain/autolabel_current.pt (328 제품코드 클래스)

데이터셋 클래스가 제품코드 체계(gather_datasets.py 기본, convert_dataset_classes.py)면 모델이 추정한 코드가 그대로 인스턴스 클래스가 된다.
현장 코드(파일명·state code)와 다른 프레임은 검수자가 제품 코드 변경으로 확정한다. 5클래스 데이터셋에 쓰려면 --map-codes.

대상은 보류 상태이고 편집본·대표가 아닌 이미지 (SAM2 전파 결과가 있으면 덮어쓴다; --keep-sam2 로 보호).
이미지는 collect_yolo_hard_cases.py 가 저장한 ROI 크롭이므로 그대로 넣는다 (운영과 같은 imgsz·conf·retina_masks).
점수는 검출 conf 의 평균, 원본 대비 변화량(diff=1-IoU)도 함께 기록된다. 툴에서 "원본과 차이 큰 순" 으로 정렬해 검수한다.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_reviewer.dataset import Dataset, Instance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--model', default=os.path.expanduser('~/jhw/data/SL/retrain/autolabel_current.pt'),
                    help='기본 = 오토라벨 기준 모델 심볼릭링크(328 제품코드 클래스 yolo26x)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--device', default='1', help='CUDA_DEVICE_ORDER=PCI_BUS_ID 기준 1 = RTX 4070 Ti')
    ap.add_argument('--codes', default='', help='제품코드 쉼표 목록 (기본 전부)')
    ap.add_argument('--keep-sam2', action='store_true', help='SAM2 전파 자동 라벨은 덮어쓰지 않음')
    ap.add_argument('--limit', type=int, default=0, help='앞에서 N장만 (검증용)')
    ap.add_argument('--map-codes', action='store_true',
                    help='12자리 제품코드 클래스 모델(codes_*)을 대분류 5클래스 데이터셋에 쓴다: 검출 클래스를 obj 이름의 부품 종류로 '
                         '바꾸고(SCALP→scalp 등), 못 바꾸면 그 프레임 원본 라벨의 가장 큰 인스턴스 클래스를 쓴다')
    a = ap.parse_args()

    from ultralytics import YOLO
    ds = Dataset(a.dataset)
    model = YOLO(a.model)
    names = {int(k): str(v) for k, v in model.names.items()}
    code_cls = None
    cls_map = None
    n_dropped_det = 0
    if a.map_codes:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reassign_codes import obj_class
        from build_holes_review import OBJ_DIR
        name_to_id = {v: k for k, v in ds.names.items()}
        code_cls = {}
        for f in os.listdir(OBJ_DIR):
            if f.lower().endswith('.obj'):
                code_cls.setdefault(f[:12], name_to_id.get(obj_class(f[:-4])))
    else:
        ## 모델 클래스 이름 → 데이터셋 id. 데이터셋에 없거나 학습 제외(dropped_names, cap)인 이름의 검출은 버린다 (2026-09-23)
        name_to_id = {v: k for k, v in ds.names.items()}
        cls_map = {k: name_to_id.get(v) for k, v in names.items()}
        for k, v in names.items():
            if v in ds.dropped_names:
                cls_map[k] = None
        if all(v is None for v in cls_map.values()):
            sys.exit(f'모델 클래스가 데이터셋에 하나도 없습니다 (--map-codes 로 코드→대분류 매핑 가능)\n  모델 {names}\n  데이터셋 {ds.names}')
        n_unmapped = sum(1 for v in cls_map.values() if v is None)
        if n_unmapped:
            print(f'모델 클래스 {n_unmapped}개는 데이터셋에 없어 그 검출은 버린다: {[names[k] for k, v in cls_map.items() if v is None][:10]}')
    codes = [c for c in a.codes.split(',') if c] or None
    targets = []
    for c in (codes or [None]):
        targets += ds.auto_targets(c)
    if a.keep_sam2:
        targets = [s for s in targets if ds.state.auto(s).get('source') != 'sam2']
    if a.limit > 0:
        targets = targets[:a.limit]
    if not targets:
        print('대상 이미지가 없습니다 (보류이고 편집본·대표가 아닌 것)')
        return 0
    src = 'yolo:' + os.path.splitext(os.path.basename(a.model))[0]
    n_empty = 0
    confs = []
    for i, stem in enumerate(targets):
        img = cv2.imread(ds.image_path(stem), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, retina_masks=True, device=a.device, verbose=False)[0]
        insts, cs = [], []
        fallback_cls = None
        if code_cls is not None:
            orig = ds.parse_instances(open(ds.original_label_path(stem), encoding='utf-8').read(), w, h) \
                if os.path.exists(ds.original_label_path(stem)) else []
            fallback_cls = max(orig, key=lambda i: i.area).cls if orig else 0
        if r.masks is not None and len(r.masks.data):
            mk = r.masks.data.cpu().numpy() > 0.5
            for k in range(mk.shape[0]):
                m = mk[k]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if m.sum() < 16:
                    continue
                cls = int(r.boxes.cls[k])
                if code_cls is not None:
                    mapped = code_cls.get(names.get(cls, ''))
                    cls = mapped if mapped is not None else fallback_cls
                elif cls_map is not None:
                    if cls_map.get(cls) is None:
                        n_dropped_det += 1
                        continue
                    cls = cls_map[cls]
                insts.append(Instance(cls, m))
                cs.append(float(r.boxes.conf[k]))
        score = float(np.mean(cs)) if cs else 0.0
        ds.write_auto(stem, insts, w, h, source=src, score=score)
        diff = ds.state.auto(stem).get('diff', 0.0)
        if not insts:
            n_empty += 1
        else:
            confs.append(score)
        print(f'\r{i + 1}/{len(targets)} {stem} {len(insts)}개 conf {score:.2f} 차이 {diff:.2f}    ', end='', flush=True)
    print(f'\n완료: {len(targets)}장 · 검출 없음 {n_empty} · conf 평균 {np.mean(confs) if confs else 0:.2f}'
          f' 최저 {min(confs) if confs else 0:.2f}\n툴에서 정렬 "원본과 차이 큰 순" 으로 검수 → 내보내기 → train_round.py 반복')
    if n_dropped_det:
        print(f'학습 제외 클래스(cap 등) 검출 {n_dropped_det}개 버림')
    return 0


if __name__ == '__main__':
    sys.exit(main())
