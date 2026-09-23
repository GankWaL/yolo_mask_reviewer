#!/usr/bin/env python3
"""YOLO(-seg) .pt → ONNX 변환 (배치 동적). ultralytics export 를 감싸고 onnxruntime 으로 배치 1·4 추론을 검증한다.

  conda activate yolo_mask_reviewer
  python scripts/export_onnx.py ~/jhw/SL_Inspection_Automation/models/yolo26s_best_260923.pt [--imgsz 640] [--out X.onnx]
                                [--dynamic-hw] [--half] [--opset 17] [--no-verify]

기본은 **배치만 동적**(입력 images: [N,3,640,640], 출력 [N,…]) 이고 높이·너비는 imgsz 로 고정한다 — 운영 입력이 640 고정이라
런타임(onnxruntime/TensorRT)이 형상을 최적화하기 좋다. `--dynamic-hw` 면 ultralytics 기본처럼 N·H·W 모두 동적.
yolo26 은 NMS 없는 end-to-end 모델이라 출력이 바로 최종 검출(seg: output0 [N, max_det, 4+1+1+32] 형식은 ultralytics 버전을 따른다)이다.
결과: 같은 폴더의 <이름>.onnx (+ 검증 요약 출력).
"""
import argparse
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('weights')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default=None, help='출력 경로 (기본: 가중치 옆 <이름>.onnx)')
    ap.add_argument('--dynamic-hw', action='store_true', help='높이·너비도 동적 (기본은 배치만 동적)')
    ap.add_argument('--half', action='store_true', help='FP16 (GPU 전용 런타임에서만 의미 있음)')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--no-verify', action='store_true')
    ap.add_argument('--device', default='cpu', help='export 장치 (cpu 권장 — 결과 동일)')
    a = ap.parse_args()

    from ultralytics import YOLO
    import onnx

    model = YOLO(a.weights)
    print(f'model: {a.weights}  task={model.task}  names={model.names}')
    # ultralytics 의 dynamic=True 는 N·H·W 를 모두 동적으로 만든다. 배치만 동적으로 두려면 export 후 H·W 차원을 imgsz 로 되돌린다.
    exported = model.export(format='onnx', imgsz=a.imgsz, dynamic=True, half=a.half, opset=a.opset, simplify=True, device=a.device)
    out = a.out or os.path.splitext(a.weights)[0] + '.onnx'
    if os.path.abspath(exported) != os.path.abspath(out):
        shutil.move(exported, out)

    m = onnx.load(out)
    inp = m.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    if not a.dynamic_hw:
        # H·W 를 고정: 입력 dim_param → dim_value. 내부 Reshape 는 ultralytics 가 -1/동적으로 내보내므로 고정 H·W 로도 동작한다.
        for d, v in zip(dims[2:4], (a.imgsz, a.imgsz)):
            d.ClearField('dim_param')
            d.dim_value = v
        onnx.save(m, out)
    m = onnx.load(out)
    onnx.checker.check_model(m)
    shape = ['N' if d.dim_param else d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    outs = [(o.name, ['N' if d.dim_param else (d.dim_value or '?') for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f'saved: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)  input {m.graph.input[0].name} {shape}  outputs {outs}')

    if a.no_verify:
        return
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    dtype = np.float16 if a.half else np.float32
    for n in (1, 4):
        x = np.random.rand(n, 3, a.imgsz, a.imgsz).astype(dtype)
        ys = sess.run(None, {name: x})
        print(f'  batch {n}: ' + ', '.join(f'{o.name}{tuple(y.shape)}' for o, y in zip(sess.get_outputs(), ys)))
    print('검증 OK (배치 1·4 추론)')


if __name__ == '__main__':
    main()
