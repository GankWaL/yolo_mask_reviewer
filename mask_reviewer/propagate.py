"""대표 이미지의 마스크를 같은 제품의 다른 이미지로 SAM2 전파해 자동 라벨(labels_auto/)을 만든다.

SAM2 비디오 예측기를 "대표 1장 + 대상 1장" 두 프레임 영상으로 돌린다 (쌍 전파). 여러 장을 한 줄로 전파하면
프레임이 연속 영상이 아니라 메모리가 흘러 뒤쪽에서 객체를 놓치므로 쓰지 않는다 (2026-09-18 실험).
대표가 여럿이면 각각 전파해 객체 점수(object_score_logits)가 가장 높은 결과를 쓴다.
torch·sam2 는 지연 import (선택 의존성, requirements-sam2.txt).
"""
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from . import sam2_helper
from .dataset import Instance

MIN_AREA = 64.0   # 이보다 작은 전파 결과는 버린다 (px)


class Sam2Propagator:
    def __init__(self, ckpt_path=None, device=None):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        ckpt_path = ckpt_path or sam2_helper.find_checkpoint()
        if not ckpt_path:
            raise FileNotFoundError('SAM2 체크포인트를 찾지 못했습니다 (MR_SAM2_CKPT / checkpoints/)')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = ckpt_path
        self.pred = build_sam2_video_predictor(sam2_helper.config_for(ckpt_path), ckpt_path, device=self.device)
        self.tmp = tempfile.mkdtemp(prefix='mr_prop_')
        self._ref_key = None

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ctx(self):
        t = self.torch
        if self.device == 'cuda':
            return t.autocast('cuda', dtype=t.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _write(self, idx, img_bgr):
        cv2.imwrite(os.path.join(self.tmp, f'{idx:04d}.jpg'), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def propagate(self, ref_img, ref_instances, tgt_img, ref_key=None):
        """대표(ref_img + 인스턴스 목록) -> 대상 이미지. [(cls, bool mask, score)] 반환 (인스턴스 순서 유지, 빈 것은 제외)."""
        if not ref_instances:
            return []
        t = self.torch
        if ref_key is None or ref_key != self._ref_key:
            self._write(0, ref_img)
            self._ref_key = ref_key
        self._write(1, tgt_img)
        out = []
        with t.inference_mode(), self._ctx():
            st = self.pred.init_state(video_path=self.tmp)
            for k, inst in enumerate(ref_instances):
                self.pred.add_new_mask(st, frame_idx=0, obj_id=k + 1, mask=t.from_numpy(np.ascontiguousarray(inst.mask)))
            masks = None
            for fi, ids, logits in self.pred.propagate_in_video(st):
                if fi == 1:
                    masks = (logits[:, 0] > 0).cpu().numpy()
            scores = None
            try:
                sl = st['output_dict']['non_cond_frame_outputs'][1]['object_score_logits']
                scores = t.sigmoid(sl.float()).reshape(-1).cpu().numpy()
            except Exception:
                pass
            self.pred.reset_state(st)
        if masks is None:
            return []
        for k, inst in enumerate(ref_instances):
            m = masks[k]
            if m.sum() < MIN_AREA:
                continue
            sc = float(scores[k]) if scores is not None and k < len(scores) else float(min(1.0, m.sum() / max(1, inst.area)))
            out.append((inst.cls, m, sc))
        return out


def propagate_dataset(ds, codes=None, max_exemplars=3, include_auto=True, max_targets=0,
                      ckpt=None, propagator=None, progress=None, should_stop=None, eps=0.7, min_area=16.0,
                      include_empty_edited=False):
    """대표가 있는 제품의 보류 이미지에 자동 라벨을 쓴다.

    codes: 제품코드 목록 (None 이면 대표가 있는 모든 제품). include_auto: 이미 자동 라벨이 있는 것도 다시 계산.
    include_empty_edited: 편집본이 비어 있는 보류 이미지도 대상에 넣고, 자동 라벨을 쓰기 전에 그 빈 편집본을 지운다.
    progress(i, n, stem, info) / should_stop() -> bool 은 선택. 요약 dict 반환.
    """
    own = propagator is None
    prop = propagator or Sam2Propagator(ckpt)
    try:
        if codes is None:
            codes = sorted({ds.code(s) for s in ds.exemplars()})
        jobs = []
        for code in codes:
            ex = ds.exemplars(code)
            if not ex:
                continue
            for s in ds.auto_targets(code, include_auto=include_auto, include_empty_edited=include_empty_edited):
                jobs.append((code, s, ex[:max_exemplars] if max_exemplars > 0 else ex))
        if max_targets > 0:
            jobs = jobs[:max_targets]
        n_done = n_empty = n_cleared = 0
        per_code = {}
        scores = []
        diffs = []
        t0 = time.time()
        ref_cache = {}
        for i, (code, stem, exs) in enumerate(jobs):
            if should_stop and should_stop():
                break
            tgt = cv2.imread(ds.image_path(stem), cv2.IMREAD_COLOR)
            if tgt is None:
                continue
            h, w = tgt.shape[:2]
            best = None
            for ex in exs:
                if ex not in ref_cache:
                    ref_cache[ex] = ds.load(ex)
                ref_img, ref_insts = ref_cache[ex]
                res = prop.propagate(ref_img, ref_insts, tgt, ref_key=ex)
                if not res:
                    continue
                sc = float(np.mean([r[2] for r in res]))
                if best is None or sc > best[0]:
                    best = (sc, ex, res)
            if include_empty_edited and ds.reviewed_is_empty(stem):
                ds.revert(stem)          # 빈 편집본은 지워야 자동 라벨이 보인다
                n_cleared += 1
            if best is None:
                ds.write_auto(stem, [], w, h, source='sam2', score=0.0, ref=exs[0], eps=eps, min_area=min_area)
                n_empty += 1
                info = '없음'
            else:
                sc, ex, res = best
                insts = [Instance(c, m) for c, m, _ in res]
                ds.write_auto(stem, insts, w, h, source='sam2', score=sc, ref=ex, eps=eps, min_area=min_area)
                scores.append(sc)
                diffs.append(ds.state.auto(stem).get('diff', 0.0))
                info = f'{len(insts)}개 원본과 차이 {diffs[-1]:.2f}'
            n_done += 1
            per_code[code] = per_code.get(code, 0) + 1
            if progress:
                progress(i, len(jobs), stem, info)
        return dict(n_jobs=len(jobs), n_done=n_done, n_empty=n_empty, n_cleared=n_cleared, per_code=per_code,
                    score_mean=float(np.mean(scores)) if scores else 0.0,
                    score_min=float(np.min(scores)) if scores else 0.0,
                    diff_mean=float(np.mean(diffs)) if diffs else 0.0,
                    n_changed=int(sum(d >= 0.05 for d in diffs)), sec=time.time() - t0)
    finally:
        if own:
            prop.close()
