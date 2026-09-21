"""SAM2 점/박스 프롬프트 분할 (선택 의존성: torch + sam2, `requirements-sam2.txt`).

labeling_tool_legacy 의 SAM2 연동을 옮긴 것이다. torch·sam2 는 이 모듈 안에서만 지연 import 하므로
설치되지 않아도 나머지 툴은 그대로 동작한다. 이미지 임베딩(encoder 출력)은 이미지당 한 번만 계산하고
`<데이터셋>/.sam2_cache/<stem>.pt` 에 캐시한다 (다시 열면 재계산 없음).
"""
import glob
import os

import numpy as np

CACHE_DIRNAME = '.sam2_cache'
# 체크포인트 파일명 -> hydra config 이름
_CFG_BY_NAME = {'sam2.1_hiera_tiny': 'configs/sam2.1/sam2.1_hiera_t.yaml',
                'sam2.1_hiera_small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
                'sam2.1_hiera_base_plus': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
                'sam2.1_hiera_large': 'configs/sam2.1/sam2.1_hiera_l.yaml',
                'sam2_hiera_tiny': 'configs/sam2/sam2_hiera_t.yaml',
                'sam2_hiera_small': 'configs/sam2/sam2_hiera_s.yaml',
                'sam2_hiera_base_plus': 'configs/sam2/sam2_hiera_b+.yaml',
                'sam2_hiera_large': 'configs/sam2/sam2_hiera_l.yaml'}
# 체크포인트를 찾아보는 기본 위치 (환경변수 MR_SAM2_CKPT 가 있으면 최우선)
DEFAULT_CKPT_DIRS = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints'),
                     os.path.expanduser('~/labeling_tool_legacy/sam2/checkpoints')]


def available():
    """torch 와 sam2 를 import 할 수 있으면 True."""
    try:
        import torch  # noqa: F401
        import sam2  # noqa: F401
        return True
    except Exception:
        return False


def find_checkpoint():
    """환경변수 → 기본 폴더 순서로 체크포인트를 찾는다. tiny 를 우선한다."""
    p = os.environ.get('MR_SAM2_CKPT')
    if p and os.path.exists(p):
        return p
    for d in DEFAULT_CKPT_DIRS:
        for name in ('sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large'):
            c = os.path.join(d, name + '.pt')
            if os.path.exists(c):
                return c
        found = sorted(glob.glob(os.path.join(d, 'sam2*.pt')))
        if found:
            return found[0]
    return None


def config_for(ckpt_path):
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    for k, v in _CFG_BY_NAME.items():
        if stem.startswith(k):
            return v
    raise ValueError(f'체크포인트 이름으로 config 를 정할 수 없습니다: {ckpt_path}')


class Sam2Segmenter:
    """이미지 하나를 set_image 로 올려 두고 점/박스 프롬프트로 마스크를 뽑는다."""

    def __init__(self, ckpt_path, device=None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = build_sam2(config_for(ckpt_path), ckpt_path, device=device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.key = None  # 현재 올라간 이미지 식별자 (cache_path)

    def _ctx(self):
        t = self.torch
        if self.device == 'cuda':
            return t.autocast('cuda', dtype=t.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def set_image(self, img_bgr, cache_path=None):
        """이미지를 인코더에 올린다. cache_path 가 있으면 임베딩을 거기서 읽거나 저장한다."""
        key = cache_path or id(img_bgr)
        if key == self.key:
            return False
        t = self.torch
        if cache_path and os.path.exists(cache_path):
            try:
                c = t.load(cache_path, map_location=self.device, weights_only=False)
                if tuple(c['orig_hw']) == tuple(img_bgr.shape[:2]):
                    self.predictor._features = c['features']
                    self.predictor._orig_hw = c['orig_hw']
                    self.predictor._is_image_set = True
                    self.key = key
                    return True
            except Exception:
                pass
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        with t.inference_mode(), self._ctx():
            self.predictor.set_image(rgb)
        self.key = key
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                t.save({'features': self.predictor._features, 'orig_hw': self.predictor._orig_hw}, cache_path)
            except Exception:
                pass
        return True

    def predict(self, points=(), labels=(), box=None):
        """points: [(x, y)], labels: [1(전경)/0(배경)], box: (x0, y0, x1, y1). bool 마스크 (H, W) 와 점수."""
        if not points and box is None:
            return None, 0.0
        t = self.torch
        pc = np.asarray(points, dtype=np.float32).reshape(-1, 2) if points else None
        pl = np.asarray(labels, dtype=np.int32).reshape(-1) if points else None
        bx = np.asarray(box, dtype=np.float32).reshape(4) if box is not None else None
        with t.inference_mode(), self._ctx():
            masks, scores, _ = self.predictor.predict(point_coords=pc, point_labels=pl, box=bx,
                                                      multimask_output=False)
        m = np.asarray(masks)
        if m.ndim == 3:
            m = m[0]
        return m > 0, float(np.asarray(scores).reshape(-1)[0])


def cache_path_for(dataset_root, stem):
    return os.path.join(dataset_root, CACHE_DIRNAME, stem + '.pt')
