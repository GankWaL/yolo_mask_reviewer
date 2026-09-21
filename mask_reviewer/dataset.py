"""데이터셋 폴더(images/ labels/ manifest.csv dataset.yaml) 읽기와 검수 상태 저장.

원본 `labels/` 는 건드리지 않는다. 라벨 층은 우선순위 순으로
  labels_reviewed/<stem>.txt   사람이 고친 편집본
  labels_auto/<stem>.txt       자동 라벨 (SAM2 대표 전파, 재학습 모델 재추론)
  labels/<stem>.txt            원본 의사 라벨
검수 상태(ok/pending/reject, 메모, 대표 여부, 자동 라벨 메타)는 `review_state.json` 에 둔다.
"""
import csv
import json
import os
import re
import tempfile
import time

import cv2
import numpy as np
import yaml

from . import maskops

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
STATUSES = ('pending', 'ok', 'reject')
STATUS_LABEL = {'pending': '보류', 'ok': '확정', 'reject': '제외'}
STATUS_MARK = {'pending': '·', 'ok': '✔', 'reject': '✖'}
REVIEWED_DIR = 'labels_reviewed'
AUTO_DIR = 'labels_auto'
STATE_FILE = 'review_state.json'
LABEL_SOURCE_LABEL = {'reviewed': '편집본', 'auto': '자동', 'original': '원본', 'none': '없음'}


class Instance:
    __slots__ = ('cls', 'mask')

    def __init__(self, cls, mask):
        self.cls = int(cls)
        self.mask = mask  # bool (H, W)

    def copy(self):
        return Instance(self.cls, self.mask.copy())

    @property
    def area(self):
        return int(self.mask.sum())


def load_dataset_yaml(root):
    p = os.path.join(root, 'dataset.yaml')
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_names(root):
    y = load_dataset_yaml(root)
    names = y.get('names')
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def load_manifest(root):
    p = os.path.join(root, 'manifest.csv')
    if not os.path.exists(p):
        return {}
    with open(p, newline='', encoding='utf-8') as f:
        return {r['stem']: r for r in csv.DictReader(f) if r.get('stem')}


_CODE_RE = re.compile(r'(\d{2}[A-Z]\d{3}\d{3}[A-Z]{2}\d)')


def code_from_stem(stem):
    m = _CODE_RE.search(stem)
    return m.group(1) if m else ''


def is_full_code(code):
    """12자리 제품코드 형식(예: 10H332000NT9)인지."""
    return bool(_CODE_RE.fullmatch(code or ''))


def _atomic_write(path, text):
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class ReviewState:
    """stem -> {status, note, updated}."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f) or {}

    def get(self, stem):
        return self.data.get(stem, {})

    def status(self, stem):
        s = self.get(stem).get('status', 'pending')
        return s if s in STATUSES else 'pending'

    def note(self, stem):
        return self.get(stem).get('note', '')

    def exemplar(self, stem):
        return bool(self.get(stem).get('exemplar', False))

    def auto(self, stem):
        """자동 라벨 메타 {source, score, ref, at} 또는 {}."""
        return self.get(stem).get('auto') or {}

    def code(self, stem):
        """검수 중 고친 제품코드 (없으면 '')."""
        return self.get(stem).get('code') or ''

    def set(self, stem, **kw):
        e = self.data.setdefault(stem, {})
        e.update(kw)
        e['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save()

    def save(self):
        _atomic_write(self.path, json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=True))

    def counts(self, stems):
        c = {s: 0 for s in STATUSES}
        for s in stems:
            c[self.status(s)] += 1
        return c


class Dataset:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.images_dir = os.path.join(self.root, 'images')
        self.labels_dir = os.path.join(self.root, 'labels')
        self.reviewed_dir = os.path.join(self.root, REVIEWED_DIR)
        self.auto_dir = os.path.join(self.root, AUTO_DIR)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f'images/ 폴더가 없습니다: {self.root}')
        self._img_path = {}
        for fn in sorted(os.listdir(self.images_dir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXTS and stem not in self._img_path:
                self._img_path[stem] = os.path.join(self.images_dir, fn)
        self.stems = sorted(self._img_path)
        self.manifest = load_manifest(self.root)
        self.names = load_names(self.root)
        if not self.names:
            self.names = {i: str(i) for i in range(self._infer_num_classes())}
        self.state = ReviewState(os.path.join(self.root, STATE_FILE))
        y = load_dataset_yaml(self.root)
        # 구멍 유지: dataset.yaml 의 keep_holes: true 면 저장·자동 라벨·내보내기 폴리곤이 구멍을 남긴다 (다리 폴리곤).
        # 읽기는 cv2.fillPoly 의 짝홀 채움이라 어느 쪽 라벨이든 그대로 마스크가 된다.
        self.keep_holes = bool(y.get('keep_holes', False))
        # 보조 구멍 레이어: aux/<stem>.png (uint8, 1 확정 구멍 / 2 모델만 / 3 CAD만). build_holes_review.py 가 만든다.
        self.aux_dir = os.path.join(self.root, y.get('aux_dir', 'aux'))
        self.has_aux = os.path.isdir(self.aux_dir)

    # ------------------------------------------------------------ 경로/메타
    def _infer_num_classes(self):
        mx = -1
        if os.path.isdir(self.labels_dir):
            for fn in os.listdir(self.labels_dir):
                if not fn.endswith('.txt'):
                    continue
                with open(os.path.join(self.labels_dir, fn)) as f:
                    for line in f:
                        v = line.split()
                        if v:
                            try:
                                mx = max(mx, int(float(v[0])))
                            except ValueError:
                                pass
        return max(mx + 1, 1)

    def image_path(self, stem):
        return self._img_path[stem]

    def aux_mask(self, stem):
        """보조 구멍 레이어 (uint8 HxW) 또는 None."""
        if not self.has_aux:
            return None
        p = os.path.join(self.aux_dir, stem + '.png')
        if not os.path.exists(p):
            return None
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    def original_label_path(self, stem):
        return os.path.join(self.labels_dir, stem + '.txt')

    def reviewed_label_path(self, stem):
        return os.path.join(self.reviewed_dir, stem + '.txt')

    def auto_label_path(self, stem):
        return os.path.join(self.auto_dir, stem + '.txt')

    def is_edited(self, stem):
        return os.path.exists(self.reviewed_label_path(stem))

    def has_auto(self, stem):
        return os.path.exists(self.auto_label_path(stem))

    def is_auto(self, stem):
        """자동 라벨이 현재 유효한 라벨인가 (편집본이 없을 때)."""
        return self.has_auto(stem) and not self.is_edited(stem)

    def label_source(self, stem):
        if self.is_edited(stem):
            return 'reviewed'
        if self.has_auto(stem):
            return 'auto'
        return 'original' if os.path.exists(self.original_label_path(stem)) else 'none'

    def effective_label_path(self, stem):
        for p in (self.reviewed_label_path(stem), self.auto_label_path(stem)):
            if os.path.exists(p):
                return p
        return self.original_label_path(stem)

    # ------------------------------------------------------------ 대표 / 자동 라벨
    def set_exemplar(self, stem, on=True):
        self.state.set(stem, exemplar=bool(on))

    def exemplars(self, code=None):
        """대표로 지정된 stem 목록 (code 를 주면 그 제품만)."""
        return [s for s in self.stems if self.state.exemplar(s) and (code is None or self.code(s) == code)]

    def reviewed_is_empty(self, stem):
        """편집본이 있는데 유효한 인스턴스 줄이 하나도 없는가 (인스턴스를 전부 지우고 넘어간 흔적)."""
        p = self.reviewed_label_path(stem)
        if not os.path.exists(p):
            return False
        with open(p, encoding='utf-8') as f:
            return not any(len(line.split()) >= 7 for line in f)

    def auto_targets(self, code=None, include_auto=True, include_empty_edited=False):
        """자동 라벨을 넣을 대상: 보류 상태이고 편집본·대표가 아닌 것.
        include_auto=False 면 이미 자동 라벨이 있는 것도 제외. include_empty_edited=True 면 편집본이 비어 있는 보류 이미지도 포함
        (자동 라벨을 쓸 때 그 빈 편집본을 지운다)."""
        out = []
        for s in self.stems:
            if code is not None and self.code(s) != code:
                continue
            if self.state.status(s) != 'pending' or self.state.exemplar(s):
                continue
            if self.is_edited(s) and not (include_empty_edited and self.reviewed_is_empty(s)):
                continue
            if not include_auto and self.has_auto(s):
                continue
            out.append(s)
        return out

    def original_union(self, stem, w, h):
        """원본 의사 라벨의 모든 인스턴스를 합친 bool 마스크 (라벨이 없으면 전부 False)."""
        p = self.original_label_path(stem)
        text = open(p, encoding='utf-8').read() if os.path.exists(p) else ''
        m = np.zeros((h, w), bool)
        for inst in self.parse_instances(text, w, h):
            m |= inst.mask
        return m

    def diff_vs_original(self, stem, instances, w, h):
        """자동 라벨과 원본 의사 라벨의 차이 = 1 - IoU (0 같음 … 1 전혀 다름). 둘 다 비어 있으면 0."""
        m = np.zeros((h, w), bool)
        for inst in instances:
            m |= inst.mask
        return round(1.0 - maskops.iou(m, self.original_union(stem, w, h)), 4)

    def write_auto(self, stem, instances, w, h, source, score=None, ref=None, eps=0.7, min_area=16.0):
        """자동 라벨을 labels_auto/ 에 쓰고 state 에 메타(source, score, ref, diff=원본 대비 변화량)를 남긴다. 반환: 라인 수."""
        os.makedirs(self.auto_dir, exist_ok=True)
        text = self.instances_to_text(instances, w, h, eps, min_area)
        with open(self.auto_label_path(stem), 'w', encoding='utf-8') as f:
            f.write(text)
        meta = {'source': source, 'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'n': text.count('\n'),
                'diff': self.diff_vs_original(stem, instances, w, h)}
        if score is not None:
            meta['score'] = round(float(score), 4)
        if ref:
            meta['ref'] = ref
        self.state.set(stem, auto=meta)
        return meta['n']

    def rescore_auto(self, progress=None):
        """이미 있는 자동 라벨의 diff(원본 대비 변화량)를 소급 계산해 state 에 넣는다. 반환: 처리 수."""
        stems = [s for s in self.stems if self.has_auto(s)]
        for i, s in enumerate(stems):
            img = cv2.imread(self.image_path(s), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(self.auto_label_path(s), encoding='utf-8') as f:
                insts = self.parse_instances(f.read(), w, h)
            meta = dict(self.state.auto(s))
            meta['diff'] = self.diff_vs_original(s, insts, w, h)
            self.state.data.setdefault(s, {})['auto'] = meta
            if progress:
                progress(i, len(stems), s)
        self.state.save()
        return len(stems)

    def clear_auto(self, stem):
        p = self.auto_label_path(stem)
        if os.path.exists(p):
            os.remove(p)
        if self.state.auto(stem):
            e = self.state.data.get(stem, {})
            e.pop('auto', None)
            self.state.save()

    def count_auto(self):
        return sum(1 for s in self.stems if self.is_auto(s))

    def orig_code(self, stem):
        """manifest.csv 의 제품코드, 없으면 파일명에서 추출."""
        r = self.manifest.get(stem)
        return (r.get('code') if r else '') or code_from_stem(stem)

    def code(self, stem):
        """유효 제품코드: 검수 중 고친 값(review_state.json) > manifest > 파일명."""
        return self.state.code(stem) or self.orig_code(stem)

    def set_code(self, stem, code):
        """제품코드를 고친다. 원본과 같거나 비면 덮어쓰기를 지운다. 바뀌었으면 True."""
        code = (code or '').strip().upper()
        before = self.code(stem)
        e = self.state.data.setdefault(stem, {})
        if not code or code == self.orig_code(stem):
            e.pop('code', None)
        else:
            e['code'] = code
        self.state.set(stem)
        return self.code(stem) != before

    def info(self, stem):
        r = dict(self.manifest.get(stem, {}))
        r['code'] = self.code(stem)
        return r

    def codes(self):
        c = {}
        for s in self.stems:
            k = self.code(s)
            c[k] = c.get(k, 0) + 1
        return c

    # ------------------------------------------------------------ 라벨 읽기/쓰기
    def label_text(self, stem):
        p = self.effective_label_path(stem)
        if not os.path.exists(p):
            return ''
        with open(p, encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def parse_instances(text, w, h):
        out = []
        for line in text.splitlines():
            r = maskops.parse_yolo_line(line, w, h)
            if r is None:
                continue
            cls, pts = r
            out.append(Instance(cls, maskops.polygon_to_mask(pts, h, w)))
        return out

    def load(self, stem):
        """(BGR 이미지, 인스턴스 목록). 라벨 파일이 없으면 빈 목록."""
        img = cv2.imread(self.image_path(stem), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'이미지를 읽을 수 없습니다: {self.image_path(stem)}')
        h, w = img.shape[:2]
        return img, self.parse_instances(self.label_text(stem), w, h)

    def instances_to_text(self, instances, w, h, eps=0.7, min_area=16.0):
        lines = []
        for inst in instances:
            line = maskops.mask_to_yolo_line(inst.cls, inst.mask, w, h, eps, min_area, keep_holes=self.keep_holes)
            if line:
                lines.append(line)
        return '\n'.join(lines) + ('\n' if lines else '')

    def save(self, stem, instances, w, h, eps=0.7, min_area=16.0):
        os.makedirs(self.reviewed_dir, exist_ok=True)
        text = self.instances_to_text(instances, w, h, eps, min_area)
        with open(self.reviewed_label_path(stem), 'w', encoding='utf-8') as f:
            f.write(text)
        return text.count('\n')

    def revert(self, stem):
        p = self.reviewed_label_path(stem)
        if os.path.exists(p):
            os.remove(p)

    def count_edited(self):
        if not os.path.isdir(self.reviewed_dir):
            return 0
        return sum(1 for fn in os.listdir(self.reviewed_dir) if fn.endswith('.txt'))
