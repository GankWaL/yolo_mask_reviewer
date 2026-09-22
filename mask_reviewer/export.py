"""검수 결과를 YOLO-seg 학습 폴더로 내보내기.

out/
  images/{train,val}/<stem>.png   (val 비율 0 이면 images/ 평면)
  labels/{train,val}/<stem>.txt   편집본은 마스크->폴리곤, 그 외는 유효 라벨(자동 > 원본) 그대로
  dataset.yaml                    path/train/val/names
  export_manifest.csv             stem, code, split, status, edited, label_src, n_inst, source
"""
import csv
import os
import random
import shutil

import cv2
import yaml

COPY_MODES = ('copy', 'hardlink', 'symlink')


def split_stems(stems, code_of, val_ratio, seed=0):
    """제품코드별 층화 분할. stem -> 'train' | 'val'."""
    groups = {}
    for s in stems:
        groups.setdefault(code_of(s), []).append(s)
    rng = random.Random(seed)
    out = {}
    for code in sorted(groups):
        g = sorted(groups[code])
        rng.shuffle(g)
        n_val = int(round(len(g) * val_ratio)) if val_ratio > 0 else 0
        n_val = min(n_val, len(g) - 1) if len(g) > 1 else 0
        for i, s in enumerate(g):
            out[s] = 'val' if i < n_val else 'train'
    return out


def _place(src, dst, mode):
    if os.path.lexists(dst):
        os.remove(dst)
    if mode == 'hardlink':
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    elif mode == 'symlink':
        os.symlink(os.path.abspath(src), dst)
        return
    shutil.copy2(src, dst)


def export_dataset(ds, out_dir, statuses=('ok',), val_ratio=0.1, seed=0, copy_mode='copy',
                   eps=0.7, min_area=16.0, progress=None, keep_excluded=False, round_name=None):
    """progress(i, n, stem) 콜백은 선택. 요약 dict 반환.
    refine_dataset.py 가 state 에 train_exclude 를 표시한 프레임(잘림·주변 물체·중앙 이탈)은 기본으로 건너뛴다 (keep_excluded 로 포함)."""
    if copy_mode not in COPY_MODES:
        raise ValueError(copy_mode)
    out_dir = os.path.abspath(out_dir)
    stems = [s for s in ds.stems if ds.state.status(s) in statuses
             and (keep_excluded or not ds.state.get(s).get('train_exclude'))]
    n_refined_out = sum(1 for s in ds.stems if ds.state.status(s) in statuses and ds.state.get(s).get('train_exclude')) if not keep_excluded else 0
    if not stems:
        raise ValueError('내보낼 이미지가 없습니다 (선택한 상태에 해당하는 항목 없음)')
    split = split_stems(stems, ds.code, val_ratio, seed) if val_ratio > 0 else {s: '' for s in stems}
    splits = sorted({v for v in split.values()})
    for sp in splits:
        os.makedirs(os.path.join(out_dir, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', sp), exist_ok=True)

    rows = []
    cls_count = {}
    n_edited = n_empty = 0
    for i, stem in enumerate(stems):
        if progress:
            progress(i, len(stems), stem)
        sp = split[stem]
        src = ds.image_path(stem)
        ext = os.path.splitext(src)[1]
        _place(src, os.path.join(out_dir, 'images', sp, stem + ext), copy_mode)
        edited = ds.is_edited(stem)
        if edited:
            n_edited += 1
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            text = ds.instances_to_text(ds.parse_instances(ds.label_text(stem), w, h), w, h, eps, min_area)
        else:
            text = ds.label_text(stem)
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            n_empty += 1
        for l in lines:
            c = int(float(l.split()[0]))
            cls_count[c] = cls_count.get(c, 0) + 1
        with open(os.path.join(out_dir, 'labels', sp, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        rows.append(dict(stem=stem, code=ds.code(stem), split=sp or 'all', status=ds.state.status(stem),
                         edited=int(edited), label_src=ds.label_source(stem), n_inst=len(lines),
                         source=ds.info(stem).get('source', '')))

    y = {'path': out_dir,
         'train': 'images/train' if val_ratio > 0 else 'images',
         'val': 'images/val' if val_ratio > 0 else 'images',
         'names': {int(k): v for k, v in sorted(ds.names.items())}}
    if ds.keep_holes:
        y['keep_holes'] = True   # 편집본 폴리곤이 구멍을 남긴 다리 폴리곤임을 표시
    with open(os.path.join(out_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'# exported by mask_reviewer from {ds.root}\n'
                f'# statuses={",".join(statuses)} val_ratio={val_ratio} seed={seed} eps={eps} min_area={min_area}'
                f' keep_holes={int(ds.keep_holes)}\n')
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(out_dir, 'export_manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    if round_name:   # 어떤 프레임이 이 학습 라운드에 들어갔는지 state 에 남긴다 (툴 필터 "최근 학습에 사용됨", 정보 패널)
        for s in stems:
            ds.state.data.setdefault(s, {})['train_round'] = round_name
            ds.state.data[s]['train_split'] = split[s] or 'all'
        ds.state.save()
    n_val = sum(1 for v in split.values() if v == 'val')
    return dict(out_dir=out_dir, n_total=len(stems), n_train=len(stems) - n_val, n_val=n_val,
                n_edited=n_edited, n_empty=n_empty, n_refine_excluded=n_refined_out,
                cls_count={ds.names.get(k, str(k)): v for k, v in sorted(cls_count.items())})
