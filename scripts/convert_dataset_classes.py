#!/usr/bin/env python3
"""검수 데이터셋의 라벨 클래스를 12자리 제품코드(328 클래스) 체계로 바꾼다 (2026-09-23).

  python scripts/convert_dataset_classes.py DATASET [--names CODES.yaml]   # 기본 = yolo26_dataset/codes_20260921/codes_nocap_20260923.yaml (322, cap 제외)

labels/·labels_auto/·labels_reviewed/ 의 모든 인스턴스 클래스 id 를 **그 프레임의 제품 코드**(검수 재지정 코드 우선, 없으면 파일명)의
id 로 바꾼다(한 프레임 = 한 제품). 코드가 목록에 없으면(UNKNOWN_* 등) 목록 뒤에 덧붙여 id 를 준다. dataset.yaml 의 names 를 교체하고
원래 names 는 `names_prev` 로 남긴다. 되돌리려면 같은 스크립트를 이전 5클래스 yaml 로 다시 돌리면 된다(클래스는 코드→대분류로 못 돌아가므로
label_src 대분류 정보가 필요하면 labels 를 미리 백업).
이후 328클래스 모델(codes_y26x_*)로 relabel_with_model.py 를 --map-codes 없이 돌리면 모델이 추정한 코드가 인스턴스 클래스로 들어간다.
gather_datasets.py 는 기본으로 이 변환을 끝에 수행한다(--no-codes 로 끔). 현장 코드와 모델 추정이 다른 프레임은 검수자가 제품 코드 변경으로 확정한다.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mask_reviewer.dataset import Dataset, load_dataset_yaml  # noqa: E402


DEFAULT_CODES_YAML = os.path.expanduser('~/jhw/data/SL/yolo26_dataset/codes_20260921/codes_none_20260929.yaml')


def convert(dataset, names_yaml=DEFAULT_CODES_YAML, by_code=False):
    """dataset 의 라벨 클래스를 names_yaml 의 코드 체계로 바꾼다. 반환 (파일 수, 인스턴스 수, 코드 없는 파일 수, 덧붙인 코드, 버린 인스턴스 수).
    데이터셋이 이미 제품코드 체계(class_scheme product_code_12)면 인스턴스의 **클래스 이름**으로 새 id 를 찾는다(모델 추정 코드 보존);
    by_code 또는 5클래스 데이터셋이면 프레임 코드로 바꾼다. names_yaml 의 dropped_names(cap 등) 인스턴스는 버린다."""
    ds = Dataset(dataset)
    yy = yaml.safe_load(open(names_yaml))
    names = yy['names']
    names = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    dropped = set(str(n) for n in (yy.get('dropped_names') or []))
    cid = {v: k for k, v in names.items()}
    old = load_dataset_yaml(ds.root)
    by_name = (not by_code) and old.get('class_scheme') == 'product_code_12'
    extra = []

    def id_of(code):
        if code not in cid:
            cid[code] = len(names) + len(extra)
            extra.append(code)
        return cid[code]
    for s in ds.stems:
        c = ds.code(s)
        if c and c not in dropped:
            id_of(c)
    n_files = n_inst = n_nocode = n_drop = 0
    for sub in ('labels', 'labels_auto', 'labels_reviewed'):
        d = os.path.join(ds.root, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.txt'):
                continue
            stem = fn[:-4]
            c = ds.code(stem)
            if not c:
                n_nocode += 1
                continue
            p = os.path.join(d, fn)
            out = []
            for line in open(p, encoding='utf-8'):
                v = line.split()
                if len(v) >= 7:
                    name = ds.names.get(int(float(v[0]))) if by_name else c
                    if name is None or name in dropped or (name in ds.dropped_names):
                        n_drop += 1
                        continue
                    out.append(f'{id_of(name)} ' + ' '.join(v[1:]))
                    n_inst += 1
            with open(p, 'w', encoding='utf-8') as f:
                f.write('\n'.join(out) + ('\n' if out else ''))
            n_files += 1
    y = load_dataset_yaml(ds.root)
    if not by_name:
        y['names_prev'] = y.get('names')
    all_names = dict(names)
    for i, c in enumerate(extra):
        all_names[len(names) + i] = c
    y['names'] = all_names
    y['class_scheme'] = 'product_code_12'
    y['dropped_names'] = sorted(dropped)
    yp = os.path.join(ds.root, 'dataset.yaml')
    head = [l for l in open(yp, encoding='utf-8') if l.startswith('#')]
    with open(yp, 'w', encoding='utf-8') as f:
        f.writelines(head)
        f.write(f'# 클래스 = 12자리 제품코드 ({len(names)} + 데이터에만 있는 {len(extra)}; 원본 {os.path.abspath(names_yaml)}), convert_dataset_classes.py\n')
        yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)
    return n_files, n_inst, n_nocode, extra, n_drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--names', default=DEFAULT_CODES_YAML, help='코드 클래스 목록을 가진 yaml (기본 %(default)s)')
    ap.add_argument('--by-code', action='store_true', help='이미 코드 체계인 데이터셋도 프레임 코드로 다시 배정 (모델 추정 코드 버림)')
    a = ap.parse_args()
    n_files, n_inst, n_nocode, extra, n_drop = convert(a.dataset, a.names, a.by_code)
    print(f'변환: 라벨 파일 {n_files}, 인스턴스 {n_inst}, 코드 없는 파일 {n_nocode}, 목록 밖 코드 덧붙임 {extra}, 제외 클래스 인스턴스 버림 {n_drop}')


if __name__ == '__main__':
    main()
