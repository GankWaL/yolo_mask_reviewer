#!/usr/bin/env python3
"""현장 YOLO-seg 폴더(대분류 5클래스)를 12자리 제품코드 클래스로 다시 라벨링해 새 폴더로 만든다.

  python scripts/remap_field_codes.py --names REF/dataset.yaml --out OUT SRC[:train_split=dst_split,...] ...

  --names  클래스 목록을 가져올 dataset.yaml (build_record_yolo.py 결과). 같은 id 를 쓴다.
  SRC      images/<split>, labels/<split> 이 있는 폴더. `:` 뒤에 split 매핑을 주면 그 split 만 옮긴다
           (예 `reviewed_20260919_valnew:val=val`). 생략하면 train→train, val→val.

코드는 SRC/export_manifest.csv 의 code 열(검수 툴에서 재지정한 코드 반영)을 우선 쓰고, 없으면 파일명(stem)의 12자리
코드(`_XXXXXXXXXXXX`)를 쓴다. 한 프레임의 인스턴스는 전부 같은 코드(한 로트 = 한 제품)로 본다.
코드가 없거나 목록에 없는 프레임은 건너뛴다. 이미지는 하드링크, 라벨은 첫 열만 바꿔 쓴다.
"""
import argparse
import csv
import os
import re
import sys

import yaml

CODE_RE = re.compile(r'_([0-9A-Z]{12})(?=[_.]|$)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('src', nargs='+')
    a = ap.parse_args()
    with open(a.names) as f:
        ref = yaml.safe_load(f)
    names = {int(k): v for k, v in ref['names'].items()}
    cid = {v: k for k, v in names.items()}

    rows = []
    stat = {}
    for spec in a.src:
        src, _, maps = spec.partition(':')
        maps = dict(m.split('=') for m in maps.split(',')) if maps else {'train': 'train', 'val': 'val'}
        tag = os.path.basename(os.path.normpath(src))
        manifest_code = {}
        mp = os.path.join(src, 'export_manifest.csv')
        if os.path.exists(mp):
            with open(mp, newline='', encoding='utf-8') as f:
                manifest_code = {r['stem']: r.get('code', '') for r in csv.DictReader(f)}
        for ssp, dsp in maps.items():
            idir = os.path.join(src, 'images', ssp)
            if not os.path.isdir(idir):
                continue
            os.makedirs(os.path.join(a.out, 'images', dsp), exist_ok=True)
            os.makedirs(os.path.join(a.out, 'labels', dsp), exist_ok=True)
            for fn in sorted(os.listdir(idir)):
                stem, ext = os.path.splitext(fn)
                code = manifest_code.get(stem) or None
                if code is None:
                    m = CODE_RE.search(stem)
                    code = m.group(1) if m else None
                lab = os.path.join(src, 'labels', ssp, stem + '.txt')
                if code is None or code not in cid:
                    st = 'skip_nocode' if code is None else 'skip_unknown_code'
                elif not os.path.exists(lab):
                    st = 'skip_nolabel'
                else:
                    lines = []
                    for row in open(lab):
                        v = row.split()
                        if len(v) >= 7:
                            lines.append(f'{cid[code]} ' + ' '.join(v[1:]))
                    if not lines:
                        st = 'skip_empty'
                    else:
                        dst_stem = f'{tag}__{stem}'
                        di = os.path.join(a.out, 'images', dsp, dst_stem + ext)
                        if not os.path.exists(di):
                            os.link(os.path.join(idir, fn), di)
                        with open(os.path.join(a.out, 'labels', dsp, dst_stem + '.txt'), 'w') as f:
                            f.write('\n'.join(lines) + '\n')
                        st = 'ok'
                rows.append({'stem': stem, 'source': tag, 'src_split': ssp, 'split': dsp, 'code': code or '', 'status': st})
                stat[(tag, dsp, st)] = stat.get((tag, dsp, st), 0) + 1
    for k, v in sorted(stat.items()):
        print(f'{k[0]:32s} {k[1]:5s} {k[2]:20s} {v}')
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['stem', 'source', 'src_split', 'split', 'code', 'status'])
        wr.writeheader()
        wr.writerows(rows)
    ds = {'path': os.path.abspath(a.out), 'train': 'images/train', 'val': 'images/val', 'names': names}
    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as f:
        f.write(f'# 현장 YOLO-seg 폴더를 12자리 제품코드 클래스로 재라벨 (클래스 목록 = {os.path.abspath(a.names)})\n')
        yaml.safe_dump(ds, f, sort_keys=False, allow_unicode=True)
    ok = [r for r in rows if r['status'] == 'ok']
    print(f'완료: ok {len(ok)} / {len(rows)}, 클래스 {len({r["code"] for r in ok})} → {a.out}/dataset.yaml')


if __name__ == '__main__':
    main()
