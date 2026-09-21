#!/usr/bin/env python3
"""검수 PC 로 넘길 재검수 묶음 만들기 (2026-09-19, §21-cb).

현장 코드가 언로딩 오류·오인식으로 믿을 수 없어 대표(★)가 잘못 지정된 제품군과, 파이프라인이 자동 판정했지만
사람 확인이 필요한 프레임을 mask_reviewer 데이터셋 형식으로 모은다 (GUI 로 열어 ★ 재지정·판정만 하면 됨).

  python scripts/make_review_bundle.py [--out ~/jhw/data/SL/for_review_<날짜>] [--codes 10H231,...]

묶음 내용 (review_state.json 의 note 에 이유, exemplar 에 원래 ★ 여부):
  A. 코드 혼동 제품군(--codes, 6자리 앞머리)의 검수 데이터셋 ok 프레임 전부 + ★  → "★ 재지정" 대상
  B. 파이프라인에서 뺀 ★ (AL_EXEMPLAR_DROP: 뒤집힌 제품·캡 gate 프레임)
  C. 자동 데이터셋에서 code-mismatch 표시된 프레임(외형이 다른 코드의 대표와 더 비슷함), 코드가 있는데 자동 제외된 프레임(뒤집힘·상자 의심), 보류 프레임
라벨은 유효 라벨(편집본 > 자동 > 원본)을 labels/ 에 복사한다. 원본 데이터셋은 건드리지 않는다.
"""
import argparse
import collections
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import field_autolabel as fa  # noqa: E402
from mask_reviewer.dataset import Dataset  # noqa: E402

DEFAULT_CODES = '10H231,10H232,10H241,10H242,10H251,10H252,10J311,10J312'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.expanduser(f'~/jhw/data/SL/for_review_{time.strftime("%Y%m%d")}'))
    ap.add_argument('--codes', default=DEFAULT_CODES)
    ap.add_argument('--per-pair', type=int, default=6, help='C-코드불일치: (현장 코드, 닮은 대표 코드) 쌍당 표본 수')
    a = ap.parse_args()
    codes6 = {c.strip() for c in a.codes.split(',') if c.strip()}
    out = a.out
    for d in ('images', 'labels'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    state, rows, cnt = {}, [], collections.Counter()

    def add(ds, stem, group, note, exemplar=False):
        if stem in state:
            state[stem]['note'] += f' | {note}'
            return
        src = ds.image_path(stem)
        dst = os.path.join(out, 'images', stem + os.path.splitext(src)[1])
        if not os.path.exists(dst):
            os.link(src, dst)
        with open(os.path.join(out, 'labels', stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write(ds.label_text(stem))
        info = ds.info(stem)
        rows.append(dict(stem=stem, source=info.get('source', ''), code=ds.code(stem), reason=info.get('reason', ''),
                         n_det=info.get('n_det', ''), top_cls=info.get('top_cls', ''), top_conf=info.get('top_conf', ''),
                         top_maskpx=info.get('top_maskpx', ''), top_fill=info.get('top_fill', ''), group=group))
        state[stem] = {'status': 'pending', 'note': f'[{group}] {note}', 'exemplar': bool(exemplar), 'orig_status': ds.state.status(stem),
                       'orig_note': ds.state.note(stem), 'orig_dataset': os.path.basename(ds.root)}
        if ds.state.code(stem):
            state[stem]['code'] = ds.state.code(stem)
        cnt[group] += 1

    ref = Dataset(fa.CFG['ref_ds'])
    for s in ref.stems:
        c = ref.code(s)
        if c[:6] in codes6 and (ref.state.status(s) == 'ok' or ref.state.exemplar(s)):
            add(ref, s, 'A-코드혼동군', f'{c} 코드 혼동 제품군 — 실제 제품 확인 후 ★ 재지정', exemplar=ref.state.exemplar(s))
    for s in fa.CFG['exemplar_drop']:
        if s in ref._img_path:
            add(ref, s, 'B-제외된★', '파이프라인에서 뺀 ★ (뒤집힌 제품 또는 캡 gate 프레임) — ★ 해제 권장', exemplar=True)
    if os.path.isdir(os.path.join(fa.CFG['ds'], 'images')):
        au = Dataset(fa.CFG['ds'])
        pair_cnt = collections.Counter()
        for s in au.stems:
            st, note = au.state.status(s), au.state.note(s)
            e = au.state.get(s)
            if e.get('code_mismatch'):
                pair = (au.code(s), e.get('ref_code', '?'))
                pair_cnt[pair] += 1
                if pair_cnt[pair] > a.per_pair:            # 같은 (현장 코드 → 닮은 대표 코드) 쌍은 표본만
                    continue
                add(au, s, 'C-코드불일치', f'외형이 {e.get("ref_code", "?")} 대표와 더 비슷함 (현장 코드 {au.code(s)}) — 언로딩 오류·오인식 의심; 자동 판정 {st}: {note}')
            elif st == 'reject' and au.code(s) not in ('none', 'gate', '') and note.startswith('auto-exclude'):
                add(au, s, 'C-자동제외(코드있음)', f'뒤집힘·상자 등 의심으로 자동 제외 — 맞으면 그대로, 아니면 확정: {note}')
            elif st == 'pending':
                add(au, s, 'C-보류', f'자동 판정 보류: {note}')
    shutil.copy2(os.path.join(fa.CFG['ref_ds'], 'dataset.yaml'), os.path.join(out, 'dataset.yaml'))
    with open(os.path.join(out, 'dataset.yaml'), 'a', encoding='utf-8') as f:
        f.write(f'# review bundle {time.strftime("%Y-%m-%d %H:%M")}: 검수 PC 에서 path 를 이 폴더 경로로 바꾼다\n')
    with open(os.path.join(out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'source', 'code', 'reason', 'n_det', 'top_cls', 'top_conf', 'top_maskpx', 'top_fill', 'group'])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, 'review_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1, sort_keys=True)
    by_code = collections.Counter((r['group'], r['code']) for r in rows)
    readme = [f'# 재검수 묶음 {time.strftime("%Y-%m-%d %H:%M")} — {len(rows)}장', '',
              '검수 PC 에서: `./run.sh <이 폴더>` 로 열고 목록의 메모([A-…]/[B-…]/[C-…])를 보며 판정한다. 원본 상태는 review_state.json 의 orig_status/orig_note 에 있다.', '',
              '- **A 코드혼동군**: 현장 코드가 언로딩 방향·오인식으로 섞인 제품군. 실제 제품을 보고 ★ 를 다시 지정한다(제품당 앞면 자세 2\\~3장). 뒤집힌 제품 프레임은 제외(X).',
              '- **B 제외된 ★**: 파이프라인이 뺀 ★. 뒤집힌 제품·캡 프레임이면 ★ 해제(R) 후 제외(X).',
              '- **C 코드불일치 / 자동제외 / 보류**: 파이프라인 자동 판정 확인. 뒤집힌 제품·상자·잘린 물체는 제외(X), 정상이면 확정(Space).', '',
              '끝나면 review_state.json 과 labels_reviewed/ 를 학습 PC 로 되돌려 보낸다 (README rsync 절). 학습 PC 는 ★ 변경분을 `~/jhw/data/SL_under_predict/review_state.json` 에 반영하고 `~/jhw/data/SL/exemplars/` 를 지우면 다음 사이클에 대표가 다시 만들어진다.', '',
              '| 그룹 | 제품 | 수 |', '|---|---|---|']
    for (g, c), n in sorted(by_code.items()):
        readme.append(f'| {g} | {c} | {n} |')
    with open(os.path.join(out, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme) + '\n')
    print(f'{out}: {len(rows)}장 {dict(cnt)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
