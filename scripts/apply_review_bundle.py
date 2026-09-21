#!/usr/bin/env python3
"""검수 PC 에서 돌아온 재검수 묶음(make_review_bundle.py 산출)을 원본 데이터셋에 반영한다 (§21-cb 3번 절차).

  python scripts/apply_review_bundle.py ~/jhw/data/SL/for_review_20260919 [--dry]

묶음의 review_state.json(status·exemplar·note)과 labels_reviewed/ 를 stem 의 출처(orig_dataset)별로 나눠 쓴다:
  · 검수 데이터셋(AL_REF_DS, SL_under_predict): status·exemplar(★) 를 그대로 덮어쓰고, 편집본은 labels_reviewed/ 에 복사.
  · 자동 수집 데이터셋(AL_DS, SL_under_predict_auto): status 를 덮어쓰고 note 를 'human: …' 로 바꿔 파이프라인이 다시 판정하지
    않게 한다(field_autolabel.is_auto_judged 는 note 가 auto- 로 시작할 때만 참). 편집본은 labels_reviewed/ 에 복사(전파 대상에서도 빠짐).
원본 파일은 덮어쓰기 전에 <파일>.bak.<시각> 으로 남긴다. 반영 후에는 exemplars/ 를 지우고 `field_autolabel.py refit` → `redo`.
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import field_autolabel as fa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bundle')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    b = os.path.abspath(a.bundle)
    with open(os.path.join(b, 'review_state.json'), encoding='utf-8') as f:
        bst = json.load(f)
    edited = {fn[:-4] for fn in os.listdir(os.path.join(b, 'labels_reviewed'))} if os.path.isdir(os.path.join(b, 'labels_reviewed')) else set()
    targets = {os.path.basename(fa.CFG['ref_ds']): fa.CFG['ref_ds'], os.path.basename(fa.CFG['ds']): fa.CFG['ds']}
    plan = collections.defaultdict(list)
    for stem, e in bst.items():
        root = targets.get(e.get('orig_dataset', ''))
        if root is None:
            print('출처 모름, 건너뜀:', stem, e.get('orig_dataset'))
            continue
        plan[root].append((stem, e))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for root, items in plan.items():
        sp = os.path.join(root, 'review_state.json')
        with open(sp, encoding='utf-8') as f:
            st = json.load(f)
        is_ref = root == fa.CFG['ref_ds']
        cnt = collections.Counter()
        for stem, e in items:
            cur = st.setdefault(stem, {})
            new_status = e.get('status', 'pending')
            if cur.get('status') != new_status:
                cnt[f'status {cur.get("status")}→{new_status}'] += 1
            cur['status'] = new_status
            if e.get('code') and cur.get('code') != e['code']:    # 검수 PC 에서 고친 제품코드
                cnt[f'code {cur.get("code") or "-"}→{e["code"]}'] += 1
                cur['code'] = e['code']
            if is_ref:
                if bool(cur.get('exemplar')) != bool(e.get('exemplar')):
                    cnt[f'★ {bool(cur.get("exemplar"))}→{bool(e.get("exemplar"))}'] += 1
                cur['exemplar'] = bool(e.get('exemplar'))
                cur['note'] = (cur.get('note') or '')
                cur['review_bundle'] = os.path.basename(b)
            else:
                cur['note'] = 'human: ' + (e.get('note') or '').split('] ', 1)[-1][:120]
                cur['review_bundle'] = os.path.basename(b)
            cur['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            if stem in edited:
                src = os.path.join(b, 'labels_reviewed', stem + '.txt')
                dst_dir = os.path.join(root, 'labels_reviewed')
                dst = os.path.join(dst_dir, stem + '.txt')
                if not a.dry:
                    os.makedirs(dst_dir, exist_ok=True)
                    if os.path.exists(dst):
                        shutil.copy2(dst, dst + f'.bak.{stamp}')
                    shutil.copy2(src, dst)
                cnt['편집본 복사' + (' (기존 덮어씀)' if os.path.exists(dst) else '')] += 1
        if not a.dry:
            shutil.copy2(sp, sp + f'.bak.{stamp}')
            tmp = sp + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(st, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, sp)
        print(f'{root}: {len(items)}장 반영{" (dry)" if a.dry else ""} — {dict(cnt)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
