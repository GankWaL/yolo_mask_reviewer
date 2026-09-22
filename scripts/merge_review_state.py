#!/usr/bin/env python3
"""review_state.json 3자 병합 — 검수 PC 의 사람 판정을 자동 처리 결과가 덮지 않게 한다 (2026-09-22).

  python scripts/merge_review_state.py LOCAL_STATE REMOTE_STATE SNAPSHOT_STATE --out MERGED

  SNAPSHOT  자동 처리(전파·재라벨·재배열)를 시작하기 전에 검수 PC 에서 가져온 상태 (review_sync.sh pull-ref 가 남기는 *.pulled).
  LOCAL     이 PC 에서 자동 처리를 마친 상태.  REMOTE  지금 검수 PC 에 있는 상태 (그 사이 사람이 검수했을 수 있음).

규칙 (stem 마다, 사람 필드 = status·code·exemplar·note, 자동 필드 = auto·cross_iou·holes·reject_reason):
  · 원격이 스냅샷과 다르면(사람이 그 사이 바꿈) 사람 필드는 **원격을 그대로** 둔다. 자동 필드만 로컬 것을 얹는다.
  · 원격이 스냅샷과 같으면 로컬 것을 쓴다 (자동 처리 결과 반영).
  · 어느 쪽이든 status 가 reject 인 stem 은 reject 로 남는다 (제외는 되살아나지 않는다).
  · **제품 코드(code)는 원격에 stem 이 있으면 항상 원격 값**이다 — 검수 후 안 바뀐 코드는 사람이 맞다고 본 것이므로 자동 처리가
    코드를 바꾸지 않는다 (jhw 2026-09-22). 원격에 없는 새 stem 만 로컬 코드를 쓴다.
"""
import argparse
import collections
import json

HUMAN = ('status', 'code', 'exemplar', 'note')
AUTO = ('auto', 'cross_iou', 'holes', 'reject_reason', 'train_exclude', 'refine_reasons', 'train_round', 'train_split')


def merge(local, remote, snap):
    out = {}
    stat = collections.Counter()
    for s in set(local) | set(remote):
        l, r, n = local.get(s, {}), remote.get(s, {}), snap.get(s, {})
        remote_changed = any(r.get(k) != n.get(k) for k in HUMAN)
        if remote_changed:
            e = dict(r)
            for k in AUTO:
                if k in l:
                    e[k] = l[k]
            stat['remote_human_kept'] += 1
        else:
            e = dict(l) if l else dict(r)
            stat['local_applied'] += 1
        if s in remote:                       # 코드는 사람 소유: 원격에 있는 stem 은 원격 코드 그대로 (없으면 키도 없음)
            if r.get('code'):
                e['code'] = r['code']
            elif 'code' in e:
                del e['code']
            if e.get('code') != l.get('code'):
                stat['code_kept_remote'] += 1
        if 'reject' in (l.get('status'), r.get('status')) and e.get('status') != 'reject':
            e['status'] = 'reject'
            e['note'] = (r.get('note') or l.get('note') or '')
            stat['reject_protected'] += 1
        if e:
            out[s] = e
    return out, stat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('local')
    ap.add_argument('remote')
    ap.add_argument('snapshot')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    load = lambda p: json.load(open(p, encoding='utf-8'))
    merged, stat = merge(load(a.local), load(a.remote), load(a.snapshot))
    json.dump(merged, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'병합 {len(merged)} 항목: 원격 사람 판정 유지 {stat["remote_human_kept"]}, 로컬 반영 {stat["local_applied"]}, 제외 보호 {stat["reject_protected"]}, 코드 원격 유지 {stat["code_kept_remote"]} → {a.out}')


if __name__ == '__main__':
    main()
