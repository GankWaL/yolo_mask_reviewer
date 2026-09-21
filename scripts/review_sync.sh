#!/usr/bin/env bash
# 검수 PC(REVIEW_PC, ~/.ssh/config, 키 id_servjhw) 와 검수 데이터 동기화 (2026-09-20, SLIA §27)
#   review_sync.sh push-bundle <묶음 폴더>   재검수 묶음(make_review_bundle.py 산출)을 검수 PC ~/jhw/data/SL/<이름>/ 로 보낸다
#   review_sync.sh pull-bundle <이름>        검수 결과(review_state.json·labels_reviewed·labels_auto)만 이 PC 로 가져온다
#   review_sync.sh push-ref                  이 PC 의 검수 데이터셋 상태(review_state.json·labels_reviewed·labels_auto)를 검수 PC 원본에 반영
#                                            (검수 PC 쪽 기존 파일은 *.bak_<일시> 로 남김). 재검수 묶음 반영(apply_review_bundle.py) 뒤에 실행
#   review_sync.sh pull-ref                  반대 방향 — 검수 PC 에서 이어서 검수한 상태를 이 PC 로 (이 PC 쪽 백업 남김)
set -euo pipefail
HOST=${REVIEW_HOST:-REVIEW_PC}
RDATA=${REVIEW_DATA:-'~/jhw/data/SL'}            # 검수 PC 의 데이터 루트
LREF=${AL_REF_DS:-$HOME/jhw/data/SL_under_predict}   # 이 PC 의 검수 데이터셋 사본
STATE_FILES=(review_state.json labels_reviewed labels_auto)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
BK="--backup --suffix=.bak_$(date +%Y%m%d_%H%M%S)"
cmd=${1:-}; shift || true
case "$cmd" in
  push-bundle)
    src=${1:?묶음 폴더}; name=$(basename "${src%/}")
    rsync -avL --info=progress2 --exclude .thumb_cache/ --exclude .sam2_cache/ "$src/" "$HOST:$RDATA/$name/"
    $SSH "$HOST" "sed -i 's|^path: .*|path: \$HOME/jhw/data/SL/$name|' $RDATA/$name/dataset.yaml" && echo "→ 검수 PC $RDATA/$name (dataset.yaml path 갱신)";;
  pull-bundle)
    name=${1:?묶음 이름}; dst=$HOME/jhw/data/SL/$name
    for f in "${STATE_FILES[@]}"; do rsync -a --ignore-missing-args $BK "$HOST:$RDATA/$name/$f" "$dst/" 2>/dev/null || true; done
    echo "← $dst (review_state.json $(python3 -c "import json;d=json.load(open('$dst/review_state.json'));import collections;print(dict(collections.Counter(v.get('status') for v in d.values())))"))";;
  push-ref)
    for f in "${STATE_FILES[@]}"; do [ -e "$LREF/$f" ] && rsync -a $BK "$LREF/$f" "$HOST:$RDATA/SL_under_predict/"; done
    echo "→ 검수 PC $RDATA/SL_under_predict (기존은 *.bak_* 로 보존)";;
  pull-ref)
    for f in "${STATE_FILES[@]}"; do rsync -a --ignore-missing-args $BK "$HOST:$RDATA/SL_under_predict/$f" "$LREF/" 2>/dev/null || true; done
    echo "← $LREF (기존은 *.bak_* 로 보존)";;
  *) sed -n 2,8p "$0"; exit 1;;
esac
