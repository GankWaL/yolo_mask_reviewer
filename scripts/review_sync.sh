#!/usr/bin/env bash
# 검수 PC(REVIEW_PC, ~/.ssh/config, 키 id_servjhw) 와 검수 데이터 동기화 (2026-09-20, SLIA §27)
#   review_sync.sh push-bundle <묶음 폴더>   재검수 묶음(make_review_bundle.py 산출)을 검수 PC ~/jhw/data/SL/<이름>/ 로 보낸다
#   review_sync.sh pull-bundle <이름>        검수 결과(review_state.json·labels_reviewed·labels_auto)만 이 PC 로 가져온다
#   review_sync.sh push-ref [이름]           이 PC 의 검수 데이터셋(기본 SL_under_predict, 이름을 주면 ~/jhw/data/SL/<이름>) 상태(review_state.json·labels_reviewed·labels_auto)를 검수 PC 원본에 반영
#                                            (검수 PC 쪽 기존 파일은 *.bak_<일시> 로 남김). 재검수 묶음 반영(apply_review_bundle.py) 뒤에 실행
#   review_sync.sh pull-ref [이름]           반대 방향 — 검수 PC 에서 이어서 검수한 상태를 이 PC 로 (이 PC 쪽 백업 남김)
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
    # 3자 병합: pull-ref 가 남긴 review_state.json.pulled(스냅샷) 과 지금 검수 PC 상태를 견줘, 그 사이 사람이 바꾼 판정은 덮지 않는다
    name=${1:-SL_under_predict}; [ "$name" = SL_under_predict ] && src=$LREF || src=$HOME/jhw/data/SL/$name
    if [ -e "$src/review_state.json.pulled" ]; then
      tmpd=$(mktemp -d); rsync -a --ignore-missing-args "$HOST:$RDATA/$name/review_state.json" "$tmpd/" 2>/dev/null || true
      if [ -e "$tmpd/review_state.json" ]; then
        PY=${PY:-$HOME/anaconda3/envs/yolo_mask_reviewer/bin/python}
        $PY "$(dirname "$0")/merge_review_state.py" "$src/review_state.json" "$tmpd/review_state.json" "$src/review_state.json.pulled" --out "$src/review_state.json" || exit 1
      fi; rm -rf "$tmpd"
    else
      echo "경고: $src/review_state.json.pulled 없음 — 병합 없이 덮어씀 (pull-ref 를 먼저 하면 스냅샷이 생긴다)"
    fi
    for f in "${STATE_FILES[@]}"; do [ -e "$src/$f" ] && rsync -a $BK "$src/$f" "$HOST:$RDATA/$name/"; done
    cp -f "$src/review_state.json" "$src/review_state.json.pulled"
    echo "→ 검수 PC $RDATA/$name (기존은 *.bak_* 로 보존, 사람 판정 병합)";;
  pull-ref)
    name=${1:-SL_under_predict}; [ "$name" = SL_under_predict ] && dst=$LREF || dst=$HOME/jhw/data/SL/$name
    for f in "${STATE_FILES[@]}"; do rsync -a --ignore-missing-args $BK "$HOST:$RDATA/$name/$f" "$dst/" 2>/dev/null || true; done
    [ -e "$dst/review_state.json" ] && cp -f "$dst/review_state.json" "$dst/review_state.json.pulled"   # push-ref 병합용 스냅샷
    echo "← $dst (기존은 *.bak_* 로 보존)";;
  *) sed -n 2,8p "$0"; exit 1;;
esac
