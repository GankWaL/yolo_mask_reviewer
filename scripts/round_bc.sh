#!/usr/bin/env bash
# B+C 한 라운드: 리허설 세트 생성(없으면) → freeze10 / freeze0 두 가지 파인튜닝 → 게이트 보고
#   FIELD=~/jhw/data/SL/field/20260918 REH=~/jhw/data/SL/rehearsal_20260918 REV=~/jhw/data/SL/reviewed_20260918 \
#   TAG=20260918 EPOCHS=15 LR=0.0001 DEV=1 scripts/round_bc.sh
# 결과: $RUNS/round_<TAG>_f10, _f0 (weights/best.pt, results.csv), $DATA/gate_report_<TAG>.md
set -uo pipefail
PY=${PY:-$HOME/anaconda3/envs/yolo_mask_reviewer/bin/python}   # 통합 환경 (docs/install.md §3)
TOOL=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$HOME/jhw/data/SL}
FIELD=${FIELD:-$DATA/field/20260918}
REH=${REH:-$DATA/rehearsal_20260918}
REV=${REV:-$DATA/reviewed_20260918}
RUNS=${RUNS:-$DATA/runs}
BASE=${BASE:-$HOME/jhw/SL_Inspection_Automation/models/yolo11s_best_20260326.pt}
TAG=${TAG:-$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-15}; LR=${LR:-0.0001}; DEV=${DEV:-1}; BATCH=${BATCH:-16}
PER_CODE=${PER_CODE:-60}; GATE_PER_CODE=${GATE_PER_CODE:-15}
FREEZES=${FREEZES:-"10 0"}
echo "[$(date +%T)] tag=$TAG epochs=$EPOCHS lr0=$LR dev=$DEV freezes=$FREEZES"
if [ ! -d "$REH/images/train" ]; then
  echo "== 리허설 세트 생성 $REH"
  $PY "$TOOL/scripts/build_rehearsal_set.py" "$FIELD" --out "$REH" --base "$BASE" --per-code "$PER_CODE" \
      --gate-per-code "$GATE_PER_CODE" --device "$DEV" || exit 1
fi
NEW=()
for FZ in $FREEZES; do
  NAME=round_${TAG}_f$FZ
  echo "== 학습 $NAME  $(date +%T)"
  $PY "$TOOL/scripts/train_round.py" "$REV" --base "$BASE" --extra-data "$REH/dataset.yaml" --runs "$RUNS" --name "$NAME" \
      --epochs "$EPOCHS" --lr0 "$LR" --freeze "$FZ" --batch "$BATCH" --device "$DEV" --patience 8 || exit 1
  NEW+=(--new "f$FZ=$RUNS/$NAME/weights/best.pt")
done
echo "== 게이트  $(date +%T)"
$PY "$TOOL/scripts/eval_gates.py" --base "$BASE" "${NEW[@]}" --gate "$REH/gate" --reviewed "$REV" --rehearsal "$REH" \
    --out "$DATA/gate_report_${TAG}.md" --device "$DEV"
echo "[$(date +%T)] 완료 → $DATA/gate_report_${TAG}.md"
