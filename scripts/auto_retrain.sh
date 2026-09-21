#!/usr/bin/env bash
# 자동 재학습 한 번 (4시간 cron 용). 로그 ~/jhw/data/SL/retrain/logs/<날짜>.log
#   0 */4 * * * ~/jhw/yolo_mask_reviewer/scripts/auto_retrain.sh
PY=${PY:-$HOME/anaconda3/envs/yolo_mask_reviewer/bin/python}   # 통합 환경 (docs/install.md §3)
TOOL=$(cd "$(dirname "$0")/.." && pwd)
LOGD=${RT_DIR:-$HOME/jhw/data/SL/retrain}/logs
mkdir -p "$LOGD"
cd "$TOOL" && TQDM_DISABLE=1 $PY scripts/auto_retrain.py "$@" 2>&1 | grep -v "Warning" >> "$LOGD/$(date +%Y%m%d).log"
