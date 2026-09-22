#!/usr/bin/env bash
# 현장 미탐 상시 회수·오토라벨 한 사이클 (cron/loop 용). 동시 실행 방지(flock), 로그는 ~/jhw/data/SL/autolabel/logs/<날짜>.log
#   */30 * * * * ~/jhw/yolo_mask_reviewer/scripts/field_autolabel.sh          # crontab 예
#   ~/jhw/yolo_mask_reviewer/scripts/field_autolabel.sh 20260919 20260920      # 날짜 지정
PY=${PY:-$HOME/anaconda3/envs/yolo_mask_reviewer/bin/python}   # 통합 환경 (docs/install.md §3)
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}   # GPU 번호를 nvidia-smi 와 같게 (0=RTX PRO 5000 은 다른 사용자, 1=RTX 4070 Ti 사용)
TOOL=$(cd "$(dirname "$0")/.." && pwd)
LOGD=${AL_STATE:-$HOME/jhw/data/SL/autolabel}/logs
mkdir -p "$LOGD"
exec 9>"$LOGD/.lock"
flock -n 9 || { echo "이미 실행 중"; exit 0; }
cd "$TOOL" && TQDM_DISABLE=1 $PY scripts/field_autolabel.py run "$@" 2>&1 | grep -v "Warning" >> "$LOGD/$(date +%Y%m%d).log"
