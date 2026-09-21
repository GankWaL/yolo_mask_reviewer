#!/usr/bin/env bash
# GUI 실행. 인자 없으면 기본 데이터셋(~/jhw/data/SL_under_predict).
cd "$(dirname "$0")"
exec python3 -m mask_reviewer "${1:-$HOME/jhw/data/SL_under_predict}"
