# yolo_mask_reviewer — YOLO-seg 의사 라벨 검수 툴

운영 YOLO-seg 모델이 놓친 프레임 묶음(`images/` + 의사 라벨 `labels/` + `manifest.csv` + `dataset.yaml`,
SL_Inspection_Automation `collect_yolo_hard_cases.py` 산출)을 불러와 마스크를 사람이 고치고, 확정한 것만 YOLO-seg 학습 폴더로 내보낸다.
대표 1장을 SAM2 로 같은 제품에 전파하는 순환 라벨링과, 현장 프레임 자동 회수 → 오토라벨 → 자동 재학습 스크립트(`scripts/`)를 함께 담고 있다.

- 원본 `labels/` 는 고치지 않는다. 편집본 `labels_reviewed/`, 자동 라벨 `labels_auto/`, 판정·메모·대표는 `review_state.json` (유효 라벨: 편집본 > 자동 > 원본).
- 코드는 git(`GankWaL/yolo_mask_reviewer`)으로, git 에 올리지 않는 체크포인트·데이터는 rsync 로 PC 간에 맞춘다.

## 빠른 시작

```bash
conda env create -f environment.yml && conda activate yolo_mask_reviewer   # SAM2 등 자세한 설치: docs/install.md
./run.sh /path/to/dataset                  # GUI (인자 없으면 ~/jhw/data/SL_under_predict)
```

1. 왼쪽 목록에서 이미지를 고르고, 가운데 캔버스에서 브러시 `B`·폴리곤 `P`·SAM2 `S`·꼭짓점 `T` 로 마스크를 고친다.
2. `Space` 확정(+다음) / `X` 제외 / `W` 보류. 편집본은 넘어갈 때 자동 저장된다.
3. `Ctrl+E` 로 확정분만 내보내고(제품별 층화 val) `scripts/train_round.py` 로 재학습한다.

GUI 없이: `python3 -m mask_reviewer stats|export|propagate|rescore DATASET …`

## 문서

- [docs/install.md](docs/install.md) — 설치, SAM2·체크포인트, 검수 PC·학습 PC 동기화(git + rsync), 테스트
- [docs/manual.md](docs/manual.md) — 화면·편집 도구·단축키, 관통 구멍, 순환 라벨링, 내보내기, 현장 오토라벨·자동 재학습 파이프라인
