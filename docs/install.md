# 설치와 동기화

실행 의존성은 python 3.10 + PyQt5, OpenCV(headless), numpy, PyYAML (`requirements.txt`). GUI 만 쓰면 여기까지면 된다.
SAM2 도구(§2)와 학습·오토라벨 스크립트(§3)는 같은 conda 환경 `yolo_mask_reviewer` 에 얹는다 (2026-09-21 부터 학습 PC 도 별도 `pro` 환경 대신 이 환경 하나로 통합).

## 1. 환경

```bash
cd ~/yolo_mask_reviewer
conda env create -f environment.yml        # python 3.10 + requirements.txt (pip)
conda activate yolo_mask_reviewer
# 이미 있는 환경에 넣을 때: pip install -r requirements.txt
```

## 2. SAM2 (선택)

점 클릭·박스로 마스크를 뽑는 SAM2 도구를 쓰려면 torch 와 sam2 를 추가로 설치한다. 없으면 도구를 눌렀을 때 안내만 하고 나머지는 그대로 동작한다.

```bash
conda activate yolo_mask_reviewer
pip install -r requirements-sam2.txt
pip install --no-build-isolation "sam2 @ git+https://github.com/facebookresearch/sam2.git"
#   로컬 사본이 있으면: pip install --no-build-isolation -e ~/labeling_tool_legacy/sam2
mkdir -p checkpoints && wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

체크포인트는 `checkpoints/` → `~/labeling_tool_legacy/sam2/checkpoints/` 순서로 찾고, 환경변수 `MR_SAM2_CKPT` 나
메뉴 `파일 → SAM2 체크포인트 선택…` 으로 바꿀 수 있다. GPU 가 있으면 자동으로 cuda 를 쓴다.
이미지 임베딩은 `<데이터셋>/.sam2_cache/<stem>.pt` 에 캐시되어 같은 이미지를 다시 열면 재계산하지 않는다 (이미지당 약 1\~2MB).

## 3. 학습·오토라벨 스크립트 (선택)

`scripts/`(train_round·eval_gates·field_autolabel·auto_retrain·build_holes_review …)를 돌리려면 §2 의 torch·SAM2 위에 ultralytics 등을 더 넣는다.

```bash
conda activate yolo_mask_reviewer
pip install -r requirements-train.txt
# ultralytics 가 끌어온 일반 opencv-python 은 PyQt5 와 Qt 플러그인이 충돌한다 → headless 로 되돌린다
pip uninstall -y opencv-python opencv-python-headless && pip install --force-reinstall --no-deps "opencv-python-headless>=4.5,<5"
```

`scripts/*.sh` 는 `PY`(기본 `~/anaconda3/envs/yolo_mask_reviewer/bin/python`)로 이 환경의 파이썬을 직접 부르므로 cron 에서도 activate 없이 돈다. 다른 환경을 쓰려면 `PY=... scripts/field_autolabel.sh`.

## 4. 실행 확인

```bash
conda activate yolo_mask_reviewer
./run.sh /path/to/dataset                  # images/ labels/ 가 있는 폴더 (인자 없으면 ~/jhw/data/SL_under_predict)
python3 -m mask_reviewer stats DATASET     # GUI 없이 검수 현황
```

## 5. 다른 PC 와 맞추기 — 코드는 git, 나머지는 rsync (2026-09-21)

- **코드**는 GitHub `git@github.com:GankWaL/yolo_mask_reviewer.git`(`main`) 으로만 맞춘다. 처음은 `git clone`, 이후는 `git pull`.
  rsync 로 코드를 밀거나 되가져오지 않는다. 다른 PC 에서 코드를 고쳤으면 거기서 commit·push 하고 이쪽에서 pull 한다 (양쪽 수정은 git 충돌로 드러난다).
- **git 에 올리지 않는 것**(`.gitignore`: `checkpoints/`, `runs/`, `*.pt`, 캐시)과 **데이터셋**은 필요한 것만 rsync 로 맞춘다.

| 역할 | 접속 | 코드 | 데이터 |
|---|---|---|---|
| 학습 PC (dxr-core-desktop, 172.30.1.3, GPU) | — | `~/jhw/yolo_mask_reviewer` | `~/jhw/data/` |
| 검수 PC (jhw-Legion, 172.30.1.90) | `ssh REVIEW_PC` (`~/.ssh/config`, 키 `id_servjhw`) | `~/yolo_mask_reviewer` (클론) | `~/jhw/data/SL/` |

```bash
# 코드 — 처음 한 번 / 이후
git clone git@github.com:GankWaL/yolo_mask_reviewer.git ~/yolo_mask_reviewer
ssh REVIEW_PC 'cd ~/yolo_mask_reviewer && git pull'           # 학습 PC 에서 push 한 뒤

# SAM2 체크포인트 (git 제외) — 없는 PC 로
rsync -av ~/jhw/yolo_mask_reviewer/checkpoints/ REVIEW_PC:~/yolo_mask_reviewer/checkpoints/

# 검수 데이터 — 학습 PC 에서 scripts/review_sync.sh 로 (묶음·판정 상태만 오간다, manual.md "재검수 묶음")
scripts/review_sync.sh push-bundle <묶음 폴더>   # 재검수 묶음 → 검수 PC
scripts/review_sync.sh pull-bundle <이름>        # 검수 결과(review_state.json·labels_reviewed·labels_auto) → 학습 PC
scripts/review_sync.sh push-ref | pull-ref       # 원본 검수셋 상태를 학습 PC ↔ 검수 PC (받는 쪽 기존 파일은 *.bak_* 로 보존)

# 내보낸 학습 폴더·데이터셋 통째로 (-L: 심볼릭링크로 내보냈어도 실제 파일 복사)
rsync -avL --info=progress2 --exclude .thumb_cache/ --exclude .sam2_cache/ SRC_DIR/ HOST:DST_DIR/
ssh HOST "sed -i 's|^path: .*|path: /받는/PC/절대경로/DST_DIR|' DST_DIR/dataset.yaml"   # dataset.yaml 의 path 는 절대경로
```

- 데이터에 `--delete` 는 붙이지 않는다. 내보내기 폴더를 통째로 교체할 때만 붙인다.
- 검수 PC 가 처음이면 위 1\~2 절대로 환경을 만들고 체크포인트를 rsync 한다 (검수만 하면 §3 은 필요 없다).

## 6. 테스트

```bash
MR_TEST_DS=/path/to/small_dataset python3 tests/test_maskops.py        # 왕복 IoU·편집 연산·내보내기
QT_QPA_PLATFORM=offscreen MR_TEST_DS=... python3 tests/test_gui_smoke.py [스크린샷 폴더]
```
GUI 스모크는 꼭짓점 편집·썸네일·대표 지정까지 확인하고, torch/sam2 와 체크포인트가 있으면 SAM2 추론과 대표 전파도 돈다 (없으면 건너뜀).
`scripts/` 는 같은 환경에서 소규모 데이터로 직접 돌려 확인한다 (`train_round.py --epochs 1`, `relabel_with_model.py --limit 4`).
`test_maskops.py` 의 내보내기 검사는 `labels_reviewed/` 가 비어 있는 원본 데이터셋을 가정한다 (검수 중인 데이터셋을 주면 편집본 수 검사가 실패한다).
실데이터 라벨 왕복(폴리곤→마스크→폴리곤) IoU 최소 0.995, 꼭짓점 평균 259 → 75.
