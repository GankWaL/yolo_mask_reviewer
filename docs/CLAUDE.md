# yolo_mask_reviewer 프로젝트 규칙

공통 규칙은 `~/claude-harness/CLAUDE.md`. 이 파일은 이 저장소에만 해당하는 것.

## 버전 관리·동기화 (2026-09-21)
- 코드는 git 으로만 관리한다. 원격 `git@github.com:GankWaL/yolo_mask_reviewer.git`, 브랜치 `main`. 커밋은 공통 규칙대로 사용자가 요청할 때만.
- 코드를 다른 PC 에 맞출 때는 `git pull` 이다. rsync 로 코드를 밀거나 되가져오지 않는다.
- git 에 올리지 않는 것(`checkpoints/`, `runs/`, `*.pt`, 데이터셋, 캐시)만 작업에 필요한 범위에서 rsync 로 맞춘다 (`docs/install.md` §4).
- PC 구성: 학습 PC = dxr-core-desktop(172.30.1.3, 이 저장소 `~/jhw/yolo_mask_reviewer`, 데이터 `~/jhw/data/`).
  검수 PC = `ssh REVIEW_PC`(jhw@172.30.1.90, 클론 `~/yolo_mask_reviewer`, 데이터 `~/jhw/data/SL/`). 검수 데이터는 `scripts/review_sync.sh` 로 오간다.
- 코드를 고쳐 검수 PC 에 반영해야 하면: 사용자가 commit·push 한 뒤 `ssh REVIEW_PC 'cd ~/yolo_mask_reviewer && git pull'`. 검수 PC 에서 GUI 가 떠 있으면 다시 띄워야 반영된다.

## 환경
- 현장 인식 PC 는 `ssh SL_perception_PC` (`~/.ssh/config`: 허브 100.66.89.55 포트 2300 포워딩, 사용자 dxr, 키 id_rsa; 2026-10-02 부터. 옛 직결 주소 dxr@100.118.154.63 은 이 PC 에서 닿지 않는다). `field_autolabel.py` 의 `AL_HOST` 와 `~/jhw/data/SL/scripts/fetch_field_data.sh` 의 `FIELD_HOST` 기본값이 이 별칭이다.
- 이 PC 의 conda 환경은 `yolo_mask_reviewer` 하나다 (GUI + SAM2 + ultralytics 학습·오토라벨). 옛 `pro`/`yolo26` 환경으로 스크립트를 돌리지 않는다. `scripts/*.sh` 의 `PY` 기본값이 이 환경을 가리킨다.
- GPU 번호는 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 로 nvidia-smi 와 같게 쓴다 (0=RTX PRO 5000 48GB, 1=4070 Ti SUPER 16GB). `~/.bashrc`·crontab·`scripts/*.sh` 에 설정돼 있고 스크립트 `--device` 기본값은 0(PRO 5000). CUDA 기본 순서(FASTEST_FIRST)는 뒤집혀 있으니 환경변수 없이 돌리지 않는다 (2026-09-22).
- ultralytics 를 다시 설치하면 일반 opencv-python 이 딸려와 PyQt5 와 충돌한다 → headless 로 되돌린다 (`docs/install.md` §3).

## 문서 배치
- `README.md` 는 소개와 빠른 시작만. 설치·동기화·테스트는 `docs/install.md`, 화면·도구·파이프라인 사용법은 `docs/manual.md`.
- 기능을 추가·변경하면 `docs/manual.md` 의 해당 절을 같이 고친다 (단축키 표, 인스턴스 연산 등).

## 테스트
- `tests/test_gui_smoke.py` 는 `MR_TEST_DS` 로 준 데이터셋을 임시 폴더에 복사해 돈다. 풀셋(2,400장)은 크므로 manifest 에서 제품 2종 6장씩 뽑은 축소본을 스크래치에 만들어 쓴다.
- 스모크의 꼭짓점 삽입·내보내기(확정 1장) 검사는 데이터셋 내용에 따라 실패할 수 있다. 실패가 변경과 무관한지는 변경 전 코드로 같은 축소본을 돌려 비교한다.

## 학습 규칙 (2026-09-23)
- YOLO 학습(train_codes.py·train_round.py·exp 스크립트·auto_retrain)은 **강건성 증강을 기본으로 포함**한다: hsv_h 0.03, hsv_s 0.8, hsv_v 0.6, 회전 ±10°, 이동 0.15, 스케일 0.7, copy_paste 0.1, bgr 0.05 + albumentations(Blur·MedianBlur·ToGray·CLAHE 자동). 사무실·현장처럼 조명과 벨트색이 다른 환경에서 마스크가 흔들리지 않게 하기 위함 (yolo_mask_reviewer 환경에 albumentations 설치됨). 12자리 코드 모델은 fliplr 0 유지, 5클래스 모델은 fliplr 0.5.

## 클래스 체계: 제품코드 325 클래스, 대분류 5클래스의 1번은 none (2026-09-29)
- 검수 데이터셋·학습 데이터의 인스턴스 클래스는 12자리 제품코드다. 클래스 목록은 `yolo26_dataset/codes_20260921/codes_none_20260929.yaml`
  (**325 = 제품코드 324 + `none`**). `gather_datasets.py`·`convert_dataset_classes.py` 의 기본 목록이다. 오토라벨은 `relabel_with_model.py`
  (기본 모델 `retrain/autolabel_current.pt`)를 `--map-codes` 없이 돌리고, 코드는 검수자가 `제품코드 변경…` 으로 확정한다. 코드는 자동으로 바꾸지 않는다.
- 대분류 모델은 **5클래스 `b_cvr, none, h_cvr, hsg, scalp`** 로 학습한다 (jhw 2026-09-29). 09-23 의 "cap 을 빼고 4클래스" 규칙은 폐기 —
  5클래스로 되돌리되 1번 클래스를 cap 대신 **none**(제품이 아닌 물체: 게이트 조각·상자 등)으로 쓴다. 목적은 제품이 아닌 물체에 제품 마스크가
  나오는 것을 줄이는 것이다. 대분류 이름 목록과 코드→대분류 조회는 `scripts/codes_to_sort5.py`(`NAMES`, `major_of`)가 기준이다.
- none 학습 프레임은 검수에서 제품코드를 `gate`·`none` 으로 바꿔 제외한 프레임에서 가져온다 (`scripts/add_none_frames.py`, 300장 안팎).
  프레임 안의 물체는 전부 none 으로 라벨한다 — 라벨 없는 물체가 남지 않게 한다.
- **고정 검증셋은 학습 데이터와 겹치지 않는다** (jhw 2026-10-01). 검증 프레임은 학습 폴더(train)에 넣지 않고, ★ 대표는 학습에 남기므로 검증셋에
  대표가 꼭 들어가야 한다는 09-28 규칙은 폐기. 검증셋에서 뺀 코드는 비학습 프레임(상한에서 빠진 검수 프레임 → 정제 제외 확정 프레임)으로 코드당 최대 5장
  채운다 (`scripts/rebuild_valid_disjoint.py`, 새 프레임은 보류로 넣어 검수 PC 에서 확인). 후보가 없는 코드는 검증셋에서 빠진다.
- **프레임 가장자리에 잘린 이웃 제품** (jhw 2026-10-01): 라벨된 인스턴스가 둘 이상이면 중심에 가장 가까운 것이 가운데 제품이고, 나머지 중
  **면적이 가운데 제품의 10% 미만이면 무조건 none** (30% 에서 10% 로, jhw 2026-10-01 밤) 으로 바꾼다 (`codes_to_sort5.edge_none_rule`, 학습·검증 생성 스크립트가 적용). 30% 이상이면
  라벨된 코드를 유지한다 (30% 이상이면 → 10% 이상이면). 라벨되지 않은 가장자리 조각은 `scripts/edge_gate_autolabel.py` 로 현재 5클래스 모델이 검출해 같은 게이트에 넣는다 —
  10% 미만이면 none 인스턴스로 추가하고, 10% 이상은 추가하지 않는다 (jhw 2026-10-01). **이미지 가장자리(3px)에 닿은 조각은 크기와 관계없이
  none 으로 추가**한다 (jhw 2026-10-06). 확정된 프레임도 대상이다. 검증셋에서는 프레임 코드와 **같은 코드**의 인스턴스가 둘 이상이면
  가장 넓은 것만 남겨 평가에서 뺀다 (`codes_to_sort5.drop_same_code_dups`, 검증셋 생성·반영 스크립트가 적용). **다른 코드나 none 으로 라벨된
  인스턴스는 빼지 않는다** (특히 none 이 섞인 프레임).
- **학습은 고정 검증셋을 val 로 쓰고, 그 밖의 모든 프레임을 train 으로 쓴다** (jhw 2026-10-01). 학습 폴더를 만든 뒤 `scripts/set_train_val.py TRAIN --valid VALID`
  로 폴더 안의 val 분할을 train 에 합치고 dataset.yaml 의 val 을 검증 폴더로 돌린다. 학습 중 val 수치 = 고정 검증 수치.
- 고정 검증셋의 none 프레임은 따로 모으지 않는다. **학습 폴더의 none 프레임을 검수한 뒤 그중 일부를 검증셋으로 옮겨** 넣는다 (jhw 2026-09-29).
  옮긴 프레임은 학습 폴더에서 뺀다 (`scripts/move_none_to_valid.py` → `cap_per_code.py --exclude-stems`). none 프레임에는 60장 상한을 걸지 않는다.
- CAP 대분류 제품은 현장에서 검사하지 않으므로 학습 프레임에 넣지 않는다. 제품코드 목록에는 10P091000NT9·10P092000NT9 만 남아 있고
  (jhw 2026-09-28 복귀) 나머지 CAP 4종은 `dropped_names` 다. 현장 수집(`field_autolabel.py`)은 CAP 코드를 받지 않는다.
  옛 5클래스 데이터셋(`cap` 이름)의 cap 인스턴스는 내보내기·재라벨에서 계속 버린다.
- SLIA 런타임은 클래스 **이름**으로 동작한다. `none` 을 내는 모델을 투입하려면 런타임이 `none` 검출을 버리는지 먼저 확인한다.

## 학습·검증 데이터 운영 (2026-09-29)
- 학습 데이터 최신본은 `~/jhw/data/SL/yolo26_dataset/train_class<N>`, 고정 검증 데이터는 `valid_class<N>` 이다 (N = 그 폴더의 클래스 수, none 포함.
  현재 `train_class137`·`valid_class130`). 자동 수집의 보유 수 기준은 `train_current` 심볼릭링크가 가리키는 폴더다. 구성·이력은 그 폴더의 `README.md`, 모델 비교는 `train_results.md`(`scripts/eval_valid.py`).
- 제품코드당 학습 프레임은 **60장 상한**이다 (`scripts/cap_per_code.py`, ★ 대표 → 사람이 고친 프레임 → 외형 다양성 순).
- 목표는 **모든 제품코드를 60장까지 채우는 것**이고 현장 수집을 계속한다 (jhw 2026-09-29). 아직 프레임이 없는 코드와 60장에 못 미치는 코드가 남아 있다.
- 검증셋에는 있는데 학습 데이터에 없는 코드는 검증 프레임이나 정제 제외 프레임을 끌어다 채우지 않는다.
  **정제(`refine_dataset.py`)를 통과하는 프레임이 새로 수집되면 그때 학습 데이터에 추가**한다.
