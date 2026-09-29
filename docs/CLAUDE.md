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

## 오토라벨 클래스 = 12자리 제품코드, CAP 은 학습 제외 (2026-09-23)
- 새 검수 데이터셋의 인스턴스 클래스는 5클래스가 아니라 제품코드 **322 클래스**(`yolo26_dataset/codes_20260921/codes_nocap_20260923.yaml`
  = 328 에서 CAP 6종을 뺀 것)다. `gather_datasets.py` 가 기본으로 변환하고, 오토라벨은 `relabel_with_model.py`(기본 모델
  `retrain/autolabel_current.pt` = yolo26x r2)를 `--map-codes` 없이 돌린다. 학습 안 된 코드는 검수자가 `제품코드 변경…`(모델 추정 코드 표시)으로
  확정한다. 코드는 자동으로 바꾸지 않는다.
- **CAP 대분류는 현장에서 검사하지 않으므로 학습에서 완전히 뺀다** (jhw 2026-09-23). 클래스 목록 yaml 의 `dropped_names`(cap 코드 6종, 5클래스는 `cap`)
  인스턴스를 변환·내보내기(`export --drop-classes`, 기본)·재라벨·`remap_field_codes.py`(`skip_dropped_class`)에서 버리고 names 에서도 빼
  id 를 당긴다. 5클래스 모델은 앞으로 4클래스(b_cvr, h_cvr, hsg, scalp)로 학습된다 — SLIA 런타임은 클래스 **이름**으로 동작하므로 코드 수정 없이 쓸 수 있다.
  현장 수집(`field_autolabel.py`)도 CAP 프레임·CAP 코드(newcodes)를 받지 않는다.

## 학습·검증 데이터 운영 (2026-09-29)
- 학습 데이터 최신본은 `~/jhw/data/SL/yolo26_dataset/train_class<N>`, 고정 검증 데이터는 `valid_class<N>` 이다 (N = 그 폴더의 제품코드 수,
  현재 `train_class136`·`valid_class143`). 구성·이력은 그 폴더의 `README.md`, 모델 비교는 `train_results.md`(`scripts/eval_valid.py`).
- 제품코드당 학습 프레임은 **60장 상한**이다 (`scripts/cap_per_code.py`, ★ 대표 → 사람이 고친 프레임 → 외형 다양성 순).
- 목표는 **모든 제품코드를 60장까지 채우는 것**이고 현장 수집을 계속한다 (jhw 2026-09-29). 아직 프레임이 없는 코드와 60장에 못 미치는 코드가 남아 있다.
- 검증셋에는 있는데 학습 데이터에 없는 코드는 검증 프레임이나 정제 제외 프레임을 끌어다 채우지 않는다.
  **정제(`refine_dataset.py`)를 통과하는 프레임이 새로 수집되면 그때 학습 데이터에 추가**한다.
