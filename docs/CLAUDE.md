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
- ultralytics 를 다시 설치하면 일반 opencv-python 이 딸려와 PyQt5 와 충돌한다 → headless 로 되돌린다 (`docs/install.md` §3).

## 문서 배치
- `README.md` 는 소개와 빠른 시작만. 설치·동기화·테스트는 `docs/install.md`, 화면·도구·파이프라인 사용법은 `docs/manual.md`.
- 기능을 추가·변경하면 `docs/manual.md` 의 해당 절을 같이 고친다 (단축키 표, 인스턴스 연산 등).

## 테스트
- `tests/test_gui_smoke.py` 는 `MR_TEST_DS` 로 준 데이터셋을 임시 폴더에 복사해 돈다. 풀셋(2,400장)은 크므로 manifest 에서 제품 2종 6장씩 뽑은 축소본을 스크래치에 만들어 쓴다.
- 스모크의 꼭짓점 삽입·내보내기(확정 1장) 검사는 데이터셋 내용에 따라 실패할 수 있다. 실패가 변경과 무관한지는 변경 전 코드로 같은 축소본을 돌려 비교한다.
