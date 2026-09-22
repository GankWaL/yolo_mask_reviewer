# 사용 매뉴얼

설치·다른 PC 동기화는 [install.md](install.md). 데이터 규약: 원본 `labels/` 는 절대 고치지 않는다. 편집본은 `<데이터셋>/labels_reviewed/<stem>.txt`,
자동 라벨(SAM2 대표 전파·재학습 모델 재추론)은 `<데이터셋>/labels_auto/<stem>.txt`,
판정·메모·대표 여부·자동 라벨 메타는 `<데이터셋>/review_state.json` 에 쌓인다 (다시 열면 이어서 작업).
유효 라벨 우선순위: 편집본 > 자동 > 원본.

## 실행

```bash
conda activate yolo_mask_reviewer
cd ~/yolo_mask_reviewer
./run.sh                                   # 기본: ~/jhw/data/SL_under_predict
./run.sh /path/to/dataset                  # images/ labels/ 가 있는 폴더
python3 -m mask_reviewer stats  DATASET    # 검수 현황 (제품코드별 확정·자동·대표 수)
python3 -m mask_reviewer export DATASET OUT --status ok --val 0.1   # GUI 없이 내보내기
python3 -m mask_reviewer propagate DATASET [--codes A,B] [--max-exemplars 3]   # GUI 없이 SAM2 대표 전파
python3 -m mask_reviewer rescore DATASET   # 기존 자동 라벨의 원본 대비 변화량(diff) 소급 계산 (툴을 닫고 실행)
```

## 화면

- 왼쪽: 이미지 목록(썸네일). 제품코드·판정 상태(보류/확정/제외/편집됨/자동 라벨/대표)·파일명으로 필터. **제품의 ★ 대표: 있음/없음** 체크박스는 같은 12자리 제품코드(재지정 코드 기준)에 ★ 대표가 있는/없는 이미지만 보여 주고, 제품 콤보에는 코드마다 대표 수(★n, 없으면 —)와 요약(대표 있는 제품 a/b)이 표시된다. 오른쪽 정보에도 그 제품의 대표 장수가 보인다.
  정렬은 파일명 순 / 원본과 차이 큰 순 / 자동 점수 낮은 순 / 제품 코드 순. `✔` 확정, `✖` 제외, `·` 보류, `✎` 편집본, `⟳` 자동 라벨, `★` 대표.
  썸네일은 백그라운드로 만들어 `<데이터셋>/.thumb_cache/` 에 캐시하며 `보기 → 목록 썸네일 표시` 로 끌 수 있다.
- 가운데: 이미지 + 마스크 오버레이. 휠 확대, 휠클릭(또는 Ctrl+드래그) 이동, `F` 화면 맞춤.
- 오른쪽: manifest 정보(제품, reason, conf), 인스턴스 목록, 편집 버튼, 판정 버튼, 메모.
  이미지에 적힌 제품코드가 틀렸으면 `제품코드 변경…` 으로 고친다: 데이터셋에 있는 12자리 코드를 고르거나 직접 입력, `원본으로` 로 되돌림.
  왼쪽 목록에서 여러 장을 선택(Shift+드래그·드래그·Shift/Ctrl+클릭)한 뒤 누르면 선택 전부에 한 번에 적용된다 (`원본으로` 는 각자 자기 원본 코드로).
  고친 값은 `review_state.json` 의 `code` 에 저장되고(manifest.csv·파일명은 그대로) 제품 필터·정렬·대표 전파·내보내기(층화 분할, export_manifest.csv)·재검수 묶음에 모두 반영된다.

## 편집 도구

| 도구 | 키 | 동작 |
|---|---|---|
| 선택 | V | 마스크 클릭으로 인스턴스 선택 |
| 브러시 | B | 좌클릭 드래그 칠하기, 우클릭 지우기. 크기 `[` `]` 또는 Ctrl+휠 |
| 지우개 | E | 양쪽 버튼 모두 지우기 |
| 폴리곤 | P | 클릭으로 점 추가 → Enter / 더블클릭 / 첫 점 클릭 / 우클릭으로 닫아 채움. Ctrl+Enter 는 빼기, Backspace 점 취소, Esc 취소 |
| 완드 | M | 좌클릭: 비슷한 색 연결 영역 추가 (허용치는 툴바), 우클릭: 제거. 검정 캡·팁처럼 색이 균일한 누락 부위에 유용 |
| SAM2 | S | 좌클릭 전경 점, 우클릭 배경 점, Shift+드래그 박스 → 청록 제안 마스크. Enter 현재 인스턴스에 추가, Ctrl+Enter 빼기, Shift+Enter 교체, Backspace 마지막 점 취소, Esc 취소 |
| 꼭짓점 | T | 현재 인스턴스 외곽선을 점으로 표시. 점 드래그 이동, 변 위 클릭으로 점 추가, 점 우클릭 삭제 (놓을 때마다 마스크에 반영) |
| 자르기 | K | 마스크를 가로지르는 선을 클릭으로 찍고 Enter(우클릭·더블클릭도 확정) → 현재 인스턴스가 선 양쪽의 두 인스턴스로 나뉜다 (같은 클래스). 물체 2개가 한 마스크로 붙은 라벨용 |

인스턴스 연산 (현재 인스턴스 대상): 새 인스턴스 `N`, 삭제 `Del`, 다음 `Tab`, 클래스 `1`\~`9`, 마스크 비우기 `C`,
목록에서 여러 개 선택(Shift+드래그·드래그·Shift/Ctrl+클릭)하면 클래스 `1`\~`9`·**클래스 적용**·삭제 `Del` 이 선택 전부에 한 번에 적용된다 (되돌리기 한 번에 묶임),
목록에서 2개 이상 선택(Ctrl+클릭) 후 **선택 병합**(같은 물체에 박스 2개 유형), **조각 분리**(떨어진 조각을 각각 인스턴스로 — 물체 2개가 한 라벨인데 서로 떨어져 있을 때; 붙어 있으면 자르기 `K`),
**GrabCut 보정** `G`(현재 마스크를 씨앗으로 색 분리 재계산; 먼저 브러시로 대충 덮고 누르면 경계가 정리됨),
구멍 채우기, 최대 조각만, 가장자리 정리, 원본 라벨 복원, **확정/모델/CAD 구멍 빼기**(아래 "관통 구멍" 절). 되돌리기 Ctrl+Z / 다시 Ctrl+Shift+Z(Ctrl+Y).

인스턴스가 없는 이미지에서 브러시·폴리곤·완드·SAM2 를 쓰면 툴바 클래스 콤보의 클래스로 새 인스턴스가 자동 생성된다.

권장 흐름: 누락 부위가 크면 `S` 로 점 한두 개 찍고 Enter(추가) → `T` 로 경계 꼭짓점을 당겨 다듬기 → `Space` 확정.

## 검수 판정

- `Space` 확정 + 다음, `X` 제외 + 다음, `W` 보류, `R` 대표 지정/해제. 이동 `A`/`D`, PgUp/PgDn.
- 이미지를 옮기거나 판정하면 편집본이 자동 저장된다 (`Ctrl+S` 수동 저장).
- 마스크를 손대지 않고 확정만 해도 된다 (내보낼 때 원본 라벨을 그대로 복사).

## 관통 구멍 (2026-09-21)

YOLO-seg 폴리곤은 구멍을 직접 표현하지 못해 기존 내보내기는 바깥 윤곽만 남겼다(구멍이 채워짐). 이제 두 가지를 지원한다.

- **구멍 유지 저장·내보내기**: `파일 → 구멍 유지 저장·내보내기` 를 켜면(데이터셋 `dataset.yaml` 의 `keep_holes: true` 가 기본값)
  편집본·자동 라벨·내보내기 폴리곤이 바깥 윤곽에서 구멍 윤곽으로 폭 0 의 다리를 넣은 **한 폴리곤**으로 저장된다.
  cv2.fillPoly 와 ultralytics 학습 래스터화(짝홀 채움)가 구멍을 그대로 남긴다. 읽기는 어느 쪽 라벨이든 그대로 마스크가 된다.
  운영 모델을 채운 마스크로 유지하려면 끄고 내보낸다.
- **보조 구멍 레이어 `aux/<stem>.png`** (표시 전용, uint8): 1 확정 구멍(청록) / 2 모델만(자홍) / 3 CAD만(주황).
  `scripts/build_holes_review.py` 가 검수본에 만든다 — 구멍을 남기도록 학습한 YOLO26-seg 추론(②)과 운영 yaw 매처로
  실루엣 캐시를 정합한 CAD 구멍(③)을 교차해, 둘이 겹치는 것만 `labels/` 에 이미 빼 둔다(확정). 정합 점수가 운영 fitness
  문턱 미만(`cad_status low`)이거나 캐시가 없으면 확정 없이 모델 구멍만 자홍으로 남긴다.
  - 보기 `J` 로 레이어 표시 전환. 목록 정렬 **"구멍 불일치 큰 순"** 은 정합 실패 → 불일치 픽셀 많은 순.
  - 버튼 **확정 구멍 빼기 / 모델 구멍 빼기 / CAD 구멍 빼기** 로 현재 인스턴스에서 해당 영역을 뺀다(되돌리기 가능).
    잘못 뚫린 건 브러시로 메우거나 `구멍 채우기` 로 되돌린다. 오른쪽 정보에 확정/모델만/CAD만 픽셀과 정합 yaw·score 가 보인다.
  - 학습 PC 에서 만들기 (`yolo_mask_reviewer` 환경, 모델은 `~/jhw/data/SL/runs/codes_y26s_holes_*/weights/best.pt`):
    ```bash
    python scripts/build_holes_review.py ~/jhw/data/SL/reviewed_20260918 ~/jhw/data/SL/reviewed_20260919_valnew \
        --out ~/jhw/data/SL/holes_review_<날짜> --model <best.pt> --device 0
    scripts/review_sync.sh push-bundle ~/jhw/data/SL/holes_review_<날짜>     # 검수 PC 로
    ```
    SLIA `core_utils/yaw_match_pool.py` 를 GL 없이 import 해 쓰므로 `~/jhw/SL_Inspection_Automation` 체크아웃과 obj 캐시가 필요하다.
  - `--into` 를 주면 새 폴더를 만들지 않고 SRC(검수 데이터셋 1개)의 보류 프레임(대표·편집본·SAM2 구멍 전파 확정분 제외)에
    결과를 `labels_auto/`(source `model-cad-holes`)와 `aux/` 로 직접 넣고 manifest 에 cad_status 등을 합친다. 코드는 state 의
    재지정 코드를 쓴다. 끝나면 `python -m mask_reviewer rescore SRC` 로 원본 대비 diff 를 갱신한다.

### 구멍 있는 대표 → SAM2 전파 (`scripts/propagate_holes.py`)

대표(★)에 구멍까지 라벨했으면 같은 제품의 보류 프레임에 SAM2 쌍 전파로 구멍 있는 자동 라벨을 만든다 (GUI 의 `SAM2 대표 전파` 와
같은 전파기, 게이트와 대표 선택만 구멍용). 2026-09-21 실험: 구멍 있는 대표 6 제품 × 5장에서 30/30 장 구멍이 그대로 옮겨졌고
(바깥 IoU 0.937), 유일한 실패는 물체 2개 프레임에서 옆 물체로 번지는 점묘였다.

- 대표 선택: SAM2 객체 점수는 거의 전부 1.0 이라 변별력이 없다 → 전파 결과(채움)와 원본 라벨(채움)의 바깥 IoU 가 큰 대표 (GUI 전파도 같음).
- 게이트: 바깥 IoU ≥ 0.85, 전경 조각 ≤ 3(점묘), 대표 대비 면적비 0.4\~2.5, 구멍 수 ≤ 대표+3, 구멍 면적비 ≤ 0.6.
  통과하면 `auto-ok`(status ok), 아니면 보류에 사유 메모. 둘 다 `labels_auto/` 에 쓴다.
- `--exemplars-from DS …` 로 다른 데이터셋(예: holes_review_*)의 대표를 빌려 쓸 수 있고, `--holed-only` 면 구멍이 있는 대표만 쓴다
  (`keep_holes: false` 로 저장된 대표가 구멍 있는 대표를 밀어내지 않도록).

```bash
python scripts/propagate_holes.py ~/jhw/data/SL/holes_review_<날짜> --device cuda:0
python scripts/propagate_holes.py ~/jhw/data/SL/field_all_<날짜> --exemplars-from ~/jhw/data/SL/holes_review_<날짜> --holed-only
```

대표가 없는 제품은 `scripts/relabel_with_model.py DS --model <구멍 모델> --map-codes` 로 모델 추론을 자동 라벨로 넣는다
(`--map-codes`: 12자리 코드 클래스 모델의 검출을 obj 이름의 부품 종류로 대분류 5클래스에 맞춘다).

### 제품 코드 자동 재배열 (`scripts/reassign_codes.py`)

현장 json 의 제품 코드가 틀린 프레임을 찾아 고친다. 두 근거를 쓴다.

- ① 기준 프레임: 사람이 코드를 확정한 프레임(state 의 `code`, ★ 대표, 사람 ok)을 `--anchors DS …` 에서 모아 DINOv2 ViT-S/14
  물체 크롭 임베딩으로 가장 닮은 기준의 코드 A 를 찾는다. 기준끼리의 LOO 정확도 92% (2026-09-21, 913장 77코드). 1위 코드와
  2위 코드의 유사도 차이(`--margin` 0.05)가 작으면 기준이 없는 제품으로 보고 바꾸지 않는다.
- ② CAD 정합: 라벨 대분류와 같은 obj 전부를 운영 yaw 매처(캐시 전용, coarse 10°)로 정합해 점수 상위 3개를 기록한다.
- 판정: A == 현재 → confirmed. A ≠ 현재이고 margin 충분하고 CAD 가 A 를 현재보다 못하게 보지 않으면 → A 로 변경. CAD 가
  현재를 강하게 지지하면 유지 + conflict. `UNKNOWN_*` 프레임은 CAD 상위 3개와 fitness 문턱 통과 여부로 obj 존재를 판단한다.
- 결과는 `TARGET/reassign_report.csv`(stem, cur, anchor, sim, margin, cad top3, decision, new_code, flags). **코드 자동 반영은 기본
  금지**다 — 검수 후 안 바뀐 코드는 사람이 맞다고 본 것이고 문제 있는 제품은 제외로 처리하므로(2026-09-22). `--apply`/`--apply-report`
  는 `--force` 를 함께 줄 때만 동작하고, 그때도 사람이 바꾼 프레임은 건드리지 않는다. `push-ref` 의 3자 병합도 코드는 항상 검수 PC 값을 쓴다.
- 부모가 CUDA 를 초기화한 뒤 fork 하면 워커가 멈추므로 CAD 정합 풀은 spawn 으로 만든다. 3,600장 × 후보 88개에 16 워커로 약 1시간.

### 검수 PC 와 상태 주고받기 — 사람 판정 보호 (`review_sync.sh pull-ref/push-ref`, `merge_review_state.py`)

자동 처리(전파·재라벨·재배열) 전에 `pull-ref <이름>` 으로 검수 PC 상태를 가져오면 `review_state.json.pulled` 스냅샷이 남는다.
`push-ref <이름>` 은 보내기 전에 검수 PC 의 현재 상태를 다시 받아 3자 병합한다: 그 사이 사람이 바꾼 stem(status·code·★·메모)은
원격 것을 그대로 두고 자동 필드(auto·cross_iou·holes)만 얹는다. 어느 쪽이든 **제외(reject)인 stem 은 reject 로 남는다** — 자동
처리가 제외를 되살리지 않는다. 전파(`propagate_holes.py`, GUI 전파)도 보류 프레임만 대상으로 하고 reject 는 건드리지 않는다.

### 학습 데이터셋 정제 (`scripts/refine_dataset.py`, 2026-09-22)

제품이 **중앙 부근에 온전히, 단독으로** 있는 프레임만 학습에 쓴다. 프레임마다 유효 라벨의 가장 큰 인스턴스로 잰다.

- `multi` 라벨 인스턴스 2개 이상 · `border` 마스크가 크롭 경계(3px)에 20px 이상 닿음 · `partial` 경계 띠(6px)에 닿으면서 면적이
  같은 코드의 온전한 프레임 중앙값의 95% 미만 · `offcenter` 마스크 중심이 화면 중심에서 폭·높이의 25% 밖 ·
  `neighbor` 현장 YOLO(conf 0.1) 재검출에서 라벨과 안 겹치는 물체, 또는 벨트색(녹색 H 18\~75, S·V 중간)이 아닌 전경 덩어리
  (2,500px 이상, 라벨 15px 밖·벨트 중앙 띠 15\~85% 안 — 가장자리에 잘린 이웃 제품도 잡힘, 제품 그림자는 뺌).
- 면적 비율만으로는 자세(yaw)가 달라 정상 프레임까지 걸리므로 경계에 닿을 때만 쓴다. 벨트색은 조명 때문에 위치마다 달라
  통계 대신 고정 임계값을 쓴다.
- 결과 `refine_report.csv`. `--apply` 면 state 에 `train_exclude`·`refine_reasons` 를 쓴다(판정은 그대로). 내보내기는 `train_exclude`
  프레임을 기본으로 건너뛰고(`--keep-excluded` 로 포함), 목록 필터 `정제 제외(학습 미사용)` 로 툴에서 볼 수 있다.
- **어느 프레임이 학습에 들어갔는지**: 내보내기에 `--round <이름>` 을 주면(예 `codes_y26x_refined_20260922`) 내보낸 프레임의 state 에
  `train_round`·`train_split`(train/val)이 남고, 목록 필터 `최근 학습에 사용됨` 과 오른쪽 정보의 "학습 사용: …" 으로 구별된다.
  학습 뒤 보류 프레임을 그 모델로 재라벨한 것은 필터 `학습 모델 자동 라벨(보류)`(자동 라벨 출처 `yolo:*` 이고 보류)로 모아 본다.
  2026-09-22 field_data_hole_all: 5,357장 중 통과 3,559(이웃 1,793·중앙 이탈 34·경계 18·복수 3 제외).

### 여러 검수 데이터셋 모으기 (`scripts/gather_datasets.py`)

`--out OUT SRC …` 로 검수 데이터셋들을 하나로 합친다: 이미지 하드링크, **유효 라벨(편집본 > 자동 > 원본)을 원본 라벨로**, 재지정
코드·판정·대표·자동 라벨 메타를 state 에 유지, `aux/` 복사, manifest 에 source·code·status·label_src·has_holes. 같은 stem 이
여러 SRC 에 있으면 앞의 SRC 가 이기고 `--prefix` 면 SRC 이름을 붙여 모두 남긴다. YOLO 폴더(images/<split>)도 SRC 로 받는다.

## 순환 라벨링 (대표 1장 → 자동 라벨 → 검수 → 재학습)

제품마다 잘 맞는 라벨 한 장만 사람이 만들고, 나머지는 자동으로 채운 뒤 검수만 하는 흐름이다.

1. **대표 지정** — 제품별로 마스크가 정확한 이미지를 골라(필요하면 SAM2·브러시로 고쳐서) `R` 로 ★ 대표 지정.
   현재 라벨이 편집본으로 저장되고 확정된다. 자세가 다른 장이 있으면 제품당 2\~3장 지정해도 된다.
2. **SAM2 대표 전파** (`자동 → SAM2 대표 전파…`, Ctrl+P) — 대표 마스크를 같은 제품의 보류 이미지로 전파해
   `labels_auto/` 에 쓴다 (장당 약 0.3초, GPU). 대표가 여럿이면 각각 전파해 객체 점수가 가장 높은 결과를 쓴다.
   대표 1장 + 대상 1장을 두 프레임 영상으로 넣는 쌍 전파라서, 여러 장을 한 줄로 전파할 때 생기는 메모리 표류가 없다.
   원본 의사 라벨이 잘린 부위(밝은 캡 등)도 대표 모양을 따라 채워진다.
   대상은 "보류이고 편집본·대표가 아닌 이미지"다. 인스턴스를 모두 지운 뒤 넘어가면 **빈 편집본**이 남아 자동 라벨보다
   우선하므로 대상에서 빠지는데, 대화상자의 "비어 있는 편집본을 가진 보류 이미지도 포함" 을 켜면 그 편집본을 지우고 전파한다
   (CLI 는 `--include-empty-edited`).
3. **검수** — 목록 필터 `자동 라벨`, 정렬 `원본과 차이 큰 순` 으로 두고 위에서부터 확인. 자동 라벨마다 원본 의사 라벨과의
   차이(1−IoU)가 기록되며, SAM2 가 원본을 크게 고친 장(잘린 부위를 채웠거나, 잘못 잡았거나)이 앞에 온다. 원본과 거의 같은 장은
   화면에서도 달라 보이지 않는 것이 정상이다. SAM2 객체 점수는 거의 항상 0.99 이상이라 정렬 기준으로는 쓸모가 적다.
   맞으면 `Space`(자동 라벨이 그대로 확정), 틀리면 고친 뒤 `Space`(편집본이 자동 라벨을 덮음). 잘 고친 장은 `R` 로 대표에
   추가해 2 를 다시 돌린다. `자동 → 현재 이미지 자동 라벨 삭제` 로 원본으로 되돌릴 수 있다.
   이 기능이 들어가기 전에 만든 자동 라벨은 툴을 닫고 `python3 -m mask_reviewer rescore DATASET` 으로 차이를 소급 계산한다
   (툴이 열려 있으면 `review_state.json` 을 서로 덮어쓴다).
4. **내보내기** (Ctrl+E, 확정만, val 비율 0.1 이상) → **재학습** (같은 환경, install.md §3):
   ```bash
   conda activate yolo_mask_reviewer
   python scripts/train_round.py OUT_DIR                       # 기준: SL_Inspection_Automation/models/yolo11s_best_20260326.pt
   python scripts/train_round.py OUT_DIR --extra-data /path/원본학습셋/data.yaml --epochs 30   # 원본 학습셋과 함께 (권장)
   ```
   결과는 `runs/round_<일시>/yolo11s_best_20260326_round_<날짜>.pt`. 검수분만으로 학습하면 다른 제품을 잊을 수 있어
   백본 10개 층을 동결하고 lr 을 낮춰 두었지만, 원본 학습셋이 있는 PC 에서는 `--extra-data` 로 함께 학습한다.
   클래스 이름·순서가 기준 모델(b_cvr, cap, h_cvr, hsg, scalp)과 다르면 중단한다.
5. **재추론** — 새 모델로 아직 보류인 이미지의 자동 라벨을 갱신하고 3 으로 돌아간다:
   ```bash
   python scripts/relabel_with_model.py DATASET --model runs/round_.../yolo11s_best_20260326_round_YYYYMMDD.pt
   ```
   SAM2 전파 결과를 남기려면 `--keep-sam2`. 점수는 검출 conf 평균, 원본 대비 차이도 함께 기록되며 툴에서 `⟳` 와 `yolo:<모델>` 로 표시된다.
6. 라운드를 반복하다 검수 통과율이 충분해지면 round 가중치를 `SL_Inspection_Automation/models/` 에 넣고
   운영 코드의 `yolo_base_model` 을 바꾼다 (스크립트가 자동으로 복사하지 않는다).

## 내보내기 (Ctrl+E)

```
OUT/
  images/train, images/val     (val 비율 0 이면 images/ 평면 — 학습 서버에서 분할)
  labels/train, labels/val     편집본은 마스크→폴리곤(단순화 eps 0.7px, 16px² 미만 조각 버림), 미편집은 원본 그대로
  dataset.yaml                 path/train/val/names (names 는 원본 dataset.yaml 과 동일)
  export_manifest.csv          stem, code, split, status, edited, label_src(reviewed/auto/original), n_inst, source
```

- 포함 상태 기본 = 확정만. 라벨은 편집본 > 자동 > 원본 순으로 유효한 것을 쓴다. val 분할은 제품코드별 층화(seed 고정).
  이미지는 복사/하드링크/심볼릭링크 선택.
- 떨어진 조각이 여럿인 마스크는 ultralytics 방식으로 한 폴리곤에 잇는다(구멍은 무시).
- `dataset.yaml` 의 `path` 는 이 PC 절대경로다. 다른 PC 로 옮기면 경로를 고친다 ([install.md](install.md) §5).

## 원본 학습셋이 없을 때 — 의사 라벨 리허설 + 보수적 파인튜닝 (B+C, 2026-09-18)

`yolo11s_best_20260326.pt` 의 원본 학습셋(`large_sort_seg.yaml`)이 없으면 검수분만으로 학습했을 때 다른 제품·클래스를 잊는다.
대신 현장 성공 프레임을 기준 모델로 의사 라벨링해 "지금 잘하는 것" 을 함께 학습시키고(자기 증류), 게이트로 잊음을 잰다.

```bash
~/jhw/data/SL/fetch_field_data.sh 20260918          # 현장 PC 하루치 회수 (json → 로그 → raw → overlay, 재실행 시 이어받기)
conda activate yolo_mask_reviewer
python scripts/build_rehearsal_set.py ~/jhw/data/SL/field/20260918 --out ~/jhw/data/SL/rehearsal_20260918   # 리허설 + 게이트셋
FIELD=... REH=... REV=~/jhw/data/SL/reviewed_20260918 TAG=20260918 scripts/round_bc.sh                          # 학습 2종 + 게이트
```

- `build_rehearsal_set.py`: 성공 프레임(raw+json)을 제품별 시간 균등으로 `--per-code`(기본 60)장 뽑아 ROI 크롭·기준 모델 추론.
  top-1 conf < 0.2 이거나 json `large_sort` 와 클래스가 다른 프레임은 **제외**(빈 라벨은 배경 학습이 되므로 쓰지 않음),
  top-1 외 인스턴스는 conf ≥ 0.3 이고 겹치지 않을 때만. 두 톤 실패 제품(§21-bt)은 성공 프레임에도 누락 편향이 있어 기본 제외(`--exclude-codes`).
  `gate/` 는 학습에 쓰지 않는 제품별 `--gate-per-code`(기본 15)장 — 보존 게이트용.
- `train_round.py`: `--optimizer AdamW` 기본. ultralytics `auto` 는 lr0 를 무시하고 0.0011 로 바꾸므로 lr0 ≤ 1e-4 를 지키려면 명시해야 한다.
  `warmup_bias_lr` 0 (기본 0.1 은 사전학습 bias 를 흔든다).
- `eval_gates.py` → `gate_report_<TAG>.md`: ① 보존 = 게이트셋에서 새 모델 top-1 이 기준 모델과 IoU ≥ 0.95 인 비율(목표 99%)·클래스 일치,
  ② 회복 = 검수 val(사람 라벨) GT 인스턴스별 IoU(기준 vs 새), ③ 클래스별 mAP50(M) (리허설 val·검수 val, 하락 0.5pt 이내 목표).
- 한계: 현장에 **cap 제품이 없어 cap 은 리허설·게이트 모두 불가**. freeze·저 lr 로 드리프트를 줄일 뿐 cap 보존은 측정하지 못한다.

## 현장 상시 회수 → 자동 제외 → SAM2 대표 전파 오토라벨 (field_autolabel, 2026-09-19)

운영 모델이 웬만한 것은 잡는 단계에서, 현장 실패 프레임만 계속 가져와 사람 손 없이 학습 폴더까지 만드는 파이프라인이다.
`scripts/field_autolabel.py` (env `yolo_mask_reviewer`, install.md §2\~3 — SAM2 의 pip 이름은 `SAM-2` 다).

```bash
scripts/field_autolabel.sh                      # 한 사이클 (어제·오늘). flock 으로 중복 실행 방지, 로그 ~/jhw/data/SL/autolabel/logs/<날짜>.log
python scripts/field_autolabel.py run 20260919  # 날짜 지정 / fetch·collect·exclude·propagate·export·status·refit 단계별 실행
*/30 * * * * ~/jhw/yolo_mask_reviewer/scripts/field_autolabel.sh   # crontab (30분 주기)
```

1. **fetch** — 현장 PC `save_pose_debug/<날짜>/` 에서 실패 프레임만(코드 raw 인데 json 없음 = fitfail, `nodet_*` = 검출 없음) rsync. 인식기가 `<stem>_skip.json` 으로 표시한 프레임(언로딩 오류·CAP, SLIA §21-cc)은 받지 않는다.
   2분 이내 파일은 건너뛰고(처리 중), 받은 것은 다시 받지 않는다. 성공 프레임·overlay 는 받지 않는다 (하루 51GB → 실패분 약 5GB).
2. **collect** — `collect_yolo_hard_cases.py` 를 이 PC 에서 돌려(현장 GPU 를 쓰지 않음) 누적 데이터셋 `~/jhw/data/SL_under_predict_auto/` 에 추가.
   검수 PC 에서 이미 본 stem(`SL_under_predict/`·`field/SL_under_predict_field/` manifest)은 뺀다. 사이클당 제품별 `--max-per-code`(기본 30) + (날짜, 제품) 누적 상한 `AL_DAILY_CAP`(60, 시간 균등) — 시간 균등 샘플이 사이클마다 달라져 무한히 늘지 않게.
3. **exclude** — 검수 PC 판정(`~/jhw/data/SL_under_predict/review_state.json` 의 ok 2,179 / reject 177)으로 학습한 로지스틱 분류기.
   특징은 **DINOv2 ViT-S/14 임베딩(전체 이미지 384 + 가장 큰 인스턴스 크롭 384)** + n_det·conf·fill·가장자리 걸침 — 현장 YOLO 임베딩은 뒤집힌 제품·상자를 거의 못 갈랐다
   (코드 있는 reject 재현율 0.19 → 0.81). 5-fold 교차검증에서 reject 정밀도 ≥ 0.9 가 되는 임계값(0.50)만 자동 제외: `none`(가장자리 잘림·빈 벨트) 97/99,
   `gate` 21/21, 코드 있는 fitfail(뒤집힘·상자·다른 부품) 46/57. 나머지는 보류로 사람에게. 보고 `autolabel/exclude_clf_report.md`, 재학습 `refit`.
   **뒤집힌 제품은 자세 추정·인식 결과를 오염시키므로 학습에서 뺀다** (2026-09-19 지시) — reject 기준에 포함돼 있다.
4. **propagate** — 대표 GT = 검수자가 ★ 로 지정한 프레임(편집본 우선) + ★ 없는 제품은 확정 export 에서 보충 → `~/jhw/data/SL/exemplars/`.
   잘못 지정된 ★(뒤집힌 제품 2장, 캡 gate 프레임 2장)은 `AL_EXEMPLAR_DROP` 로 뺀다(검수 PC 에서 ★ 를 고치면 비운다).
   **대표 선택은 코드가 아니라 외형(DINOv2 전체 이미지 코사인 유사도)으로 한다** — 현장 코드는 언로딩 방향·오인식으로 섞여 있어(10H23x/24x 6종, 10J311/312 등) 믿을 수 없다.
   같은 6자리 코드 대표와 유사도 ≥ 0.5 이면 그중 상위 3장, 아니면 전체 대표에서 상위 3장. 다른 코드 대표가 같은 코드 최고보다 0.1 이상 더 비슷하면 `code-mismatch(<코드>)` 로 표시(라벨은 닮은 대표의 클래스로 — 프레임에 실제로 있는 물체 기준).
   인스턴스 수가 비슷한 순으로 SAM2 쌍 전파해 객체 점수 최고를 채택. 확정 조건: 점수 ≥ 0.9, 후보 YOLO(`retrain/current.pt`) 와 클래스 무관 합집합 IoU ≥ 0.6(검출 없으면 생략), 대표 인스턴스 대비 면적 비율 0.4\~2.5(캡 인스턴스가 하우징에 얹히는 표류 방지).
   클래스는 사람이 만든 대표를 믿는다(후보 YOLO 클래스와 교차 확인하면 실패 프레임의 정상 마스크가 대거 보류됨). 대표·검출 모두 없으면 pending(auto-none).
5. **export** — ok 만 `~/jhw/data/SL_under_predict_auto_yolo/` 로 (하드링크, 제품 층화 val 0.1, 매번 새로). 학습은 `train_round.py ... --extra-data` 로 섞는다.

- 누적 데이터셋은 이 툴 형식 그대로라 GUI 로 열어 자동 판정을 뒤집을 수 있다 (메모 `auto-ok …`/`auto-exclude p=…`/`auto-pending …`).
  사람이 정한 상태·편집본은 다음 사이클에서 건드리지 않는다 (상태 없는 새 이미지만 판정).
- 사이클 요약 `autolabel/cycles.csv`, 현황 `python scripts/field_autolabel.py status`. 규칙·분류기·대표를 바꾼 뒤에는 `redo` 로 파이프라인 판정분(사람 판정 제외)을 전부 다시 판정한다.
- **재검수 묶음** `scripts/make_review_bundle.py` → `~/jhw/data/SL/for_review_<날짜>/` (툴 형식): A 코드 혼동 제품군의 확정 프레임·★, B 뺀 ★, C 코드불일치 표본(쌍당 6장)·코드 있는 자동 제외·보류.
  검수 PC 와는 `scripts/review_sync.sh` 로 학습 PC 에서 직접 주고받는다(SSH 별칭 `REVIEW_PC`, 2026-09-20): `push-bundle <묶음>` → 검수 PC 에서 `./run.sh` 로 ★ 재지정·판정 →
  `pull-bundle <이름>` → `scripts/apply_review_bundle.py <묶음>`(검수셋·자동셋에 반영) → `exemplars/` 삭제 → `refit` → `redo` → `push-ref`(검수 PC 원본을 이 PC 상태로 맞춤).
- 한계: 대표 1장에 인스턴스 1개면 화면 가장자리의 두 번째 물체는 라벨되지 않는다(배경 학습). 제외 분류기는 `none` 유형 외에는 보수적이라 보류가 쌓이면 GUI 로 정리한다.

### 제품코드별 상한 채우기 (`newcodes`, 2026-09-22)

`run` 은 fetch 다음에 `newcodes` 를 돈다: 현장 `save_pose_debug/<날짜>/` 의 **인식 성공 프레임**(json 있음, 인식기가 `_skip.json` 으로
수집 제외 표시한 프레임은 뺌) 파일명에서 12자리 코드를 뽑아, 코드마다 지금까지 모은 수 — `AL_KNOWN_DS`(기본
`field_data_hole_all_20260921` + `SL_under_predict`) 와 DS 자체를 합쳐 재지정 코드 기준으로 세고, reject 와 데이터셋끼리 겹치는 stem 은
빼거나 한 번만 — 가 **`AL_NEWCODE_CAP`(60) 미만이면 부족분만큼** 시간 균등으로 raw+json 을 회수한다. 미수집 코드(보유 0)와 60장 미만
코드를 같은 규칙으로 채우므로, 이미 모은 코드도 60장이 될 때까지 계속 모인다. json 의 런타임 마스크(`mask_rle`)를 ROI 로 잘라
라벨(대분류 = json `large_sort`)로 써서 DS 에 넣는다 (reason `newcode`, 보류, 메모 "★ 대표 필요"). 검수 PC 에서 이 프레임들로 ★ 대표를
만들면 이후 전파에 쓰인다. 누적 수집 수는 `$AL_STATE/newcodes.json`(코드별 n·first·last).
`python scripts/field_autolabel.py newcodes [날짜...]` 로 단독 실행할 수 있다.

주기: 현장 PC 의 `save_pose_debug` 는 50GB 상한에서 오래된 파일부터 지워지는데 2026-09-22 실측으로 시간당 raw 약 1,400\~1,600장
(프레임당 raw+png+json 약 6.4MB)이라 **보존 창이 약 6시간**이다. 30분 cron(사이클 약 2분)이면 한 사이클을 건너뛰어도 사라지기 전에
본다. 보존 창이 30분 근처로 줄면(촬영량 증가·상한 축소) cron 간격을 줄여야 한다.

## 자동 재학습 (auto_retrain, 4시간 cron, 2026-09-19)

field_autolabel 이 모은 ok 데이터로 직전 기준 모델에서 이어 파인튜닝하고, 게이트를 통과하면 다음 기준 모델로 승격한다 (`scripts/auto_retrain.py`, env `yolo_mask_reviewer`).

```bash
scripts/auto_retrain.sh                    # 한 번 (새 ok 데이터가 20장 미만이면 건너뜀)
python scripts/auto_retrain.py --force     # 데이터 변화 없어도 학습
python scripts/auto_retrain.py --status    # current.pt 와 이력
0 */4 * * * ~/jhw/yolo_mask_reviewer/scripts/auto_retrain.sh    # crontab
```

1. **스냅샷** — `SL_under_predict_auto_yolo/` 를 field_autolabel 잠금(flock) 아래 하드링크 복사해 학습 중 덮어쓰기를 막는다. stem 목록 해시가 직전과 같으면 건너뜀.
2. **학습** — `train_round.py`: 검수 확정 `SL_under_predict_yolo1` + 리허설(18일·신규 제품) + 스냅샷, base = `~/jhw/data/SL/retrain/current.pt`
   (처음엔 `runs/round_20260919_b0918`), freeze 0 · AdamW lr0 1e-4 · 15 epoch. 결과 `runs/auto_<TAG>/`.
3. **게이트** — `eval_gates.py` base = current.pt 대비: ① 18일 게이트셋 보존(top-1 IoU≥0.95 비율) ≥ 0.90, ② 19일 신규 val 87장 평균 IoU 가 base 보다 0.01 이상 안 떨어짐.
4. **승격** — 통과하면 `current.pt` 심볼릭링크를 새 best.pt 로 바꾼다. 다음 재학습의 base 이자 field_autolabel 의 후보 YOLO(`AL_CAND_MODEL` 기본)가 된다.
   실패하면 유지. **현장 적용(models/ 등록, main() 교체)은 자동으로 하지 않는다** — `retrain/history.csv` 와 `gate_<TAG>.md` 를 보고 사람이 결정.

- 이력 `~/jhw/data/SL/retrain/history.csv` (tag, base, 자동 라벨 수, 게이트 값, promoted), 학습 로그 `train_<TAG>.out`, 래퍼 로그 `retrain/logs/<날짜>.log`.
- 리허설 세트는 고정(18일 + 19일 신규 제품)이다. 제품군이 크게 바뀌면 `build_rehearsal_set.py` 로 새 리허설을 만들고 `RT_REH` 에 추가한다.
- 임계값·경로는 환경변수(`RT_GATE1_MIN`, `RT_GATE2_DROP`, `RT_MIN_NEW`, `RT_REH`, `RT_INIT_BASE` …)로 바꾼다.
