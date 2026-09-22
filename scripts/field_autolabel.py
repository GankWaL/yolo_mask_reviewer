#!/usr/bin/env python3
"""현장 미탐 프레임 상시 회수 → 자동 제외 → SAM2 대표 전파 오토라벨 → 학습 폴더 내보내기 (2026-09-19).

한 사이클(`run`):
  1. fetch     현장 PC save_pose_debug/<날짜>/ 에서 실패 프레임만 골라 회수 (fitfail = 코드 raw 인데 json 없음, nodet_* = 검출 없음 캡처).
               <stem>_skip.json 이 있는 프레임(인식기가 언로딩 오류·CAP 으로 표시, SLIA §21-cc)은 받지 않는다.
               2분 이내 파일은 아직 처리 중일 수 있어 건너뛴다. 이미 받은 파일은 건너뛴다.
  1b. newcodes 인식 성공 프레임(json 있음, _skip.json 없음)의 12자리 제품코드마다 지금까지 모은 수(AL_KNOWN_DS + DS, reject 제외,
               재지정 코드 기준)가 AL_NEWCODE_CAP(60) 미만이면 부족분만큼 시간 균등으로 회수하고, json 의 런타임 마스크(mask_rle)를
               ROI 로 잘라 라벨로 써서 DS 에 넣는다(reason newcode, 보류 상태 — 검수 PC 에서 ★ 대표를 만들어야 함).
               미수집 코드와 60장 미만 코드를 같은 규칙(부족분 = 상한 − 보유)으로 채운다. 누적 수는 $AL_STATE/newcodes.json.
               현장 json 의 코드는 **원래 코드 → 검수 재지정 코드 대응표**(code_alias: 검수 데이터셋에서 사람이 바꾼 코드를 집계,
               $AL_STATE/code_alias.json, 수동 code_alias_manual.json 우선)를 거쳐 세고 저장한다 (`codes` 명령으로 표 확인·DS 적용).
  2. collect   새로 받은 프레임만 심볼릭링크 스테이징에 모아 collect_yolo_hard_cases.py 로 누적 데이터셋(DS)에 추가
               (ROI 크롭 + 현장 모델 의사 라벨 + manifest). 사이클당 제품별 --max-per-code, (날짜, 제품) 누적 상한 AL_DAILY_CAP(60).
  3. exclude   기준 데이터(현장 manifest 2,423 = 확정 export 2,179 + 검수에서 빠진 244)로 학습한 제외 분류기
               (현장 모델 임베딩 512 + manifest 수치 + 가장자리 걸침)로 새 이미지를 판정 → 확신하는 것만 reject.
  4. propagate 제품별 대표 GT(확정 export 에서 제품코드별 최대 5장, 사람 편집본 우선)를 같은 12자리 코드(없으면 6자리) 의
               보류 이미지로 SAM2 쌍 전파 → labels_auto/. 대표가 없는 코드(none/gate 등)는 후보 YOLO 모델 추론으로 대체.
               SAM2 결과는 후보 YOLO 추론과 합집합 IoU 로 교차 확인해 어긋나면 보류로 남긴다.
               점수·교차확인을 통과하면 status ok(메모 auto-ok), 아니면 pending(사람 검수 필요).
  5. export    status ok 만 YOLO 학습 폴더로 (하드링크, 제품 층화 val 0.1). 매번 새로 만든다.

  conda activate pro
  python scripts/field_autolabel.py run                 # 오늘·어제 날짜
  python scripts/field_autolabel.py run 20260919        # 날짜 지정
  python scripts/field_autolabel.py refit               # 제외 분류기 재학습 (기준 데이터가 바뀌었을 때)
  python scripts/field_autolabel.py fetch|collect|exclude|propagate|export [날짜...]
  python scripts/field_autolabel.py status              # 누적 현황

경로는 환경변수로 바꿀 수 있다 (아래 CFG). 결과·로그 폴더: $AL_STATE (기본 ~/jhw/data/SL/autolabel).
검수 GUI(yolo_mask_reviewer)로 DS 를 열면 reject/ok/보류·자동 라벨·메모가 그대로 보이고, 사람이 뒤집을 수 있다.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
sys.path.insert(0, TOOL)
HOME = os.path.expanduser('~')
E = os.environ.get
CFG = dict(
    host=E('AL_HOST', 'dxr@100.118.154.63'),
    remote=E('AL_REMOTE', 'SL_Inspection_Automation/save_pose_debug'),
    field_dir=E('AL_FIELD', f'{HOME}/jhw/data/SL/field'),
    ds=E('AL_DS', f'{HOME}/jhw/data/SL_under_predict_auto'),
    export=E('AL_EXPORT', f'{HOME}/jhw/data/SL_under_predict_auto_yolo'),
    state_dir=E('AL_STATE', f'{HOME}/jhw/data/SL/autolabel'),
    ref_field=E('AL_REF_FIELD', f'{HOME}/jhw/data/SL/field/SL_under_predict_field'),
    ref_ok=E('AL_REF_OK', f'{HOME}/jhw/data/SL_under_predict_yolo1'),
    ref_ds=E('AL_REF_DS', f'{HOME}/jhw/data/SL_under_predict'),          # 검수 PC 데이터셋 사본 (review_state.json, labels_reviewed/, labels_auto/)
    ref_state=E('AL_REF_STATE', f'{HOME}/jhw/data/SL_under_predict/review_state.json'),   # 검수 PC 의 실제 판정 (있으면 우선)
    exemplars=E('AL_EXEMPLARS', f'{HOME}/jhw/data/SL/exemplars'),
    collector=E('AL_COLLECTOR', f'{HOME}/jhw/SL_Inspection_Automation/core_utils/collect_yolo_hard_cases.py'),
    field_model=E('AL_FIELD_MODEL', f'{HOME}/jhw/SL_Inspection_Automation/models/yolo11s_best_20260918.pt'),
    cand_model=E('AL_CAND_MODEL', f'{HOME}/jhw/data/SL/retrain/current.pt'
                 if os.path.lexists(f'{HOME}/jhw/data/SL/retrain/current.pt')
                 else f'{HOME}/jhw/data/SL/runs/round_20260919_b0918/weights/best.pt'),   # auto_retrain 이 승격한 최신 모델
    sam_ckpt=E('AL_SAM_CKPT', f'{TOOL}/checkpoints/sam2.1_hiera_tiny.pt'),
    roi=E('AL_ROI', '605,190,685,685'),
    # 잘못 지정된 ★ (뒤집힌 제품 2, 캡 gate 프레임 2) — 검수 PC 에서 ★ 를 고치면 비운다
    exemplar_drop=[x for x in E('AL_EXEMPLAR_DROP', '').split(',') if x],   # 2026-09-19 재검수 반영 후 비움 (검수 PC 가 ★ 를 직접 정리)
    device=E('AL_DEVICE', '1'),   # CUDA_DEVICE_ORDER=PCI_BUS_ID 기준 1 = RTX 4070 Ti (0 = PRO 5000 은 다른 사용자, 2026-09-22)
    known_ds=[x for x in E('AL_KNOWN_DS', f'{HOME}/jhw/data/SL/field_data_hole_all_20260921,{HOME}/jhw/data/SL_under_predict').split(',') if x],
    newcode_cap=int(E('AL_NEWCODE_CAP', '60')),
)
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=20']
CODE_RE = re.compile(r'^\d{5}_([0-9A-Z]{12})_raw\.png$')
SAM_OK_SCORE = 0.9      # SAM2 객체 점수 이상이면 확정 후보
YOLO_OK_CONF = 0.6      # 대표 없는 코드: 후보 YOLO 평균 conf 이상이면 확정 후보
CROSS_IOU = 0.6         # SAM2 ↔ 후보 YOLO 합집합 IoU(클래스 무관) 이하이면 보류
SIZE_RATIO = (0.4, 2.5) # SAM2 인스턴스 면적 / 대표 인스턴스 면적 허용 범위 (벗어나면 표류로 보고 보류)
SIM_SAME_MIN = 0.5      # 같은 6자리 코드 대표와의 DINOv2 유사도가 이보다 낮으면 전체 대표에서 고른다
SIM_CODE_MARGIN = 0.1   # 다른 코드 대표가 같은 코드 최고보다 이만큼 더 비슷하면 '코드 불일치'(언로딩 오류·오인식 의심) 표시
DAILY_CAP = int(E('AL_DAILY_CAP', '60'))   # (날짜, 제품코드) 별 누적 상한 — 시간 균등 샘플이 사이클마다 달라져도 하루 60장을 넘지 않게
EXCL_PRECISION = 0.9    # 제외 분류기 임계값: 교차검증에서 reject 정밀도가 이 값 이상이 되는 가장 낮은 임계값


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def today_dates(args_dates):
    if args_dates:
        return args_dates
    t = dt.date.today()
    return [(t - dt.timedelta(days=1)).strftime('%Y%m%d'), t.strftime('%Y%m%d')]


# ----------------------------------------------------------------------------- 1. fetch
def fetch(dates):
    """실패 프레임(fitfail raw, nodet raw+json)만 회수. 반환: 새로 받은 raw 경로 목록(로컬)."""
    field = CFG['field_dir']
    new_files = []
    for d in dates:
        cmd = SSH + [CFG['host'], f"cd {CFG['remote']}/{d} 2>/dev/null && find . -type f -mmin +2 \\( -name '*_raw.png' -o -name '*.json' \\)"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not r.stdout:
            log(f'fetch {d}: 원격 폴더 없음/접속 실패 ({r.stderr.strip()[:80]})')
            continue
        files = set(l.strip()[2:] for l in r.stdout.splitlines() if l.strip())
        want = []
        n_skip = 0
        for f in sorted(files):
            base = os.path.basename(f)
            if not base.endswith('_raw.png'):
                continue
            stem = base[:-8]
            js = os.path.join(os.path.dirname(f), stem + '.json')
            if os.path.join(os.path.dirname(f), stem + '_skip.json') in files:   # 인식기가 수집 제외로 표시 (언로딩 오류·CAP, §21-cc)
                n_skip += 1
                continue
            if stem.startswith('nodet_'):
                want.append(f)
                if js in files:
                    want.append(js)
            elif CODE_RE.match(base) and js not in files:
                want.append(f)
        todo = [f for f in want if not os.path.exists(os.path.join(field, d, 'save_pose_debug', f))]
        log(f'fetch {d}: 원격 raw {sum(f.endswith("_raw.png") for f in files)} → 실패 프레임 {sum(f.endswith("_raw.png") for f in want)} (수집 제외 표시 {n_skip}), 새로 받을 파일 {len(todo)}')
        if not todo:
            continue
        dest = os.path.join(field, d, 'save_pose_debug')
        os.makedirs(dest, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix='al_fetch_')
        chunks = [todo[i::4] for i in range(4)]
        procs = []
        for i, c in enumerate(chunks):
            if not c:
                continue
            lp = os.path.join(tmp, f'c{i}')
            with open(lp, 'w') as f:
                f.write('\n'.join(c) + '\n')
            procs.append(subprocess.Popen(['rsync', '-a', '--partial', '--ignore-missing-args', f'--files-from={lp}',
                                           '-e', ' '.join(SSH), f"{CFG['host']}:{CFG['remote']}/{d}/", dest + '/'],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
        for p in procs:
            _, err = p.communicate()
            if p.returncode not in (0, 24):
                log(f'  rsync rc={p.returncode}: {err.strip()[:120]}')
        shutil.rmtree(tmp, ignore_errors=True)
        got = [os.path.join(dest, f) for f in todo if f.endswith('_raw.png') and os.path.exists(os.path.join(dest, f))]
        log(f'  받음 raw {len(got)}')
        new_files += got
    return new_files


# ----------------------------------------------------------------------------- 1b. newcodes
JSON_RE = re.compile(r'^(.+)/(\d{5})_([0-9A-Z]{12})\.json$')
SORT_TO_CLS = {'SCALP': 'scalp', 'HSG': 'hsg', 'H/CVR': 'h_cvr', 'B/CVR': 'b_cvr', 'CAP': 'cap'}


def code_counts(ds):
    """지금까지 모은 데이터셋(AL_KNOWN_DS)과 DS 자체의 12자리 코드별 프레임 수.
    재지정 코드 기준, reject 는 빼고, 데이터셋끼리 겹치는 stem 은 한 번만 센다."""
    from mask_reviewer.dataset import Dataset, is_full_code
    stems = collections.defaultdict(set)
    for root in CFG['known_ds'] + [ds.root]:
        if not os.path.isdir(os.path.join(root, 'images')):
            continue
        try:
            d = ds if os.path.abspath(root) == os.path.abspath(ds.root) else Dataset(root)
        except Exception as e:
            log(f'newcodes: {root} 열기 실패 ({e})')
            continue
        for s in d.stems:
            c = d.code(s)
            if is_full_code(c) and d.state.status(s) != 'reject':
                stems[c].add(s)
    return {c: len(v) for c, v in stems.items()}


def code_alias(ds, min_n=3, min_share=0.8):
    """원래(현장 json) 코드 → 검수 재지정 코드 대응표. AL_KNOWN_DS(검수된 기준 데이터셋)의 state 에서 사람이 코드를 바꾼 프레임을 세어,
    한 원래 코드의 재지정이 min_n 장 이상이고 최다 코드 비율이 min_share 이상이면 채택. $AL_STATE/code_alias_manual.json
    ({원래: 재지정}) 이 있으면 그것이 우선. 결과는 $AL_STATE/code_alias.json 에 남긴다 (읽기용)."""
    from mask_reviewer.dataset import Dataset, is_full_code, code_from_stem
    # 원래 코드 = 파일명(현장 json) 코드. 비율 분모는 그 원래 코드의 전체 프레임(안 바꾼 것 포함, reject 제외) —
    # 일부 프레임만 바꾼 것(LH/RH 개별 교정 등)은 대응표가 아니다. 서로 맞바꾼 쌍(A→B 와 B→A)도 대응표가 아니다.
    tally = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    seen = set()
    for root in CFG['known_ds']:          # 검수된 기준 데이터셋만 (DS 자체의 미검수 자동 수집분은 분모에 넣지 않는다)
        if not os.path.isdir(os.path.join(root, 'images')):
            continue
        try:
            d = ds if os.path.abspath(root) == os.path.abspath(ds.root) else Dataset(root)
        except Exception:
            continue
        for s_ in d.stems:
            if s_ in seen or d.state.status(s_) == 'reject':
                continue
            seen.add(s_)
            orig = code_from_stem(s_)
            if not is_full_code(orig):
                continue
            total[orig] += 1
            new = d.code(s_)
            if is_full_code(new) and new != orig:
                tally[orig][new] += 1
    alias = {}
    for orig, cnt in tally.items():
        new, n = cnt.most_common(1)[0]
        if n >= min_n and n / max(total[orig], 1) >= min_share:
            alias[orig] = new
    for a_, b_ in list(alias.items()):
        if alias.get(b_) == a_:      # 맞바꾼 쌍 제거
            alias.pop(a_, None)
            alias.pop(b_, None)
    mp = os.path.join(CFG['state_dir'], 'code_alias_manual.json')
    if os.path.isfile(mp):
        alias.update({k: v for k, v in json.load(open(mp)).items() if is_full_code(k) and is_full_code(v)})
    os.makedirs(CFG['state_dir'], exist_ok=True)
    json.dump(dict(sorted(alias.items())), open(os.path.join(CFG['state_dir'], 'code_alias.json'), 'w'), ensure_ascii=False, indent=1)
    return alias


def apply_alias(ds, alias):
    """DS 에서 사람이 코드를 안 바꾼 프레임 중 원래 코드가 대응표에 있으면 재지정 코드로 (state code). 반환: 바꾼 수."""
    n = 0
    for s_ in ds.stems:
        if ds.state.code(s_):
            continue
        new = alias.get(ds.orig_code(s_))
        if new and ds.set_code(s_, new):
            n += 1
    return n


def uniform_pick(seq, n):
    if n <= 0 or not seq:
        return []
    if len(seq) <= n:
        return list(seq)
    idx = (np.arange(n) * (len(seq) / n)).astype(int)   # 간격 >= 1 이라 항상 n 개가 서로 다르다 (round 는 n 이 len 에 가까우면 겹쳐 모자랐음)
    return [seq[i] for i in idx.tolist()]


def newcodes(dates, ds):
    """제품코드별 상한 채우기: 성공 프레임(json 있음, _skip.json 없음)의 코드마다 보유 수(code_counts)가 AL_NEWCODE_CAP 미만이면
    부족분만큼 시간 균등으로 회수 → ROI 크롭 + 런타임 마스크(mask_rle) 라벨 → DS. 미수집 코드도 이미 모은 코드도 같은 규칙.
    반환: 추가한 stem 수."""
    from mask_reviewer import maskops
    sys.path.insert(0, HERE)
    from build_record_yolo import rle_decode
    have = code_counts(ds)
    alias = code_alias(ds)
    cap = CFG['newcode_cap']
    cnt_path = os.path.join(CFG['state_dir'], 'newcodes.json')
    counts = json.load(open(cnt_path)) if os.path.isfile(cnt_path) else {}
    rx, ry, rw, rh = [int(v) for v in CFG['roi'].split(',')]
    name_to_id = {v: k for k, v in ds.names.items()}
    rows = load_manifest(ds.root)
    existing = {r['stem'] for r in rows} | set(ds.stems)
    n_added = 0
    found = {}
    for d in dates:
        # 프레임 json 은 2분 이내면 아직 쓰는 중일 수 있어 제외. _skip.json(인식기의 수집 제외 표시, §21-cc/§25)은
        # 프레임보다 늦게 비동기로 생기므로 시간 조건 없이 전부 본다.
        cmd = SSH + [CFG['host'], f"cd {CFG['remote']}/{d} 2>/dev/null && "
                     f"find . -type f \\( -name '*_skip.json' -o \\( -mmin +2 -name '*.json' \\) \\)"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and not r.stdout:
            log(f'newcodes {d}: 원격 폴더 없음/접속 실패')
            continue
        skip = set()
        by_code = collections.defaultdict(list)
        for l in r.stdout.splitlines():
            rel = l.strip()
            rel = rel[2:] if rel.startswith('./') else rel
            if rel.endswith('_skip.json'):
                skip.add(rel[:-len('_skip.json')])
                continue
            m = JSON_RE.match(rel)
            if not m or os.path.basename(m.group(1)).startswith('nodet_'):
                continue
            by_code[alias.get(m.group(3), m.group(3))].append(rel)   # 현장 코드 → 재지정 코드 (대응표)
        short = {c: cap - have.get(c, 0) for c in by_code if have.get(c, 0) < cap}
        if not short:
            log(f'newcodes {d}: 성공 프레임 코드 {len(by_code)}종, 전부 상한({cap}) 충족')
            continue
        picks = {}
        n_skip = 0
        for c in sorted(short):
            cand = []
            for js in sorted(by_code[c]):
                if js[:-5] in skip:
                    n_skip += 1
                    continue
                if f'{d}_{os.path.basename(os.path.dirname(js))}_{os.path.basename(js)[:-5]}' in existing:
                    continue
                cand.append(js)
            picks[c] = uniform_pick(cand, short[c])
        want = [p for pk in picks.values() for js in pk for p in (js, js[:-5] + '_raw.png')]
        detail = ' '.join(f'{c}:{have.get(c, 0)}+{len(picks[c])}' for c in sorted(short))
        log(f'newcodes {d}: 성공 프레임 코드 {len(by_code)}종 중 상한({cap}) 미달 {len(short)}종 → 회수 {len(want) // 2}장 '
            f'(skip 표시 제외 {n_skip}) [{detail}]')
        if not want:
            continue
        dest = os.path.join(CFG['field_dir'], d, 'save_pose_debug')
        os.makedirs(dest, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix='al_newcode_')
        lp = os.path.join(tmp, 'files')
        with open(lp, 'w') as f:
            f.write('\n'.join(want) + '\n')
        p = subprocess.run(['rsync', '-a', '--partial', '--ignore-missing-args', f'--files-from={lp}', '-e', ' '.join(SSH),
                            f"{CFG['host']}:{CFG['remote']}/{d}/", dest + '/'], capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
        if p.returncode not in (0, 24):
            log(f'  rsync rc={p.returncode}: {p.stderr.strip()[:120]}')
        for c, pk in picks.items():
            for js in pk:
                jp = os.path.join(dest, js)
                ip = jp[:-5] + '_raw.png'
                if not (os.path.isfile(jp) and os.path.isfile(ip)):
                    continue
                run_dir = os.path.basename(os.path.dirname(js))
                stem = f'{d}_{run_dir}_{os.path.basename(js)[:-5]}'
                if stem in existing:
                    continue
                try:
                    meta = json.load(open(jp))
                    img = cv2.imread(ip)
                    m = rle_decode(meta['mask_rle'])
                except Exception as e:
                    log(f'  {stem}: json/마스크 읽기 실패 ({e})')
                    continue
                crop = img[ry:ry + rh, rx:rx + rw]
                mc = m[ry:ry + rh, rx:rx + rw] > 0
                cls = name_to_id.get(SORT_TO_CLS.get(str(meta.get('large_sort', '')).upper(), ''), None)
                if cls is None or mc.sum() < 64:
                    continue
                line = maskops.mask_to_yolo_line(cls, mc, rw, rh, keep_holes=ds.keep_holes)
                if not line:
                    continue
                cv2.imwrite(os.path.join(ds.root, 'images', stem + '.png'), crop)
                with open(os.path.join(ds.root, 'labels', stem + '.txt'), 'w', encoding='utf-8') as f:
                    f.write(line + '\n')
                orig = os.path.basename(js)[6:18]
                rows.append(dict(stem=stem, source=f'save_pose_debug/{d}/{js[:-5]}_raw.png', code=orig, reason='newcode', n_det=1,
                                 top_cls=ds.names[cls], top_conf=round(float(meta.get('fitness', 0) or 0), 3),
                                 top_maskpx=int(mc.sum()), top_fill=''))
                note = 'newcode 자동 수집 — 제품코드별 상한 채우기, ★ 대표 필요'
                if orig != c:
                    note += f' | 현장 코드 {orig} → 대응표 {c}'
                ds.state.set(stem, status='pending', note=note)
                if orig != c:
                    ds.set_code(stem, c)
                existing.add(stem)
                have[c] = have.get(c, 0) + 1
                e = counts.setdefault(c, {'n': 0, 'first': d})
                e['n'] += 1
                e['last'] = d
                found[c] = found.get(c, 0) + 1
                n_added += 1
    if n_added:
        write_manifest(ds.root, rows)
        ds.stems = sorted(set(ds.stems) | existing)   # 같은 프로세스의 다음 단계가 새 stem 을 보도록
        ds._img_path.update({s_: os.path.join(ds.root, 'images', s_ + '.png') for s_ in existing if s_ not in ds._img_path})
    os.makedirs(CFG['state_dir'], exist_ok=True)
    json.dump(counts, open(cnt_path, 'w'), ensure_ascii=False, indent=1)
    n_short = sum(1 for c, n in have.items() if n < cap)
    log(f'newcodes: 추가 {n_added}장 {dict(found)} (누적 수집 코드 {len(counts)}종, 보유 코드 {len(have)}종 중 상한 미달 {n_short}종)')
    return n_added


# ----------------------------------------------------------------------------- 2. collect
def local_failures(dates):
    """로컬 field/<날짜>/save_pose_debug 의 실패 프레임 raw 경로 (nodet_*, 코드 raw 인데 json 없음)."""
    out = []
    for d in dates:
        for dp, _, fs in os.walk(os.path.join(CFG['field_dir'], d, 'save_pose_debug')):
            for f in fs:
                if f.endswith('_raw.png') and (f.startswith('nodet_') or not os.path.exists(os.path.join(dp, f[:-8] + '.json'))) \
                        and not os.path.exists(os.path.join(dp, f[:-8] + '_skip.json')):
                    out.append(os.path.join(dp, f))
    return sorted(out)


def collect(raw_paths, max_per_code):
    """새 raw 만 <날짜>/<실행>/ 구조의 심볼릭링크 스테이징으로 모아 collector 를 돌린다. 반환: 추가된 stem 수."""
    if not raw_paths:
        log('collect: 새 프레임 없음')
        return 0
    ds_rows = load_manifest(CFG['ds'])
    known = {r['stem'] for r in load_manifest(CFG['ref_field'])} | {r['stem'] for r in load_manifest(CFG['ref_ds'])} \
        | {r['stem'] for r in ds_rows}
    have = collections.Counter((r['stem'][:8], r['code']) for r in ds_rows)     # (날짜, 코드) 별 이미 누적된 수
    groups = collections.defaultdict(list)
    n_known = 0
    for p in raw_paths:
        run_dir = os.path.dirname(p)
        date = date_of(p)
        base = os.path.basename(p)[:-8]
        if f'{date}_{os.path.basename(run_dir)}_{base}' in known:   # 이미 검수 PC 에서 봤거나 DS 에 있는 프레임
            n_known += 1
            continue
        code = base.rsplit('_', 1)[-1] if base.startswith('nodet_') else base.split('_', 1)[-1]
        groups[(date, code)].append(p)
    # (날짜, 코드) 별 일일 상한 DAILY_CAP: 시간 균등 샘플이 사이클마다 달라져도 하루 누적이 상한을 넘지 않게
    picked, n_capped = [], 0
    for (date, code), lst in sorted(groups.items()):
        quota = min(max_per_code, DAILY_CAP - have[(date, code)])
        if quota <= 0:
            n_capped += len(lst)
            continue
        lst.sort()
        idx = sorted(set(np.linspace(0, len(lst) - 1, quota).round().astype(int).tolist())) if len(lst) > quota else range(len(lst))
        picked += [lst[i] for i in idx]
    stage = tempfile.mkdtemp(prefix='al_stage_')
    for p in picked:
        run_dir = os.path.dirname(p)
        date = date_of(p)
        d = os.path.join(stage, date, os.path.basename(run_dir))
        os.makedirs(d, exist_ok=True)
        for f in (p, p[:-8] + '.json'):
            if os.path.exists(f):
                os.symlink(os.path.abspath(f), os.path.join(d, os.path.basename(f)))
    before = len(load_manifest(CFG['ds']))
    if not os.listdir(stage):
        shutil.rmtree(stage, ignore_errors=True)
        log(f'collect: 실패 프레임 {len(raw_paths)} 전부 이미 처리됨')
        return 0
    log(f'collect: 실패 프레임 {len(raw_paths)} 중 이미 처리 {n_known}, 일일 상한 초과 {n_capped} 제외 → 후보 {len(picked)}')
    cmd = [sys.executable, CFG['collector'], stage, '--out', CFG['ds'], '--model', CFG['field_model'],
           '--roi', CFG['roi'], '--device', CFG['device'], '--max-per-code', '0']    # 샘플링은 위에서 끝냄
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip() and 'Warning' not in line:
            log('  ' + line.rstrip()[:160])
    shutil.rmtree(stage, ignore_errors=True)
    # collector 는 source 를 스테이징(임시) 경로로 적으므로 실제 경로로 바꿔 둔다
    rows = load_manifest(CFG['ds'])
    changed = False
    for row in rows:
        if 'al_stage_' in row.get('source', ''):
            m = re.search(r'(\d{8})/([^/]+)/([^/]+)$', row['source'])
            if m:
                row['source'] = os.path.join(CFG['field_dir'], m.group(1), 'save_pose_debug', m.group(2), m.group(3))
                changed = True
    if changed:
        write_manifest(CFG['ds'], rows)
    n = len(rows) - before
    log(f'collect: +{n} (누적 {len(rows)})')
    return n


def date_of(path):
    """field/<날짜>/save_pose_debug/<실행>/<파일> → 날짜 (8자리 폴더명 중 파일에 가장 가까운 것)."""
    for c in reversed(os.path.dirname(path).split(os.sep)):
        if re.fullmatch(r'\d{8}', c):
            return c
    return 'nodate'


def load_manifest(root):
    p = os.path.join(root, 'manifest.csv')
    if not os.path.isfile(p):
        return []
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_manifest(root, rows):
    fields = ['stem', 'source', 'code', 'reason', 'n_det', 'top_cls', 'top_conf', 'top_maskpx', 'top_fill']
    tmp = os.path.join(root, 'manifest.csv.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, os.path.join(root, 'manifest.csv'))


# ----------------------------------------------------------------------------- 3. exclude
_models = {}


def yolo(kind):
    """kind: embed(현장 모델 임베딩) / cand(후보 모델 추론). embed 모드는 predictor 를 바꾸므로 인스턴스를 분리한다."""
    if kind not in _models:
        from ultralytics import YOLO
        _models[kind] = YOLO(CFG['field_model'] if kind == 'embed' else CFG['cand_model'])
    return _models[kind]


_dino = {}


def dino():
    """DINOv2 ViT-S/14 (torch.hub, ~/.cache/torch/hub). 외형 구분용 — 현장 YOLO 임베딩보다 뒤집힘·상자·코드 혼동을 훨씬 잘 가른다."""
    if 'm' not in _dino:
        import torch
        _dino['torch'] = torch
        dev = f"cuda:{CFG['device']}" if CFG['device'].isdigit() else CFG['device']
        _dino['m'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).to(dev).eval()
        _dino['dev'] = dev
    return _dino['m']


_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])


def _prep(img):
    x = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB) / 255.0
    return _dino['torch'].from_numpy(((x - _MEAN) / _STD).transpose(2, 0, 1)).float()


def _crop_top(img, label_path, pad=0.08):
    """라벨에서 가장 큰 인스턴스의 bbox(+여백)로 자른다. 라벨이 없으면 전체."""
    h, w = img.shape[:2]
    best = None
    if os.path.isfile(label_path):
        with open(label_path, encoding='utf-8') as f:
            for line in f:
                v = line.split()
                if len(v) < 7:
                    continue
                pts = np.array(v[1:], dtype=np.float32).reshape(-1, 2)
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                a = (x1 - x0) * (y1 - y0)
                if best is None or a > best[0]:
                    best = (a, x0, y0, x1, y1)
    if best is None:
        return img
    _, x0, y0, x1, y1 = best
    X0, Y0 = int(max(0, (x0 - pad) * w)), int(max(0, (y0 - pad) * h))
    X1, Y1 = int(min(w, (x1 + pad) * w)), int(min(h, (y1 + pad) * h))
    return img[Y0:Y1, X0:X1] if X1 - X0 > 10 and Y1 - Y0 > 10 else img


def dino_embed(items, cache_path=None):
    """items: [(key, image_path, label_path|None)] → {key: (whole 384, crop 384)} (L2 정규화). cache_path(npz) 에 key 별 캐시."""
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        cache = {k: (w, c) for k, w, c in zip(z['keys'].tolist(), z['whole'], z['crop'])}
    need = [it for it in items if it[0] not in cache]
    if need:
        m = dino()
        torch = _dino['torch']
        with torch.no_grad():
            for i in range(0, len(need), 32):
                b = need[i:i + 32]
                imgs = [cv2.imread(p) for _, p, _ in b]
                xw = torch.stack([_prep(im) for im in imgs]).to(_dino['dev'])
                xc = torch.stack([_prep(_crop_top(im, lp or '')) for im, (_, _, lp) in zip(imgs, b)]).to(_dino['dev'])
                ew, ec = m(xw).cpu().numpy(), m(xc).cpu().numpy()
                for (k, _, _), w_, c_ in zip(b, ew, ec):
                    cache[k] = (w_ / (np.linalg.norm(w_) + 1e-6), c_ / (np.linalg.norm(c_) + 1e-6))
        if cache_path:
            ks = list(cache)
            np.savez(cache_path, keys=np.array(ks), whole=np.stack([cache[k][0] for k in ks]), crop=np.stack([cache[k][1] for k in ks]))
    return {it[0]: cache[it[0]] for it in items}


def border_feats(label_path, w=685, h=685):
    """원본 의사 라벨 폴리곤의 가장자리 걸침: (걸친 인스턴스 비율, 최대 걸친 변 수, 최대 인스턴스 bbox 면적비)."""
    if not os.path.isfile(label_path):
        return 0.0, 0.0, 0.0
    touch, edges, area = [], [], []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            v = line.split()
            if len(v) < 7:
                continue
            pts = np.array(v[1:], dtype=np.float32).reshape(-1, 2)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            e = int(x0 <= 0.005) + int(y0 <= 0.005) + int(x1 >= 0.995) + int(y1 >= 0.995)
            touch.append(e > 0)
            edges.append(e)
            area.append((x1 - x0) * (y1 - y0))
    if not touch:
        return 0.0, 0.0, 0.0
    return float(np.mean(touch)), float(max(edges)), float(max(area))


def manifest_feats(row, label_path):
    n = float(row.get('n_det') or 0)
    tc = float(row.get('top_conf') or 0)
    tf = float(row.get('top_fill') or 0)
    px = float(row.get('top_maskpx') or 0)
    bt, be, ba = border_feats(label_path)
    return [n, min(n, 3), tc, tf, np.log1p(px), bt, be, ba, float(row.get('reason') == 'nodet'),
            float(row.get('code') in ('none', 'gate'))]


def features(rows, root, cache_path=None):
    """rows(manifest 행) → (N, 384+384+10) 특징: DINOv2 전체 + 물체 크롭 + manifest 수치. 임베딩은 cache_path(npz) 에 캐시."""
    emb = dino_embed([(r['stem'], os.path.join(root, 'images', r['stem'] + '.png'), os.path.join(root, 'labels', r['stem'] + '.txt'))
                      for r in rows], cache_path)
    X = []
    for r in rows:
        w_, c_ = emb[r['stem']]
        X.append(np.concatenate([w_, c_, manifest_feats(r, os.path.join(root, 'labels', r['stem'] + '.txt'))]))
    return np.array(X, dtype=np.float32)


def clf_path():
    return os.path.join(CFG['state_dir'], 'exclude_clf_dino.npz')


def refit():
    """기준 라벨: ref_state(검수 PC review_state.json) 가 있으면 status ok=0 / reject=1 (보류·미판정 제외),
    없으면 근사치 — ref_field manifest 전체 중 ref_ok export 에 있으면 0(채택), 없으면 1(제외)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    os.makedirs(CFG['state_dir'], exist_ok=True)
    rows = [r for r in load_manifest(CFG['ref_field'])
            if os.path.isfile(os.path.join(CFG['ref_field'], 'images', r['stem'] + '.png'))]
    if os.path.isfile(CFG['ref_state']):
        with open(CFG['ref_state'], encoding='utf-8') as f:
            st = json.load(f)
        lab = {s: {'ok': 0, 'reject': 1}.get(v.get('status')) for s, v in st.items()}
        rows = [r for r in rows if lab.get(r['stem']) is not None]
        y = np.array([lab[r['stem']] for r in rows])
        basis = f'검수 PC 판정 {CFG["ref_state"]}'
        ## 자동 수집 데이터셋에서 사람이 판정한 프레임(apply_review_bundle: note 'human: …')도 기준에 더한다
        hp = os.path.join(CFG['ds'], 'review_state.json')
        if os.path.isfile(hp):
            with open(hp, encoding='utf-8') as f:
                hst = json.load(f)
            hrows = [r for r in load_manifest(CFG['ds'])
                     if str(hst.get(r['stem'], {}).get('note', '')).startswith('human:')
                     and hst[r['stem']].get('status') in ('ok', 'reject')
                     and os.path.isfile(os.path.join(CFG['ds'], 'images', r['stem'] + '.png'))]
            if hrows:
                X2 = features(hrows, CFG['ds'], os.path.join(CFG['state_dir'], 'ds_dino_feat.npz'))
                y2 = np.array([1 if hst[r['stem']]['status'] == 'reject' else 0 for r in hrows])
                basis += f' + 자동 수집분 사람 판정 {len(hrows)}장'
                extra = (X2, y2)
            else:
                extra = None
        else:
            extra = None
    else:
        ok = {r['stem'] for r in load_manifest_any(os.path.join(CFG['ref_ok'], 'export_manifest.csv'))}
        y = np.array([0 if r['stem'] in ok else 1 for r in rows])
        basis = '근사치 (현장 manifest − 확정 export; 보류·미판정도 제외로 셈)'
    log(f'refit: 기준 {len(rows)}장 (채택 {int((y == 0).sum())}, 제외 {int(y.sum())}) — {basis}')
    X = features(rows, CFG['ref_field'], os.path.join(CFG['state_dir'], 'ref_dino_feat.npz'))
    if locals().get('extra') is not None:
        X = np.concatenate([X, extra[0]])
        y = np.concatenate([y, extra[1]])
        rows = rows + hrows
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    mk = lambda: LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        p[te] = mk().fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    thr, best = 0.99, None
    lines = ['| 임계값 | 제외 판정 수 | 정밀도 | 재현율 | 잘못 제외(채택인데 제외) |', '|---|---|---|---|---|']
    for t in np.arange(0.5, 0.991, 0.05):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, int(y.sum()))
        lines.append(f'| {t:.2f} | {int(pred.sum())} | {prec:.3f} | {rec:.3f} | {fp} |')
        if prec >= EXCL_PRECISION and (best is None or rec > best[1]):
            best, thr = (t, rec), float(t)
    clf = mk().fit(Xs, y)
    np.savez(clf_path(), coef=clf.coef_[0], intercept=clf.intercept_, mu=mu, sd=sd, thr=thr,
             n=len(y), at=time.strftime('%Y-%m-%d %H:%M:%S'))
    rep = [f'# 제외 분류기 (기준 {len(y)}장: 채택 {int((y == 0).sum())} / 제외 {int(y.sum())}, 5-fold 교차검증, {time.strftime("%Y-%m-%d %H:%M")})', '',
           f'기준 라벨: {basis}', '',
           '특징: DINOv2 ViT-S/14 임베딩(전체 384 + 가장 큰 인스턴스 크롭 384) + n_det·top_conf·top_fill·mask px·가장자리 걸침·nodet·코드없음. 로지스틱 회귀(C=1, balanced).', '',
           f'채택 임계값 **{thr:.2f}** (교차검증 정밀도 ≥ {EXCL_PRECISION} 중 재현율 최대). 이 값 이상만 자동 제외, 나머지는 보류로 사람에게.', ''] + lines
    # 유형별 재현율
    pred = p >= thr
    rep += ['', '| 유형 | 제외 기준 수 | 자동 제외 | 잘못 제외 |', '|---|---|---|---|']
    kinds = collections.defaultdict(lambda: [0, 0, 0])
    for r, yy, pp in zip(rows, y, pred):
        k = r['code'] if r['code'] in ('none', 'gate') else ('coded ' + r['reason'])
        kinds[k][0] += yy == 1
        kinds[k][1] += (yy == 1) and pp
        kinds[k][2] += (yy == 0) and pp
    for k, v in sorted(kinds.items()):
        rep.append(f'| {k} | {v[0]} | {v[1]} | {v[2]} |')
    with open(os.path.join(CFG['state_dir'], 'exclude_clf_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')
    log(f'refit: 임계값 {thr:.2f} (재현율 {best[1] if best else 0:.3f}) → {clf_path()}')
    print('\n'.join(rep))


def load_manifest_any(p):
    if not os.path.isfile(p):
        return []
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def is_auto_judged(ds, s):
    """파이프라인이 정한 상태(메모 auto-*)이고 사람이 손대지 않은 것."""
    return ds.state.note(s).startswith('auto-') and not ds.is_edited(s) and not ds.state.exemplar(s)


def exclude(ds, rejudge=False):
    """상태가 없는(처음 보는) 이미지만 판정한다. rejudge=True 면 파이프라인이 판정한 것도 다시 (사람이 정한 상태는 절대 안 건드림)."""
    if not os.path.isfile(clf_path()):
        refit()
    z = np.load(clf_path(), allow_pickle=True)
    rows = {r['stem']: r for r in load_manifest(CFG['ds'])}
    todo = [s for s in ds.stems if s in rows and (not ds.state.get(s) or (rejudge and is_auto_judged(ds, s)))]
    if not todo:
        log('exclude: 판정할 새 이미지 없음')
        return 0
    X = features([rows[s] for s in todo], CFG['ds'], os.path.join(CFG['state_dir'], 'ds_dino_feat.npz'))
    logit = ((X - z['mu']) / z['sd']) @ z['coef'] + z['intercept'][0]
    p = 1 / (1 + np.exp(-logit))
    n = 0
    for s, pp in zip(todo, p):
        if pp >= float(z['thr']):
            ds.state.set(s, status='reject', note=f'auto-exclude p={pp:.2f}', auto_exclude=round(float(pp), 3))
            n += 1
        else:
            ds.state.set(s, status='pending', note='auto-pending (미판정)', auto_exclude=round(float(pp), 3))
    log(f'exclude: {len(todo)}장 판정 → 제외 {n}, 보류 {len(todo) - n} (임계값 {float(z["thr"]):.2f})')
    return n


# ----------------------------------------------------------------------------- 4. propagate
def build_exemplars(max_per_code=5):
    """확정 export(ref_ok)에서 제품코드별 대표를 고른다: 사람 편집본(reviewed) > auto > original, 인스턴스 수가 다양하게."""
    out = CFG['exemplars']
    man = os.path.join(out, 'manifest.csv')
    if os.path.isfile(man):
        return
    from mask_reviewer.dataset import Dataset
    for d in ('images', 'labels'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    sel, starred = [], collections.Counter()
    if os.path.isfile(os.path.join(CFG['ref_ds'], 'review_state.json')):     # 검수자가 지정한 대표 ★ (유효 라벨 = 편집본 > 자동 > 원본)
        rds = Dataset(CFG['ref_ds'])
        for st in rds.exemplars():
            code = rds.code(st)
            if code in ('none', 'gate', '') or rds.state.status(st) != 'ok' or st in CFG['exemplar_drop']:
                continue
            text = rds.label_text(st)
            n = sum(len(l.split()) >= 7 for l in text.splitlines())
            if n == 0:
                continue
            dst = os.path.join(out, 'images', st + '.png')
            if not os.path.lexists(dst):
                os.link(rds.image_path(st), dst)
            with open(os.path.join(out, 'labels', st + '.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
            sel.append(dict(stem=st, code=code, n_inst=n, label_src='star:' + rds.label_source(st)))
            starred[code] += 1
        log(f'exemplars: 검수자 대표 ★ {len(sel)}장 ({len(starred)} 제품)')
    rows = load_manifest_any(os.path.join(CFG['ref_ok'], 'export_manifest.csv'))
    rank = {'reviewed': 0, 'auto': 1, 'original': 2}
    by = collections.defaultdict(list)
    for r in rows:
        if r.get('status', 'ok') == 'ok' and r['code'] not in ('none', 'gate', '') and starred[r['code']] == 0:
            by[r['code']].append(r)
    for code, lst in by.items():
        lst.sort(key=lambda r: (rank.get(r['label_src'], 3), r['stem']))
        picked, seen_n = [], collections.Counter()
        for r in lst:                       # 인스턴스 수별로 최대 2장씩, 총 max_per_code
            n = r['n_inst']
            if seen_n[n] >= 2:
                continue
            picked.append(r)
            seen_n[n] += 1
            if len(picked) >= max_per_code:
                break
        for r in picked:
            sp = r['split']
            for d, ext in (('images', '.png'), ('labels', '.txt')):
                src = os.path.join(CFG['ref_ok'], d, sp, r['stem'] + ext)
                dst = os.path.join(out, d, r['stem'] + ext)
                if not os.path.lexists(dst):
                    os.link(src, dst)
            sel.append(dict(stem=r['stem'], code=r['code'], n_inst=r['n_inst'], label_src=r['label_src']))
    shutil.copy2(os.path.join(CFG['ref_ok'], 'dataset.yaml'), os.path.join(out, 'dataset.yaml'))
    with open(man, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'code', 'n_inst', 'label_src'])
        w.writeheader()
        w.writerows(sel)
    log(f'exemplars: ★ 없는 제품 {len(by)}개는 확정 export 에서 보충 → 총 {len(sel)}장 → {out}')


def union(masks, h, w):
    u = np.zeros((h, w), bool)
    for m in masks:
        u |= m
    return u


def iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0



def yolo_instances(img):
    """후보 모델 추론 → [(cls, mask, conf)] (conf 0.1, retina)."""
    from mask_reviewer.dataset import Instance
    r = yolo('cand').predict(img, imgsz=640, conf=0.1, retina_masks=True, device=CFG['device'], verbose=False)[0]
    out = []
    if r.masks is not None:
        md = r.masks.data.cpu().numpy().astype(bool)
        for i in range(len(r.boxes)):
            if md[i].sum() >= 64:
                out.append((int(r.boxes.cls[i]), md[i], float(r.boxes.conf[i])))
    return out


def propagate(ds, include_auto=False, limit=0):
    from mask_reviewer.dataset import Dataset, Instance
    from mask_reviewer.propagate import Sam2Propagator
    build_exemplars()
    ex_ds = Dataset(CFG['exemplars'])
    ex_by_code = collections.defaultdict(list)
    for r in load_manifest_any(os.path.join(CFG['exemplars'], 'manifest.csv')):
        ex_by_code[r['code']].append((r['stem'], int(r['n_inst'])))
    ex_by_6 = collections.defaultdict(list)
    for c, l in ex_by_code.items():
        ex_by_6[c[:6]] += l
    rows = {r['stem']: r for r in load_manifest(CFG['ds'])}
    todo = [s for s in ds.stems if ds.state.status(s) == 'pending' and not ds.state.exemplar(s) and not ds.is_edited(s)
            and (include_auto or not ds.has_auto(s))]
    if limit:
        todo = todo[:limit]
    if not todo:
        log('propagate: 대상 없음')
        return {}
    # 대표 선택은 코드가 아니라 외형(DINOv2 전체 이미지 유사도)으로: 현장 코드는 언로딩 오류·오인식으로 믿을 수 없다 (§21-cb)
    ex_stems = sorted({st for l in ex_by_code.values() for st, _ in l})
    ex_code = {st: c for c, l in ex_by_code.items() for st, _ in l}
    ex_n = {st: n for l in ex_by_code.values() for st, n in l}
    ex_emb = dino_embed([(st, ex_ds.image_path(st), ex_ds.effective_label_path(st)) for st in ex_stems],
                        os.path.join(CFG['exemplars'], 'dino.npz'))
    EX = np.stack([ex_emb[st][0] for st in ex_stems])
    tg_emb = dino_embed([(s, ds.image_path(s), ds.original_label_path(s)) for s in todo],
                        os.path.join(CFG['state_dir'], 'ds_dino_feat.npz'))
    prop = Sam2Propagator(CFG['sam_ckpt'], device='cuda')
    cnt = collections.Counter()
    ref_cache = {}
    try:
        for i, s in enumerate(todo):
            code = ds.code(s)
            n_det = int(rows.get(s, {}).get('n_det') or 0) or 1
            sims = EX @ tg_emb[s][0]
            order = np.argsort(-sims)
            same = [i for i in order if ex_code[ex_stems[i]][:6] == code[:6]]
            best_same = float(sims[same[0]]) if same else -1.0
            best_all = float(sims[order[0]])
            code_mismatch = bool(same) and best_all >= best_same + SIM_CODE_MARGIN and ex_code[ex_stems[order[0]]][:6] != code[:6]
            if same and best_same >= SIM_SAME_MIN and not code_mismatch:
                pick, via = same[:3], 'code'
            else:
                pick, via = list(order[:3]), 'sim'
            exs = [(ex_stems[i], ex_n[ex_stems[i]]) for i in pick]
            top_sim = float(sims[pick[0]]) if pick else 0.0
            tgt = cv2.imread(ds.image_path(s), cv2.IMREAD_COLOR)
            if tgt is None:
                continue
            h, w = tgt.shape[:2]
            ydet = yolo_instances(tgt)
            status, note, src, score, insts, ref = 'pending', '', None, 0.0, [], None
            if exs:
                cand = sorted(exs, key=lambda t: (abs(t[1] - n_det), -float(sims[ex_stems.index(t[0])])))[:3]
                best = None
                for ex_stem, _n in cand:
                    if ex_stem not in ref_cache:
                        ref_cache[ex_stem] = ex_ds.load(ex_stem)
                    ref_img, ref_insts = ref_cache[ex_stem]
                    res = prop.propagate(ref_img, ref_insts, tgt, ref_key=ex_stem)
                    if not res:
                        continue
                    sc = float(np.mean([r[2] for r in res]))
                    if best is None or sc > best[0]:
                        best = (sc, ex_stem, res)
                if best:
                    score, ref, res = best
                    insts = [Instance(c, m) for c, m, _ in res]
                    src = f'sam2:{via}'
                    # 형태: 클래스 무관 합집합 IoU (클래스는 사람이 만든 대표를 믿는다 — 후보 YOLO 가 틀리는 프레임이 곧 실패 프레임)
                    x = iou(union([m for _, m, _ in res], h, w), union([m for _, m, _ in ydet], h, w)) if ydet else -1.0
                    # 크기: 대표의 같은 클래스 인스턴스와 면적 비율이 크게 다르면 표류 (예: cap 인스턴스가 하우징에 얹힘)
                    ref_areas = collections.defaultdict(list)
                    for ri in ref_cache[ref][1]:
                        ref_areas[ri.cls].append(float(ri.area))
                    ratios = [m.sum() / min(ref_areas[c], key=lambda a: abs(a - m.sum())) for c, m, _ in res if ref_areas.get(c)]
                    size_ok = all(SIZE_RATIO[0] <= r <= SIZE_RATIO[1] for r in ratios)
                    cls_diff = bool(ydet) and {c for c, _, _ in res} != {c for c, _, _ in ydet}
                    flags = ('' if size_ok else ' size!') + (' cls-diff' if cls_diff else '') \
                        + (f' code-mismatch({ex_code[ref][:6]})' if code_mismatch else '') + f' sim={top_sim:.2f} via={via}'
                    if score >= SAM_OK_SCORE and (x >= CROSS_IOU or not ydet) and size_ok:
                        status, note = 'ok', f'auto-ok sam2 score={score:.2f} yolo-iou={x:.2f}{flags}'
                    else:
                        note = f'auto-pending sam2 score={score:.2f} yolo-iou={x:.2f}{flags}'
                    ds.state.set(s, cross_iou=round(x, 3), size_ratio=[round(r, 2) for r in ratios], sim=round(top_sim, 3),
                                 code_mismatch=code_mismatch, ref_code=ex_code[ref])
            if not insts and ydet:                    # 대표 없음 또는 SAM2 실패 → 후보 YOLO 라벨
                insts = [Instance(c, m) for c, m, _ in ydet]
                score = float(np.mean([c for _, _, c in ydet]))
                src = f'yolo:{os.path.basename(os.path.dirname(os.path.dirname(CFG["cand_model"])))}'
                if score >= YOLO_OK_CONF:
                    status, note = 'ok', f'auto-ok yolo conf={score:.2f}'
                else:
                    note = f'auto-pending yolo conf={score:.2f}'
            if insts:
                ds.write_auto(s, insts, w, h, source=src, score=score, ref=ref)
                ds.state.set(s, status=status, note=note)
            else:
                ds.state.set(s, status='pending', note='auto-none: 대표·검출 모두 없음 (사람 라벨 필요)')
            key = f'{status}:{(src or "none").split(":")[0]}'
            cnt[key] += 1
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                log(f'  propagate {i + 1}/{len(todo)} {dict(cnt)}')
    finally:
        prop.close()
    log(f'propagate: {len(todo)}장 → {dict(cnt)}')
    return dict(cnt)


# ----------------------------------------------------------------------------- 5. export
def export(ds):
    from mask_reviewer.export import export_dataset
    out = CFG['export']
    if not any(ds.state.status(s) == 'ok' for s in ds.stems):
        log('export: ok 상태 이미지 없음')
        return None
    if os.path.isdir(out):
        shutil.rmtree(out)
    r = export_dataset(ds, out, statuses=('ok',), val_ratio=0.1, copy_mode='hardlink')
    log(f'export: {out} {r}')
    return r


def status(ds):
    c = ds.state.counts(ds.stems)
    auto = sum(ds.has_auto(s) for s in ds.stems)
    ok_auto = sum(ds.state.status(s) == 'ok' and ds.has_auto(s) for s in ds.stems)
    print(f'{CFG["ds"]}: 전체 {len(ds.stems)} · 확정 {c["ok"]}(자동 {ok_auto}) · 보류 {c["pending"]} · 제외 {c["reject"]} · 자동 라벨 {auto}')
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for s in ds.stems:
        e = per[ds.code(s)]
        e[0] += 1
        e[{'ok': 1, 'pending': 2, 'reject': 3}[ds.state.status(s)]] += 1
    print(f'  {"제품":16s} {"확정":>5s} {"보류":>5s} {"제외":>5s} / 전체')
    for code, (n, ok, pe, rj) in sorted(per.items()):
        print(f'  {code or "(없음)":16s} {ok:5d} {pe:5d} {rj:5d} / {n}')
    return c


def open_ds():
    from mask_reviewer.dataset import Dataset
    root = CFG['ds']
    os.makedirs(os.path.join(root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
    if not os.path.isfile(os.path.join(root, 'dataset.yaml')):
        shutil.copy2(os.path.join(CFG['ref_ok'], 'dataset.yaml'), os.path.join(root, 'dataset.yaml'))
    return Dataset(root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'fetch', 'newcodes', 'collect', 'refit', 'exclude', 'propagate', 'export', 'status', 'redo', 'codes'],
                    help='redo: 자동 판정(auto-ok/auto-pending/auto-none) 이미지를 보류로 되돌려 전파·내보내기를 다시 (사람 판정·자동 제외는 유지)')
    ap.add_argument('dates', nargs='*', help='YYYYMMDD (기본 어제·오늘)')
    ap.add_argument('--max-per-code', type=int, default=30, help='collect: 사이클당 제품별 상한')
    ap.add_argument('--include-auto', action='store_true', help='propagate: 이미 자동 라벨이 있는 보류도 다시')
    ap.add_argument('--limit', type=int, default=0, help='propagate: 대상 수 제한 (시험용)')
    ap.add_argument('--apply-alias', action='store_true', help='codes: 대응표를 DS 에 적용')
    a = ap.parse_args()
    os.makedirs(CFG['state_dir'], exist_ok=True)
    dates = today_dates(a.dates)
    t0 = time.time()
    if a.cmd == 'refit':
        return refit()
    if a.cmd == 'codes':
        ds_ = open_ds()
        al = code_alias(ds_)
        print(f'대응표 {len(al)}건 (원래 현장 코드 → 검수 재지정 코드), $AL_STATE/code_alias.json / 수동 code_alias_manual.json')
        for k, v in sorted(al.items()):
            print(f'  {k} → {v}')
        if a.apply_alias:
            print(f'DS 에 적용: {apply_alias(ds_, al)}장 코드 변경')
        else:
            print('(--apply-alias 를 주면 DS 의 미변경 프레임에 적용; run 은 매 사이클 자동 적용)')
        return 0
    if a.cmd == 'newcodes':
        newcodes(dates, open_ds())
        return 0
    if a.cmd in ('run', 'fetch', 'collect'):
        if a.cmd != 'collect':
            fetch(dates)
            if a.cmd == 'run':
                try:
                    newcodes(dates, open_ds())
                except Exception as e:
                    log(f'newcodes 실패 (계속 진행): {e}')
        if a.cmd == 'fetch':
            return 0
        # 대상 = 로컬에 있는 해당 날짜 실패 프레임 전부. collector 가 DS 에 이미 있는 stem 을 건너뛰고,
        # collect() 가 검수 PC 에서 이미 본 stem 을 빼므로 매 사이클 전체를 넘겨도 새 것만 추가된다.
        pool = local_failures(dates)
        n_new = collect(pool, a.max_per_code)
        try:   # 실패 프레임(collector)도 현장 코드 → 재지정 코드 대응표 적용 (사람이 바꾼 코드는 그대로)
            ds_ = open_ds()
            n_al = apply_alias(ds_, code_alias(ds_))
            if n_al:
                log(f'code alias: {n_al}장 코드를 대응표로 재지정')
        except Exception as e:
            log(f'code alias 실패 (계속 진행): {e}')
        if a.cmd == 'collect':
            return 0
    ds = open_ds()
    if a.cmd == 'status':
        status(ds)
        return 0
    summary = {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'dates': ' '.join(dates)}
    if a.cmd in ('run', 'exclude'):
        summary['excluded'] = exclude(ds)
    if a.cmd == 'redo':
        n = 0
        for s_ in ds.stems:
            if ds.state.note(s_).startswith('auto-') and ds.state.status(s_) != 'reject' and not ds.is_edited(s_):
                ds.state.set(s_, status='pending')
                n += 1
        log(f'redo: 자동 판정 {n}장 보류로 되돌림')
        a.include_auto = True
        summary['excluded'] = exclude(ds, rejudge=True)
    if a.cmd in ('run', 'propagate', 'redo'):
        summary['propagate'] = json.dumps(propagate(ds, include_auto=a.include_auto, limit=a.limit), ensure_ascii=False)
    if a.cmd in ('run', 'export', 'redo'):
        r = export(ds)
        summary['export'] = json.dumps(r, ensure_ascii=False) if r else ''
    c = ds.state.counts(ds.stems)
    summary.update(total=len(ds.stems), ok=c['ok'], pending=c['pending'], reject=c['reject'], sec=round(time.time() - t0))
    if a.cmd == 'run':
        summary['collected'] = n_new
        p = os.path.join(CFG['state_dir'], 'cycles.csv')
        new_file = not os.path.isfile(p)
        with open(p, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['at', 'dates', 'collected', 'excluded', 'propagate', 'export', 'total', 'ok', 'pending', 'reject', 'sec'])
            if new_file:
                w.writeheader()
            w.writerow(summary)
    log(f'완료 {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
