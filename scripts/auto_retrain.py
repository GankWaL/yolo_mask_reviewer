#!/usr/bin/env python3
"""자동 재학습 (2026-09-19): field_autolabel 이 모은 자동 라벨(ok) 데이터를 스냅샷 → 직전 기준 모델에서 이어 파인튜닝 →
게이트 통과 시 다음 기준 모델(current.pt)로 승격. 4시간 cron 용 (`auto_retrain.sh`).

  conda activate pro
  python scripts/auto_retrain.py            # 새 ok 데이터가 없으면 건너뜀
  python scripts/auto_retrain.py --force    # 데이터 변화 없어도 학습
  python scripts/auto_retrain.py --status   # 이력 출력

흐름:
  1. 스냅샷  field_autolabel 의 export(`SL_under_predict_auto_yolo`)를 autolabel 잠금 아래 하드링크 복사 (학습 중 덮어쓰기 방지).
             stem 목록 해시가 직전 학습과 같으면 건너뜀.
  2. 학습    train_round.py: 검수 확정(REV) + 리허설(18일·신규 제품) + 스냅샷, base = current.pt (처음엔 round_20260919_b0918),
             freeze 0 · AdamW lr0 1e-4 · 15 epoch (§21-bx 채택 설정). 결과 runs/auto_<TAG>/.
  3. 게이트  eval_gates.py (base = current.pt 대비): ① 18일 게이트셋 보존 top-1 IoU≥0.95 비율, ② 19일 신규 val 87장(사람 라벨) 평균 IoU.
  4. 승격    ① ≥ GATE1_MIN 이고 ② 가 base 보다 GATE2_DROP 이상 나빠지지 않으면 current.pt → 새 best.pt (다음 재학습·field_autolabel 후보 모델이 됨).
             실패하면 current.pt 유지, 결과만 기록. 현장 적용(models/ 등록·main() 교체)은 자동으로 하지 않는다.
결과: $RT/history.csv, $RT/gate_<TAG>.md, $RT/current.pt (심볼릭링크), 로그는 래퍼가 $RT/logs/ 에.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
HOME = os.path.expanduser('~')
E = os.environ.get
CFG = dict(
    rt=E('RT_DIR', f'{HOME}/jhw/data/SL/retrain'),
    export=E('AL_EXPORT', f'{HOME}/jhw/data/SL_under_predict_auto_yolo'),
    al_lock=E('AL_LOCK', f'{HOME}/jhw/data/SL/autolabel/logs/.lock'),
    rev=E('RT_REV', f'{HOME}/jhw/data/SL_under_predict_yolo1'),
    reh=E('RT_REH', f'{HOME}/jhw/data/SL/rehearsal_20260918/dataset.yaml {HOME}/jhw/data/SL/rehearsal_20260919_newcodes/dataset.yaml').split(),
    gate=E('RT_GATE', f'{HOME}/jhw/data/SL/rehearsal_20260918/gate'),
    gate_reh=E('RT_GATE_REH', f'{HOME}/jhw/data/SL/rehearsal_20260918'),
    valnew=E('RT_VALNEW', f'{HOME}/jhw/data/SL/reviewed_20260919_valnew'),
    runs=E('RT_RUNS', f'{HOME}/jhw/data/SL/runs'),
    init_base=E('RT_INIT_BASE', f'{HOME}/jhw/data/SL/runs/round_20260919_b0918/weights/best.pt'),
    device=E('AL_DEVICE', '1'),
    epochs=int(E('RT_EPOCHS', '15')), lr0=float(E('RT_LR0', '0.0001')), batch=int(E('RT_BATCH', '16')),
)
GATE1_MIN = float(E('RT_GATE1_MIN', '0.90'))   # 보존: 직전 모델 대비 top-1 IoU≥0.95 비율 (§21-bu/bx 실측 93%)
GATE2_DROP = float(E('RT_GATE2_DROP', '0.01'))  # 회복: 신규 val 평균 IoU 가 base 보다 이만큼 이상 떨어지면 실패
MIN_NEW = int(E('RT_MIN_NEW', '20'))            # 직전 학습 대비 새 ok 이미지가 이보다 적으면 건너뜀
PY = sys.executable


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def current_base():
    p = os.path.join(CFG['rt'], 'current.pt')
    return os.path.realpath(p) if os.path.lexists(p) else CFG['init_base']


def snapshot(tag):
    """export 폴더를 autolabel 잠금 아래 하드링크 복사. 반환 (스냅샷 경로, stem 해시, n) 또는 None."""
    exp = CFG['export']
    man = os.path.join(exp, 'export_manifest.csv')
    if not os.path.isfile(man):
        log('export 없음 (아직 ok 데이터가 없음)')
        return None
    os.makedirs(os.path.dirname(CFG['al_lock']), exist_ok=True)
    with open(CFG['al_lock'], 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)           # field_autolabel 사이클이 끝날 때까지 기다린다
        with open(man, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        stems = sorted(r['stem'] for r in rows)
        h = hashlib.md5('\n'.join(stems).encode()).hexdigest()[:12]
        snap = os.path.join(CFG['rt'], f'data_{tag}')
        for sp in ('train', 'val'):
            for d in ('images', 'labels'):
                src = os.path.join(exp, d, sp)
                dst = os.path.join(snap, d, sp)
                os.makedirs(dst, exist_ok=True)
                for fn in os.listdir(src) if os.path.isdir(src) else []:
                    os.link(os.path.join(src, fn), os.path.join(dst, fn))
        with open(os.path.join(exp, 'dataset.yaml'), encoding='utf-8') as f:
            y = f.read()
        y = re.sub(r'^path: .*$', f'path: {snap}', y, flags=re.M)
        with open(os.path.join(snap, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(y)
        shutil.copy2(man, os.path.join(snap, 'export_manifest.csv'))
    return snap, h, len(stems)


def parse_gates(path, name):
    """gate_report 에서 ① name 의 top-1 IoU≥0.95 비율, ② base/name 의 검수 val 평균 IoU."""
    g1 = g2b = g2n = None
    sec = ''
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('## '):
                sec = line
                continue
            if not line.startswith('| '):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if '게이트 1' in sec and cells[0] == name:
                g1 = float(cells[1].rstrip('%')) / 100
            elif '게이트 2' in sec and cells[0] in ('base', name):
                v = float(cells[2])
                if cells[0] == 'base':
                    g2b = v
                else:
                    g2n = v
    return g1, g2b, g2n


def history_rows():
    p = os.path.join(CFG['rt'], 'history.csv')
    if not os.path.isfile(p):
        return []
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def append_history(row):
    p = os.path.join(CFG['rt'], 'history.csv')
    fields = ['tag', 'at', 'base', 'n_auto', 'data_hash', 'gate1', 'gate2_base', 'gate2_new', 'promoted', 'best', 'sec', 'note']
    new = not os.path.isfile(p)
    with open(p, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    a = ap.parse_args()
    os.makedirs(CFG['rt'], exist_ok=True)
    if a.status:
        print(f'current.pt → {current_base()}')
        for r in history_rows():
            print('  ' + ' '.join(f'{k}={v}' for k, v in r.items() if k not in ('best',)))
        return 0
    lock = open(os.path.join(CFG['rt'], '.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log('이미 재학습 중')
        return 0
    t0 = time.time()
    tag = time.strftime('%Y%m%d_%H%M')
    base = current_base()
    snap = snapshot(tag)
    if not snap:
        return 0
    snap_dir, h, n = snap
    hist = history_rows()
    last = hist[-1] if hist else None
    if last and not a.force:
        if last['data_hash'] == h:
            log(f'새 ok 데이터 없음 (해시 {h}, {n}장) → 건너뜀')
            shutil.rmtree(snap_dir, ignore_errors=True)
            return 0
        if n - int(last['n_auto']) < MIN_NEW:
            log(f'새 ok 데이터 {n - int(last["n_auto"])}장 < {MIN_NEW} → 건너뜀')
            shutil.rmtree(snap_dir, ignore_errors=True)
            return 0
    name = f'auto_{tag}'
    log(f'학습 {name}: base={base} 자동 라벨 {n}장 (해시 {h}) + 검수 {CFG["rev"]} + 리허설 {len(CFG["reh"])}개')
    cmd = [PY, os.path.join(HERE, 'train_round.py'), CFG['rev'], '--base', base,
           '--extra-data', *CFG['reh'], os.path.join(snap_dir, 'dataset.yaml'),
           '--runs', CFG['runs'], '--name', name, '--epochs', str(a.epochs), '--lr0', str(CFG['lr0']),
           '--freeze', '0', '--batch', str(CFG['batch']), '--device', CFG['device'], '--patience', '8']
    r = subprocess.run(cmd, stdout=open(os.path.join(CFG['rt'], f'train_{tag}.out'), 'w'), stderr=subprocess.STDOUT)
    best = os.path.join(CFG['runs'], name, 'weights', 'best.pt')
    if r.returncode != 0 or not os.path.isfile(best):
        log(f'학습 실패 rc={r.returncode} (train_{tag}.out 참고)')
        append_history(dict(tag=tag, at=time.strftime('%Y-%m-%d %H:%M'), base=base, n_auto=n, data_hash=h, promoted=0,
                            sec=round(time.time() - t0), note=f'train failed rc={r.returncode}'))
        shutil.rmtree(snap_dir, ignore_errors=True)
        return 1
    log(f'게이트 (base {os.path.basename(os.path.dirname(os.path.dirname(base)))} 대비)')
    rep = os.path.join(CFG['rt'], f'gate_{tag}.md')
    cmd = [PY, os.path.join(HERE, 'eval_gates.py'), '--base', base, '--new', f'auto={best}', '--gate', CFG['gate'],
           '--reviewed', CFG['valnew'], '--rehearsal', CFG['gate_reh'], '--out', rep, '--device', CFG['device']]
    r = subprocess.run(cmd, stdout=open(os.path.join(CFG['rt'], f'gate_{tag}.out'), 'w'), stderr=subprocess.STDOUT)
    g1, g2b, g2n = parse_gates(rep, 'auto') if os.path.isfile(rep) else (None, None, None)
    ok = g1 is not None and g2n is not None and g1 >= GATE1_MIN and g2n >= (g2b or 0) - GATE2_DROP
    note = f'gate1={g1} gate2 base={g2b} new={g2n}'
    if ok:
        cur = os.path.join(CFG['rt'], 'current.pt')
        tmp = cur + '.tmp'
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(best, tmp)
        os.replace(tmp, cur)
        log(f'승격: current.pt → {best}')
    else:
        log(f'게이트 실패 → current.pt 유지 ({note})')
    append_history(dict(tag=tag, at=time.strftime('%Y-%m-%d %H:%M'), base=base, n_auto=n, data_hash=h,
                        gate1=None if g1 is None else round(g1, 4), gate2_base=g2b, gate2_new=g2n,
                        promoted=int(ok), best=best, sec=round(time.time() - t0), note=note))
    shutil.rmtree(snap_dir, ignore_errors=True)
    log(f'완료 {name} promoted={int(ok)} {round(time.time() - t0)}s')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
