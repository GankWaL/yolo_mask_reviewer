"""실행 진입점.

  python -m mask_reviewer [DATASET_DIR]                       GUI
  python -m mask_reviewer export DATASET_DIR OUT_DIR [옵션]   GUI 없이 내보내기
  python -m mask_reviewer stats DATASET_DIR                   검수 현황 출력
  python -m mask_reviewer propagate DATASET_DIR [옵션]        대표 → 보류 이미지 SAM2 전파 (GUI 없이)
  python -m mask_reviewer rescore DATASET_DIR                 기존 자동 라벨의 원본 대비 변화량(diff) 소급 계산
"""
import argparse
import faulthandler
import signal
import sys

# 멈춤 진단용: `kill -USR1 <pid>` 로 모든 스레드의 파이썬 스택을 stderr 에 찍는다 (sudo·py-spy 불필요)
if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1, all_threads=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('export', 'stats', 'propagate', 'rescore'):
        return _cli(argv)
    ap = argparse.ArgumentParser(description='YOLO-seg 마스크 검수 GUI')
    ap.add_argument('dataset', nargs='?', help='images/ labels/ 가 있는 데이터셋 폴더')
    a = ap.parse_args(argv)
    from PyQt5.QtWidgets import QApplication
    from .app import MainWindow
    app = QApplication(sys.argv)
    win = MainWindow(a.dataset)
    win.show()
    return app.exec_()


def _cli(argv):
    from .dataset import Dataset, STATUSES, STATUS_LABEL
    ap = argparse.ArgumentParser(prog='mask_reviewer')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export', help='검수 결과를 YOLO 학습 폴더로 내보내기')
    e.add_argument('dataset')
    e.add_argument('out')
    e.add_argument('--status', default='ok', help='포함할 상태, 쉼표 구분 (ok,pending,reject)')
    e.add_argument('--val', type=float, default=0.1, help='val 비율 (0 이면 분할 없음)')
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--mode', default='copy', choices=('copy', 'hardlink', 'symlink'))
    e.add_argument('--eps', type=float, default=0.7)
    e.add_argument('--min-area', type=float, default=16.0)
    s = sub.add_parser('stats', help='검수 현황')
    s.add_argument('dataset')
    pp = sub.add_parser('propagate', help='대표(★) 라벨을 같은 제품의 보류 이미지로 SAM2 전파 → labels_auto/')
    pp.add_argument('dataset')
    pp.add_argument('--codes', default='', help='제품코드 쉼표 목록 (기본: 대표가 있는 모든 제품)')
    pp.add_argument('--max-exemplars', type=int, default=3)
    pp.add_argument('--skip-auto', action='store_true', help='이미 자동 라벨이 있는 이미지는 건너뜀')
    pp.add_argument('--include-empty-edited', action='store_true',
                    help='편집본이 비어 있는 보류 이미지도 포함 (그 빈 편집본은 삭제)')
    pp.add_argument('--ckpt', default=None, help='SAM2 체크포인트 (.pt)')
    rs = sub.add_parser('rescore', help='labels_auto/ 의 자동 라벨마다 원본 대비 변화량(1-IoU)을 계산해 state 에 기록')
    rs.add_argument('dataset')
    a = ap.parse_args(argv)
    ds = Dataset(a.dataset)
    if a.cmd == 'stats':
        c = ds.state.counts(ds.stems)
        print(' · '.join(f'{STATUS_LABEL[k]} {c[k]}' for k in STATUSES)
              + f' · 편집됨 {ds.count_edited()} · 자동 {ds.count_auto()} · 대표 {len(ds.exemplars())} / {len(ds.stems)}')
        per = {}
        for st in ds.stems:
            e = per.setdefault(ds.code(st), [0, 0, 0, 0])
            e[0] += 1
            e[1] += ds.state.status(st) == 'ok'
            e[2] += ds.is_auto(st)
            e[3] += ds.state.exemplar(st)
        print(f'  {"제품":16s} {"확정":>5s} {"자동":>5s} {"대표":>5s} / 전체')
        for code, (n, ok, au, ex) in sorted(per.items()):
            print(f'  {code or "(없음)":16s} {ok:5d} {au:5d} {ex:5d} / {n}')
        return 0
    if a.cmd == 'rescore':
        n = ds.rescore_auto(progress=lambda i, m, st: print(f'\r{i + 1}/{m} {st}    ', end='', flush=True))
        ds2 = Dataset(a.dataset)
        d = sorted(ds2.state.auto(s).get('diff', 0.0) for s in ds2.stems if ds2.has_auto(s))
        print(f'\n{n}장 계산 · 차이 평균 {sum(d) / max(len(d), 1):.3f} · 0.05 이상 {sum(v >= 0.05 for v in d)} · 0.2 이상 {sum(v >= 0.2 for v in d)}')
        return 0
    if a.cmd == 'propagate':
        from .propagate import propagate_dataset
        codes = [c for c in a.codes.split(',') if c] or None
        r = propagate_dataset(ds, codes, a.max_exemplars, include_auto=not a.skip_auto, ckpt=a.ckpt,
                              include_empty_edited=a.include_empty_edited,
                              progress=lambda i, n, st, info: print(f'\r{i + 1}/{n} {st} {info}    ', end='', flush=True))
        print()
        for k, v in r.items():
            print(f'{k}: {v}')
        return 0
    from .export import export_dataset
    r = export_dataset(ds, a.out, statuses=tuple(a.status.split(',')), val_ratio=a.val, seed=a.seed,
                       copy_mode=a.mode, eps=a.eps, min_area=a.min_area,
                       progress=lambda i, n, st: print(f'\r{i + 1}/{n}', end='', flush=True))
    print()
    for k, v in r.items():
        print(f'{k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
