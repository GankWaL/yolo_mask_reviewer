"""마스크<->폴리곤 왕복, 편집 연산, 데이터셋/내보내기 검증. 실행: python3 -m pytest tests -q  또는 python3 tests/test_maskops.py DATASET"""
import os, sys, glob, tempfile, shutil
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mask_reviewer import maskops
from mask_reviewer.dataset import Dataset
from mask_reviewer.export import export_dataset, split_stems

DS = os.environ.get('MR_TEST_DS', '')


def test_roundtrip_real_labels():
    assert DS, 'MR_TEST_DS 환경변수로 데이터셋 폴더 지정'
    ious = []
    nverts = []
    for p in sorted(glob.glob(os.path.join(DS, 'labels', '*.txt')))[:40]:
        img = cv2.imread(os.path.join(DS, 'images', os.path.basename(p)[:-4] + '.png'))
        h, w = img.shape[:2]
        for line in open(p):
            r = maskops.parse_yolo_line(line, w, h)
            if r is None:
                continue
            cls, pts = r
            m = maskops.polygon_to_mask(pts, h, w)
            poly = maskops.mask_to_polygon(m, eps=0.7, min_area=16)
            assert poly is not None
            m2 = maskops.polygon_to_mask(poly, h, w)
            ious.append(maskops.iou(m, m2))
            nverts.append((len(pts), len(poly)))
            # 라인 문자열 재파싱
            line2 = maskops.yolo_line(cls, poly, w, h)
            cls2, pts2 = maskops.parse_yolo_line(line2, w, h)
            assert cls2 == cls and len(pts2) == len(poly)
    assert ious and min(ious) > 0.97, (min(ious), np.mean(ious))
    print(f'roundtrip n={len(ious)} iou min {min(ious):.4f} mean {np.mean(ious):.4f} '
          f'verts {np.mean([a for a, _ in nverts]):.0f} -> {np.mean([b for _, b in nverts]):.0f}')


def test_edit_ops():
    m = np.zeros((100, 100), bool)
    maskops.brush(m, (10, 10), (60, 10), 5, True)
    assert m[10, 10] and m[10, 60] and m[10, 35] and not m[30, 30]
    a0 = m.sum()
    maskops.brush(m, (30, 10), (40, 10), 5, False)
    assert m.sum() < a0 and not m[10, 35]
    maskops.fill_polygon(m, [(50, 50), (90, 50), (90, 90), (50, 90)], True)
    assert m[70, 70]
    maskops.fill_polygon(m, [(60, 60), (80, 60), (80, 80), (60, 80)], False)
    assert not m[70, 70] and m[55, 55]
    filled = maskops.fill_holes(m)
    assert filled[70, 70]
    big = maskops.keep_largest(filled)
    assert big[70, 70] and not big[10, 10]
    # 조각 2개 -> 폴리곤 1개로 이어짐
    two = np.zeros((100, 100), bool)
    two[10:30, 10:30] = True
    two[60:90, 60:90] = True
    poly = maskops.mask_to_polygon(two, eps=0.5)
    back = maskops.polygon_to_mask(poly, 100, 100)
    assert maskops.iou(two, back) > 0.95, maskops.iou(two, back)
    # 완드
    img = np.full((50, 50, 3), 200, np.uint8)
    img[10:20, 10:20] = (20, 20, 20)
    r = maskops.wand_region(img, (15, 15), 10, blur=0)
    assert r.sum() == 100 and r[15, 15]
    # 조각 분리 / 자르기
    two = np.zeros((60, 60), bool); two[5:25, 5:25] = True; two[35:55, 35:55] = True
    parts = maskops.split_components(two)
    assert len(parts) == 2 and parts[0].sum() == 400
    bar = np.zeros((60, 60), bool); bar[20:40, 5:55] = True   # 가로 막대 하나를 세로선으로 자르기
    cut = maskops.cut_mask(bar, [(30, 0), (30, 59)], thickness=3)
    assert len(cut) == 2 and abs(int(cut[0].sum()) - int(cut[1].sum())) <= 60 and (cut[0] | cut[1]).sum() == bar.sum(), \
        [int(c.sum()) for c in cut]
    assert len(maskops.cut_mask(bar, [(0, 10), (59, 10)])) == 1   # 안 가로지르면 그대로
    # grabcut 은 실행만 확인
    g = maskops.grabcut_refine(img, r, margin=5, band=5, iters=1)
    assert g.shape == r.shape


def test_dataset_and_export():
    assert DS
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'ds')
        shutil.copytree(DS, root)
        # 검수 중인 데이터셋을 줘도 되게 복사본의 검수 상태·편집본·자동 라벨은 지우고 시작
        for d in ('labels_reviewed', 'labels_auto', '.thumb_cache', '.sam2_cache'):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        if os.path.exists(os.path.join(root, 'review_state.json')):
            os.remove(os.path.join(root, 'review_state.json'))
        ds = Dataset(root)
        assert ds.names and ds.stems
        s = ds.stems[0]
        img, insts = ds.load(s)
        h, w = img.shape[:2]
        n0 = len(insts)
        # 편집 저장 -> 편집본 경로 사용
        maskops.brush(insts[0].mask if insts else None, (5, 5), (5, 5), 3, True) if insts else None
        ds.save(s, insts, w, h)
        assert ds.is_edited(s) and ds.effective_label_path(s).endswith('labels_reviewed/' + s + '.txt')
        _, insts2 = ds.load(s)
        assert len(insts2) == n0
        ds.state.set(s, status='ok', note='테스트')
        ds.state.set(ds.stems[1], status='ok')
        ds.state.set(ds.stems[2], status='reject')
        ds2 = Dataset(root)
        assert ds2.state.status(s) == 'ok' and ds2.state.note(s) == '테스트'
        # 되돌리기
        ds.revert(s)
        assert not ds.is_edited(s)
        ds.save(s, insts, w, h)
        # 자동 라벨 층: 편집본 > 자동 > 원본
        s2 = ds.stems[1]
        img2, insts2b = ds.load(s2)
        h2, w2 = img2.shape[:2]
        ds.write_auto(s2, insts2b, w2, h2, source='sam2', score=0.9, ref=s)
        assert ds.is_auto(s2) and ds.label_source(s2) == 'auto' and ds.state.auto(s2)['score'] == 0.9
        assert ds.state.auto(s2)['diff'] == 0.0            # 원본 그대로 넣었으니 차이 0
        insts2c = [i.copy() for i in insts2b]
        if insts2c:
            insts2c[0].mask[:] = False
            insts2c[0].mask[:h2 // 2, :w2 // 2] = True
        ds.write_auto(s2, insts2c, w2, h2, source='sam2', score=0.9)
        assert ds.state.auto(s2)['diff'] > 0.3
        ds.state.data[s2]['auto'].pop('diff'); ds.state.save()
        assert ds.rescore_auto() >= 1 and ds.state.auto(s2)['diff'] > 0.3
        ds.write_auto(s2, insts2b, w2, h2, source='sam2', score=0.9, ref=s)
        assert ds.effective_label_path(s2).endswith('labels_auto/' + s2 + '.txt')
        ds.save(s2, insts2b, w2, h2)
        assert ds.label_source(s2) == 'reviewed' and not ds.is_auto(s2)
        ds.revert(s2)
        assert ds.label_source(s2) == 'auto'
        ds.set_exemplar(s)
        assert ds.exemplars(ds.code(s)) == [s] and s2 not in ds.exemplars()
        assert s not in ds.auto_targets() and s2 not in ds.auto_targets(include_auto=False)
        ds.clear_auto(s2)
        assert ds.label_source(s2) == 'original' and not ds.state.auto(s2)
        # 빈 편집본(인스턴스 전부 삭제 후 저장)은 기본 대상에서 빠지고, include_empty_edited 로만 들어온다
        s3 = ds.stems[3]
        ds.save(s3, [], w2, h2)
        assert ds.is_edited(s3) and ds.reviewed_is_empty(s3) and not ds.reviewed_is_empty(s2)
        assert s3 not in ds.auto_targets() and s3 in ds.auto_targets(include_empty_edited=True)
        ds.revert(s3)
        # 층화 분할
        sp = split_stems(ds.stems, ds.code, 0.2, 0)
        assert set(sp.values()) <= {'train', 'val'}
        # 내보내기 (ok 만)
        out = os.path.join(tmp, 'out')
        r = export_dataset(ds, out, statuses=('ok',), val_ratio=0.5, seed=1, copy_mode='hardlink')
        assert r['n_total'] == 2 and r['n_edited'] == 1, r
        assert 'label_src' in open(os.path.join(out, 'export_manifest.csv')).readline()
        assert os.path.exists(os.path.join(out, 'dataset.yaml'))
        lbl = glob.glob(os.path.join(out, 'labels', '*', '*.txt'))
        imgs = glob.glob(os.path.join(out, 'images', '*', '*.png'))
        assert len(lbl) == 2 and len(imgs) == 2
        for p in lbl:
            for line in open(p):
                v = line.split()
                assert len(v) >= 7 and len(v) % 2 == 1 and all(0 <= float(x) <= 1 for x in v[1:])
        # 분할 없음
        out2 = os.path.join(tmp, 'out2')
        r2 = export_dataset(ds, out2, statuses=('ok', 'reject'), val_ratio=0.0, copy_mode='symlink')
        assert r2['n_total'] == 3 and os.path.islink(glob.glob(os.path.join(out2, 'images', '*.png'))[0])
        print('export', r, r2['n_total'])
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DS = sys.argv[1]
    test_roundtrip_real_labels()
    test_edit_ops()
    test_dataset_and_export()
    print('OK')
