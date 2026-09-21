"""오프스크린 Qt 스모크: 열기 -> 브러시/폴리곤/완드/꼭짓점/SAM2/undo -> 판정 -> 저장 -> 썸네일 -> 내보내기. 
실행: QT_QPA_PLATFORM=offscreen MR_TEST_DS=<ds> python3 tests/test_gui_smoke.py [screenshot_dir]"""
import os, sys, shutil, tempfile, glob
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from mask_reviewer import maskops
from mask_reviewer.app import MainWindow
from mask_reviewer.export import export_dataset


def run(ds_src, shot_dir=None):
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, 'ds')
    shutil.copytree(ds_src, root)
    for d in ('labels_reviewed', 'labels_auto', '.thumb_cache', '.sam2_cache'):   # 검수 중인 데이터셋도 쓸 수 있게 초기화
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    if os.path.exists(os.path.join(root, 'review_state.json')):
        os.remove(os.path.join(root, 'review_state.json'))
    app = QApplication.instance() or QApplication([])
    w = MainWindow(root)
    w.resize(1500, 900)
    w.show()
    QTest.qWaitForWindowExposed(w)
    app.processEvents()
    assert w.ds is not None and w.stem == w.ds.stems[0]
    c = w.canvas
    n0 = len(w.instances)
    a0 = w.instances[0].area if n0 else 0

    def wpt(x, y):
        p = c.to_widget(QPointF(x, y))
        return QPoint(int(p.x()), int(p.y()))

    # 브러시 칠하기 (좌클릭 드래그)
    w.set_tool('brush')
    w.brush_spin.setValue(15)
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, wpt(20, 20))
    QTest.mouseMove(c, wpt(120, 20))
    c.mouseMoveEvent(type('E', (), {'pos': lambda s: wpt(120, 20), 'buttons': lambda s: Qt.LeftButton})())
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, wpt(120, 20))
    app.processEvents()
    assert w.dirty and w.instances[0].mask[20, 70], 'brush stroke failed'
    a1 = w.instances[0].area
    assert a1 > a0
    # undo / redo
    w.undo(); assert w.instances[0].area == a0
    w.redo(); assert w.instances[0].area == a1
    # 지우개 (우클릭)
    QTest.mousePress(c, Qt.RightButton, Qt.NoModifier, wpt(70, 20))
    QTest.mouseRelease(c, Qt.RightButton, Qt.NoModifier, wpt(70, 20))
    assert not w.instances[0].mask[20, 70]
    # 새 인스턴스 + 폴리곤
    w.set_class(1)
    w.new_instance()
    assert len(w.instances) == n0 + 1 and w.instances[-1].cls == 1
    w.set_tool('polygon')
    for x, y in ((300, 300), (400, 300), (400, 400), (300, 400)):
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(x, y))
    QTest.keyClick(c, Qt.Key_Return)
    assert w.instances[-1].mask[350, 350] and w.instances[-1].area > 9000, w.instances[-1].area
    # Ctrl+Enter 빼기
    for x, y in ((320, 320), (380, 320), (380, 380), (320, 380)):
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(x, y))
    QTest.keyClick(c, Qt.Key_Return, Qt.ControlModifier)
    assert not w.instances[-1].mask[350, 350] and w.instances[-1].mask[305, 305]
    # 완드
    w.set_tool('wand')
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(600, 600))
    assert w.instances[-1].mask[600, 600]
    # 꼭짓점 편집: 드래그 이동 / 변 클릭 추가 / 우클릭 삭제
    w.set_tool('vertex')
    assert c.vert_polys, 'vertex trace failed'
    nv = len(c.vert_polys[0])
    vx, vy = c.vert_polys[0][0]
    a_v = w.instances[-1].area
    QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, wpt(vx, vy))
    assert c._vert_drag == (0, 0)
    c.mouseMoveEvent(type('E', (), {'pos': lambda s: wpt(vx + 30, vy + 30), 'buttons': lambda s: Qt.LeftButton})())
    QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, wpt(vx + 30, vy + 30))
    assert c._vert_drag is None and w.instances[-1].area != a_v, 'vertex drag failed'
    assert len(c.vert_polys[0]) == nv, 'vertex list re-traced after own edit'
    w.undo()
    assert w.instances[-1].area == a_v
    w.set_tool('vertex')
    import numpy as np
    poly = c.vert_polys[0]
    k = int(np.argmax(((np.roll(poly, -1, axis=0) - poly) ** 2).sum(1)))  # 가장 긴 변 (꼭짓점과 안 겹치게)
    mx, my = (poly[k] + poly[(k + 1) % len(poly)]) / 2.0
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(mx, my))
    assert len(c.vert_polys[0]) == nv + 1, 'vertex insert failed'
    QTest.mouseClick(c, Qt.RightButton, Qt.NoModifier, wpt(mx, my))
    assert len(c.vert_polys[0]) == nv, 'vertex delete failed'
    # 현재 인스턴스 비우기 (C) + 되돌리기
    a_c = w.instances[-1].area
    w.clear_instance()
    assert w.instances[-1].area == 0
    w.undo()
    assert w.instances[-1].area == a_c
    # SAM2 (torch/sam2 + 체크포인트가 있을 때만 실제 추론, 없으면 브러시로 되돌아가는지만 확인)
    from mask_reviewer import sam2_helper
    sam_ok = sam2_helper.available() and sam2_helper.find_checkpoint() is not None
    if not sam_ok:
        w._sam_failed = True  # 안내 대화상자 없이 폴백
    w.set_tool('sam')
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(350, 350))
    if sam_ok:
        assert c.sam_proposal is not None and c.sam_proposal.any(), 'SAM proposal empty'
        n_i = len(w.instances)
        w.new_instance()
        w.set_tool('sam')
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(350, 350))
        QTest.keyClick(c, Qt.Key_Return)
        assert len(w.instances) == n_i + 1 and w.instances[-1].area > 0 and c.sam_proposal is None
        w.delete_instance()
        print('SAM2 smoke OK')
    else:
        assert c.tool == 'brush' and not c.sam_points, 'SAM fallback failed'
        print('SAM2 skipped (torch/sam2 or checkpoint missing)')
    # 자르기: 마지막 인스턴스(폴리곤 사각형 + 완드 영역)를 세로선으로 가르기 → 인스턴스 증가
    n_c = len(w.instances)
    w.set_tool('cut')
    for x, y in ((350, 280), (350, 420)):
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(x, y))
    QTest.keyClick(c, Qt.Key_Return)
    assert len(w.instances) > n_c, 'cut failed'
    w.undo()
    assert len(w.instances) == n_c
    # 조각 분리: 사각형과 완드 영역은 떨어져 있으면 분리된다
    n_p = len(maskops.split_components(w.instances[-1].mask, 16))
    w.canvas.set_current(len(w.instances) - 1)
    w.split_instance()
    assert len(w.instances) == n_c + n_p - 1
    w.undo()
    # 선택 도구로 고르기
    w.set_tool('select')
    QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, wpt(305, 305))
    assert c.current == len(w.instances) - 1
    # 일괄 클래스 변경: 인스턴스 목록에서 Shift+드래그로 범위 선택 → 숫자키(1~9) 한 번에 적용 → 되돌리기 한 번
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent
    assert len(w.instances) >= 2
    il = w.inst_list
    vp = il.viewport()
    rc = lambda r: il.visualItemRect(il.item(r)).center()
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, rc(0))          # 기준 행 클릭 (Shift 범위의 시작점)
    QTest.mousePress(vp, Qt.LeftButton, Qt.ShiftModifier, rc(min(1, len(w.instances) - 1)))
    app.sendEvent(vp, QMouseEvent(QEvent.MouseMove, rc(len(w.instances) - 1), Qt.LeftButton, Qt.LeftButton, Qt.ShiftModifier))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.ShiftModifier, rc(len(w.instances) - 1))
    sel = sorted(i.row() for i in il.selectedIndexes())
    assert sel == list(range(len(w.instances))), sel
    cls_before = [i.cls for i in w.instances]
    new_cls = next(k for k in sorted(w.ds.names) if k not in cls_before) if len(w.ds.names) > len(set(cls_before)) else sorted(w.ds.names)[-1]
    n_undo = len(w.undo_stack)
    w.set_class(new_cls)
    assert all(i.cls == new_cls for i in w.instances), [i.cls for i in w.instances]
    assert len(w.undo_stack) == n_undo + 1, 'batch class change must be one undo step'
    assert sorted(i.row() for i in il.selectedIndexes()) == sel, 'selection lost after batch class change'
    w.set_class(new_cls)                              # 같은 클래스면 변화 없음 (undo 스택 그대로)
    assert len(w.undo_stack) == n_undo + 1
    w.undo()
    assert [i.cls for i in w.instances] == cls_before
    # 병합: 목록에서 2개 선택
    w.inst_list.selectAll()
    w.merge_instances()
    assert len(w.instances) == 1
    # 기타 연산
    w.fill_holes(); w.keep_largest(); w.smooth(); w.grabcut()
    # 판정 + 저장 + 다음
    s0 = w.stem
    w.set_status('ok', advance=True)
    assert w.ds.state.status(s0) == 'ok' and w.ds.is_edited(s0) and w.stem == w.ds.stems[1]
    w.set_status('reject', advance=True)
    w.navigate(-1); w.navigate(-1)
    assert w.stem == s0 and not w.dirty
    # 필터
    w.status_filter.setCurrentIndex(2)  # 확정
    assert w.list.count() == 1
    w.status_filter.setCurrentIndex(0)
    w.code_filter.setCurrentIndex(1)
    assert 0 < w.list.count() < len(w.ds.stems)
    w.code_filter.setCurrentIndex(0)
    # 원본 복원 (대화상자 없이 직접)
    w.ds.revert(s0)
    assert not w.ds.is_edited(s0)
    w.stem = None; w.load_stem(s0)
    assert len(w.instances) == n0
    # 대표 지정 -> SAM2 전파 (헤드리스 함수) -> 자동 라벨 층 / 필터 / 정렬
    w.status_filter.setCurrentIndex(0); w.code_filter.setCurrentIndex(0)

    def goto(stem):
        for i in range(w.list.count()):
            if w.list.item(i).data(Qt.UserRole) == stem:
                w.list.setCurrentRow(i)
                if w.stem != stem:          # 이미 현재 행이면 신호가 안 나므로 직접 로드
                    w.stem = None; w.load_stem(stem)
                return
        raise AssertionError(stem)
    codes = w.ds.codes()
    ex = max((t for t in w.ds.stems if t != s0), key=lambda t: codes[w.ds.code(t)])   # 이미지가 가장 많은 제품의 한 장을 대표로 (s0 는 마지막 내보내기 검사용 확정분이라 제외)
    goto(ex)
    w.toggle_exemplar()
    assert w.ds.state.exemplar(ex) and w.ds.state.status(ex) == 'ok' and w.ds.is_edited(ex)
    assert w.exemplar_btn.isChecked() and '★' in w.list.currentItem().text()
    if sam_ok:
        from mask_reviewer.propagate import propagate_dataset
        # 대상 하나에 빈 편집본을 만들어 두고 include_empty_edited 로 지워지는지 본다
        t_empty = w.ds.auto_targets(w.ds.code(ex))[0]
        w.ds.save(t_empty, [], 10, 10)
        assert t_empty not in w.ds.auto_targets(w.ds.code(ex))
        r = propagate_dataset(w.ds, [w.ds.code(ex)], max_exemplars=1, max_targets=3, propagator=w.propagator,
                              include_empty_edited=True)
        w.propagator = None
        assert r['n_done'] == 3 and r['n_empty'] == 0 and r['n_cleared'] == 1, r
        assert not w.ds.is_edited(t_empty) and w.ds.is_auto(t_empty)
        w.refresh_list()
        tgt = [t for t in w.ds.stems if w.ds.is_auto(t)][0]
        assert w.ds.label_source(tgt) == 'auto' and w.ds.state.auto(tgt)['source'] == 'sam2'
        assert '⟳' in [w.list.item(i).text() for i in range(w.list.count()) if w.list.item(i).data(Qt.UserRole) == tgt][0]
        w.status_filter.setCurrentIndex(5)  # 자동 라벨
        assert w.list.count() == 3
        w.sort_combo.setCurrentIndex(1)   # 원본과 차이 큰 순
        df = [w.ds.state.auto(w.list.item(i).data(Qt.UserRole))['diff'] for i in range(w.list.count())]
        assert df == sorted(df, reverse=True)
        w.sort_combo.setCurrentIndex(2)   # 점수 낮은 순
        sc = [w.ds.state.auto(w.list.item(i).data(Qt.UserRole))['score'] for i in range(w.list.count())]
        assert sc == sorted(sc)
        w.status_filter.setCurrentIndex(0)
        w.sort_combo.setCurrentIndex(3)   # 제품 코드 순 (코드 없는 stem 은 맨 뒤, 같은 제품 안은 파일명 순)
        order = [w.list.item(i).data(Qt.UserRole) for i in range(w.list.count())]
        assert order == sorted(order, key=lambda t: ((0, w.ds.code(t)) if w.ds.code(t) else (1, '')) + (t,))
        w.sort_combo.setCurrentIndex(0)
        # 제품코드 변경: 덮어쓰기 → 필터·정보·코드 목록 반영, 원본으로 되돌리면 덮어쓰기 삭제
        goto(tgt)
        orig_code = w.ds.orig_code(tgt)
        other = sorted(c for c in codes if c and c != orig_code)[0]
        assert w.apply_code(other) and w.ds.code(tgt) == other and w.ds.state.code(tgt) == other
        assert w.ds.info(tgt)['code'] == other and '수정됨' in w.info.text()
        assert w.code_filter.findData(other) > 0 and w.ds.codes()[other] == codes[other] + 1
        assert not w.apply_code(other)                     # 같은 값이면 변화 없음
        assert w.apply_code(orig_code) and not w.ds.state.code(tgt) and w.ds.code(tgt) == orig_code and '수정됨' not in w.info.text()
        assert w.ds.set_code(tgt, ' ' + other.lower() + ' ') and w.ds.code(tgt) == other   # 공백·소문자 정규화
        w.ds.set_code(tgt, '')                             # 비우면 원본
        assert w.ds.code(tgt) == orig_code
        # 제품코드 일괄 변경: 이미지 목록에서 Shift+클릭으로 2장 이상 선택 → 한 번에 덮어쓰기 → 각자 원본으로
        goto(tgt)
        r0 = w.list.currentRow()
        r1 = min(r0 + 2, w.list.count() - 1)
        QTest.mouseClick(w.list.viewport(), Qt.LeftButton, Qt.ShiftModifier, w.list.visualItemRect(w.list.item(r1)).center())
        stems_sel = w._selected_stems()
        assert len(stems_sel) == r1 - r0 + 1 and tgt in stems_sel, stems_sel
        assert w.apply_code(other, stems_sel) and all(w.ds.code(t) == other for t in stems_sel)
        assert sorted(w._selected_stems()) == sorted(stems_sel), '일괄 변경 뒤 선택 유지 실패'
        assert not w.apply_code(other, stems_sel)                       # 전부 같은 값이면 변화 없음
        assert w.apply_code(None, stems_sel) and all(not w.ds.state.code(t) for t in stems_sel)   # 각자 원본으로
        w.list.clearSelection()
        goto(tgt)
        assert len(w.instances) >= 1 and '자동' in w.info.text()
        # 확정하면 자동 라벨이 그대로 유효 (편집본 없음)
        w.set_status('ok', advance=False)
        assert w.ds.label_source(tgt) == 'auto'
        w.clear_auto_current()
        assert not w.ds.has_auto(tgt) and w.ds.label_source(tgt) == 'original'
        w.ds.state.set(tgt, status='pending')
        for t in [t for t in w.ds.stems if w.ds.has_auto(t)]:
            w.ds.clear_auto(t)
        print('propagate smoke OK', r['n_done'])
    w.status_filter.setCurrentIndex(6)  # 대표
    assert w.list.count() == 1
    w.status_filter.setCurrentIndex(0)
    goto(ex)
    w.toggle_exemplar()
    assert not w.ds.state.exemplar(ex) and not w.exemplar_btn.isChecked()
    w.ds.revert(ex); w.ds.state.set(ex, status='pending')
    goto(s0)
    # 썸네일: 백그라운드 생성 후 목록 아이콘 + 캐시 폴더
    for _ in range(200):
        if len(w._thumbs) >= len(w.ds.stems):
            break
        QTest.qWait(50)
    assert len(w._thumbs) == len(w.ds.stems), f'thumbs {len(w._thumbs)}/{len(w.ds.stems)}'
    assert not w.list.item(0).icon().isNull() and os.path.isdir(os.path.join(root, '.thumb_cache'))
    w.act_thumbs.setChecked(False); w.toggle_thumbs()
    assert w.list.item(0).icon().isNull()
    w.act_thumbs.setChecked(True); w.toggle_thumbs()
    assert not w.list.item(0).icon().isNull()
    # 스크린샷 (썸네일·꼭짓점 도구가 보이게)
    if shot_dir:
        os.makedirs(shot_dir, exist_ok=True)
        w.list.scrollToTop()
        w.set_tool('vertex')
        c.zoom_at(1.6, wpt(342, 342))
        app.processEvents()
        w.grab().save(os.path.join(shot_dir, 'main.png'))
        w.set_tool('brush')
    # 내보내기 (헤드리스)
    out = os.path.join(tmp, 'out')
    r = export_dataset(w.ds, out, statuses=('ok',), val_ratio=0.0)
    assert r['n_total'] == 1
    assert 'label_src' in open(os.path.join(out, 'export_manifest.csv')).readline()
    w.close()
    shutil.rmtree(tmp)
    print('GUI smoke OK', r['n_total'])


if __name__ == '__main__':
    run(os.environ['MR_TEST_DS'], sys.argv[1] if len(sys.argv) > 1 else None)
