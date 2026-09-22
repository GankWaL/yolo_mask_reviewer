"""YOLO-seg 마스크 검수 GUI 메인 윈도우."""
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PyQt5.QtCore import QSettings, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (QAction, QActionGroup, QApplication, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider, QSpinBox,
                             QSplitter, QToolBar, QVBoxLayout, QWidget, QAbstractItemView)

from . import maskops, propagate, sam2_helper
from .canvas import TOOL_KEY, TOOL_LABEL, TOOLS, MaskCanvas, class_color
from .dataset import LABEL_SOURCE_LABEL, STATUSES, STATUS_LABEL, STATUS_MARK, Dataset, Instance, is_full_code
from .export import COPY_MODES, export_dataset

MAX_UNDO = 40
THUMB_PX = 72                    # 목록 썸네일 한 변 (px)
THUMB_CACHE_DIRNAME = '.thumb_cache'  # <데이터셋>/.thumb_cache/<stem>.jpg
SAVE_EPS = 0.7       # 저장 시 폴리곤 단순화 (px)
SAVE_MIN_AREA = 16.0  # 저장 시 버리는 조각 면적 (px²)


class ExportDialog(QDialog):
    def __init__(self, parent, ds, default_out):
        super().__init__(parent)
        self.setWindowTitle('YOLO 형식으로 내보내기')
        self.ds = ds
        lay = QFormLayout(self)
        row = QHBoxLayout()
        self.out = QLineEdit(default_out)
        b = QPushButton('찾기…')
        b.clicked.connect(self._browse)
        row.addWidget(self.out)
        row.addWidget(b)
        lay.addRow('출력 폴더', row)
        cnt = ds.state.counts(ds.stems)
        self.chk = {}
        srow = QHBoxLayout()
        for s in STATUSES:
            c = QCheckBox(f'{STATUS_LABEL[s]} ({cnt[s]})')
            c.setChecked(s == 'ok')
            self.chk[s] = c
            srow.addWidget(c)
        lay.addRow('포함할 상태', srow)
        self.val = QDoubleSpinBox()
        self.val.setRange(0.0, 0.5)
        self.val.setSingleStep(0.05)
        self.val.setValue(0.1)
        self.val.setToolTip('0 이면 분할 없이 images/ labels/ 평면 구조 (학습 서버에서 나눔)')
        lay.addRow('val 비율 (제품코드별 층화)', self.val)
        self.seed = QSpinBox()
        self.seed.setRange(0, 99999)
        lay.addRow('분할 seed', self.seed)
        self.mode = QComboBox()
        for m, t in zip(COPY_MODES, ('복사', '하드링크', '심볼릭 링크')):
            self.mode.addItem(t, m)
        lay.addRow('이미지 복사 방식', self.mode)
        self.eps = QDoubleSpinBox()
        self.eps.setRange(0.0, 5.0)
        self.eps.setSingleStep(0.1)
        self.eps.setValue(SAVE_EPS)
        lay.addRow('편집본 폴리곤 단순화 eps (px)', self.eps)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 5000)
        self.min_area.setValue(int(SAVE_MIN_AREA))
        lay.addRow('버릴 조각 면적 (px²)', self.min_area)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, '출력 폴더', self.out.text() or os.path.expanduser('~'))
        if d:
            self.out.setText(d)

    def params(self):
        return dict(out_dir=self.out.text().strip(),
                    statuses=tuple(s for s, c in self.chk.items() if c.isChecked()),
                    val_ratio=float(self.val.value()), seed=int(self.seed.value()),
                    copy_mode=self.mode.currentData(), eps=float(self.eps.value()),
                    min_area=float(self.min_area.value()))


class PropagateWorker(QThread):
    """SAM2 대표 전파를 백그라운드에서 돌린다. 모델(propagator)은 호출자가 넘기거나 여기서 만든다."""
    progress = pyqtSignal(int, int, str, str)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ds, codes, max_exemplars, include_auto, propagator=None, ckpt=None, include_empty_edited=False):
        super().__init__()
        self.ds, self.codes, self.max_exemplars, self.include_auto = ds, codes, max_exemplars, include_auto
        self.include_empty_edited = include_empty_edited
        self.propagator, self.ckpt = propagator, ckpt
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.propagator is None:
                self.propagator = propagate.Sam2Propagator(self.ckpt)
            r = propagate.propagate_dataset(self.ds, self.codes, self.max_exemplars, self.include_auto,
                                            propagator=self.propagator, progress=self.progress.emit,
                                            should_stop=lambda: self._stop,
                                            include_empty_edited=self.include_empty_edited)
            self.done.emit(r)
        except Exception as e:
            self.failed.emit(f'{e}\n{traceback.format_exc()[-800:]}')


class PropagateDialog(QDialog):
    def __init__(self, parent, ds, code):
        super().__init__(parent)
        self.setWindowTitle('SAM2 대표 전파')
        self.ds, self.code = ds, code
        lay = QFormLayout(self)
        self.scope = QComboBox()
        lay.addRow('범위', self.scope)
        self.max_ex = QSpinBox()
        self.max_ex.setRange(1, 10)
        self.max_ex.setValue(3)
        self.max_ex.setToolTip('제품당 대표를 최대 몇 장까지 써서 가장 점수 높은 결과를 고를지 (장수만큼 시간 증가)')
        lay.addRow('대표 최대 사용 수', self.max_ex)
        self.redo = QCheckBox('이미 자동 라벨이 있는 이미지도 다시 계산')
        self.redo.setChecked(True)
        lay.addRow(self.redo)
        n_empty = sum(1 for s in ds.stems if ds.state.status(s) == 'pending' and ds.reviewed_is_empty(s)
                      and not ds.state.exemplar(s))
        self.empty_edited = QCheckBox(f'비어 있는 편집본을 가진 보류 이미지도 포함 — 그 편집본은 삭제 ({n_empty}장)')
        self.empty_edited.setToolTip('인스턴스를 모두 지운 뒤 넘어가면 빈 편집본이 남아 자동 라벨보다 우선합니다. '
                                     '체크하면 그런 보류 이미지도 대상에 넣고 빈 편집본을 지웁니다.')
        self.empty_edited.setChecked(n_empty > 0)
        self.empty_edited.toggled.connect(self._fill_scope)
        lay.addRow(self.empty_edited)
        lay.addRow(QLabel('대상: 보류 상태이고 편집본·대표가 아닌 이미지. 결과는 labels_auto/ 에 쓰고 원본은 건드리지 않습니다.'))
        self._fill_scope()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def _fill_scope(self):
        ds, code = self.ds, self.code
        ie = self.empty_edited.isChecked()
        cur = self.scope.currentIndex()
        self.scope.blockSignals(True)
        self.scope.clear()
        n_cur = len(ds.auto_targets(code, include_empty_edited=ie)) if code and ds.exemplars(code) else 0
        all_codes = sorted({ds.code(s) for s in ds.exemplars()})
        n_all = sum(len(ds.auto_targets(c, include_empty_edited=ie)) for c in all_codes)
        self.scope.addItem(f'현재 제품 {code} (대표 {len(ds.exemplars(code)) if code else 0}장 → 보류 {n_cur}장)', [code] if code else [])
        self.scope.addItem(f'대표가 있는 모든 제품 ({len(all_codes)}종 → 보류 {n_all}장)', all_codes)
        self.scope.setCurrentIndex(max(cur, 0))
        self.scope.blockSignals(False)

    def params(self):
        return dict(codes=self.scope.currentData(), max_exemplars=int(self.max_ex.value()),
                    include_auto=self.redo.isChecked(), include_empty_edited=self.empty_edited.isChecked())


class MainWindow(QMainWindow):
    def __init__(self, dataset_dir=None):
        super().__init__()
        self.setWindowTitle('YOLO 마스크 검수')
        self.settings = QSettings('jhw', 'mask_reviewer')
        self.ds = None
        self.stem = None
        self.img = None
        self.instances = []
        self.dirty = False
        self.undo_stack = []
        self.redo_stack = []
        self._loading = False
        self.sam = None            # sam2_helper.Sam2Segmenter (지연 로드)
        self._sam_failed = False
        self._thumbs = {}          # stem -> QIcon
        self._thumb_gen = 0        # 데이터셋을 다시 열면 증가 (옛 배치 폐기)
        self._thumb_exec = None
        self.propagator = None     # propagate.Sam2Propagator (전파 후 재사용)
        self._prop_worker = None
        self._build_ui()
        self._build_actions()
        g = self.settings.value('geometry')
        if g:
            self.restoreGeometry(g)
        else:
            self.resize(1500, 900)
        if dataset_dir:
            self.open_dataset(dataset_dir)

    # ================================================================ UI 구성
    def _build_ui(self):
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        # --- 왼쪽: 이미지 목록
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        fr = QHBoxLayout()
        self.code_filter = QComboBox()
        self.status_filter = QComboBox()
        for k, t in (('all', '전체'), ('pending', '보류'), ('ok', '확정'), ('reject', '제외'), ('edited', '편집됨'),
                     ('auto', '자동 라벨'), ('exemplar', '대표'), ('train_exclude', '정제 제외(학습 미사용)')):
            self.status_filter.addItem(t, k)
        self.search = QLineEdit()
        self.search.setPlaceholderText('파일명 검색')
        self.search.setClearButtonEnabled(True)
        fr.addWidget(self.code_filter, 2)
        fr.addWidget(self.status_filter, 1)
        ll.addLayout(fr)
        er = QHBoxLayout()
        er.addWidget(QLabel('제품의 ★ 대표:'))
        self.ex_has = QCheckBox('있음')
        self.ex_none = QCheckBox('없음')
        self.ex_has.setToolTip('같은 12자리 제품코드에 ★ 대표가 한 장이라도 있는 이미지만 (재지정 코드 기준)')
        self.ex_none.setToolTip('같은 12자리 제품코드에 ★ 대표가 없는 이미지만 — 아직 대표를 만들어야 하는 제품')
        er.addWidget(self.ex_has)
        er.addWidget(self.ex_none)
        self.ex_summary = QLabel('')
        self.ex_summary.setStyleSheet('color:#666')
        er.addWidget(self.ex_summary, 1)
        ll.addLayout(er)
        sr = QHBoxLayout()
        sr.addWidget(self.search, 1)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem('파일명 순', 'name')
        self.sort_combo.addItem('원본과 차이 큰 순', 'diff')
        self.sort_combo.addItem('자동 점수 낮은 순', 'score')
        self.sort_combo.addItem('제품 코드 순', 'code')
        self.sort_combo.addItem('구멍 불일치 큰 순', 'holes')
        self.sort_combo.setToolTip('원본과 차이 큰 순: 자동 라벨이 원본 의사 라벨을 많이 바꾼 것(1-IoU)부터 검수\n'
                                   '자동 점수 낮은 순: SAM2 객체 점수 / YOLO conf 가 낮은 것부터\n'
                                   '제품 코드 순: 같은 제품끼리 모아서 (코드 없는 이미지는 맨 뒤), 제품 안에서는 파일명 순')
        sr.addWidget(self.sort_combo)
        ll.addLayout(sr)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)   # 드래그·Shift/Ctrl+클릭 다중 선택 → 제품코드 일괄 변경
        self.list.setToolTip('드래그·Shift/Ctrl+클릭으로 여러 장을 고르면 [제품코드 변경…] 이 선택 전부에 적용된다')
        self.list.setIconSize(QSize(THUMB_PX, THUMB_PX))
        self.list.setUniformItemSizes(True)   # 모든 행을 같은 높이로 (썸네일이 늦게 와도 행이 안 흔들림)
        self.list.currentItemChanged.connect(self._on_list_changed)
        ll.addWidget(self.list, 1)
        self.progress = QLabel('-')
        self.progress.setWordWrap(True)
        ll.addWidget(self.progress)
        self.code_filter.currentIndexChanged.connect(self.refresh_list)
        self.status_filter.currentIndexChanged.connect(self.refresh_list)
        self.ex_has.toggled.connect(self.refresh_list)
        self.ex_none.toggled.connect(self.refresh_list)
        self.sort_combo.currentIndexChanged.connect(self.refresh_list)
        self.search.textChanged.connect(self.refresh_list)

        # --- 가운데: 캔버스 + 도구 막대
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        self.toolbar = QToolBar('도구')
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        self.canvas = MaskCanvas()
        cl.addWidget(self.canvas, 1)
        self.canvas.editBegan.connect(self.push_undo)
        self.canvas.edited.connect(self._on_edited)
        self.canvas.instancePicked.connect(self._on_picked)
        self.canvas.cursorMoved.connect(self._on_cursor)
        self.canvas.message.connect(self.say)
        self.canvas.zoomChanged.connect(lambda s: self.zoom_lbl.setText(f'{s * 100:.0f}%'))
        self.canvas.samPromptChanged.connect(self._on_sam_prompt)
        self.canvas.cutRequested.connect(self.cut_instance)

        # --- 오른쪽: 정보 / 인스턴스 / 상태
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        gi = QGroupBox('이미지 정보')
        gil = QVBoxLayout(gi)
        self.info = QLabel('-')
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gil.addWidget(self.info)
        crow = QHBoxLayout()
        crow.addStretch(1)
        self.code_btn = QPushButton('제품코드 변경…')
        self.code_btn.setToolTip('이미지에 적힌(manifest·파일명) 제품코드가 틀렸을 때 고친다. '
                                 '목록에 있는 12자리 코드를 고르거나 직접 입력. review_state.json 에 저장되며 필터·내보내기·전파에 반영.\n'
                                 '왼쪽 목록에서 여러 장을 선택(드래그·Shift/Ctrl+클릭)했으면 선택 전부에 한 번에 적용')
        self.code_btn.clicked.connect(self.edit_code)
        crow.addWidget(self.code_btn)
        gil.addLayout(crow)
        rl.addWidget(gi)

        ginst = QGroupBox('인스턴스 (Tab: 다음, 1~9: 클래스, Del: 삭제 — 드래그·Shift/Ctrl+클릭으로 여러 개 선택 시 일괄)')
        gl = QVBoxLayout(ginst)
        self.inst_list = QListWidget()
        self.inst_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.inst_list.currentRowChanged.connect(self._on_inst_row)
        gl.addWidget(self.inst_list)
        crow = QHBoxLayout()
        self.cls_combo = QComboBox()
        self.cls_combo.setToolTip('새 인스턴스 클래스 / [적용] 으로 선택된 인스턴스(없으면 현재) 클래스 변경')
        self.cls_combo.currentIndexChanged.connect(self._on_cls_combo)
        b = QPushButton('클래스 적용')
        b.clicked.connect(self.apply_class)
        b.setToolTip('목록에서 선택된 인스턴스 전부의 클래스를 콤보 값으로 (Shift+드래그로 범위 선택). 단축키 1~9 도 같음')
        crow.addWidget(self.cls_combo, 1)
        crow.addWidget(b)
        gl.addLayout(crow)
        grid = QGridLayout()
        btns = [('새 인스턴스 (N)', self.new_instance), ('삭제 (Del)', self.delete_instance),
                ('선택 병합', self.merge_instances), ('조각 분리', self.split_instance),
                ('자르기 (K)', lambda: self.set_tool('cut')), ('GrabCut 보정 (G)', self.grabcut),
                ('구멍 채우기', self.fill_holes), ('최대 조각만', self.keep_largest),
                ('가장자리 정리', self.smooth), ('원본 라벨 복원', self.revert),
                ('확정 구멍 빼기', lambda: self.apply_aux_holes((1,))),
                ('모델 구멍 빼기', lambda: self.apply_aux_holes((1, 2))),
                ('CAD 구멍 빼기', lambda: self.apply_aux_holes((1, 3)))]
        for i, (t, fn) in enumerate(btns):
            b = QPushButton(t)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
        gl.addLayout(grid)
        rl.addWidget(ginst, 1)

        gs = QGroupBox('검수 판정')
        sl = QVBoxLayout(gs)
        srow = QHBoxLayout()
        self.status_btn = {}
        for s, key in (('ok', 'Space'), ('pending', 'W'), ('reject', 'X')):
            b = QPushButton(f'{STATUS_MARK[s]} {STATUS_LABEL[s]} ({key})')
            b.setCheckable(True)
            b.clicked.connect(lambda _, s=s: self.set_status(s, advance=(s != 'pending')))
            self.status_btn[s] = b
            srow.addWidget(b)
        self.exemplar_btn = QPushButton('★ 대표 (R)')
        self.exemplar_btn.setCheckable(True)
        self.exemplar_btn.setToolTip('이 이미지를 제품 대표 라벨로 지정 (현재 라벨을 편집본으로 저장하고 확정). '
                                     '자동 → SAM2 대표 전파 로 같은 제품의 보류 이미지에 전파한다')
        self.exemplar_btn.clicked.connect(self.toggle_exemplar)
        srow.addWidget(self.exemplar_btn)
        sl.addLayout(srow)
        self.note = QLineEdit()
        self.note.setPlaceholderText('메모 (예: 캡 누락, 박스 2개)')
        self.note.editingFinished.connect(self._save_note)
        sl.addWidget(self.note)
        brow = QHBoxLayout()
        b = QPushButton('저장 (Ctrl+S)')
        b.clicked.connect(lambda: self.save_current(force=True))
        brow.addWidget(b)
        b = QPushButton('내보내기 (Ctrl+E)')
        b.clicked.connect(self.export)
        brow.addWidget(b)
        sl.addLayout(brow)
        rl.addWidget(gs)
        help_lbl = QLabel('이동 A/D · PgUp/PgDn   확대 휠, 이동 휠클릭/Ctrl+드래그   브러시 크기 [ ] 또는 Ctrl+휠\n'
                          '브러시: 좌클릭 칠하기 / 우클릭 지우기   폴리곤: 클릭으로 점, Enter·더블클릭·첫점 클릭으로 닫기,'
                          ' Ctrl+Enter 는 빼기, Esc 취소\n완드: 좌클릭 비슷한 색 추가 / 우클릭 제거   '
                          'SAM2: 좌클릭 전경점 / 우클릭 배경점 / Shift+드래그 박스 → Enter 추가, Ctrl+Enter 빼기, Shift+Enter 교체\n'
                          '꼭짓점: 드래그 이동, 변 클릭 추가, 우클릭 삭제   자르기: 선 찍고 Enter → 두 인스턴스로   '
                          '조각 분리: 떨어진 조각을 인스턴스로   C 현재 인스턴스 비우기   H 채움, O 외곽선, F 화면 맞춤')
        help_lbl.setStyleSheet('color: #888; font-size: 11px;')
        help_lbl.setWordWrap(True)
        rl.addWidget(help_lbl)

        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([330, 900, 330])

        self.zoom_lbl = QLabel('100%')
        self.pos_lbl = QLabel('')
        self.statusBar().addPermanentWidget(self.pos_lbl)
        self.statusBar().addPermanentWidget(self.zoom_lbl)

    def _build_actions(self):
        mb = self.menuBar()
        fm = mb.addMenu('파일')
        em = mb.addMenu('편집')
        vm = mb.addMenu('보기')

        def act(text, fn, key=None, menu=None, checkable=False):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.setCheckable(checkable)
            a.triggered.connect(fn)
            (menu or self).addAction(a)
            if menu is None:
                self.addAction(a)
            return a

        act('데이터셋 열기…', self.open_dialog, 'Ctrl+O', fm)
        act('저장', lambda: self.save_current(force=True), 'Ctrl+S', fm)
        act('YOLO 로 내보내기…', self.export, 'Ctrl+E', fm)
        fm.addSeparator()
        act('종료', self.close, 'Ctrl+Q', fm)
        act('되돌리기', self.undo, 'Ctrl+Z', em)
        act('다시 실행', self.redo, 'Ctrl+Shift+Z', em)
        act('다시 실행 ', self.redo, 'Ctrl+Y')
        em.addSeparator()
        act('새 인스턴스', self.new_instance, 'N', em)
        act('인스턴스 삭제', self.delete_instance, 'Delete', em)
        act('다음 인스턴스', self.next_instance, 'Tab', em)
        act('GrabCut 보정', self.grabcut, 'G', em)
        act('현재 인스턴스 비우기', self.clear_instance, 'C', em)
        act('원본 라벨 복원', self.revert, None, em)
        em.addSeparator()
        act('확정 + 다음', lambda: self.set_status('ok', True), 'Space', em)
        act('보류', lambda: self.set_status('pending', False), 'W', em)
        act('제외 + 다음', lambda: self.set_status('reject', True), 'X', em)
        act('대표 지정/해제', self.toggle_exemplar, 'R', em)
        am = mb.addMenu('자동')
        act('SAM2 대표 전파…', self.run_propagate, 'Ctrl+P', am)
        act('현재 이미지 자동 라벨 삭제', self.clear_auto_current, None, am)
        am.addSeparator()
        act('SAM2 체크포인트 선택…', self.choose_sam_checkpoint, None, am)
        act('이전 이미지', lambda: self.navigate(-1), 'A', vm)
        act('다음 이미지', lambda: self.navigate(1), 'D', vm)
        act('이전 이미지 ', lambda: self.navigate(-1), 'PgUp')
        act('다음 이미지 ', lambda: self.navigate(1), 'PgDown')
        vm.addSeparator()
        self.act_fill = act('마스크 채움 표시', self.toggle_fill, 'H', vm, checkable=True)
        self.act_fill.setChecked(True)
        self.act_outline = act('외곽선 표시', self.toggle_outline, 'O', vm, checkable=True)
        self.act_outline.setChecked(True)
        self.act_aux = act('보조 구멍 레이어 표시 (청록 확정 / 자홍 모델만 / 주황 CAD만)', self.toggle_aux, 'J', vm, checkable=True)
        self.act_aux.setChecked(True)
        self.act_keep_holes = act('구멍 유지 저장·내보내기 (다리 폴리곤)', self.toggle_keep_holes, None, fm, checkable=True)
        self.act_keep_holes.setToolTip('켜면 편집본·자동 라벨·내보내기 폴리곤이 관통 구멍을 남긴다. dataset.yaml 의 keep_holes 가 기본값.')
        act('화면 맞춤', self.canvas.fit, 'F', vm)
        self.act_thumbs = act('목록 썸네일 표시', self.toggle_thumbs, None, vm, checkable=True)
        self.act_thumbs.setChecked(self.settings.value('thumbs', 'true') in ('true', True, 'True', '1'))
        act('브러시 크게', lambda: self.brush_spin.setValue(self.brush_spin.value() + 2), ']')
        act('브러시 작게', lambda: self.brush_spin.setValue(self.brush_spin.value() - 2), '[')
        for i in range(1, 10):
            act(f'클래스 {i}', lambda _, i=i: self.set_class(i - 1), str(i))

        # 도구 막대
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        for t in TOOLS:
            a = QAction(TOOL_LABEL[t], self)
            a.setCheckable(True)
            a.setShortcut(QKeySequence(TOOL_KEY[t]))
            a.triggered.connect(lambda _, t=t: self.set_tool(t))
            self.tool_group.addAction(a)
            self.toolbar.addAction(a)
            self.tool_actions[t] = a
        self.tool_actions['brush'].setChecked(True)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel(' 브러시 '))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 150)
        self.brush_spin.setValue(self.canvas.brush_radius)
        self.brush_spin.setSuffix(' px')
        self.brush_spin.valueChanged.connect(self.canvas.set_brush_radius)
        self.toolbar.addWidget(self.brush_spin)
        self.toolbar.addWidget(QLabel(' 완드 허용 '))
        self.wand_spin = QSpinBox()
        self.wand_spin.setRange(1, 120)
        self.wand_spin.setValue(self.canvas.wand_tol)
        self.wand_spin.valueChanged.connect(self.canvas.set_wand_tol)
        self.toolbar.addWidget(self.wand_spin)
        self.toolbar.addWidget(QLabel(' 채움 투명도 '))
        self.alpha = QSlider(Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(int(self.canvas.fill_alpha * 100))
        self.alpha.setFixedWidth(120)
        self.alpha.valueChanged.connect(lambda v: self.canvas.set_fill_alpha(v / 100.0))
        self.toolbar.addWidget(self.alpha)
        self.toolbar.addSeparator()
        a = QAction('화면 맞춤 (F)', self)
        a.triggered.connect(self.canvas.fit)
        self.toolbar.addAction(a)
        self.set_tool('brush')

    # ================================================================ 데이터셋
    def open_dialog(self):
        d = QFileDialog.getExistingDirectory(self, '데이터셋 폴더 (images/ labels/ 포함)',
                                             self.settings.value('last_dir', os.path.expanduser('~')))
        if d:
            self.open_dataset(d)

    def open_dataset(self, path):
        self.save_current()
        try:
            ds = Dataset(path)
        except Exception as e:
            QMessageBox.critical(self, '열기 실패', str(e))
            return
        self.ds = ds
        self.stem = None
        self.settings.setValue('last_dir', ds.root)
        self.setWindowTitle(f'YOLO 마스크 검수 — {ds.root}')
        self.cls_combo.blockSignals(True)
        self.cls_combo.clear()
        for k in sorted(ds.names):
            px = QPixmap(14, 14)
            px.fill(class_color(k))
            self.cls_combo.addItem(QIcon(px), f'{k}: {ds.names[k]}', k)
        self.cls_combo.blockSignals(False)
        self.canvas.default_cls = self.cls_combo.currentData() or 0
        self.act_keep_holes.setChecked(ds.keep_holes)
        if ds.has_aux:
            self.say('보조 구멍 레이어(aux/) 있음 — J 로 표시 전환, "구멍 불일치 큰 순" 정렬로 검수')
        self._refresh_code_filter()
        self._thumbs = {}
        self.refresh_list()
        if self.list.count():
            self.list.setCurrentRow(0)
        self.say(f'{len(ds.stems)}장 · 클래스 {len(ds.names)}종 · 편집됨 {ds.count_edited()}')
        QTimer.singleShot(0, self._start_thumbs)

    def _refresh_code_filter(self):
        """제품 필터 항목을 다시 채운다 (선택은 유지)."""
        ds = self.ds
        keep = self.code_filter.currentData()
        self.code_filter.blockSignals(True)
        self.code_filter.clear()
        self.code_filter.addItem(f'전체 제품 ({len(ds.stems)})', '')
        ex_by_code = {}
        for s_ in ds.exemplars():
            ex_by_code[ds.code(s_)] = ex_by_code.get(ds.code(s_), 0) + 1
        codes = ds.codes()
        for code, n in sorted(codes.items()):
            k = ex_by_code.get(code, 0)
            self.code_filter.addItem(f'{code or "(코드 없음)"} ({n})' + (f' ★{k}' if k else ' —'), code)
        n_has = sum(1 for c in codes if ex_by_code.get(c))
        self.ex_summary.setText(f'대표 있는 제품 {n_has} / {len(codes)}  (없는 제품 {len(codes) - n_has}, {sum(n for c, n in codes.items() if not ex_by_code.get(c))}장)')
        i = self.code_filter.findData(keep) if keep else 0
        self.code_filter.setCurrentIndex(max(i, 0))
        self.code_filter.blockSignals(False)

    def _selected_stems(self):
        """이미지 목록에서 선택된 stem 들 (목록 순서). 선택이 없으면 현재 이미지."""
        stems = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count()) if self.list.item(i).isSelected()]
        if not stems and self.stem is not None:
            stems = [self.stem]
        return stems

    def _select_stems(self, stems):
        """목록 갱신 뒤 다중 선택을 되살린다 (현재 행은 그대로)."""
        want = set(stems)
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) in want:
                it.setSelected(True)
        self.list.blockSignals(False)

    def edit_code(self):
        """선택된 이미지(없으면 현재 이미지)의 제품코드를 고친다: 데이터셋에 있는 12자리 코드 선택 또는 직접 입력."""
        if self.ds is None or self.stem is None:
            return
        stems = self._selected_stems()
        origs = {self.ds.orig_code(t) for t in stems}
        curs = {self.ds.code(t) for t in stems}
        d = QDialog(self)
        d.setWindowTitle('제품코드 변경')
        f = QFormLayout(d)
        f.addRow('이미지', QLabel(self.stem if len(stems) == 1 else f'{len(stems)}장 선택 ({stems[0]} …)'))
        f.addRow('원본 코드', QLabel((next(iter(origs)) or '(없음)') if len(origs) == 1 else f'(여러 값: {len(origs)}종)'))
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        for code in sorted(c for c in self.ds.codes() if is_full_code(c)):
            combo.addItem(code)
        combo.setCurrentText(self.ds.code(self.stem) if len(curs) != 1 else next(iter(curs)))
        combo.lineEdit().selectAll()
        combo.setToolTip('목록: 이 데이터셋에 있는 12자리 제품코드. 직접 입력도 가능 (대문자로 저장)')
        f.addRow('제품코드', combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = bb.addButton('원본으로', QDialogButtonBox.ResetRole)
        reset.setEnabled(any(self.ds.state.code(t) for t in stems))
        reset.setToolTip('각 이미지를 자기 원본 코드(manifest·파일명)로 되돌린다')
        do_reset = []
        reset.clicked.connect(lambda: (do_reset.append(True), d.accept()))
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec_() != QDialog.Accepted:
            return
        self.apply_code(None if do_reset else combo.currentText(), stems)

    def apply_code(self, code, stems=None):
        """stems(없으면 현재 이미지)의 제품코드를 code 로. code 가 None 이면 각자 원본으로. 하나라도 바뀌면 True."""
        stems = list(stems) if stems else [self.stem]
        if code is not None:
            code = (code or '').strip().upper()
            if not code:
                QMessageBox.warning(self, '제품코드 변경', '제품코드를 입력하세요.')
                return False
            if not is_full_code(code) and any(code != self.ds.orig_code(t) for t in stems):
                if QMessageBox.question(self, '제품코드 변경', f'"{code}" 는 12자리 제품코드 형식(예: 10H332000NT9)이 아닙니다. 그래도 저장할까요?') \
                        != QMessageBox.Yes:
                    return False
        n = sum(1 for t in stems if self.ds.set_code(t, code if code is not None else self.ds.orig_code(t)))
        if not n:
            return False
        self._refresh_code_filter()
        self.refresh_list()
        if len(stems) > 1:
            self._select_stems(stems)
        self._refresh_info()
        if len(stems) == 1:
            self.say(f'제품코드 {self.stem}: {self.ds.code(self.stem)}' + ('' if self.ds.state.code(self.stem) else ' (원본)'))
        else:
            self.say(f'제품코드 {len(stems)}장 → ' + (code if code is not None else '각자 원본') + f' ({n}장 변경)')
        return True

    # ================================================================ 썸네일 (legacy 툴 방식: 캐시 + 백그라운드 생성)
    @staticmethod
    def _make_thumb(img_path, cache_path, size=THUMB_PX):
        """워커 스레드에서 실행. 캐시가 있으면 그대로, 없으면 축소해 jpg 로 저장."""
        if os.path.exists(cache_path):
            return cache_path
        try:
            img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_4)
            if img is None:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            f = size / float(max(h, w))
            if f < 1.0:
                img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cv2.imwrite(cache_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return cache_path
        except Exception:
            return None

    def _start_thumbs(self):
        if self.ds is None or not self.act_thumbs.isChecked():
            return
        self._thumb_gen += 1
        gen = self._thumb_gen
        if self._thumb_exec is not None:
            self._thumb_exec.shutdown(wait=False)
        self._thumb_exec = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        cache_dir = os.path.join(self.ds.root, THUMB_CACHE_DIRNAME)
        futs = [(s, self._thumb_exec.submit(self._make_thumb, self.ds.image_path(s), os.path.join(cache_dir, s + '.jpg')))
                for s in self.ds.stems if s not in self._thumbs]
        self._poll_thumbs(futs, 0, gen)

    def _poll_thumbs(self, futs, idx, gen):
        if gen != self._thumb_gen or self.ds is None:
            return
        n = 0
        while idx < len(futs) and n < 30:
            stem, fut = futs[idx]
            if not fut.done():
                break
            path = fut.result()
            px = QPixmap(path) if path else QPixmap()
            if not px.isNull():
                self._thumbs[stem] = QIcon(px)
            idx += 1
            n += 1
        if n:
            self._apply_thumbs()
        if idx < len(futs):
            QTimer.singleShot(30, lambda: self._poll_thumbs(futs, idx, gen))

    def _apply_thumbs(self):
        show = self.act_thumbs.isChecked()
        for i in range(self.list.count()):
            it = self.list.item(i)
            ic = self._thumbs.get(it.data(Qt.UserRole)) if show else None
            if ic is not None:
                if it.icon().isNull():
                    it.setIcon(ic)
            elif not it.icon().isNull():
                it.setIcon(QIcon())

    def toggle_thumbs(self):
        on = self.act_thumbs.isChecked()
        self.settings.setValue('thumbs', 'true' if on else 'false')
        self.list.setIconSize(QSize(THUMB_PX, THUMB_PX) if on else QSize(0, 0))
        self.refresh_list()   # 행 높이 힌트를 다시 잡는다
        if on:
            self._start_thumbs()

    def _visible_stems(self):
        code = self.code_filter.currentData() or ''
        st = self.status_filter.currentData() or 'all'
        q = self.search.text().strip().lower()
        ex_has, ex_none = self.ex_has.isChecked(), self.ex_none.isChecked()
        codes_with_ex = {self.ds.code(s_) for s_ in self.ds.exemplars()} if ex_has != ex_none else None
        out = []
        for s in self.ds.stems:
            if code and self.ds.code(s) != code:
                continue
            if codes_with_ex is not None and ((self.ds.code(s) in codes_with_ex) != ex_has):
                continue
            if st == 'edited':
                if not self.ds.is_edited(s):
                    continue
            elif st == 'auto':
                if not self.ds.is_auto(s):
                    continue
            elif st == 'exemplar':
                if not self.ds.state.exemplar(s):
                    continue
            elif st == 'train_exclude':
                if not self.ds.state.get(s).get('train_exclude'):
                    continue
            elif st != 'all' and self.ds.state.status(s) != st:
                continue
            if q and q not in s.lower():
                continue
            out.append(s)
        mode = self.sort_combo.currentData()
        if mode == 'score':
            def key(s):
                a = self.ds.state.auto(s) if self.ds.is_auto(s) else None
                return (0, a.get('score', 0.0)) if a else (1, 0.0)
            out.sort(key=key)
        elif mode == 'diff':
            def key(s):
                a = self.ds.state.auto(s) if self.ds.is_auto(s) else None
                return (0, -a.get('diff', 0.0)) if a else (1, 0.0)
            out.sort(key=key)
        elif mode == 'code':
            out.sort(key=lambda s: ((0, self.ds.code(s)) if self.ds.code(s) else (1, '')))
        elif mode == 'holes':
            def key(s):
                r = self.ds.manifest.get(s, {})
                try:
                    d = float(r.get('disagree_px') or 0)
                except ValueError:
                    d = 0.0
                bad = 1 if r.get('cad_status') not in ('ok', None, '') else 0
                return (-bad, -d)
            out.sort(key=key)
        return out

    def refresh_list(self):
        if self.ds is None:
            return
        keep = self.stem
        self.list.blockSignals(True)
        self.list.clear()
        row = -1
        thumbs = self.act_thumbs.isChecked()
        for i, s in enumerate(self._visible_stems()):
            it = QListWidgetItem(self._item_text(s))
            it.setData(Qt.UserRole, s)
            if thumbs:
                it.setSizeHint(QSize(0, THUMB_PX + 6))
                ic = self._thumbs.get(s)
                if ic is not None:
                    it.setIcon(ic)
            self._style_item(it, s)
            self.list.addItem(it)
            if s == keep:
                row = i
        self.list.blockSignals(False)
        if row >= 0:
            self.list.setCurrentRow(row)
        elif self.list.count() and keep is not None:
            self.list.setCurrentRow(0)
        self._update_progress()

    def _item_text(self, s):
        src = '✎' if self.ds.is_edited(s) else '⟳' if self.ds.is_auto(s) else ' '
        ex = '★' if self.ds.state.exemplar(s) else ''
        return f'{STATUS_MARK[self.ds.state.status(s)]}{src}{ex} {s}'

    def _label_desc(self, s):
        src = self.ds.label_source(s)
        t = LABEL_SOURCE_LABEL[src]
        if src == 'auto':
            a = self.ds.state.auto(s)
            t += f' ({a.get("source", "?")}'
            if 'diff' in a:
                t += f' · 원본과 차이 {a["diff"]:.2f}'
            if 'score' in a:
                t += f' · score {a["score"]:.2f}'
            if a.get('ref'):
                t += f' · 대표 {a["ref"]}'
            t += ')'
        return t

    def _style_item(self, it, s):
        st = self.ds.state.status(s)
        it.setForeground(QColor(60, 160, 60) if st == 'ok' else QColor(170, 60, 60) if st == 'reject'
                         else QColor(0, 0, 0))
        it.setToolTip(f'{self.ds.code(s)} · {STATUS_LABEL[st]} · 라벨 {self._label_desc(s)}'
                      + (' · ★대표' if self.ds.state.exemplar(s) else ''))

    def _refresh_item(self, s):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == s:
                it.setText(self._item_text(s))
                self._style_item(it, s)
                break
        self._update_progress()

    def _update_progress(self):
        c = self.ds.state.counts(self.ds.stems)
        self.progress.setText(f'확정 {c["ok"]} · 보류 {c["pending"]} · 제외 {c["reject"]} · '
                              f'편집됨 {self.ds.count_edited()} · 자동 {self.ds.count_auto()} · 대표 {len(self.ds.exemplars())}'
                              f' / 전체 {len(self.ds.stems)}   (목록 {self.list.count()})')

    # ================================================================ 이미지 로드/저장
    def _on_list_changed(self, cur, prev):
        if cur is None:
            return
        s = cur.data(Qt.UserRole)
        if s != self.stem:
            self.load_stem(s)

    def load_stem(self, stem):
        self.save_current()
        self._loading = True
        try:
            img, insts = self.ds.load(stem)
        except Exception as e:
            self._loading = False
            QMessageBox.warning(self, '읽기 실패', str(e))
            return
        self.stem = stem
        self.img = img
        self.instances = insts
        self.dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.canvas.set_image(img)
        cur = int(np.argmax([i.area for i in insts])) if insts else -1
        self.canvas.set_instances(self.instances, cur)
        self.canvas.set_aux(self.ds.aux_mask(stem))
        self.canvas.fit()
        self._refresh_inst_list()
        self._refresh_info()
        self._loading = False
        self.canvas.setFocus()

    def _refresh_info(self):
        s = self.stem
        r = self.ds.info(s)
        st = self.ds.state.status(s)
        h, w = self.img.shape[:2]
        src = self.ds.label_source(s)
        color = {'reviewed': '#c60', 'auto': '#07a', 'original': '#333', 'none': '#999'}[src]
        lines = [f'<b>{s}</b>' + (' <span style="color:#b80">★ 대표</span>' if self.ds.state.exemplar(s) else ''),
                 f'제품 <b>{r.get("code", "")}</b>'
                 + (f' <span style="color:#c60">(수정됨, 원본 {self.ds.orig_code(s) or "없음"})</span>' if self.ds.state.code(s) else '')
                 + (lambda k: f' · ★ 대표 {k}장' if k else ' · <span style="color:#c00">★ 대표 없음</span>')(len(self.ds.exemplars(r.get('code', ''))))
                 + f' · {w}×{h} · 라벨: <span style="color:{color}">{self._label_desc(s)}</span>']
        if 'reason' in r:
            lines.append(f'reason {r.get("reason")} · 검출 {r.get("n_det")} · top {r.get("top_cls")} '
                         f'conf {r.get("top_conf")} · fill {r.get("top_fill")}')
        if self.ds.state.get(s).get('train_exclude'):
            lines.append(f'<span style="color:#c00">정제 제외(학습 미사용): {self.ds.state.get(s).get("refine_reasons", "")}</span>')
        if 'cad_status' in r:
            st_col = '#080' if r.get('cad_status') == 'ok' else '#c00'
            lines.append(f'구멍: 확정 <b>{r.get("holes_confirmed_px")}</b>px · 모델만 {r.get("model_only_px")} · CAD만 {r.get("cad_only_px")}'
                         f' · CAD 정합 <span style="color:{st_col}">{r.get("cad_status")}</span>'
                         f' (yaw {r.get("match_yaw")}, score {r.get("match_score")}, iou {r.get("match_iou")})')
        if r.get('source'):
            lines.append(f'<span style="color:#777">{r["source"]}</span>')
        self.info.setText('<br>'.join(lines))
        for k, b in self.status_btn.items():
            b.setChecked(k == st)
        self.exemplar_btn.setChecked(self.ds.state.exemplar(s))
        self.note.setText(self.ds.state.note(s))

    def _refresh_inst_list(self):
        self.inst_list.blockSignals(True)
        self.inst_list.clear()
        for i, inst in enumerate(self.instances):
            name = self.ds.names.get(inst.cls, str(inst.cls)) if self.ds else str(inst.cls)
            it = QListWidgetItem(f'#{i + 1}  {inst.cls}: {name}   {inst.area:,} px')
            px = QPixmap(14, 14)
            px.fill(class_color(inst.cls))
            it.setIcon(QIcon(px))
            self.inst_list.addItem(it)
        if 0 <= self.canvas.current < len(self.instances):
            self.inst_list.setCurrentRow(self.canvas.current)
        self.inst_list.blockSignals(False)

    def save_current(self, force=False):
        """편집이 있으면 labels_reviewed 에 저장. force 면 편집이 없어도 저장."""
        if self.ds is None or self.stem is None:
            return
        if not (self.dirty or force):
            return
        h, w = self.img.shape[:2]
        n = self.ds.save(self.stem, self.instances, w, h, SAVE_EPS, SAVE_MIN_AREA)
        self.dirty = False
        self._refresh_item(self.stem)
        self._refresh_info()
        self.say(f'저장: labels_reviewed/{self.stem}.txt ({n}개 인스턴스)')

    def revert(self):
        if self.ds is None or self.stem is None:
            return
        if not self.ds.is_edited(self.stem) and not self.dirty:
            self.say('편집본이 없습니다' + (' (자동 라벨은 자동 → 현재 이미지 자동 라벨 삭제)' if self.ds.has_auto(self.stem) else ''))
            return
        if QMessageBox.question(self, '원본 복원', '편집본을 지우고 ' + ('자동' if self.ds.has_auto(self.stem) else '원본')
                                + ' 라벨로 되돌릴까요?') != QMessageBox.Yes:
            return
        self.ds.revert(self.stem)
        if self.ds.state.exemplar(self.stem):
            self.ds.state.set(self.stem, exemplar=False)
        self.dirty = False
        s = self.stem
        self.stem = None
        self.load_stem(s)
        self._refresh_item(s)

    def _save_note(self):
        if self.ds is None or self.stem is None or self._loading:
            return
        t = self.note.text()
        if t != self.ds.state.note(self.stem):
            self.ds.state.set(self.stem, note=t)

    def set_status(self, status, advance=False):
        if self.ds is None or self.stem is None:
            return
        self.save_current()
        self.ds.state.set(self.stem, status=status)
        self._refresh_item(self.stem)
        for k, b in self.status_btn.items():
            b.setChecked(k == status)
        self.say(f'{self.stem}: {STATUS_LABEL[status]}')
        if advance:
            self.navigate(1)

    def navigate(self, delta):
        if self.list.count() == 0:
            return
        row = self.list.currentRow()
        new = row + delta
        if new < 0 or new >= self.list.count():
            self.say('목록 끝')
            return
        self.list.setCurrentRow(new)

    def toggle_exemplar(self):
        if self.ds is None or self.stem is None:
            return
        on = not self.ds.state.exemplar(self.stem)
        if on:
            if not self.instances:
                self.say('인스턴스가 없는 이미지는 대표로 지정할 수 없습니다')
                self.exemplar_btn.setChecked(False)
                return
            self.save_current(force=True)          # 현재 라벨을 편집본으로 고정
            self.ds.state.set(self.stem, status='ok', exemplar=True)
            n = len(self.ds.exemplars(self.ds.code(self.stem)))
            self.say(f'★ 대표 지정: {self.ds.code(self.stem)} (대표 {n}장) — 자동 → SAM2 대표 전파 (Ctrl+P)')
        else:
            self.ds.state.set(self.stem, exemplar=False)
            self.say('대표 해제')
        self._refresh_item(self.stem)
        self._refresh_code_filter()   # 제품 콤보의 ★ 수·요약 갱신
        self._refresh_info()
        if self.ex_has.isChecked() != self.ex_none.isChecked():
            self.refresh_list()       # 대표 유무 필터가 켜져 있으면 목록도 다시
        self._refresh_info()

    # ================================================================ 자동 라벨 (SAM2 대표 전파)
    def run_propagate(self):
        if self.ds is None or self._prop_worker is not None:
            return
        self.save_current()
        if not self.ds.exemplars():
            QMessageBox.information(self, 'SAM2 대표 전파', '대표로 지정된 이미지가 없습니다.\n'
                                    '제품마다 잘 맞는 라벨 한 장을 골라 ★ 대표 (R) 로 지정한 뒤 실행하세요.')
            return
        if not sam2_helper.available():
            QMessageBox.information(self, 'SAM2 사용 불가', 'torch / sam2 가 설치되어 있지 않습니다.\npip install -r requirements-sam2.txt')
            return
        code = self.ds.code(self.stem) if self.stem else None
        dlg = PropagateDialog(self, self.ds, code)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.params()
        if not p['codes'] or not any(self.ds.exemplars(c) for c in p['codes']):
            QMessageBox.information(self, 'SAM2 대표 전파', '선택한 범위에 대표가 없습니다.')
            return
        ckpt = self.settings.value('sam2_ckpt') or sam2_helper.find_checkpoint()
        if not ckpt:
            QMessageBox.information(self, 'SAM2 체크포인트 없음', '자동 → SAM2 체크포인트 선택… 으로 지정하세요.')
            return
        prog = QProgressDialog('SAM2 모델 로드 중…', '중지', 0, 100, self)
        prog.setWindowTitle('SAM2 대표 전파')
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        # 100 에 닿으면 자동으로 reset/hide 되면서 setValue 안의 processEvents 와 겹친다 → 끄고 finish 에서만 닫는다
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.setValue(0)
        w = PropagateWorker(self.ds, p['codes'], p['max_exemplars'], p['include_auto'], self.propagator, ckpt,
                            include_empty_edited=p['include_empty_edited'])
        self._prop_worker = w
        prog.canceled.connect(w.stop)

        def on_progress(i, n, st, info):
            prog.setLabelText(f'{i + 1}/{n}  {st}\n{info}')
            prog.setValue(min(int(100 * (i + 1) / max(n, 1)), 99))   # 모달 setValue 는 processEvents 를 부른다
        w.progress.connect(on_progress)

        def finish(r=None, err=None):
            # 워커 신호 슬롯 안(진행률 창의 processEvents 중첩)이 아니라 메인 루프에서 실행되도록 0ms 뒤로 미룬다
            QTimer.singleShot(0, lambda: _finish(r, err))

        def _finish(r, err):
            self.propagator = w.propagator
            self._prop_worker = None
            try:
                w.progress.disconnect(on_progress)
                prog.canceled.disconnect(w.stop)
            except TypeError:
                pass
            prog.reset()
            prog.close()
            prog.deleteLater()
            cur = self.stem
            self.stem = None
            self.refresh_list()
            if cur:
                self.load_stem(cur)
            if err:
                QMessageBox.warning(self, 'SAM2 대표 전파 실패', err)
                return
            per = ', '.join(f'{k} {v}' for k, v in sorted(r['per_code'].items()))
            QMessageBox.information(self, 'SAM2 대표 전파 완료',
                                    f'{r["n_done"]}/{r["n_jobs"]}장 자동 라벨 작성 ({r["sec"]:.0f}초)\n'
                                    f'빈 결과 {r["n_empty"]} · 빈 편집본 삭제 {r["n_cleared"]} · 원본과 차이(1-IoU) 평균 {r["diff_mean"]:.2f} · '
                                    f'0.05 이상 바뀐 장 {r["n_changed"]}\n{per}\n\n'
                                    f'목록 정렬을 "원본과 차이 큰 순" 으로 두고 검수하세요. '
                                    f'Space 로 확정하면 자동 라벨이 그대로 내보내집니다.')

        w.done.connect(lambda r: finish(r=r))
        w.failed.connect(lambda e: finish(err=e))
        w.start()

    def clear_auto_current(self):
        if self.ds is None or self.stem is None or not self.ds.has_auto(self.stem):
            self.say('자동 라벨이 없습니다')
            return
        self.ds.clear_auto(self.stem)
        s = self.stem
        self.dirty = False
        self.stem = None
        self.load_stem(s)
        self._refresh_item(s)
        self.say('자동 라벨 삭제 → 원본 라벨 표시')

    def closeEvent(self, e):
        if self._prop_worker is not None:
            self._prop_worker.stop()
            self._prop_worker.wait(5000)
        self.save_current()
        self.settings.setValue('geometry', self.saveGeometry())
        self._thumb_gen += 1
        if self._thumb_exec is not None:
            self._thumb_exec.shutdown(wait=True, cancel_futures=True)  # 진행 중인 썸네일 쓰기만 끝내고 종료
        super().closeEvent(e)

    # ================================================================ 편집
    def push_undo(self):
        self.undo_stack.append(([i.copy() for i in self.instances], self.canvas.current))
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore(self, snap):
        insts, cur = snap
        self.instances[:] = [i.copy() for i in insts]
        self.canvas.set_instances(self.instances, cur)
        self.dirty = True
        self._refresh_inst_list()

    def undo(self):
        if not self.undo_stack:
            self.say('되돌릴 것이 없습니다')
            return
        self.redo_stack.append(([i.copy() for i in self.instances], self.canvas.current))
        self._restore(self.undo_stack.pop())

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(([i.copy() for i in self.instances], self.canvas.current))
        self._restore(self.redo_stack.pop())

    def _on_edited(self):
        self.dirty = True
        self._refresh_inst_list()

    def _on_picked(self, i):
        self.canvas.set_current(i)
        self._refresh_inst_list()

    def _on_inst_row(self, row):
        if self._loading:
            return
        self.canvas.set_current(row)

    def _on_cursor(self, x, y):
        w, h = self.canvas.image_size()
        if 0 <= x < w and 0 <= y < h and self.img is not None:
            b, g, r = self.img[y, x]
            self.pos_lbl.setText(f'({x}, {y}) RGB {r},{g},{b}  ')
        else:
            self.pos_lbl.setText('')

    def _on_cls_combo(self, _):
        self.canvas.default_cls = self.cls_combo.currentData() or 0

    def set_tool(self, t):
        self.canvas.set_tool(t)
        self.tool_actions[t].setChecked(True)
        self.say(TOOL_LABEL[t])

    def _current_inst(self):
        i = self.canvas.current
        return self.instances[i] if 0 <= i < len(self.instances) else None

    def _selected_rows(self):
        """인스턴스 목록에서 선택된 행들 (드래그·Shift/Ctrl+클릭). 선택이 없으면 현재 인스턴스."""
        rows = sorted({i.row() for i in self.inst_list.selectedIndexes()})
        rows = [r for r in rows if 0 <= r < len(self.instances)]
        if not rows and 0 <= self.canvas.current < len(self.instances):
            rows = [self.canvas.current]
        return rows

    def _select_rows(self, rows):
        """목록 갱신 뒤 다중 선택을 되살린다 (현재 행은 그대로)."""
        self.inst_list.blockSignals(True)
        for r in rows:
            it = self.inst_list.item(r)
            if it is not None:
                it.setSelected(True)
        self.inst_list.blockSignals(False)

    def set_class(self, cls):
        """선택된 인스턴스 전부(없으면 현재 인스턴스)의 클래스를 바꾼다. 되돌리기 한 번에 묶임."""
        if self.ds is None or cls not in self.ds.names:
            return
        idx = self.cls_combo.findData(cls)
        if idx >= 0:
            self.cls_combo.setCurrentIndex(idx)
        rows = self._selected_rows()
        changed = [r for r in rows if self.instances[r].cls != cls]
        if not changed:
            return
        self.push_undo()
        for r in changed:
            self.instances[r].cls = cls
        self.canvas.rebuild()
        self._on_edited()
        if len(rows) > 1:
            self._select_rows(rows)
        name = self.ds.names[cls]
        if len(changed) == 1:
            self.say(f'#{changed[0] + 1} 클래스 → {cls}: {name}')
        else:
            self.say(f'인스턴스 {len(changed)}개 클래스 → {cls}: {name} (선택 {len(rows)}개)')

    def apply_class(self):
        self.set_class(self.cls_combo.currentData())

    def new_instance(self):
        if self.img is None:
            return
        self.push_undo()
        h, w = self.img.shape[:2]
        self.instances.append(Instance(self.canvas.default_cls, np.zeros((h, w), bool)))
        self.canvas.set_current(len(self.instances) - 1)
        self._on_edited()
        if self.canvas.tool == 'select':
            self.set_tool('brush')
        self.say(f'새 인스턴스 #{len(self.instances)} — 브러시/폴리곤/완드로 칠하세요')

    def delete_instance(self):
        rows = self._selected_rows()[::-1]
        if not rows:
            return
        self.push_undo()
        for r in rows:
            del self.instances[r]
        self.canvas.set_instances(self.instances, min(rows[-1], len(self.instances) - 1))
        self._on_edited()
        self.say(f'인스턴스 {len(rows)}개 삭제')

    def merge_instances(self):
        rows = sorted({i.row() for i in self.inst_list.selectedIndexes()})
        if len(rows) < 2:
            self.say('인스턴스 목록에서 2개 이상 선택(Ctrl+클릭)한 뒤 병합하세요')
            return
        self.push_undo()
        base = self.instances[rows[0]]
        for r in rows[1:]:
            base.mask |= self.instances[r].mask
        for r in reversed(rows[1:]):
            del self.instances[r]
        self.canvas.set_instances(self.instances, rows[0])
        self._on_edited()
        self.say(f'{len(rows)}개 병합 → #{rows[0] + 1}')

    def _replace_instance(self, idx, masks, label):
        """인스턴스 idx 를 같은 클래스의 masks 여러 개로 바꾼다 (첫 조각이 그 자리, 나머지는 뒤에 추가)."""
        inst = self.instances[idx]
        self.push_undo()
        new = [Instance(inst.cls, m) for m in masks]
        self.instances[idx:idx + 1] = new[:1]
        self.instances.extend(new[1:])
        self.canvas.set_instances(self.instances, idx)
        self._on_edited()
        self.say(f'{label}: #{idx + 1} → {len(masks)}개 인스턴스 ({", ".join(f"{int(m.sum()):,}" for m in masks)} px)')

    def split_instance(self):
        """현재 인스턴스의 떨어진 조각을 각각 인스턴스로 (마스크 2개가 하나로 합쳐진 라벨용)."""
        inst = self._current_inst()
        if inst is None:
            self.say('선택된 인스턴스가 없습니다')
            return
        parts = maskops.split_components(inst.mask, min_area=SAVE_MIN_AREA)
        if len(parts) < 2:
            self.say('떨어진 조각이 없습니다 — 붙어 있으면 자르기 (K) 로 선을 그어 나누세요')
            return
        self._replace_instance(self.canvas.current, parts, '조각 분리')

    def cut_instance(self, pts):
        """자르기 도구의 선으로 현재 인스턴스를 가른다."""
        inst = self._current_inst()
        if inst is None:
            self.say('선택된 인스턴스가 없습니다')
            return
        parts = maskops.cut_mask(inst.mask, pts, thickness=3, min_area=SAVE_MIN_AREA)
        if len(parts) < 2:
            self.say('선이 마스크를 가르지 못했습니다 — 마스크 바깥에서 바깥까지 가로지르게 그어 주세요')
            return
        self._replace_instance(self.canvas.current, parts, '자르기')

    def next_instance(self):
        if not self.instances:
            return
        self.canvas.set_current((self.canvas.current + 1) % len(self.instances))
        self._refresh_inst_list()

    def _apply_mask_op(self, fn, label):
        inst = self._current_inst()
        if inst is None:
            self.say('선택된 인스턴스가 없습니다')
            return
        self.push_undo()
        before = inst.area
        try:
            inst.mask = fn(inst.mask)
        except Exception as e:
            self.undo_stack.pop()
            QMessageBox.warning(self, label, f'실패: {e}\n{traceback.format_exc()[-400:]}')
            return
        self.canvas.rebuild()
        self._on_edited()
        self.say(f'{label}: {before:,} → {inst.area:,} px')

    def grabcut(self):
        r = int(self.canvas.brush_radius)
        self._apply_mask_op(lambda m: maskops.grabcut_refine(self.img, m, margin=10, band=max(8, 2 * r)),
                            'GrabCut 보정')

    def fill_holes(self):
        self._apply_mask_op(maskops.fill_holes, '구멍 채우기')

    def keep_largest(self):
        self._apply_mask_op(maskops.keep_largest, '최대 조각만')

    def smooth(self):
        self._apply_mask_op(lambda m: maskops.smooth(m, 2), '가장자리 정리')

    def clear_instance(self):
        inst = self._current_inst()
        if inst is None:
            return
        if self.canvas.tool == 'sam' and (self.canvas.sam_points or self.canvas.sam_box is not None):
            self.canvas.clear_sam()
            return
        self.push_undo()
        n = inst.area
        inst.mask[:] = False
        self.canvas.rebuild()
        self._on_edited()
        self.say(f'#{self.canvas.current + 1} 마스크 비움 ({n:,} px) — Ctrl+Z 로 되돌리기')

    # ================================================================ SAM2
    def choose_sam_checkpoint(self):
        start = self.settings.value('sam2_ckpt') or sam2_helper.find_checkpoint() or os.path.expanduser('~')
        p, _ = QFileDialog.getOpenFileName(self, 'SAM2 체크포인트 (.pt)', start, 'SAM2 checkpoint (*.pt)')
        if not p:
            return
        self.settings.setValue('sam2_ckpt', p)
        self.sam = None
        self._sam_failed = False
        self.say(f'SAM2 체크포인트: {p} (다음 SAM 클릭 때 로드)')

    def _ensure_sam(self):
        """SAM2 모델을 지연 로드. 실패하면 이유를 한 번만 알리고 False."""
        if self.sam is not None:
            return True
        if self._sam_failed:
            return False
        if not sam2_helper.available():
            self._sam_failed = True
            QMessageBox.information(self, 'SAM2 사용 불가',
                                    'torch / sam2 가 설치되어 있지 않습니다.\n\n'
                                    'pip install -r requirements-sam2.txt\n\n(README 의 SAM2 절 참고)')
            return False
        ckpt = self.settings.value('sam2_ckpt') or sam2_helper.find_checkpoint()
        if not ckpt or not os.path.exists(ckpt):
            self._sam_failed = True
            QMessageBox.information(self, 'SAM2 체크포인트 없음',
                                    '체크포인트(.pt)를 찾지 못했습니다.\n파일 → SAM2 체크포인트 선택… 으로 지정하세요.\n'
                                    '(checkpoints/ 폴더 또는 MR_SAM2_CKPT 환경변수도 인식)')
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.say('SAM2 모델 로드 중…')
        QApplication.processEvents()
        try:
            self.sam = sam2_helper.Sam2Segmenter(ckpt)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._sam_failed = True
            QMessageBox.warning(self, 'SAM2 로드 실패', f'{ckpt}\n{e}\n{traceback.format_exc()[-600:]}')
            return False
        QApplication.restoreOverrideCursor()
        self.say(f'SAM2 로드: {os.path.basename(ckpt)} ({self.sam.device})')
        return True

    def _on_sam_prompt(self):
        c = self.canvas
        if self.img is None or self.ds is None or self.stem is None:
            return
        if not self._ensure_sam():
            c.clear_sam(emit=False)
            self.set_tool('brush')
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.sam.set_image(self.img, sam2_helper.cache_path_for(self.ds.root, self.stem)):
                QApplication.processEvents()
            mask, score = self.sam.predict(c.sam_points, c.sam_labels, c.sam_box)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'SAM2 추론 실패', f'{e}\n{traceback.format_exc()[-600:]}')
            return
        QApplication.restoreOverrideCursor()
        c.set_sam_proposal(mask, score)
        if mask is not None:
            self.say(f'SAM 제안 {int(mask.sum()):,}px (score {score:.2f}) — Enter 추가 / Ctrl+Enter 빼기 / Shift+Enter 교체 / Esc 취소')

    def toggle_fill(self):
        self.canvas.show_fill = self.act_fill.isChecked()
        self.canvas.update()

    def toggle_outline(self):
        self.canvas.show_outline = self.act_outline.isChecked()
        self.canvas.update()

    def toggle_aux(self):
        self.canvas.show_aux = self.act_aux.isChecked()
        self.canvas.update()

    def toggle_keep_holes(self):
        if self.ds is not None:
            self.ds.keep_holes = self.act_keep_holes.isChecked()
            self.say(f'구멍 유지 저장·내보내기: {"켜짐" if self.ds.keep_holes else "꺼짐"}')

    def apply_aux_holes(self, values):
        """보조 구멍 레이어의 값(values) 영역을 현재 인스턴스에서 뺀다."""
        aux = self.canvas.aux
        if aux is None:
            self.say('이 이미지에는 보조 구멍 레이어가 없습니다')
            return
        region = np.isin(aux, values)
        self._apply_mask_op(lambda m: maskops.subtract(m, region), '구멍 빼기')

    # ================================================================ 내보내기
    def export(self):
        if self.ds is None:
            return
        self.save_current()
        default = self.settings.value('last_export', os.path.join(os.path.dirname(self.ds.root),
                                                                  os.path.basename(self.ds.root) + '_yolo'))
        dlg = ExportDialog(self, self.ds, default)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.params()
        if not p['out_dir']:
            return
        if not p['statuses']:
            QMessageBox.warning(self, '내보내기', '포함할 상태를 하나 이상 고르세요')
            return
        if os.path.isdir(p['out_dir']) and os.listdir(p['out_dir']):
            if QMessageBox.question(self, '내보내기', f'{p["out_dir"]} 이 비어 있지 않습니다. 같은 이름 파일은 덮어씁니다. 계속할까요?') \
                    != QMessageBox.Yes:
                return
        prog = QProgressDialog('내보내는 중…', '취소', 0, 100, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        cancelled = {'v': False}

        def cb(i, n, stem):
            prog.setValue(int(100 * i / max(n, 1)))
            prog.setLabelText(f'{i + 1}/{n}  {stem}')
            QApplication.processEvents()
            if prog.wasCanceled():
                cancelled['v'] = True
                raise RuntimeError('사용자 취소')

        try:
            r = export_dataset(self.ds, progress=cb, **p)
        except Exception as e:
            prog.close()
            if not cancelled['v']:
                QMessageBox.critical(self, '내보내기 실패', str(e))
            return
        prog.setValue(100)
        prog.close()
        self.settings.setValue('last_export', p['out_dir'])
        cls = ', '.join(f'{k} {v}' for k, v in r['cls_count'].items())
        QMessageBox.information(self, '내보내기 완료',
                                f'{r["out_dir"]}\n\n이미지 {r["n_total"]} (train {r["n_train"]} / val {r["n_val"]})\n'
                                f'편집본 {r["n_edited"]} · 라벨 없음 {r["n_empty"]}\n인스턴스: {cls}\n\n'
                                f'dataset.yaml 의 path 는 이 PC 절대경로입니다. 학습 서버에서 경로를 고쳐 쓰세요.')

    # ================================================================ 기타
    def say(self, msg):
        self.statusBar().showMessage(msg, 6000)

