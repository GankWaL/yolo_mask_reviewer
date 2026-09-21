"""이미지 + 마스크 오버레이 캔버스 (확대/이동, 브러시·지우개·폴리곤·완드·SAM2·꼭짓점 편집)."""
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF, QBrush
from PyQt5.QtWidgets import QWidget

from . import maskops
from .dataset import Instance

TOOLS = ('select', 'brush', 'eraser', 'polygon', 'wand', 'sam', 'vertex', 'cut')
TOOL_LABEL = {'select': '선택 (V)', 'brush': '브러시 (B)', 'eraser': '지우개 (E)',
              'polygon': '폴리곤 (P)', 'wand': '완드 (M)', 'sam': 'SAM2 (S)', 'vertex': '꼭짓점 (T)', 'cut': '자르기 (K)'}
TOOL_KEY = {'select': 'V', 'brush': 'B', 'eraser': 'E', 'polygon': 'P', 'wand': 'M', 'sam': 'S', 'vertex': 'T', 'cut': 'K'}
VERTEX_EPS = 1.2          # 꼭짓점 편집용 외곽선 단순화 (px)
VERTEX_GRAB_PX = 9        # 꼭짓점 잡기 반경 (화면 px)
VERTEX_EDGE_PX = 12       # 변 위 클릭으로 꼭짓점 추가 인정 거리 (화면 px)

PALETTE = [(235, 64, 64), (250, 200, 40), (60, 200, 235), (220, 80, 220), (80, 130, 255),
           (70, 210, 100), (255, 140, 0), (170, 110, 255), (0, 190, 170), (200, 200, 200)]


# 보조 구멍 레이어 색 (BGR): 1 확정 구멍(모델·CAD 일치) / 2 모델만 / 3 CAD만
AUX_COLORS = {1: (255, 255, 0), 2: (255, 0, 255), 3: (0, 140, 255)}
AUX_LABELS = {1: '확정(모델=CAD)', 2: '모델만', 3: 'CAD만'}


def class_color(cls):
    r, g, b = PALETTE[int(cls) % len(PALETTE)]
    return QColor(r, g, b)


class MaskCanvas(QWidget):
    editBegan = pyqtSignal()          # 마스크를 바꾸기 직전 (undo 스냅샷용)
    edited = pyqtSignal()             # 마스크가 바뀐 뒤
    instancePicked = pyqtSignal(int)  # 선택 도구로 인스턴스를 고름 (-1 = 빈 곳)
    cursorMoved = pyqtSignal(int, int)
    message = pyqtSignal(str)
    zoomChanged = pyqtSignal(float)
    samPromptChanged = pyqtSignal()   # SAM 점/박스 프롬프트가 바뀜 -> 앱이 predict 후 set_sam_proposal
    cutRequested = pyqtSignal(list)   # 자르기 선 확정 [(x, y), ...] -> 앱이 현재 인스턴스를 가른다

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)
        self.img = None
        self.qimg = None
        self.instances = []
        self.current = -1
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.tool = 'brush'
        self.brush_radius = 8
        self.wand_tol = 20
        self.fill_alpha = 0.45
        self.show_fill = True
        self.show_outline = True
        self.default_cls = 0
        self.poly_pts = []
        self._overlay = None
        self._overlay_buf = None
        self._contours = []
        # 보조 구멍 레이어 (uint8 HxW: 1 확정 / 2 모델만 / 3 CAD만) — 표시 전용, 편집 대상 아님
        self.aux = None
        self.show_aux = True
        self._aux_overlay = None
        self._aux_buf = None
        self._drag = None
        self._last_pt = None
        self._pan_start = None
        self._mouse = None
        # SAM2 프롬프트 상태
        self.sam_points = []      # [(x, y)]
        self.sam_labels = []      # 1 전경 / 0 배경
        self.sam_box = None       # (x0, y0, x1, y1)
        self.sam_proposal = None  # bool (H, W)
        self.sam_score = 0.0
        self._sam_box_drag = None
        # 꼭짓점 편집 상태
        self.vert_polys = []      # [(N,2) float 배열] 현재 인스턴스 외곽선
        self._vert_drag = None    # (poly_idx, pt_idx)
        self._vert_hover = None
        self._vert_self_edit = False

    # ------------------------------------------------------------ 데이터
    def has_image(self):
        return self.img is not None

    def image_size(self):
        if self.img is None:
            return 0, 0
        h, w = self.img.shape[:2]
        return w, h

    def set_image(self, img_bgr):
        self.img = img_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        self.qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.poly_pts = []
        self._drag = None
        self.clear_sam(emit=False)
        self.vert_polys = []
        self._vert_drag = None

    def set_aux(self, aux):
        """보조 구멍 레이어 설정 (None 이면 없음). 색: 1 청록(확정) / 2 자홍(모델만) / 3 주황(CAD만)."""
        self.aux = aux
        self._aux_overlay = None
        self._aux_buf = None
        if aux is not None and self.img is not None and aux.shape[:2] == self.img.shape[:2]:
            h, w = aux.shape[:2]
            buf = np.zeros((h, w, 4), np.uint8)
            for v, (b, g, r) in AUX_COLORS.items():
                m = aux == v
                buf[m, 0] = b
                buf[m, 1] = g
                buf[m, 2] = r
                buf[m, 3] = 150
            self._aux_buf = buf
            self._aux_overlay = QImage(buf.data, w, h, 4 * w, QImage.Format_ARGB32)
        self.update()

    def set_instances(self, instances, current=-1):
        self.instances = instances
        self.current = current if 0 <= current < len(instances) else (len(instances) - 1 if instances else -1)
        self.rebuild()

    def set_current(self, i):
        self.current = i if 0 <= i < len(self.instances) else -1
        self.rebuild()

    def rebuild(self):
        """오버레이 이미지와 외곽선 캐시를 다시 만든다."""
        if self.img is None:
            self._overlay = None
            self._contours = []
            self.update()
            return
        h, w = self.img.shape[:2]
        buf = np.zeros((h, w, 4), np.uint8)  # BGRA (Format_ARGB32, little-endian)
        self._contours = []
        for i, inst in enumerate(self.instances):
            c = class_color(inst.cls)
            a = self.fill_alpha * (1.35 if i == self.current else 0.8)
            m = inst.mask
            buf[m, 0] = c.blue()
            buf[m, 1] = c.green()
            buf[m, 2] = c.red()
            buf[m, 3] = int(255 * min(1.0, a))
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self._contours.append([QPolygonF([QPointF(float(x) + 0.5, float(y) + 0.5) for x, y in cc.reshape(-1, 2)])
                                   for cc in cs if len(cc) >= 3])
        self._overlay_buf = buf
        self._overlay = QImage(buf.data, w, h, 4 * w, QImage.Format_ARGB32)
        if self.tool == 'vertex' and not self._vert_self_edit:
            self._trace_vertices()
        self.update()

    # ------------------------------------------------------------ 좌표
    def fit(self):
        if self.img is None:
            return
        w, h = self.image_size()
        s = min(self.width() / w, self.height() / h) * 0.98
        self.scale = max(s, 0.05)
        self.offset = QPointF((self.width() - w * self.scale) / 2, (self.height() - h * self.scale) / 2)
        self.zoomChanged.emit(self.scale)
        self.update()

    def to_img(self, p):
        return QPointF((p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale)

    def to_widget(self, p):
        return QPointF(p.x() * self.scale + self.offset.x(), p.y() * self.scale + self.offset.y())

    def _img_xy(self, p):
        q = self.to_img(QPointF(p))
        return int(np.floor(q.x())), int(np.floor(q.y()))

    def _inside(self, x, y):
        w, h = self.image_size()
        return 0 <= x < w and 0 <= y < h

    def zoom_at(self, factor, pos):
        if self.img is None:
            return
        before = self.to_img(pos)
        self.scale = float(np.clip(self.scale * factor, 0.05, 80.0))
        after = self.to_widget(before)
        self.offset += QPointF(pos) - after
        self.zoomChanged.emit(self.scale)
        self.update()

    # ------------------------------------------------------------ 도구 설정
    def set_tool(self, tool):
        if tool not in TOOLS:
            return
        if self.tool in ('polygon', 'cut') and tool != self.tool:
            self.cancel_polygon()
        if self.tool == 'sam' and tool != 'sam':
            self.clear_sam(emit=False)
        self.tool = tool
        self.setCursor(Qt.ArrowCursor if tool in ('select', 'vertex') else Qt.CrossCursor)
        if tool == 'cut':
            self.message.emit('자르기: 마스크를 가로지르는 선을 클릭으로 찍고 Enter (우클릭도 확정), Esc 취소')
        if tool == 'vertex':
            self._trace_vertices()
        else:
            self.vert_polys = []
            self._vert_drag = None
        self.update()

    def set_brush_radius(self, r):
        self.brush_radius = max(1, int(r))
        self.update()

    def set_wand_tol(self, t):
        self.wand_tol = max(1, int(t))

    def set_fill_alpha(self, a):
        self.fill_alpha = float(np.clip(a, 0.0, 1.0))
        self.rebuild()

    # ------------------------------------------------------------ 편집 헬퍼
    def _ensure_instance(self):
        """편집 대상 인스턴스가 없으면 default_cls 로 빈 인스턴스를 만든다."""
        if 0 <= self.current < len(self.instances):
            return self.instances[self.current]
        w, h = self.image_size()
        inst = Instance(self.default_cls, np.zeros((h, w), bool))
        self.instances.append(inst)
        self.current = len(self.instances) - 1
        self.message.emit(f'새 인스턴스 #{self.current + 1} 생성 (클래스 {self.default_cls})')
        return inst

    def _apply_region(self, region, add):
        inst = self._ensure_instance()
        if add:
            inst.mask |= region
        else:
            inst.mask &= ~region
        self.rebuild()
        self.edited.emit()

    def cancel_polygon(self):
        if self.poly_pts:
            self.poly_pts = []
            self.update()

    def finish_polygon(self, add=True):
        if len(self.poly_pts) < 3:
            self.cancel_polygon()
            return
        pts = np.array(self.poly_pts, dtype=np.float64)
        self.poly_pts = []
        self.editBegan.emit()
        inst = self._ensure_instance()
        maskops.fill_polygon(inst.mask, pts, add)
        self.rebuild()
        self.edited.emit()

    # ------------------------------------------------------------ SAM2 프롬프트
    def clear_sam(self, emit=True):
        had = bool(self.sam_points or self.sam_box is not None or self.sam_proposal is not None)
        self.sam_points = []
        self.sam_labels = []
        self.sam_box = None
        self.sam_proposal = None
        self.sam_score = 0.0
        self._sam_box_drag = None
        if had and emit:
            self.message.emit('SAM 프롬프트 지움')
        self.update()

    def set_sam_proposal(self, mask, score=0.0):
        """앱이 SAM predict 결과를 넣는다 (None 이면 제안 없음)."""
        self.sam_proposal = None if mask is None else np.asarray(mask, bool)
        self.sam_score = float(score)
        self.update()

    def sam_undo_point(self):
        if self.sam_points:
            self.sam_points.pop()
            self.sam_labels.pop()
        elif self.sam_box is not None:
            self.sam_box = None
        else:
            return
        if self.sam_points or self.sam_box is not None:
            self.samPromptChanged.emit()
        else:
            self.sam_proposal = None
        self.update()

    def apply_sam(self, mode='add'):
        """제안 마스크를 현재 인스턴스에 반영. mode: add / sub / replace. 그 뒤 프롬프트를 지운다."""
        if self.sam_proposal is None:
            self.message.emit('SAM 제안 마스크가 없습니다 (좌클릭 전경 / 우클릭 배경 / Shift+드래그 박스)')
            return False
        self.editBegan.emit()
        inst = self._ensure_instance()
        if mode == 'replace':
            inst.mask[:] = self.sam_proposal
        elif mode == 'sub':
            inst.mask &= ~self.sam_proposal
        else:
            inst.mask |= self.sam_proposal
        n = int(self.sam_proposal.sum())
        self.clear_sam(emit=False)
        self.rebuild()
        self.edited.emit()
        self.message.emit(f'SAM {"교체" if mode == "replace" else "빼기" if mode == "sub" else "추가"}: {n:,}px')
        return True

    # ------------------------------------------------------------ 꼭짓점 편집
    def _trace_vertices(self):
        """현재 인스턴스 마스크의 외곽선(구멍 무시)을 단순화해 편집용 꼭짓점 목록으로."""
        self.vert_polys = []
        self._vert_drag = None
        self._vert_hover = None
        if not (0 <= self.current < len(self.instances)):
            return
        for c in maskops.mask_contours(self.instances[self.current].mask, min_area=4.0):
            poly = maskops.simplify(c, VERTEX_EPS)
            if len(poly) >= 3:
                self.vert_polys.append(np.asarray(poly, dtype=np.float64) + 0.5)

    def _rasterize_vertices(self):
        """편집한 꼭짓점 폴리곤들로 현재 인스턴스 마스크를 다시 채운다."""
        if not (0 <= self.current < len(self.instances)):
            return
        inst = self.instances[self.current]
        inst.mask[:] = False
        for poly in self.vert_polys:
            if len(poly) >= 3:
                maskops.fill_polygon(inst.mask, poly - 0.5, True)
        self._vert_self_edit = True
        try:
            self.rebuild()
        finally:
            self._vert_self_edit = False
        self.edited.emit()

    def _nearest_vertex(self, wpos, radius=VERTEX_GRAB_PX):
        """화면 좌표 wpos 에서 radius px 안의 가장 가까운 (poly_idx, pt_idx)."""
        best, best_d = None, radius * radius
        for pi, poly in enumerate(self.vert_polys):
            w = poly * self.scale + np.array([self.offset.x(), self.offset.y()])
            d = ((w - np.array([wpos.x(), wpos.y()])) ** 2).sum(1)
            j = int(np.argmin(d))
            if d[j] <= best_d:
                best, best_d = (pi, j), d[j]
        return best

    def _nearest_edge(self, wpos, radius=VERTEX_EDGE_PX):
        """화면 좌표에서 radius px 안의 가장 가까운 변 (poly_idx, edge_idx). 새 점은 edge_idx+1 에 들어간다."""
        p = np.array([wpos.x(), wpos.y()])
        best, best_d = None, radius
        for pi, poly in enumerate(self.vert_polys):
            w = poly * self.scale + np.array([self.offset.x(), self.offset.y()])
            a = w
            b = np.roll(w, -1, axis=0)
            ab = b - a
            ll = (ab ** 2).sum(1)
            t = np.clip(((p - a) * ab).sum(1) / np.where(ll > 0, ll, 1), 0, 1)
            d = np.sqrt(((a + ab * t[:, None] - p) ** 2).sum(1))
            j = int(np.argmin(d))
            if d[j] <= best_d:
                best, best_d = (pi, j), d[j]
        return best

    def vertex_insert(self, wpos):
        hit = self._nearest_edge(wpos)
        if hit is None:
            return False
        pi, j = hit
        q = self.to_img(QPointF(wpos))
        self.editBegan.emit()
        self.vert_polys[pi] = np.insert(self.vert_polys[pi], j + 1, [q.x(), q.y()], axis=0)
        self._rasterize_vertices()
        return True

    def vertex_delete(self, wpos):
        hit = self._nearest_vertex(wpos, VERTEX_GRAB_PX * 1.6)
        if hit is None:
            return False
        pi, j = hit
        self.editBegan.emit()
        if len(self.vert_polys[pi]) <= 3:
            del self.vert_polys[pi]
        else:
            self.vert_polys[pi] = np.delete(self.vert_polys[pi], j, axis=0)
        self._rasterize_vertices()
        return True

    def finish_cut(self):
        if len(self.poly_pts) < 2:
            self.cancel_polygon()
            return
        pts = list(self.poly_pts)
        self.poly_pts = []
        self.update()
        self.cutRequested.emit(pts)

    def pick_instance(self, x, y):
        """점을 포함하는 가장 작은 인스턴스 index, 없으면 -1."""
        best, best_area = -1, None
        for i, inst in enumerate(self.instances):
            if self._inside(x, y) and inst.mask[y, x]:
                a = inst.area
                if best_area is None or a < best_area:
                    best, best_area = i, a
        return best

    # ------------------------------------------------------------ 이벤트
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.img is not None and self._overlay is not None and self.scale <= 0.05001:
            self.fit()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d == 0:
            return
        if e.modifiers() & Qt.ControlModifier:
            self.set_brush_radius(self.brush_radius + (1 if d > 0 else -1))
            self.message.emit(f'브러시 반지름 {self.brush_radius}px')
            return
        self.zoom_at(1.15 ** (d / 120.0), e.pos())

    def mousePressEvent(self, e):
        self.setFocus()
        if self.img is None:
            return
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.ControlModifier):
            self._pan_start = (e.pos(), QPointF(self.offset))
            self.setCursor(Qt.ClosedHandCursor)
            return
        x, y = self._img_xy(e.pos())
        add = e.button() == Qt.LeftButton
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        if self.tool == 'select':
            if add:
                self.instancePicked.emit(self.pick_instance(x, y))
            return
        if self.tool in ('brush', 'eraser'):
            if self.tool == 'eraser':
                add = False
            self.editBegan.emit()
            inst = self._ensure_instance()
            maskops.brush(inst.mask, (x, y), (x, y), self.brush_radius, add)
            self._drag = add
            self._last_pt = (x, y)
            self.rebuild()
            return
        if self.tool == 'polygon':
            if not add:
                self.finish_polygon(True)
                return
            if len(self.poly_pts) >= 3:
                p0 = self.to_widget(QPointF(*self.poly_pts[0]))
                if (p0 - QPointF(e.pos())).manhattanLength() < 10:
                    self.finish_polygon(True)
                    return
            q = self.to_img(QPointF(e.pos()))
            self.poly_pts.append((q.x(), q.y()))
            self.update()
            return
        if self.tool == 'wand':
            if not self._inside(x, y):
                return
            region = maskops.wand_region(self.img, (x, y), self.wand_tol)
            self.editBegan.emit()
            self._apply_region(region, add)
            self.message.emit(f'완드 {"추가" if add else "제거"}: {int(region.sum())}px (허용 {self.wand_tol})')
            return
        if self.tool == 'cut':
            if not add:
                self.finish_cut()
                return
            q = self.to_img(QPointF(e.pos()))
            self.poly_pts.append((q.x(), q.y()))
            self.update()
            return
        if self.tool == 'sam':
            if not self._inside(x, y):
                return
            if add and e.modifiers() & Qt.ShiftModifier:
                self._sam_box_drag = ((x, y), (x, y))
                self.update()
                return
            self.sam_points.append((x, y))
            self.sam_labels.append(1 if add else 0)
            self.samPromptChanged.emit()
            self.update()
            return
        if self.tool == 'vertex':
            if not self.vert_polys:
                self.message.emit('현재 인스턴스에 외곽선이 없습니다 (브러시·SAM 으로 먼저 만드세요)')
                return
            if add:
                hit = self._nearest_vertex(e.pos())
                if hit is not None:
                    self.editBegan.emit()
                    self._vert_drag = hit
                    self.setCursor(Qt.ClosedHandCursor)
                    return
                if not self.vertex_insert(e.pos()):
                    self.message.emit('꼭짓점을 드래그하거나, 변 위를 클릭해 점을 추가하세요 (우클릭 삭제)')
            else:
                self.vertex_delete(e.pos())

    def mouseMoveEvent(self, e):
        self._mouse = e.pos()
        if self.img is None:
            return
        if self._pan_start is not None:
            start, off = self._pan_start
            self.offset = off + QPointF(e.pos() - start)
            self.update()
            return
        x, y = self._img_xy(e.pos())
        self.cursorMoved.emit(x, y)
        if self._drag is not None and 0 <= self.current < len(self.instances):
            maskops.brush(self.instances[self.current].mask, self._last_pt, (x, y), self.brush_radius, self._drag)
            self._last_pt = (x, y)
            self.rebuild()
        elif self._sam_box_drag is not None:
            w, h = self.image_size()
            self._sam_box_drag = (self._sam_box_drag[0], (min(max(x, 0), w - 1), min(max(y, 0), h - 1)))
            self.update()
        elif self._vert_drag is not None:
            pi, j = self._vert_drag
            q = self.to_img(QPointF(e.pos()))
            w, h = self.image_size()
            self.vert_polys[pi][j] = [min(max(q.x(), 0.0), float(w)), min(max(q.y(), 0.0), float(h))]
            self.update()
        else:
            if self.tool == 'vertex':
                self._vert_hover = self._nearest_vertex(e.pos())
            self.update()

    def mouseReleaseEvent(self, e):
        if self._pan_start is not None and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor if self.tool in ('select', 'vertex') else Qt.CrossCursor)  # 꼭짓점 재추적 없이
            return
        if self._drag is not None:
            self._drag = None
            self._last_pt = None
            self.edited.emit()
            return
        if self._sam_box_drag is not None and e.button() == Qt.LeftButton:
            (x0, y0), (x1, y1) = self._sam_box_drag
            self._sam_box_drag = None
            if abs(x1 - x0) >= 3 and abs(y1 - y0) >= 3:
                self.sam_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                self.samPromptChanged.emit()
            self.update()
            return
        if self._vert_drag is not None and e.button() == Qt.LeftButton:
            self._vert_drag = None
            self.setCursor(Qt.ArrowCursor)
            self._rasterize_vertices()

    def mouseDoubleClickEvent(self, e):
        if self.tool in ('polygon', 'cut') and e.button() == Qt.LeftButton:
            if self.poly_pts:
                self.poly_pts.pop()  # 더블클릭의 두 번째 클릭으로 들어간 점 제거
            if self.tool == 'cut':
                self.finish_cut()
            else:
                self.finish_polygon(True)

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def keyPressEvent(self, e):
        if self.tool == 'sam':
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                mods = e.modifiers()
                self.apply_sam('sub' if mods & Qt.ControlModifier else 'replace' if mods & Qt.ShiftModifier else 'add')
                return
            if e.key() == Qt.Key_Escape:
                self.clear_sam()
                return
            if e.key() == Qt.Key_Backspace:
                self.sam_undo_point()
                return
        if self.tool in ('polygon', 'cut') and self.poly_pts:
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self.tool == 'cut':
                    self.finish_cut()
                else:
                    self.finish_polygon(not (e.modifiers() & Qt.ControlModifier))
                return
            if e.key() == Qt.Key_Escape:
                self.cancel_polygon()
                return
            if e.key() == Qt.Key_Backspace:
                self.poly_pts.pop()
                self.update()
                return
        e.ignore()

    # ------------------------------------------------------------ 그리기
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 44))
        if self.qimg is None:
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignCenter, '데이터셋을 열어 주세요 (Ctrl+O)')
            return
        w, h = self.image_size()
        target = QRectF(self.offset.x(), self.offset.y(), w * self.scale, h * self.scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1.0)
        p.drawImage(target, self.qimg)
        if self.show_fill and self._overlay is not None:
            p.drawImage(target, self._overlay)
        if self.show_aux and self._aux_overlay is not None:
            p.drawImage(target, self._aux_overlay)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.show_outline:
            for i, polys in enumerate(self._contours):
                c = class_color(self.instances[i].cls)
                if i == self.current:
                    pen = QPen(QColor(255, 255, 255), 2.5)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    for poly in polys:
                        p.drawPolygon(self._map(poly))
                    pen = QPen(c, 1.2)
                else:
                    pen = QPen(c, 1.5)
                pen.setCosmetic(True)
                p.setPen(pen)
                for poly in polys:
                    p.drawPolygon(self._map(poly))
        if self.poly_pts:
            cutting = self.tool == 'cut'
            col = QColor(255, 140, 0) if cutting else QColor(255, 255, 0)
            pen = QPen(col, 2.0 if cutting else 1.5)
            pen.setCosmetic(True)
            p.setPen(pen)
            pts = [self.to_widget(QPointF(x, y)) for x, y in self.poly_pts]
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(a, b)
            if self._mouse is not None:
                pen.setStyle(Qt.DashLine)
                p.setPen(pen)
                p.drawLine(pts[-1], QPointF(self._mouse))
                if len(pts) >= 2 and not cutting:
                    p.drawLine(QPointF(self._mouse), pts[0])
            p.setPen(QPen(col, 1))
            p.setBrush(QBrush(col))
            for q in pts:
                p.drawEllipse(q, 3, 3)
            p.setBrush(Qt.NoBrush)
            if not cutting:
                p.setPen(QPen(QColor(255, 80, 80), 1))
                p.drawEllipse(pts[0], 6, 6)
        if self.tool == 'sam':
            self._paint_sam(p, target)
        if self.tool == 'vertex' and self.vert_polys:
            self._paint_vertices(p)
        if self._mouse is not None and self.tool in ('brush', 'eraser') and self._pan_start is None:
            r = self.brush_radius * self.scale
            pen = QPen(QColor(255, 255, 255) if self.tool == 'brush' else QColor(255, 120, 120), 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(self._mouse), r, r)
        p.end()

    def _map(self, poly):
        return QPolygonF([self.to_widget(q) for q in poly])

    def _paint_sam(self, p, target):
        if self.sam_proposal is not None:
            h, w = self.sam_proposal.shape
            buf = np.zeros((h, w, 4), np.uint8)
            buf[self.sam_proposal] = (255, 230, 0, 110)  # BGRA: 청록
            p.drawImage(target, QImage(buf.data, w, h, 4 * w, QImage.Format_ARGB32))
            cs, _ = cv2.findContours(self.sam_proposal.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pen = QPen(QColor(0, 230, 255), 2)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            for cc in cs:
                if len(cc) >= 3:
                    p.drawPolygon(self._map(QPolygonF([QPointF(float(x) + 0.5, float(y) + 0.5) for x, y in cc.reshape(-1, 2)])))
        box = self.sam_box
        if self._sam_box_drag is not None:
            (x0, y0), (x1, y1) = self._sam_box_drag
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if box is not None:
            pen = QPen(QColor(0, 230, 255), 1.5, Qt.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            a = self.to_widget(QPointF(box[0], box[1]))
            b = self.to_widget(QPointF(box[2] + 1, box[3] + 1))
            p.drawRect(QRectF(a, b))
        for (x, y), lab in zip(self.sam_points, self.sam_labels):
            q = self.to_widget(QPointF(x + 0.5, y + 0.5))
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
            p.setBrush(QBrush(QColor(40, 220, 60) if lab else QColor(230, 40, 40)))
            p.drawEllipse(q, 6, 6)
        p.setBrush(Qt.NoBrush)

    def _paint_vertices(self, p):
        pen = QPen(QColor(0, 230, 255), 1.5)
        pen.setCosmetic(True)
        for pi, poly in enumerate(self.vert_polys):
            pts = [self.to_widget(QPointF(x, y)) for x, y in poly]
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF(pts))
            for j, q in enumerate(pts):
                active = (self._vert_drag == (pi, j)) or (self._vert_drag is None and self._vert_hover == (pi, j))
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.setBrush(QBrush(QColor(255, 60, 255) if active else QColor(0, 230, 255)))
                r = 6 if active else 4
                p.drawEllipse(q, r, r)
        p.setBrush(Qt.NoBrush)

    def grab_image(self):
        """오프스크린 검증용: 위젯을 QImage 로."""
        return self.grab().toImage()
