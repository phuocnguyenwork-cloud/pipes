"""
Pipes – UI: Pygame Application
================================
Chế độ hỗ trợ:
  - Chơi thủ công: bấm chuột trái vào ô để xoay CW 90°
  - Phím R: reset về trạng thái ban đầu
  - Phím N: tạo puzzle mới
  - Phím T: chạy solver và xem toàn bộ quá trình suy luận (Trace Replay)
  - Phím S: áp dụng ngay nghiệm giải (Solution Replay)
  - Phím SPACE: phát/dừng replay
  - Phím LEFT/RIGHT: bước replay thủ công
  - Phím UP/DOWN: tăng/giảm tốc độ replay
  - Phím F: chuyển đổi Active Branch / Full Tree trong cây tìm kiếm
  - Phím ESC: thoát

Yêu cầu: pygame >= 2.x
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    print("Cần cài pygame: pip install pygame")
    sys.exit(1)

from core.puzzle import State, N, E, S, W
from core.engine import rotate, flood_fill, is_solved, count_violations
from generator.tree_gen import generate_puzzle, make_initial_state, make_solution_state
from solvers.base import Budget, TraceStep, get_solver
import solvers.backtracking  # noqa: F401 – đăng ký backtrack solver
import solvers.astar         # noqa: F401 – đăng ký astar solver
import solvers.dfs           # noqa: F401 – đăng ký blind dfs solver


# ---------------------------------------------------------------------------
# Hằng số giao diện
# ---------------------------------------------------------------------------

CELL_SIZE = 80          # pixel mỗi ô
MARGIN = 20             # margin xung quanh
STATUS_H = 60           # chiều cao thanh trạng thái trên
PANEL_H = 125           # chiều cao bảng giải thích dưới cùng

# Khu vực Tree Panel bên phải
TREE_PANEL_W = 420      # chiều rộng panel cây

# Màu sắc (R, G, B)
BG          = (18, 18, 32)
GRID_BG     = (30, 30, 50)
GRID_LINE   = (60, 60, 90)
PIPE_DEF    = (140, 140, 180)    # pipe chưa nối nguồn
PIPE_CON    = (80, 220, 120)     # pipe đã nối nguồn
SOURCE      = (255, 200, 0)      # ô nguồn
SOLVED_C    = (80, 255, 160)     # khi giải xong
LOCKED      = (220, 100, 80)
TEXT_C      = (200, 220, 255)
TEXT_SOLVED = (80, 255, 160)
REPLAY_C    = (255, 165, 0)

PANEL_BG    = (24, 25, 42)
PANEL_BORDER= (55, 60, 95)

# Màu action badge
ACTION_COLORS = {
    "TRY":       (240, 190, 60),    # Vàng
    "PRUNE":     (240, 80, 80),     # Đỏ cam (vi phạm)
    "FORWARD":   (80, 190, 240),    # Cyan (hợp lệ đi tiếp)
    "BACKTRACK": (210, 100, 230),   # Tím hồng (quay lui)
    "SOLVED":    (80, 255, 160),    # Xanh lá
}

# Màu cho Tree Panel
TREE_BG         = (22, 24, 38)
TREE_BORDER     = (50, 55, 85)
TREE_LINE_C     = (55, 60, 90)        # đường nối nhánh
TREE_HIGHLIGHT  = (45, 48, 75)        # nền node đang chọn
TREE_ACTIVE_BG  = (60, 55, 30)        # nền node hiện tại (đang replay)
TREE_HEADER_BG  = (30, 32, 52)
TREE_SCROLL_BG  = (35, 38, 58)
TREE_SCROLL_FG  = (80, 85, 120)

PIPE_W = 12       # độ dày ống (px)

# Chiều cao mỗi dòng trong tree panel
TREE_ROW_H = 22
TREE_INDENT = 18   # pixel per depth level


# ---------------------------------------------------------------------------
# Cấu trúc node cho cây tìm kiếm
# ---------------------------------------------------------------------------

class TreeNode:
    """Một node trong cây tìm kiếm (search tree)."""
    __slots__ = ("trace_idx", "cell", "rotation", "action", "reason",
                 "depth", "children", "parent", "grid_rotations")

    def __init__(self, trace_idx: int, cell: Tuple[int, int], rotation: int,
                 action: str, reason: str, depth: int,
                 grid_rotations: Optional[np.ndarray] = None):
        self.trace_idx = trace_idx
        self.cell = cell
        self.rotation = rotation
        self.action = action
        self.reason = reason
        self.depth = depth
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.grid_rotations = grid_rotations


def build_search_tree(trace_steps: List[TraceStep]) -> Tuple[Optional[TreeNode], List[TreeNode]]:
    """Xây dựng cây tìm kiếm từ danh sách trace steps.

    Trả về (root_node, flat_list) trong đó flat_list là danh sách các node
    theo thứ tự DFS pre-order để dễ render.
    """
    if not trace_steps:
        return None, []

    # Root node (bước đầu tiên)
    first = trace_steps[0]
    root = TreeNode(
        trace_idx=0,
        cell=(first.r, first.c),
        rotation=first.rotation,
        action=first.action,
        reason=first.reason,
        depth=0,
        grid_rotations=first.grid_rotations,
    )

    # Xây cây bằng cách theo dõi stack depth.
    # Quy ước: FORWARD / TRY tại ô mới = đi sâu hơn,
    # BACKTRACK = quay lại node cha, PRUNE = node lá rồi tiếp tục ở cùng cha.
    all_nodes = [root]
    current_parent = root
    prev_depth = 0

    for i in range(1, len(trace_steps)):
        step = trace_steps[i]
        action = step.action

        # Tính depth dựa trên cell index (row-major order)
        # Mỗi ô là 1 tầng depth trong cây
        # Sử dụng heuristic: nếu action=BACKTRACK -> quay lên cha
        # nếu FORWARD -> đi sâu xuống con
        # nếu PRUNE/TRY -> anh em (sibling) hoặc mới

        node = TreeNode(
            trace_idx=i,
            cell=(step.r, step.c),
            rotation=step.rotation,
            action=action,
            reason=step.reason,
            depth=0,  # sẽ tính lại
            grid_rotations=step.grid_rotations,
        )

        if action == "SOLVED":
            # Node đặc biệt: gắn vào nhánh hiện tại
            node.depth = current_parent.depth + 1
            node.parent = current_parent
            current_parent.children.append(node)
        elif action == "BACKTRACK":
            # Backtrack: gắn vào node cha hiện tại, rồi quay lên ông
            node.depth = current_parent.depth + 1
            node.parent = current_parent
            current_parent.children.append(node)
            # Quay lên 1 hoặc 2 cấp (tùy thuộc cha đã xử lý xong chưa)
            if current_parent.parent is not None:
                current_parent = current_parent.parent
        elif action == "PRUNE":
            # Prune: lá node, gắn vào cha hiện tại, không đi sâu
            node.depth = current_parent.depth + 1
            node.parent = current_parent
            current_parent.children.append(node)
        elif action in ("FORWARD", "TRY"):
            # Đi sâu: tạo con mới dưới cha hiện tại
            node.depth = current_parent.depth + 1
            node.parent = current_parent
            current_parent.children.append(node)
            # Nếu FORWARD (hợp lệ, đi tiếp) thì node này trở thành cha mới
            if action == "FORWARD":
                current_parent = node
        else:
            # Fallback
            node.depth = current_parent.depth + 1
            node.parent = current_parent
            current_parent.children.append(node)

        all_nodes.append(node)

    # Flatten theo thứ tự DFS pre-order để render
    flat: List[TreeNode] = []

    def flatten(node: TreeNode):
        flat.append(node)
        for child in node.children:
            flatten(child)

    flatten(root)

    return root, flat


def get_active_path(flat_nodes: List[TreeNode], current_trace_idx: int) -> set:
    """Trả về tập trace_idx thuộc đường đi từ root đến node hiện tại."""
    path_set = set()
    # Tìm node hiện tại
    target = None
    for node in flat_nodes:
        if node.trace_idx == current_trace_idx:
            target = node
            break
    if target is None:
        return path_set

    # Đi ngược lên root
    n = target
    while n is not None:
        path_set.add(n.trace_idx)
        n = n.parent
    return path_set


# ---------------------------------------------------------------------------
# Vẽ ô
# ---------------------------------------------------------------------------

def _draw_cell(
    surf: pygame.Surface,
    state: State,
    r: int, c: int,
    connected: set,
    source: Tuple[int, int],
    ox: int, oy: int,  # offset lưới
    cell_size: int,
    is_active: bool = False,
    active_color: Tuple[int, int, int] = (255, 220, 50),
):
    cx = ox + c * cell_size + cell_size // 2
    cy = oy + r * cell_size + cell_size // 2
    rect = pygame.Rect(ox + c * cell_size, oy + r * cell_size, cell_size, cell_size)

    # Nền ô
    bg = (45, 45, 75) if is_active else GRID_BG
    pygame.draw.rect(surf, bg, rect)
    pygame.draw.rect(surf, GRID_LINE, rect, 1)

    # Viền nổi bật cho ô đang xét
    if is_active:
        pygame.draw.rect(surf, active_color, rect, 3)

    mask = state.current_mask(r, c)
    is_src = (r, c) == source
    is_conn = (r, c) in connected
    color = SOURCE if is_src else (PIPE_CON if is_conn else PIPE_DEF)
    half = cell_size // 2
    end_len = half - PIPE_W // 2

    def draw_pipe(x1, y1, x2, y2):
        pygame.draw.line(surf, color, (x1, y1), (x2, y2), PIPE_W)

    if mask & N:
        draw_pipe(cx, cy, cx, cy - end_len)
    if mask & S:
        draw_pipe(cx, cy, cx, cy + end_len)
    if mask & E:
        draw_pipe(cx, cy, cx + end_len, cy)
    if mask & W:
        draw_pipe(cx, cy, cx - end_len, cy)

    # Chấm trung tâm
    r_dot = PIPE_W // 2 + 2
    pygame.draw.circle(surf, color, (cx, cy), r_dot)

    # Ngôi sao cho nguồn
    if is_src:
        font = pygame.font.SysFont("segoeui", 14, bold=True)
        txt = font.render("★", True, (0, 0, 0))
        surf.blit(txt, txt.get_rect(center=(cx, cy)))


# ---------------------------------------------------------------------------
# App chính
# ---------------------------------------------------------------------------

class PipesApp:
    def __init__(self, h: int = 5, w: int = 5, seed: int | None = None):
        pygame.init()
        pygame.display.set_caption("Pipes Puzzle – Algorithm Tracer")

        self.h = h
        self.w = w
        self._seed = seed
        self._seed_counter = seed if seed is not None else 0

        # Tính kích thước cửa sổ
        self.cell_size = CELL_SIZE
        self.ox = MARGIN
        self.oy = MARGIN + STATUS_H
        grid_w = w * self.cell_size
        grid_h = h * self.cell_size

        left_w = max(grid_w + 2 * MARGIN, 540)
        self.win_w = left_w + TREE_PANEL_W
        self.win_h = grid_h + 2 * MARGIN + STATUS_H + PANEL_H
        self._left_w = left_w   # lưu lại vị trí chia tách
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_badge = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_tree = pygame.font.SysFont("consolas", 12)
        self.font_tree_header = pygame.font.SysFont("consolas", 13, bold=True)

        # State
        self._new_puzzle()

        # Replay & Trace
        self._trace_steps: List[TraceStep] = []
        self._solution_moves: List[Tuple] = []
        self._replay_mode = "play"  # "play", "trace", "solution"
        self._replay_idx = 0
        self._replay_playing = False
        self._replay_speed = 0.08   # giây mỗi bước
        self._replay_last_t = 0.0
        self._active_cell: Optional[Tuple[int, int]] = None
        self._active_action: str = ""
        self._active_reason: str = ""
        self._active_solver_name: str = ""

        # Search Tree state
        self._tree_root: Optional[TreeNode] = None
        self._tree_flat: List[TreeNode] = []
        self._tree_scroll_y = 0       # pixel offset cuộn
        self._tree_view_mode = "active"  # "active" hoặc "full"
        self._tree_hover_idx = -1     # index trong flat list đang hover

    def _new_puzzle(self, seed: int | None = None):
        s = seed if seed is not None else self._seed_counter
        self._seed_counter += 1
        self._puzzle, self._solution_rotations = generate_puzzle(
            h=self.h, w=self.w, seed=s
        )
        self._state = make_initial_state(self._puzzle)
        self._initial_state = self._state.copy()
        self._replay_mode = "play"
        self._trace_steps = []
        self._solution_moves = []
        self._replay_idx = 0
        self._active_cell = None
        self._active_action = ""
        self._active_reason = "San sang. Bam T de xem suy luan, S de giai ngay."

        # Reset tree
        self._tree_root = None
        self._tree_flat = []
        self._tree_scroll_y = 0

    def _reset(self):
        self._state = self._initial_state.copy()
        self._replay_mode = "play"
        self._replay_playing = False
        self._active_cell = None
        self._active_action = ""
        self._active_reason = "Da dat lai trang thai ban dau."

    def _run_solver(self, solver_name: str = "backtrack", mode: str = "trace"):
        """Chạy solver được chọn và nạp trace/moves vào bộ phát."""
        budget = Budget(max_time=25.0, max_nodes=500000000) #50000
        solver = get_solver(solver_name)
        result = solver.solve(self._puzzle, budget)

        if result.solved:
            self._trace_steps = result.trace
            self._solution_moves = result.moves
            self._replay_idx = 0
            self._replay_playing = True
            self._replay_last_t = time.perf_counter()
            self._active_solver_name = solver_name.upper()

            # Xây cây tìm kiếm từ trace
            self._tree_root, self._tree_flat = build_search_tree(result.trace)
            self._tree_scroll_y = 0

            if mode == "trace" and self._trace_steps:
                self._replay_mode = "trace"
                self._apply_trace_step(0)
            else:
                self._replay_mode = "solution"
                self._state = make_initial_state(self._puzzle)
                self._active_action = "SOLVED"
                self._active_reason = f"[{self._active_solver_name}] Da giai xong ({result.stats.get('nodes')} nodes, {result.stats.get('time_s', 0):.3f}s)"

            print(f"[{self._active_solver_name}] Thanh cong! nodes={result.stats.get('nodes')}, "
                  f"trace_steps={len(result.trace)}, time={result.stats.get('time_s', 0):.3f}s")
        else:
            self._active_action = "PRUNE"
            self._active_reason = f"[{solver_name.upper()}] Khong tim duoc nghiem trong gioi han budget!"
            # Vẫn xây tree cho phần đã duyệt
            if result.trace:
                self._trace_steps = result.trace
                self._tree_root, self._tree_flat = build_search_tree(result.trace)
                self._tree_scroll_y = 0

    def _apply_trace_step(self, idx: int):
        """Áp dụng bước thứ idx trong trace."""
        if not (0 <= idx < len(self._trace_steps)):
            return
        step = self._trace_steps[idx]
        self._active_cell = (step.r, step.c)
        self._active_action = step.action
        self._active_reason = step.reason

        if step.grid_rotations is not None:
            self._state = State(puzzle=self._puzzle, rotations=step.grid_rotations.copy())
        else:
            new_rot = self._state.rotations.copy()
            new_rot[step.r, step.c] = step.rotation
            self._state = State(puzzle=self._puzzle, rotations=new_rot)

        # Auto-scroll cây tìm kiếm để node hiện tại nằm trong vùng nhìn thấy
        self._tree_auto_scroll_to(idx)

    def _apply_solution_step(self, idx: int):
        """Áp dụng bước thứ idx trong solution moves."""
        if not (0 <= idx < len(self._solution_moves)):
            return
        move = self._solution_moves[idx]
        r, c, k = move[0], move[1], move[2]
        self._active_cell = (r, c)
        self._active_action = "FORWARD"
        self._active_reason = f"Xoay o ({r}, {c}) toi goc {k * 90} theo nghiem"

        new_rot = self._state.rotations.copy()
        new_rot[r, c] = k
        self._state = State(puzzle=self._puzzle, rotations=new_rot)

    def _step_forward(self):
        if self._replay_mode == "trace":
            if self._replay_idx + 1 < len(self._trace_steps):
                self._replay_idx += 1
                self._apply_trace_step(self._replay_idx)
            else:
                self._replay_playing = False
        elif self._replay_mode == "solution":
            if self._replay_idx + 1 < len(self._solution_moves):
                self._replay_idx += 1
                self._apply_solution_step(self._replay_idx)
            else:
                self._replay_playing = False

    def _step_back(self):
        if self._replay_mode == "trace":
            if self._replay_idx > 0:
                self._replay_idx -= 1
                self._apply_trace_step(self._replay_idx)
        elif self._replay_mode == "solution":
            if self._replay_idx > 0:
                self._replay_idx -= 1
                # Tái lập từ đầu
                self._state = make_initial_state(self._puzzle)
                for i in range(self._replay_idx + 1):
                    self._apply_solution_step(i)

    # ------------------------------------------------------------------
    # Tree Panel helpers
    # ------------------------------------------------------------------

    def _tree_auto_scroll_to(self, trace_idx: int):
        """Cuộn tree panel để node có trace_idx nằm trong vùng nhìn thấy."""
        # Tìm vị trí flat index tương ứng
        flat_idx = -1
        visible_nodes = self._get_visible_tree_nodes()
        for vi, node in enumerate(visible_nodes):
            if node.trace_idx == trace_idx:
                flat_idx = vi
                break
        if flat_idx < 0:
            return

        panel_h = self.win_h - 60  # trừ header
        visible_rows = panel_h // TREE_ROW_H

        row_y = flat_idx * TREE_ROW_H
        if row_y < self._tree_scroll_y:
            self._tree_scroll_y = max(0, row_y - TREE_ROW_H * 2)
        elif row_y > self._tree_scroll_y + panel_h - TREE_ROW_H * 3:
            self._tree_scroll_y = row_y - panel_h // 2

        # Clamp
        max_scroll = max(0, len(visible_nodes) * TREE_ROW_H - panel_h + TREE_ROW_H)
        self._tree_scroll_y = max(0, min(self._tree_scroll_y, max_scroll))

    def _get_visible_tree_nodes(self) -> List[TreeNode]:
        """Trả về danh sách node cần hiển thị theo chế độ view."""
        if not self._tree_flat:
            return []

        if self._tree_view_mode == "full":
            return self._tree_flat

        # Active branch: chỉ hiện đường từ root -> current + anh em trực tiếp
        active_path = get_active_path(self._tree_flat, self._replay_idx)
        if not active_path:
            return self._tree_flat[:50]  # fallback

        visible = []
        for node in self._tree_flat:
            # Luôn hiện node nằm trên active path
            if node.trace_idx in active_path:
                visible.append(node)
            # Hiện sibling (anh em) của các node trên active path
            elif node.parent is not None and node.parent.trace_idx in active_path:
                visible.append(node)
        return visible

    def _handle_tree_click(self, mx: int, my: int):
        """Xử lý click chuột vào tree panel -> time-travel tới node đó."""
        if self._replay_mode != "trace" or not self._tree_flat:
            return

        tree_x = self._left_w
        tree_header_h = 36
        tree_content_y = tree_header_h

        # Tính vị trí click tương đối trong nội dung cây
        rel_y = my - tree_content_y + self._tree_scroll_y

        visible = self._get_visible_tree_nodes()
        row_idx = int(rel_y // TREE_ROW_H)
        if 0 <= row_idx < len(visible):
            clicked_node = visible[row_idx]
            # Time-travel: nhảy tới bước trace tương ứng
            self._replay_idx = clicked_node.trace_idx
            self._replay_playing = False
            self._apply_trace_step(self._replay_idx)

    def _handle_tree_scroll(self, dy: int):
        """Xử lý cuộn chuột trong tree panel."""
        self._tree_scroll_y -= dy * TREE_ROW_H * 3
        # Clamp
        visible = self._get_visible_tree_nodes()
        panel_h = self.win_h - 60
        max_scroll = max(0, len(visible) * TREE_ROW_H - panel_h + TREE_ROW_H)
        self._tree_scroll_y = max(0, min(self._tree_scroll_y, max_scroll))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        connected = flood_fill(self._state)
        source = self._puzzle.source
        solved = is_solved(self._state)

        # 1. Vẽ các ô trên lưới
        for r in range(self.h):
            for c in range(self.w):
                is_active = (self._active_cell == (r, c))
                act_color = ACTION_COLORS.get(self._active_action, (255, 200, 50))
                _draw_cell(
                    self.screen, self._state,
                    r, c, connected, source,
                    self.ox, self.oy, self.cell_size,
                    is_active=is_active,
                    active_color=act_color
                )

        # 2. Thanh trạng thái trên đỉnh
        viol = count_violations(self._state)
        mode_titles = {
            "play": "[CHE DO CHOI TAY]",
            "trace": f"[SUY LUAN AI: {self._replay_idx + 1}/{len(self._trace_steps)}]",
            "solution": f"[REPLAY NGHIEM: {self._replay_idx + 1}/{len(self._solution_moves)}]"
        }
        mode_str = mode_titles.get(self._replay_mode, "")

        if solved:
            status = "DA GIAI XONG TOAN BO LUOI!"
            color = TEXT_SOLVED
        else:
            status = f"Vi pham: {viol['total']} (Cong ho: {viol['open_ports']}, Mat noi: {viol['disconnected']})"
            color = TEXT_C

        surf_mode = self.font.render(mode_str, True, REPLAY_C if self._replay_mode != "play" else PIPE_DEF)
        surf_status = self.font_big.render(status, True, color)
        self.screen.blit(surf_mode, (MARGIN, 8))
        self.screen.blit(surf_status, (MARGIN, 30))

        # 3. Bảng giải thích chi tiết phía dưới (Explanation Panel)
        panel_y = self.oy + self.h * self.cell_size + 10
        panel_w = self._left_w - 2 * MARGIN
        panel_rect = pygame.Rect(MARGIN, panel_y, panel_w, PANEL_H)

        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, 1, border_radius=8)

        # Tiêu đề Panel + Badge hành động
        if self._active_action:
            badge_color = ACTION_COLORS.get(self._active_action, (200, 200, 200))
            badge_text = f" {self._active_action} "
            surf_badge = self.font_badge.render(badge_text, True, (0, 0, 0))
            badge_rect = surf_badge.get_rect(topleft=(MARGIN + 12, panel_y + 10))
            pygame.draw.rect(self.screen, badge_color, badge_rect, border_radius=3)
            self.screen.blit(surf_badge, badge_rect)

            cell_str = f"O: {self._active_cell}" if self._active_cell else ""
            surf_cell = self.font.render(cell_str, True, (180, 200, 230))
            self.screen.blit(surf_cell, (MARGIN + 110, panel_y + 10))
        else:
            surf_title = self.font.render("BANG GIAI THICH QUA TRINH SUY LUAN", True, (130, 140, 175))
            self.screen.blit(surf_title, (MARGIN + 12, panel_y + 10))

        # Dòng lý do (Reason text wrap đơn giản)
        reason_text = f"Ly do: {self._active_reason}" if self._active_reason else "Bam T de xem thuat toan suy luan tung buoc."
        # Chia 2 dòng nếu quá dài
        max_chars = int((panel_w - 24) / 8.5)
        if len(reason_text) > max_chars:
            line1 = reason_text[:max_chars]
            line2 = reason_text[max_chars:]
        else:
            line1 = reason_text
            line2 = ""

        surf_r1 = self.font.render(line1, True, (230, 240, 255))
        self.screen.blit(surf_r1, (MARGIN + 12, panel_y + 36))
        if line2:
            surf_r2 = self.font.render(line2, True, (200, 215, 240))
            self.screen.blit(surf_r2, (MARGIN + 12, panel_y + 56))

        # Hướng dẫn phím ở đáy panel
        speed_str = f"{1.0/self._replay_speed:.1f}x"
        hints = f"1/T:BT | 2/A:A* | 3/D:DFS | S:Nghiem | SPC:Play({speed_str}) | F:Tree"
        surf_hint = self.font.render(hints, True, (110, 125, 160))
        self.screen.blit(surf_hint, (MARGIN + 12, panel_y + PANEL_H - 24))

        # 4. Vẽ Tree Panel (bên phải)
        self._draw_tree_panel()

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Vẽ Tree Panel
    # ------------------------------------------------------------------

    def _draw_tree_panel(self):
        """Vẽ panel cây tìm kiếm bên phải."""
        tree_x = self._left_w
        tree_y = 0
        tree_w = TREE_PANEL_W
        tree_h = self.win_h

        # Nền panel
        panel_rect = pygame.Rect(tree_x, tree_y, tree_w, tree_h)
        pygame.draw.rect(self.screen, TREE_BG, panel_rect)
        pygame.draw.line(self.screen, TREE_BORDER, (tree_x, 0), (tree_x, tree_h), 2)

        # Header
        header_h = 36
        header_rect = pygame.Rect(tree_x, 0, tree_w, header_h)
        pygame.draw.rect(self.screen, TREE_HEADER_BG, header_rect)
        pygame.draw.line(self.screen, TREE_BORDER, (tree_x, header_h), (tree_x + tree_w, header_h), 1)

        # Tiêu đề + mode indicator
        mode_label = "ACTIVE BRANCH" if self._tree_view_mode == "active" else "FULL TREE"
        header_text = f"CAY TIM KIEM [{mode_label}]"
        surf_header = self.font_tree_header.render(header_text, True, (180, 190, 220))
        self.screen.blit(surf_header, (tree_x + 10, 10))

        # Số node
        if self._tree_flat:
            visible_nodes = self._get_visible_tree_nodes()
            count_text = f"{len(visible_nodes)}/{len(self._tree_flat)} nodes"
            surf_count = self.font_tree.render(count_text, True, (120, 130, 160))
            self.screen.blit(surf_count, (tree_x + tree_w - 120, 12))

        if not self._tree_flat:
            # Không có dữ liệu
            empty_text = "Chua co du lieu."
            surf_empty = self.font_tree.render(empty_text, True, (100, 110, 140))
            self.screen.blit(surf_empty, (tree_x + 20, header_h + 20))
            hint_text = "Bam T/A/D de chay solver."
            surf_hint = self.font_tree.render(hint_text, True, (80, 90, 120))
            self.screen.blit(surf_hint, (tree_x + 20, header_h + 42))
            return

        # Vùng nội dung cây (clipped)
        content_y = header_h
        content_h = tree_h - header_h
        content_rect = pygame.Rect(tree_x, content_y, tree_w, content_h)

        # Clip để các phần tử ngoài vùng không bị vẽ ra ngoài
        self.screen.set_clip(content_rect)

        visible_nodes = self._get_visible_tree_nodes()
        active_path = get_active_path(self._tree_flat, self._replay_idx)

        # Tìm max depth để tính indent
        max_depth = max((n.depth for n in visible_nodes), default=0)

        for vi, node in enumerate(visible_nodes):
            row_y = content_y + vi * TREE_ROW_H - self._tree_scroll_y

            # Bỏ qua nếu ngoài vùng nhìn
            if row_y + TREE_ROW_H < content_y or row_y > content_y + content_h:
                continue

            indent = 10 + node.depth * TREE_INDENT
            x = tree_x + indent
            y = row_y

            # Nền highlight cho node hiện tại hoặc node trong active path
            is_current = (node.trace_idx == self._replay_idx)
            is_on_path = (node.trace_idx in active_path)

            if is_current:
                hl_rect = pygame.Rect(tree_x + 1, y, tree_w - 2, TREE_ROW_H)
                pygame.draw.rect(self.screen, TREE_ACTIVE_BG, hl_rect)
                # Viền nhấn mạnh
                pygame.draw.rect(self.screen, ACTION_COLORS.get(node.action, (200, 200, 200)),
                                 hl_rect, 1, border_radius=2)
            elif is_on_path:
                hl_rect = pygame.Rect(tree_x + 1, y, tree_w - 2, TREE_ROW_H)
                pygame.draw.rect(self.screen, TREE_HIGHLIGHT, hl_rect)

            # Đường nối nhánh (tree lines)
            if node.depth > 0:
                # Đường ngang từ indent cha đến node
                branch_x = tree_x + 10 + (node.depth - 1) * TREE_INDENT + TREE_INDENT // 2
                mid_y = y + TREE_ROW_H // 2
                pygame.draw.line(self.screen, TREE_LINE_C, (branch_x, mid_y), (x - 4, mid_y), 1)
                # Đường dọc từ cha xuống
                pygame.draw.line(self.screen, TREE_LINE_C, (branch_x, y), (branch_x, mid_y), 1)

            # Badge màu action (hình tròn nhỏ)
            badge_color = ACTION_COLORS.get(node.action, (150, 150, 150))
            circle_x = x + 4
            circle_y = y + TREE_ROW_H // 2
            pygame.draw.circle(self.screen, badge_color, (circle_x, circle_y), 5)

            # Viền tròn cho node hiện tại
            if is_current:
                pygame.draw.circle(self.screen, (255, 255, 255), (circle_x, circle_y), 7, 2)

            # Text: "(r,c) R=k"
            cell_r, cell_c = node.cell
            deg = node.rotation * 90
            label = f"({cell_r},{cell_c}) {deg:3d}"

            # Thêm indicator cho action
            action_short = {
                "TRY": "?",
                "PRUNE": "X",
                "FORWARD": ">",
                "BACKTRACK": "<",
                "SOLVED": "!",
            }.get(node.action, " ")

            label = f"{action_short} ({cell_r},{cell_c}) {deg}"

            # Chọn màu text theo trạng thái
            if is_current:
                text_color = (255, 255, 230)
            elif is_on_path:
                text_color = (200, 210, 240)
            elif node.action == "PRUNE":
                text_color = (160, 90, 90)
            elif node.action == "BACKTRACK":
                text_color = (150, 90, 160)
            else:
                text_color = (140, 150, 175)

            surf_label = self.font_tree.render(label, True, text_color)
            self.screen.blit(surf_label, (circle_x + 10, y + 3))

        # Reset clip
        self.screen.set_clip(None)

        # Scrollbar
        if visible_nodes:
            total_content_h = len(visible_nodes) * TREE_ROW_H
            if total_content_h > content_h:
                scrollbar_h = max(20, int(content_h * content_h / total_content_h))
                scrollbar_y = content_y + int(self._tree_scroll_y / total_content_h * content_h)
                scrollbar_rect = pygame.Rect(tree_x + tree_w - 8, scrollbar_y, 6, scrollbar_h)
                pygame.draw.rect(self.screen, TREE_SCROLL_BG,
                                 pygame.Rect(tree_x + tree_w - 8, content_y, 6, content_h))
                pygame.draw.rect(self.screen, TREE_SCROLL_FG, scrollbar_rect, border_radius=3)

        # Legend ở dưới đáy
        legend_y = tree_h - 22
        pygame.draw.rect(self.screen, TREE_HEADER_BG,
                         pygame.Rect(tree_x, legend_y - 4, tree_w, 26))
        legends = [
            ("?=TRY", ACTION_COLORS["TRY"]),
            (">FWD", ACTION_COLORS["FORWARD"]),
            ("X=CUT", ACTION_COLORS["PRUNE"]),
            ("<BT", ACTION_COLORS["BACKTRACK"]),
            ("!=OK", ACTION_COLORS["SOLVED"]),
        ]
        lx = tree_x + 8
        for ltxt, lcol in legends:
            pygame.draw.circle(self.screen, lcol, (lx + 4, legend_y + 6), 4)
            surf_l = self.font_tree.render(ltxt, True, (140, 150, 175))
            self.screen.blit(surf_l, (lx + 12, legend_y - 2))
            lx += surf_l.get_width() + 20

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            # Tự động phát Replay
            if self._replay_mode in ("trace", "solution") and self._replay_playing:
                now = time.perf_counter()
                if now - self._replay_last_t >= self._replay_speed:
                    max_len = len(self._trace_steps) if self._replay_mode == "trace" else len(self._solution_moves)
                    if self._replay_idx + 1 < max_len:
                        self._step_forward()
                        self._replay_last_t = now
                    else:
                        self._replay_playing = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._reset()
                    elif event.key == pygame.K_n:
                        self._new_puzzle()
                    elif event.key in (pygame.K_t, pygame.K_1):
                        # Chạy Backtracking tracer
                        self._run_solver(solver_name="backtrack", mode="trace")
                    elif event.key in (pygame.K_a, pygame.K_2):
                        # Chạy A* Search tracer
                        self._run_solver(solver_name="astar", mode="trace")
                    elif event.key in (pygame.K_d, pygame.K_3):
                        # Chạy Blind DFS tracer
                        self._run_solver(solver_name="dfs", mode="trace")
                    elif event.key == pygame.K_s:
                        # Xem trực tiếp nghiệm
                        self._run_solver(solver_name="backtrack", mode="solution")
                    elif event.key == pygame.K_f:
                        # Chuyển đổi chế độ cây
                        if self._tree_view_mode == "active":
                            self._tree_view_mode = "full"
                        else:
                            self._tree_view_mode = "active"
                        self._tree_scroll_y = 0
                    elif event.key == pygame.K_SPACE:
                        if self._replay_mode in ("trace", "solution"):
                            self._replay_playing = not self._replay_playing
                            self._replay_last_t = time.perf_counter()
                    elif event.key == pygame.K_RIGHT:
                        self._step_forward()
                    elif event.key == pygame.K_LEFT:
                        self._step_back()
                    elif event.key == pygame.K_UP:
                        self._replay_speed = max(0.01, self._replay_speed * 0.7)
                    elif event.key == pygame.K_DOWN:
                        self._replay_speed = min(1.0, self._replay_speed * 1.4)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if event.button == 1:  # chuột trái
                        if mx >= self._left_w:
                            # Click trong tree panel
                            self._handle_tree_click(mx, my)
                        elif self._replay_mode == "play":
                            c = (mx - self.ox) // self.cell_size
                            r = (my - self.oy) // self.cell_size
                            if 0 <= r < self.h and 0 <= c < self.w:
                                self._state = rotate(self._state, r, c)
                                self._active_cell = (r, c)
                                self._active_action = "TRY"
                                self._active_reason = f"Nguoi choi tu tay xoay o ({r}, {c}) them 90 do"

                    elif event.button in (4, 5):
                        # Mouse wheel (legacy pygame)
                        if mx >= self._left_w:
                            dy = 1 if event.button == 4 else -1
                            self._handle_tree_scroll(dy)

                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if mx >= self._left_w:
                        self._handle_tree_scroll(event.y)

            self._draw()

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipes Puzzle UI with Algorithm Tracer")
    parser.add_argument("--rows", "-r", type=int, default=5)
    parser.add_argument("--cols", "-c", type=int, default=5)
    parser.add_argument("--seed", "-s", type=int, default=None)
    args = parser.parse_args()

    app = PipesApp(h=args.rows, w=args.cols, seed=args.seed)
    app.run()


if __name__ == "__main__":
    main()
