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

PIPE_W = 12       # độ dày ống (px)


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

        self.win_w = max(grid_w + 2 * MARGIN, 540)
        self.win_h = grid_h + 2 * MARGIN + STATUS_H + PANEL_H
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_badge = pygame.font.SysFont("consolas", 13, bold=True)

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
        self._active_reason = "Sẵn sàng. Bấm T để xem suy luận, S để giải ngay."

    def _reset(self):
        self._state = self._initial_state.copy()
        self._replay_mode = "play"
        self._replay_playing = False
        self._active_cell = None
        self._active_action = ""
        self._active_reason = "Đã đặt lại trạng thái ban đầu."

    def _run_solver(self, solver_name: str = "backtrack", mode: str = "trace"):
        """Chạy solver được chọn và nạp trace/moves vào bộ phát."""
        budget = Budget(max_time=10.0, max_nodes=50000)
        solver = get_solver(solver_name)
        result = solver.solve(self._puzzle, budget)

        if result.solved:
            self._trace_steps = result.trace
            self._solution_moves = result.moves
            self._replay_idx = 0
            self._replay_playing = True
            self._replay_last_t = time.perf_counter()
            self._active_solver_name = solver_name.upper()

            if mode == "trace" and self._trace_steps:
                self._replay_mode = "trace"
                self._apply_trace_step(0)
            else:
                self._replay_mode = "solution"
                self._state = make_initial_state(self._puzzle)
                self._active_action = "SOLVED"
                self._active_reason = f"[{self._active_solver_name}] Đã giải xong ({result.stats.get('nodes')} nodes, {result.stats.get('time_s', 0):.3f}s)"

            print(f"[{self._active_solver_name}] Thành công! nodes={result.stats.get('nodes')}, "
                  f"trace_steps={len(result.trace)}, time={result.stats.get('time_s', 0):.3f}s")
        else:
            self._active_action = "PRUNE"
            self._active_reason = f"[{solver_name.upper()}] Không tìm được nghiệm trong giới hạn budget!"

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

    def _apply_solution_step(self, idx: int):
        """Áp dụng bước thứ idx trong solution moves."""
        if not (0 <= idx < len(self._solution_moves)):
            return
        move = self._solution_moves[idx]
        r, c, k = move[0], move[1], move[2]
        self._active_cell = (r, c)
        self._active_action = "FORWARD"
        self._active_reason = f"Xoay ô ({r}, {c}) tới góc {k * 90}° theo nghiệm"

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
            "play": "[CHẾ ĐỘ CHƠI TAY]",
            "trace": f"[SUY LUẬN AI: {self._replay_idx + 1}/{len(self._trace_steps)}]",
            "solution": f"[REPLAY NGHIỆM: {self._replay_idx + 1}/{len(self._solution_moves)}]"
        }
        mode_str = mode_titles.get(self._replay_mode, "")

        if solved:
            status = "✅ ĐÃ GIẢI XONG TOÀN BỘ LƯỚI!"
            color = TEXT_SOLVED
        else:
            status = f"Vi phạm: {viol['total']} (Cổng hở: {viol['open_ports']}, Mất nối: {viol['disconnected']})"
            color = TEXT_C

        surf_mode = self.font.render(mode_str, True, REPLAY_C if self._replay_mode != "play" else PIPE_DEF)
        surf_status = self.font_big.render(status, True, color)
        self.screen.blit(surf_mode, (MARGIN, 8))
        self.screen.blit(surf_status, (MARGIN, 30))

        # 3. Bảng giải thích chi tiết phía dưới (Explanation Panel)
        panel_y = self.oy + self.h * self.cell_size + 10
        panel_w = self.win_w - 2 * MARGIN
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

            cell_str = f"Ô: {self._active_cell}" if self._active_cell else ""
            surf_cell = self.font.render(cell_str, True, (180, 200, 230))
            self.screen.blit(surf_cell, (MARGIN + 110, panel_y + 10))
        else:
            surf_title = self.font.render("BẢNG GIẢI THÍCH QUÁ TRÌNH SUY LUẬN", True, (130, 140, 175))
            self.screen.blit(surf_title, (MARGIN + 12, panel_y + 10))

        # Dòng lý do (Reason text wrap đơn giản)
        reason_text = f"Lý do: {self._active_reason}" if self._active_reason else "Bấm T để xem thuật toán suy luận từng bước."
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
        hints = f"1/T:Backtrack | 2/A:A* | 3/D:DFS | S:Nghiệm | SPACE:Chạy({speed_str}) | ←→:Bước | N:Mới"
        surf_hint = self.font.render(hints, True, (110, 125, 160))
        self.screen.blit(surf_hint, (MARGIN + 12, panel_y + PANEL_H - 24))

        pygame.display.flip()

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

                elif event.type == pygame.MOUSEBUTTONDOWN and self._replay_mode == "play":
                    if event.button == 1:  # chuột trái
                        mx, my = event.pos
                        c = (mx - self.ox) // self.cell_size
                        r = (my - self.oy) // self.cell_size
                        if 0 <= r < self.h and 0 <= c < self.w:
                            self._state = rotate(self._state, r, c)
                            self._active_cell = (r, c)
                            self._active_action = "TRY"
                            self._active_reason = f"Người chơi tự tay xoay ô ({r}, {c}) thêm 90°"

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
