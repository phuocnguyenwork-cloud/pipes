"""
Pipes – UI: Pygame Application
================================
Chế độ hỗ trợ:
  - Chơi thủ công: bấm chuột trái vào ô để xoay CW 90°
  - Phím R: reset về trạng thái ban đầu
  - Phím N: tạo puzzle mới
  - Phím S: chạy solver backtrack và chuyển sang chế độ replay
  - Phím SPACE: phát/dừng replay
  - Phím LEFT/RIGHT: bước replay thủ công
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
from solvers.base import Budget, get_solver
import solvers.backtracking  # noqa: F401 – đăng ký solver


# ---------------------------------------------------------------------------
# Hằng số giao diện
# ---------------------------------------------------------------------------

CELL_SIZE = 80          # pixel mỗi ô
MARGIN = 20             # margin xung quanh
STATUS_H = 60           # chiều cao thanh trạng thái

# Màu sắc (R, G, B)
BG       = (18, 18, 32)
GRID_BG  = (30, 30, 50)
GRID_LINE= (60, 60, 90)
PIPE_DEF = (140, 140, 180)    # pipe chưa nối nguồn
PIPE_CON = (80, 220, 120)     # pipe đã nối nguồn
SOURCE   = (255, 200, 0)      # ô nguồn
SOLVED_C = (80, 255, 160)     # khi giải xong
LOCKED   = (220, 100, 80)
TEXT_C   = (200, 220, 255)
TEXT_SOLVED = (80, 255, 160)
REPLAY_C = (255, 165, 0)

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
    highlight: bool = False,
):
    cx = ox + c * cell_size + cell_size // 2
    cy = oy + r * cell_size + cell_size // 2
    rect = pygame.Rect(ox + c * cell_size, oy + r * cell_size, cell_size, cell_size)

    # Nền ô
    bg = (50, 50, 80) if highlight else GRID_BG
    pygame.draw.rect(surf, bg, rect)
    pygame.draw.rect(surf, GRID_LINE, rect, 1)

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
        pygame.display.set_caption("Pipes Puzzle")

        self.h = h
        self.w = w
        self._seed = seed
        self._seed_counter = seed if seed is not None else 0

        # Tính kích thước cửa sổ
        self.cell_size = CELL_SIZE
        self.ox = MARGIN
        self.oy = MARGIN + STATUS_H
        win_w = w * self.cell_size + 2 * MARGIN
        win_h = h * self.cell_size + 2 * MARGIN + STATUS_H
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 15)
        self.font_big = pygame.font.SysFont("consolas", 20, bold=True)

        # State
        self._new_puzzle()

        # Replay
        self._replay_moves: List[Tuple] = []
        self._replay_idx = 0
        self._replay_playing = False
        self._replay_speed = 0.15  # giây/bước
        self._replay_last_t = 0.0
        self._in_replay = False
        self._replay_state_initial: State | None = None

    def _new_puzzle(self, seed: int | None = None):
        s = seed if seed is not None else self._seed_counter
        self._seed_counter += 1
        self._puzzle, self._solution_rotations = generate_puzzle(
            h=self.h, w=self.w, seed=s
        )
        self._state = make_initial_state(self._puzzle)
        self._initial_state = self._state.copy()
        self._in_replay = False
        self._replay_moves = []

    def _reset(self):
        self._state = self._initial_state.copy()
        self._in_replay = False
        self._replay_playing = False

    def _run_solver(self):
        budget = Budget(max_time=10.0)
        solver = get_solver("backtrack")
        result = solver.solve(self._puzzle, budget)
        if result.solved:
            # Tái tạo danh sách move để replay
            self._replay_moves = list(result.moves)
            self._replay_idx = 0
            self._replay_playing = False
            self._in_replay = True
            self._replay_state_initial = make_initial_state(self._puzzle)
            self._state = self._replay_state_initial.copy()
            print(f"[Solver] Giải được! nodes={result.stats.get('nodes')}, "
                  f"time={result.stats.get('time_s', 0):.3f}s")
        else:
            print("[Solver] Không tìm được nghiệm trong budget.")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        connected = flood_fill(self._state)
        source = self._puzzle.source
        solved = is_solved(self._state)

        # Ô
        for r in range(self.h):
            for c in range(self.w):
                _draw_cell(
                    self.screen, self._state,
                    r, c, connected, source,
                    self.ox, self.oy, self.cell_size
                )

        # Thanh trạng thái
        viol = count_violations(self._state)
        mode_str = "[REPLAY]" if self._in_replay else "[PLAY]"
        if self._in_replay:
            mode_str += f" {self._replay_idx}/{len(self._replay_moves)}"

        if solved:
            status = "✅ SOLVED!"
            color = TEXT_SOLVED
        else:
            status = (f"Vi phạm: {viol['total']}  "
                      f"(cổng hở: {viol['open_ports']}, "
                      f"ngắt: {viol['disconnected']})")
            color = TEXT_C

        surf_mode = self.font.render(mode_str, True, REPLAY_C if self._in_replay else PIPE_DEF)
        surf_status = self.font_big.render(status, True, color)
        self.screen.blit(surf_mode, (MARGIN, 5))
        self.screen.blit(surf_status, (MARGIN, 28))

        # Hướng dẫn phím
        hints = "N=Mới  R=Reset  S=Solver  SPACE=Play  ←→=Bước  ESC=Thoát"
        surf_hint = self.font.render(hints, True, (100, 110, 140))
        self.screen.blit(surf_hint, (MARGIN, self.oy + self.h * self.cell_size + 5))

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Replay logic
    # ------------------------------------------------------------------

    def _replay_step_forward(self):
        if not self._in_replay:
            return
        if self._replay_idx < len(self._replay_moves):
            move = self._replay_moves[self._replay_idx]
            r, c = move[0], move[1]
            k = move[2] if len(move) > 2 else 1
            # Xoay về đúng vị trí target
            self._state = self._state.__class__(
                puzzle=self._state.puzzle,
                rotations=self._state.rotations.copy()
            )
            self._state.rotations[r, c] = k
            self._replay_idx += 1

    def _replay_step_back(self):
        if not self._in_replay or self._replay_idx == 0:
            return
        self._replay_idx -= 1
        # Tái tạo state từ đầu
        self._state = self._replay_state_initial.copy()
        for i in range(self._replay_idx):
            move = self._replay_moves[i]
            r, c = move[0], move[1]
            k = move[2] if len(move) > 2 else 1
            self._state.rotations[r, c] = k

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            # Auto replay
            if self._in_replay and self._replay_playing:
                now = time.perf_counter()
                if now - self._replay_last_t >= self._replay_speed:
                    if self._replay_idx < len(self._replay_moves):
                        self._replay_step_forward()
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
                    elif event.key == pygame.K_s:
                        self._run_solver()
                    elif event.key == pygame.K_SPACE:
                        if self._in_replay:
                            self._replay_playing = not self._replay_playing
                            self._replay_last_t = time.perf_counter()
                    elif event.key == pygame.K_RIGHT:
                        self._replay_step_forward()
                    elif event.key == pygame.K_LEFT:
                        self._replay_step_back()

                elif event.type == pygame.MOUSEBUTTONDOWN and not self._in_replay:
                    if event.button == 1:  # chuột trái
                        mx, my = event.pos
                        # Tọa độ ô
                        c = (mx - self.ox) // self.cell_size
                        r = (my - self.oy) // self.cell_size
                        if 0 <= r < self.h and 0 <= c < self.w:
                            self._state = rotate(self._state, r, c)

            self._draw()

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipes Puzzle UI")
    parser.add_argument("--rows", "-r", type=int, default=5)
    parser.add_argument("--cols", "-c", type=int, default=5)
    parser.add_argument("--seed", "-s", type=int, default=None)
    args = parser.parse_args()

    app = PipesApp(h=args.rows, w=args.cols, seed=args.seed)
    app.run()


if __name__ == "__main__":
    main()
