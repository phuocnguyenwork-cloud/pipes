"""
Pipes – Solver: Backtracking (Baseline)
========================================
Giải bằng backtracking đơn giản: duyệt từng ô, thử 4 hướng xoay,
kiểm tra tính nhất quán cục bộ (local consistency) trước khi đệ quy.

Đây là solver baseline để kiểm chứng pipeline và test đúng đắn của engine.
Chỉ thực tế với lưới nhỏ (≤ 7×7).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from core.puzzle import Puzzle, State, N, E, S, W, DIRS, DR, DC, OPPOSITE, rotate_mask
from core.engine import is_solved, count_violations
from generator.tree_gen import make_initial_state
from solvers.base import Budget, Solver, SolveResult, register


@register("backtrack")
class BacktrackSolver(Solver):
    """Backtracking + kiểm tra nhất quán cục bộ.

    Duyệt ô theo thứ tự row-major. Tại mỗi ô, thử 4 giá trị xoay (0..3).
    Trước khi đệ quy, kiểm tra:
    - Ô hiện tại không có cổng hở về phía ô đã được xử lý.
    - Ô hiện tại không gây mâu thuẫn với ô kề đã xác định.
    """
    name = "backtrack"

    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult:
        h, w = puzzle.h, puzzle.w
        rotations = np.zeros((h, w), dtype=np.int8)
        cells = [(r, c) for r in range(h) for c in range(w)]

        timer = budget.timer()
        nodes = [0]
        move_log: List[Tuple] = []

        def is_consistent(idx: int) -> bool:
            """Kiểm tra ô hiện tại không mâu thuẫn với ô đã xử lý."""
            r, c = cells[idx]
            mask = rotate_mask(int(puzzle.base[r, c]), int(rotations[r, c]))

            for d in DIRS:
                nr, nc = r + DR[d], c + DC[d]

                # Ô kề ngoài biên
                if not (0 <= nr < h and 0 <= nc < w):
                    if mask & d:  # cổng hướng ra biên → vi phạm
                        return False
                    continue

                # Tính chỉ số của ô kề trong danh sách cells (row-major)
                nb_idx = nr * w + nc

                # Ô kề chưa được xử lý → bỏ qua
                if nb_idx >= idx:
                    continue

                # Ô kề đã xử lý → kiểm tra nhất quán
                nb_mask = rotate_mask(int(puzzle.base[nr, nc]), int(rotations[nr, nc]))
                has_fwd = bool(mask & d)
                has_bwd = bool(nb_mask & OPPOSITE[d])
                if has_fwd != has_bwd:
                    return False

            return True

        def backtrack(idx: int) -> bool:
            if timer.expired():
                return False
            if idx == len(cells):
                # Kiểm tra lần cuối: mọi ô đều giải xong
                state = State(puzzle=puzzle, rotations=rotations.copy())
                return is_solved(state)

            r, c = cells[idx]
            nodes[0] += 1

            for k in range(4):
                rotations[r, c] = k
                if is_consistent(idx):
                    move_log.append((r, c, k))
                    if backtrack(idx + 1):
                        return True
                    move_log.pop()

            rotations[r, c] = 0
            return False

        start = time.perf_counter()
        found = backtrack(0)
        elapsed = time.perf_counter() - start

        final_state = State(puzzle=puzzle, rotations=rotations.copy())
        return SolveResult(
            solved=found,
            rotations=rotations.copy(),
            moves=list(move_log),
            stats={
                "nodes": nodes[0],
                "time_s": elapsed,
                "solver": self.name,
                "grid": f"{h}x{w}",
            },
        )
