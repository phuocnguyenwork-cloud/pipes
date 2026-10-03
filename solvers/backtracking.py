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

from core.puzzle import (
    Puzzle, State, N, E, S, W, DIRS, DIR_NAMES, DR, DC, OPPOSITE, rotate_mask
)
from core.engine import is_solved, count_violations
from generator.tree_gen import make_initial_state
from solvers.base import Budget, Solver, SolveResult, TraceStep, register


@register("backtrack")
class BacktrackSolver(Solver):
    """Backtracking + kiểm tra nhất quán cục bộ + ghi vết chi tiết.

    Duyệt ô theo thứ tự row-major. Tại mỗi ô, thử 4 giá trị xoay (0..3).
    Ghi nhận lại toàn bộ:
      - Ô nào được chọn (r, c)
      - Thử góc nào (rotation * 90°)
      - Tại sao (lý do khớp, lý do vi phạm, hoặc lý do quay lui)
    """
    name = "backtrack"

    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult:
        h, w = puzzle.h, puzzle.w
        rotations = np.zeros((h, w), dtype=np.int8)
        cells = [(r, c) for r in range(h) for c in range(w)]

        timer = budget.timer()
        nodes = [0]
        move_log: List[Tuple] = []
        trace_log: List[TraceStep] = []

        def get_inconsistency_reason(idx: int) -> Optional[str]:
            """Kiểm tra ô hiện tại có mâu thuẫn không. Trả về lý do nếu vi phạm, None nếu hợp lệ."""
            r, c = cells[idx]
            mask = rotate_mask(int(puzzle.base[r, c]), int(rotations[r, c]))

            for d in DIRS:
                nr, nc = r + DR[d], c + DC[d]
                d_name = DIR_NAMES[d]

                # Ô kề ngoài biên
                if not (0 <= nr < h and 0 <= nc < w):
                    if mask & d:
                        return f"Cổng hướng {d_name} đâm vào tường biên ngoài bàn cờ"
                    continue

                # Chỉ số của ô kề trong thứ tự duyệt
                nb_idx = nr * w + nc

                # Ô kề chưa duyệt -> bỏ qua (sẽ được kiểm tra khi duyệt tới ô đó)
                if nb_idx >= idx:
                    continue

                # Ô kề đã xử lý -> kiểm tra 2 đầu ống
                nb_mask = rotate_mask(int(puzzle.base[nr, nc]), int(rotations[nr, nc]))
                opp_name = DIR_NAMES[OPPOSITE[d]]
                has_fwd = bool(mask & d)
                has_bwd = bool(nb_mask & OPPOSITE[d])

                if has_fwd and not has_bwd:
                    return f"Mở cổng {d_name} sang ({nr}, {nc}), nhưng ({nr}, {nc}) không mở cổng {opp_name} đón"
                if not has_fwd and has_bwd:
                    return f"Ô ({nr}, {nc}) có cổng {opp_name} chĩa sang, nhưng ({r}, {c}) không mở cổng {d_name} đón"

            return None

        def backtrack(idx: int) -> bool:
            if timer.expired():
                return False

            if idx == len(cells):
                # Kiểm tra toàn cục lần cuối
                state = State(puzzle=puzzle, rotations=rotations.copy())
                if is_solved(state):
                    last_r, last_c = cells[-1]
                    trace_log.append(TraceStep(
                        r=last_r, c=last_c,
                        rotation=int(rotations[last_r, last_c]),
                        action="SOLVED",
                        reason="Đã duyệt hết các ô! Toàn bộ ống khớp kín và nối thông với nguồn",
                        grid_rotations=rotations.copy(),
                    ))
                    return True
                return False

            r, c = cells[idx]
            nodes[0] += 1

            for k in range(4):
                rotations[r, c] = k
                deg = k * 90
                reason_fail = get_inconsistency_reason(idx)

                if reason_fail:
                    # Ghi nhận vi phạm & cắt tỉa (prune)
                    trace_log.append(TraceStep(
                        r=r, c=c, rotation=k,
                        action="PRUNE",
                        reason=f"Thử góc {deg}° thất bại: {reason_fail}",
                        grid_rotations=rotations.copy(),
                    ))
                else:
                    # Hợp lệ -> Tiếp tục đào sâu
                    trace_log.append(TraceStep(
                        r=r, c=c, rotation=k,
                        action="FORWARD",
                        reason=f"Thử góc {deg}° hợp lệ (khớp các ô kề đã duyệt) -> Tiếp tục ô sau",
                        grid_rotations=rotations.copy(),
                    ))
                    move_log.append((r, c, k))

                    if backtrack(idx + 1):
                        return True

                    move_log.pop()

            # Thử cả 4 góc xoay đều thất bại -> Quay lui
            rotations[r, c] = 0
            trace_log.append(TraceStep(
                r=r, c=c, rotation=0,
                action="BACKTRACK",
                reason=f"Cả 4 góc xoay của ô ({r}, {c}) đều không hợp lệ -> Quay lui về ô trước",
                grid_rotations=rotations.copy(),
            ))
            return False

        start = time.perf_counter()
        found = backtrack(0)
        elapsed = time.perf_counter() - start

        return SolveResult(
            solved=found,
            rotations=rotations.copy(),
            moves=list(move_log),
            trace=trace_log,
            stats={
                "nodes": nodes[0],
                "time_s": elapsed,
                "solver": self.name,
                "grid": f"{h}x{w}",
                "trace_steps": len(trace_log),
            },
        )
