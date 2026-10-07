"""
Pipes – Solver: Blind Depth-First Search (DFS)
==============================================
Thuật toán tìm kiếm theo chiều sâu duyệt mù (Blind DFS / Generate-and-Test):
- Sử dụng ngăn xếp (Stack) duyệt qua không gian trạng thái gán góc xoay cho từng ô.
- ĐẶC TRƯNG "MÙ": Không kiểm tra mâu thuẫn cục bộ (không check biên, không check cổng ô kề).
  Thuật toán đi thẳng một mạch tới khi tất cả các ô đều đã có góc xoay (chạm đáy).
- Tại đáy (tất cả ô đã gán): Gọi hàm kiểm tra toàn cục `is_solved(state)`.
  + Nếu thỏa mãn: Trả về nghiệm đúng.
  + Nếu sai: Lùi bước (backtrack) để thử góc xoay tiếp theo.

Dùng để đối chứng thực nghiệm với Backtracking (CSP) và A* Search.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from core.puzzle import (
    Puzzle, State, N, E, S, W, DIRS, DIR_NAMES, DR, DC, OPPOSITE, rotate_mask
)
from core.engine import is_solved, count_violations
from solvers.base import Budget, Solver, SolveResult, TraceStep, register


@register("dfs")
class DFSSolver(Solver):
    """Blind Depth-First Search (Duyệt mù theo chiều sâu)."""
    name = "dfs"

    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult:
        h, w = puzzle.h, puzzle.w
        cells = [(r, c) for r in range(h) for c in range(w)]
        total_cells = len(cells)

        timer = budget.timer()
        nodes_count = 0
        trace_log: List[TraceStep] = []
        max_trace_records = 2000  # Giới hạn số lượng log để tránh tràn RAM khi duyệt hàng chục ngàn node

        # Ngăn xếp (Stack) lưu các phần tử: (cell_idx, rotations_matrix)
        # Trong Python, kiểu list với append() và pop() hoạt động chuẩn như Stack (LIFO)
        initial_rotations = np.zeros((h, w), dtype=np.int8)
        stack: List[Tuple[int, np.ndarray]] = [(0, initial_rotations)]

        solution_rotations: Optional[np.ndarray] = None

        # Ghi nhận bước bắt đầu
        trace_log.append(TraceStep(
            r=cells[0][0], c=cells[0][1], rotation=0,
            action="TRY",
            reason="Bắt đầu Blind DFS từ ô (0, 0) với ngăn xếp (Stack)",
            grid_rotations=initial_rotations.copy(),
        ))

        while stack and not timer.expired():
            idx, current_rot = stack.pop()
            nodes_count += 1

            # Kiểm tra giới hạn số node nếu budget có đặt
            if budget.max_nodes and nodes_count >= budget.max_nodes:
                break

            # -------------------------------------------------------------
            # TRƯỜNG HỢP 1: ĐÃ CHẠM ĐÁY (Tất cả N ô đều đã được gán góc xoay)
            # -------------------------------------------------------------
            if idx == total_cells:
                candidate_state = State(puzzle=puzzle, rotations=current_rot)
                if is_solved(candidate_state):
                    # Tìm thấy nghiệm hợp lệ!
                    solution_rotations = current_rot.copy()
                    last_r, last_c = cells[-1]
                    trace_log.append(TraceStep(
                        r=last_r, c=last_c,
                        rotation=int(current_rot[last_r, last_c]),
                        action="SOLVED",
                        reason=f"Blind DFS tìm thấy cấu hình thỏa mãn toàn bộ lưới sau {nodes_count} nodes!",
                        grid_rotations=solution_rotations.copy(),
                    ))
                    break
                else:
                    # Chạm đáy nhưng không khớp -> Thất bại, lùi lại
                    if len(trace_log) < max_trace_records:
                        last_r, last_c = cells[-1]
                        trace_log.append(TraceStep(
                            r=last_r, c=last_c,
                            rotation=int(current_rot[last_r, last_c]),
                            action="BACKTRACK",
                            reason=f"Cấu hình chạm đáy nhưng vi phạm (is_solved=False) -> Lùi bước",
                            grid_rotations=current_rot.copy(),
                        ))
                continue

            # -------------------------------------------------------------
            # TRƯỜNG HỢP 2: ĐANG Ở NODE TRUNG GIAN (Ô thứ idx)
            # -------------------------------------------------------------
            r, c = cells[idx]

            # Ghi nhận vết suy luận nếu chưa vượt quá giới hạn log
            if len(trace_log) < max_trace_records:
                trace_log.append(TraceStep(
                    r=r, c=c,
                    rotation=int(current_rot[r, c]),
                    action="TRY",
                    reason=f"Duyệt ô ({r}, {c}) ở độ sâu {idx}/{total_cells} (Duyệt mù, không lọc ràng buộc)",
                    grid_rotations=current_rot.copy(),
                ))

            # Đẩy 4 nhánh con (tương ứng 4 góc xoay: 0, 1, 2, 3) vào Stack
            # Đẩy theo thứ tự đảo ngược (3, 2, 1, 0) để khi pop() ra sẽ duyệt từ 0 trước
            for k in (3, 2, 1, 0):
                next_rot = current_rot.copy()
                next_rot[r, c] = k
                stack.append((idx + 1, next_rot))

        elapsed = timer.elapsed()
        is_solved_flag = solution_rotations is not None
        final_rot = solution_rotations if is_solved_flag else initial_rotations

        # Tái hiện danh sách moves để replay
        moves_list: List[Tuple[int, int, int]] = []
        if is_solved_flag:
            for r in range(h):
                for c in range(w):
                    k = int(final_rot[r, c])
                    if k > 0:
                        moves_list.append((r, c, k))

        return SolveResult(
            solved=is_solved_flag,
            rotations=final_rot,
            moves=moves_list,
            trace=trace_log,
            stats={
                "nodes": nodes_count,
                "time_s": elapsed,
                "solver": self.name,
                "grid": f"{h}x{w}",
                "trace_steps": len(trace_log),
            },
        )
