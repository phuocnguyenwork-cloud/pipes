"""
Pipes – Core: Classic Search Problem Interface
===============================================
Mô hình hóa bài toán Pipes theo chuẩn State Space Search (AIMA standard).
Dành riêng cho các thuật toán tìm kiếm cổ điển:
  - Thuật toán đồ thị (Graph Search): BFS, DFS, UCS, A*, IDA*
  - Thuật toán tìm kiếm cục bộ (Local Search): Hill Climbing, Simulated Annealing, Tabu Search
  - Thuật toán di truyền (Genetic Algorithm)

Không phụ thuộc vào bất kỳ framework RL bên ngoài nào (Gym/Gymnasium).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from core.puzzle import Puzzle, State
from core.engine import rotate, is_solved, count_violations, count_open_ports


@dataclass(frozen=True)
class Action:
    """Hành động trong không gian trạng thái của Pipes.

    Attributes
    ----------
    r, c : tọa độ ô cần xoay
    k    : số lần xoay 90° theo chiều kim đồng hồ (1, 2 hoặc 3)
    """
    r: int
    c: int
    k: int = 1

    def __repr__(self) -> str:
        return f"Action(({self.r},{self.c}), +{self.k * 90}°)"


class PipesProblem:
    """Định nghĩa bài toán tìm kiếm chuẩn cho Pipes.

    Attributes
    ----------
    puzzle : Puzzle đề bài
    """

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def initial_state(self) -> State:
        """Trạng thái ban đầu của bài toán (các ô chưa xoay thêm)."""
        return State.from_puzzle(self.puzzle)

    def actions(self, state: State) -> List[Action]:
        """Tập các hành động khả thi từ state hiện tại.

        Mỗi ô có thể xoay 1 lần CW (90°).
        Tùy chọn: chỉ trả về các ô đang có vi phạm (cổng hở) để thu hẹp không gian tìm kiếm.
        """
        acts: List[Action] = []
        h, w = self.puzzle.h, self.puzzle.w
        for r in range(h):
            for c in range(w):
                acts.append(Action(r=r, c=c, k=1))
        return acts

    def result(self, state: State, action: Action) -> State:
        """Hàm chuyển trạng thái: Transition model T(s, a) -> s'."""
        return rotate(state, action.r, action.c, action.k)

    def is_goal(self, state: State) -> bool:
        """Kiểm tra điều kiện đích: toàn bộ lưới nối nguồn, không hở, không vòng."""
        return is_solved(state)

    def step_cost(self, state: State, action: Action) -> float:
        """Chi phí thực hiện một hành động (mặc định = 1.0 cho mỗi lần xoay)."""
        return 1.0

    def heuristic(self, state: State) -> float:
        """Hàm đánh giá heuristic h(n): ước lượng chi phí còn lại tới đích.

        Heuristic: Tổng số vi phạm (cổng hở + ô mất kết nối nguồn).
        Khi h(s) = 0 thì s là trạng thái đích.
        """
        viol = count_violations(state)
        return float(viol["total"])

    def successors(self, state: State) -> Iterator[Tuple[Action, State, float]]:
        """Tiện ích sinh các trạng thái kế tiếp: (action, next_state, cost)."""
        for act in self.actions(state):
            nxt = self.result(state, act)
            cost = self.step_cost(state, act)
            yield act, nxt, cost
