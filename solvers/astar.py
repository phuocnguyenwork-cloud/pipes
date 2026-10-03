"""
Pipes – Solver: A* Search
=========================
Thuật toán tìm kiếm A* sử dụng Search Interface thuần túy (PipesProblem).
Hàm đánh giá: f(n) = g(n) + h(n)
  - g(n): số bước xoay từ trạng thái ban đầu
  - h(n): heuristic ước lượng vi phạm (cổng hở + ô mất kết nối)

Có ghi vết TraceStep chi tiết: giá trị f, g, h và lý do chọn mở từng node.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.puzzle import Puzzle, State
from core.problem import PipesProblem, Action
from solvers.base import Budget, Solver, SolveResult, TraceStep, register


@register("astar")
class AStarSolver(Solver):
    """A* Search trên không gian trạng thái Pipes."""
    name = "astar"

    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult:
        problem = PipesProblem(puzzle)
        initial_state = problem.initial_state()

        timer = budget.timer()
        nodes_expanded = 0
        trace_log: List[TraceStep] = []

        # Priority Queue: lưu tuple (f_score, counter, state, g_score, path_moves)
        counter = 0
        h_init = problem.heuristic(initial_state)
        f_init = 0.0 + h_init

        frontier: List[Tuple[float, int, State, float, List[Action]]] = []
        heapq.heappush(frontier, (f_init, counter, initial_state, 0.0, []))

        # Best g-score cho từng state
        best_g: Dict[State, float] = {initial_state: 0.0}

        solution_state: Optional[State] = None
        solution_moves: List[Action] = []

        trace_log.append(TraceStep(
            r=puzzle.source[0], c=puzzle.source[1], rotation=0,
            action="TRY",
            reason=f"Bắt đầu A* từ trạng thái khởi tạo. Heuristic h = {h_init:.0f}",
            grid_rotations=initial_state.rotations.copy(),
        ))

        while frontier and not timer.expired():
            f, _, current_state, g, path = heapq.heappop(frontier)
            nodes_expanded += 1

            # Nếu g hiện tại lớn hơn best_g đã biết -> bỏ qua
            if g > best_g.get(current_state, float("inf")):
                continue

            # Kiểm tra trạng thái đích
            if problem.is_goal(current_state):
                solution_state = current_state
                solution_moves = path
                last_act = path[-1] if path else Action(0, 0, 0)
                trace_log.append(TraceStep(
                    r=last_act.r, c=last_act.c, rotation=int(current_state.rotations[last_act.r, last_act.c]),
                    action="SOLVED",
                    reason=f"A* tìm thấy lời giải tối ưu! Chi phí g = {g:.0f}, mở rộng {nodes_expanded} nodes",
                    grid_rotations=current_state.rotations.copy(),
                ))
                break

            # Giới hạn số nodes nếu vượt budget
            if budget.max_nodes and nodes_expanded >= budget.max_nodes:
                break

            # Mở rộng các trạng thái kế tiếp
            h, w = puzzle.h, puzzle.w
            for act, next_state, cost in problem.successors(current_state):
                next_g = g + cost
                if next_g < best_g.get(next_state, float("inf")):
                    best_g[next_state] = next_g
                    next_h = problem.heuristic(next_state)
                    next_f = next_g + next_h
                    counter += 1
                    next_path = path + [act]
                    heapq.heappush(frontier, (next_f, counter, next_state, next_g, next_path))

                    # Ghi nhận bước tiến triển nếu có cải thiện heuristic
                    if next_h < h_init:
                        trace_log.append(TraceStep(
                            r=act.r, c=act.c,
                            rotation=int(next_state.rotations[act.r, act.c]),
                            action="FORWARD",
                            reason=f"Xoay ({act.r},{act.c}) -> Giảm vi phạm h={next_h:.0f} (f={next_f:.0f}, g={next_g:.0f})",
                            grid_rotations=next_state.rotations.copy(),
                        ))

        elapsed = timer.elapsed()
        is_solved_flag = solution_state is not None
        final_rotations = solution_state.rotations.copy() if solution_state else initial_state.rotations.copy()
        moves_tuples = [(act.r, act.c, act.k) for act in solution_moves]

        return SolveResult(
            solved=is_solved_flag,
            rotations=final_rotations,
            moves=moves_tuples,
            trace=trace_log,
            stats={
                "nodes": nodes_expanded,
                "time_s": elapsed,
                "solver": self.name,
                "grid": f"{puzzle.h}x{puzzle.w}",
                "trace_steps": len(trace_log),
            },
        )
