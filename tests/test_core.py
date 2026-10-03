"""
Pipes – Unit Tests
===================
Chạy: pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from core.puzzle import (
    N, E, S, W, rotate_mask, mask_to_dirs, cell_type,
    Puzzle, State,
)
from core.engine import (
    rotate, are_connected, flood_fill, is_solved,
    count_violations, count_open_ports,
)
from generator.tree_gen import generate_puzzle, make_initial_state, make_solution_state


# ---------------------------------------------------------------------------
# Test rotate_mask
# ---------------------------------------------------------------------------

class TestRotateMask:
    def test_4_rotations_identity(self):
        """Xoay 4 lần trở về ban đầu."""
        for mask in range(16):
            assert rotate_mask(mask, 4) == mask, f"mask={mask}"

    def test_n_becomes_e(self):
        """N xoay 1 lần CW → E."""
        assert rotate_mask(N, 1) == E

    def test_e_becomes_s(self):
        assert rotate_mask(E, 1) == S

    def test_s_becomes_w(self):
        assert rotate_mask(S, 1) == W

    def test_w_becomes_n(self):
        assert rotate_mask(W, 1) == N

    def test_straight_ns_becomes_ew(self):
        assert rotate_mask(N | S, 1) == E | W

    def test_elbow(self):
        # N+E xoay 1 → E+S
        assert rotate_mask(N | E, 1) == E | S


# ---------------------------------------------------------------------------
# Test cell_type
# ---------------------------------------------------------------------------

class TestCellType:
    def test_dead_end(self):
        assert cell_type(N) == "dead_end"

    def test_straight(self):
        assert cell_type(N | S) == "straight"
        assert cell_type(E | W) == "straight"

    def test_elbow(self):
        assert cell_type(N | E) == "elbow"

    def test_tee(self):
        assert cell_type(N | E | S) == "tee"

    def test_cross(self):
        assert cell_type(N | E | S | W) == "cross"


# ---------------------------------------------------------------------------
# Puzzle 2×2 đơn giản để test engine
# ---------------------------------------------------------------------------

def _make_2x2_solved() -> State:
    """
    Dùng generator tạo puzzle 2x2 và trả về solution state (guaranteed valid).
    """
    from generator.tree_gen import generate_puzzle, make_solution_state
    puzzle, sol_rot = generate_puzzle(h=2, w=2, seed=0)
    return make_solution_state(puzzle, sol_rot)


class TestEngine:
    def test_is_solved_2x2(self):
        state = _make_2x2_solved()
        assert is_solved(state)

    def test_count_violations_zero_when_solved(self):
        state = _make_2x2_solved()
        v = count_violations(state)
        assert v["total"] == 0

    def test_rotate_changes_mask(self):
        state = _make_2x2_solved()
        new_state = rotate(state, 0, 0, 1)
        assert new_state.current_mask(0, 0) != state.current_mask(0, 0)

    def test_rotate_4_times_identity(self):
        state = _make_2x2_solved()
        s = state
        for _ in range(4):
            s = rotate(s, 0, 0)
        assert s.current_mask(0, 0) == state.current_mask(0, 0)

    def test_flood_fill_solved(self):
        state = _make_2x2_solved()
        connected = flood_fill(state)
        assert len(connected) == 4

    def test_flood_fill_partial(self):
        """Xoay một ô dead-end khỏi vị trí đúng → ô đó mất kết nối."""
        from generator.tree_gen import generate_puzzle, make_solution_state
        puzzle, sol_rot = generate_puzzle(h=3, w=3, seed=42)
        state = make_solution_state(puzzle, sol_rot)
        # Tìm một ô dead-end (1 cổng) để xoay
        h, w = puzzle.h, puzzle.w
        dead_end = None
        for r in range(h):
            for c in range(w):
                if bin(state.current_mask(r, c)).count('1') == 1 and (r, c) != puzzle.source:
                    dead_end = (r, c)
                    break
            if dead_end:
                break
        assert dead_end is not None, "Không tìm thấy ô dead-end"
        dr, dc = dead_end
        state2 = rotate(state, dr, dc, 1)  # xoay → mất kết nối
        connected = flood_fill(state2)
        assert dead_end not in connected

    def test_open_ports_on_boundary(self):
        """Ô có cổng hướng ra biên → bị tính là hở."""
        # Tạo ô đơn giản: mask=N (hướng lên) ở hàng 0 → ra biên
        base = np.array([[N]], dtype=np.uint8)
        puzzle = Puzzle(h=1, w=1, base=base, source=(0, 0))
        state = State(puzzle=puzzle, rotations=np.zeros((1, 1), dtype=np.int8))
        assert count_open_ports(state, 0, 0) == 1

    def test_not_solved_when_open_port(self):
        base = np.array([[N]], dtype=np.uint8)
        puzzle = Puzzle(h=1, w=1, base=base, source=(0, 0))
        state = State(puzzle=puzzle, rotations=np.zeros((1, 1), dtype=np.int8))
        assert not is_solved(state)

    def test_solved_single_cell_no_ports(self):
        """Ô 1×1 không có cổng nào → coi như giải được (cây rỗng)."""
        base = np.array([[0]], dtype=np.uint8)
        puzzle = Puzzle(h=1, w=1, base=base, source=(0, 0))
        state = State(puzzle=puzzle, rotations=np.zeros((1, 1), dtype=np.int8))
        assert is_solved(state)


# ---------------------------------------------------------------------------
# Test Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_generate_5x5_seed(self):
        puzzle, sol_rot = generate_puzzle(h=5, w=5, seed=42)
        assert puzzle.h == 5
        assert puzzle.w == 5

    def test_solution_is_valid(self):
        """Nghiệm gốc phải giải được."""
        for seed in range(10):
            puzzle, sol_rot = generate_puzzle(h=5, w=5, seed=seed)
            state = make_solution_state(puzzle, sol_rot)
            assert is_solved(state), f"seed={seed}: nghiệm gốc không giải được."

    def test_initial_state_not_solved(self):
        """Trạng thái ban đầu (xáo trộn) thường không giải được."""
        not_solved_count = 0
        for seed in range(20):
            puzzle, _ = generate_puzzle(h=5, w=5, seed=seed)
            state = make_initial_state(puzzle)
            if not is_solved(state):
                not_solved_count += 1
        # Phần lớn phải chưa giải được
        assert not_solved_count >= 15

    def test_deterministic_with_seed(self):
        """Cùng seed → cùng puzzle."""
        p1, r1 = generate_puzzle(h=5, w=5, seed=99)
        p2, r2 = generate_puzzle(h=5, w=5, seed=99)
        assert np.array_equal(p1.base, p2.base)
        assert np.array_equal(r1, r2)

    def test_different_seeds_different_puzzles(self):
        p1, _ = generate_puzzle(h=5, w=5, seed=1)
        p2, _ = generate_puzzle(h=5, w=5, seed=2)
        assert not np.array_equal(p1.base, p2.base)

    @pytest.mark.parametrize("size", [(3, 3), (4, 4), (5, 5), (6, 6)])
    def test_various_sizes(self, size):
        h, w = size
        puzzle, sol_rot = generate_puzzle(h=h, w=w, seed=0)
        state = make_solution_state(puzzle, sol_rot)
        assert is_solved(state), f"size={size}: nghiệm không hợp lệ"


# ---------------------------------------------------------------------------
# Test Solver (backtrack)
# ---------------------------------------------------------------------------

class TestBacktrackSolver:
    def test_solve_3x3(self):
        from solvers.base import Budget, get_solver
        import solvers.backtracking  # noqa

        for seed in range(5):
            puzzle, _ = generate_puzzle(h=3, w=3, seed=seed)
            solver = get_solver("backtrack")
            result = solver.solve(puzzle, Budget(max_time=5.0))
            assert result.solved, f"seed={seed}: solver không giải được."

    def test_solve_result_is_valid(self):
        from solvers.base import Budget, get_solver
        import solvers.backtracking  # noqa

        puzzle, _ = generate_puzzle(h=4, w=4, seed=7)
        solver = get_solver("backtrack")
        result = solver.solve(puzzle, Budget(max_time=10.0))
        assert result.solved

        state = State(puzzle=puzzle, rotations=result.rotations)
        assert is_solved(state)

    def test_trace_contains_reasons_and_actions(self):
        """Kiểm tra trace lưu đúng cấu trúc: ô, hướng, lý do, action."""
        from solvers.base import Budget, get_solver
        import solvers.backtracking  # noqa

        puzzle, _ = generate_puzzle(h=3, w=3, seed=12)
        solver = get_solver("backtrack")
        result = solver.solve(puzzle, Budget(max_time=5.0))
        assert result.solved
        assert len(result.trace) > 0

        # Kiểm tra các loại action có mặt
        actions = {s.action for s in result.trace}
        assert "FORWARD" in actions
        assert "SOLVED" in actions

        # Kiểm tra mỗi step đều có lý do và tọa độ hợp lệ
        for step in result.trace:
            assert 0 <= step.r < puzzle.h
            assert 0 <= step.c < puzzle.w
            assert 0 <= step.rotation <= 3
            assert isinstance(step.reason, str) and len(step.reason) > 0


# ---------------------------------------------------------------------------
# Test Pure Search Problem Interface & A* Search
# ---------------------------------------------------------------------------

class TestSearchProblem:
    def test_problem_initial_state_and_is_goal(self):
        from core.problem import PipesProblem, Action
        puzzle, sol_rot = generate_puzzle(h=3, w=3, seed=5)
        problem = PipesProblem(puzzle)

        s0 = problem.initial_state()
        assert s0.puzzle == puzzle
        assert problem.heuristic(s0) >= 0

        # Trạng thái nghiệm phải là goal và heuristic = 0
        sol_state = make_solution_state(puzzle, sol_rot)
        assert problem.is_goal(sol_state)
        assert problem.heuristic(sol_state) == 0.0

    def test_problem_actions_and_result(self):
        from core.problem import PipesProblem, Action
        puzzle, _ = generate_puzzle(h=2, w=2, seed=1)
        problem = PipesProblem(puzzle)
        s0 = problem.initial_state()

        acts = problem.actions(s0)
        assert len(acts) == 4  # 2x2 = 4 actions

        # Áp dụng action xoay ô (0, 0)
        a = Action(0, 0, 1)
        s1 = problem.result(s0, a)
        assert s1.rotations[0, 0] == (s0.rotations[0, 0] + 1) % 4
        assert s1 != s0

    def test_state_hashable_in_set_and_dict(self):
        puzzle, _ = generate_puzzle(h=2, w=2, seed=2)
        s1 = State.from_puzzle(puzzle)
        s2 = s1.copy()
        s3 = rotate(s1, 0, 0, 1)

        visited = {s1}
        assert s2 in visited
        assert s3 not in visited


class TestAStarSolver:
    def test_astar_solve_2x2(self):
        from solvers.base import Budget, get_solver
        import solvers.astar  # noqa

        for seed in range(3):
            puzzle, _ = generate_puzzle(h=2, w=2, seed=seed)
            solver = get_solver("astar")
            result = solver.solve(puzzle, Budget(max_time=5.0))
            assert result.solved, f"seed={seed}: A* không giải được lưới 2x2."
            final_state = State(puzzle=puzzle, rotations=result.rotations)
            assert is_solved(final_state)


