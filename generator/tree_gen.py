"""
Pipes – Generator: Tree-based Puzzle Generator
================================================
Thuật toán:
1. Sinh cây bao trùm ngẫu nhiên trên lưới bằng DFS ngẫu nhiên (randomized DFS).
2. Từ cây, suy ra base_mask của mỗi ô theo các cạnh trong cây.
3. Xáo trộn: xoay mỗi ô ngẫu nhiên 0–3 lần (solution_rotations lưu nghiệm gốc).
4. Trả về (Puzzle, solution_rotations) để có thể kiểm chứng.

Luôn nhận seed để tái lập kết quả.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np

from core.puzzle import N, E, S, W, DIRS, DR, DC, OPPOSITE, rotate_mask, Puzzle, State


# ---------------------------------------------------------------------------
# Sinh cây bao trùm ngẫu nhiên (Randomized DFS / Recursive Backtracker)
# ---------------------------------------------------------------------------

def _random_spanning_tree(
    h: int, w: int, source: Tuple[int, int], rng: random.Random
) -> set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Trả về tập cạnh của cây bao trùm ngẫu nhiên.

    Mỗi cạnh là (ô_cha, ô_con) – không có thứ tự.
    """
    visited = set()
    edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def dfs(r: int, c: int):
        visited.add((r, c))
        # Xáo trộn thứ tự duyệt hướng
        dirs = list(DIRS)
        rng.shuffle(dirs)
        for d in dirs:
            nr, nc = r + DR[d], c + DC[d]
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                edges.add(((r, c), (nr, nc)))
                dfs(nr, nc)

    dfs(*source)
    return edges


# ---------------------------------------------------------------------------
# Tạo base_mask từ cây
# ---------------------------------------------------------------------------

def _build_base_masks(
    h: int, w: int, edges: set
) -> np.ndarray:
    """Tính base_mask cho mỗi ô từ tập cạnh của cây."""
    base = np.zeros((h, w), dtype=np.uint8)

    # Xác định hướng từ dr,dc
    _dir_from_delta = {(-1, 0): N, (0, 1): E, (1, 0): S, (0, -1): W}

    for (r1, c1), (r2, c2) in edges:
        dr, dc = r2 - r1, c2 - c1
        d_fwd = _dir_from_delta[(dr, dc)]
        d_bwd = OPPOSITE[d_fwd]
        base[r1, c1] |= d_fwd
        base[r2, c2] |= d_bwd

    return base


# ---------------------------------------------------------------------------
# Generator chính
# ---------------------------------------------------------------------------

def generate_puzzle(
    h: int = 5,
    w: int = 5,
    source: Tuple[int, int] | None = None,
    seed: int | None = None,
    wrap: bool = False,
    allow_cycles: bool = False,
) -> Tuple[Puzzle, np.ndarray]:
    """Sinh một puzzle pipes ngẫu nhiên.

    Parameters
    ----------
    h, w         : kích thước lưới
    source       : ô nguồn; mặc định là góc trên-trái (0, 0)
    seed         : seed cho RNG (để tái lập kết quả)
    wrap         : wrap edges (hiện tại luôn False theo plan)
    allow_cycles : cho phép vòng lặp (luôn False)

    Returns
    -------
    puzzle           : Puzzle object
    solution_rotations: mảng H×W int8 với số lần xoay để giải
                        (trong trường hợp này, tất cả 0 vì base đã là nghiệm gốc,
                         nhưng sau khi xáo trộn rotations = (4 - shuffle_k) % 4)
    """
    if wrap:
        raise NotImplementedError("Wrap chưa được hỗ trợ.")

    if source is None:
        source = (0, 0)

    rng = random.Random(seed)

    # 1. Sinh cây bao trùm ngẫu nhiên
    edges = _random_spanning_tree(h, w, source, rng)

    # 2. Tính base_mask
    base = _build_base_masks(h, w, edges)

    # 3. Xáo trộn: chọn ngẫu nhiên số lần xoay cho mỗi ô
    shuffle_rotations = np.array(
        [[rng.randint(0, 3) for _ in range(w)] for _ in range(h)],
        dtype=np.int8,
    )

    # base_shuffled[r,c] = rotate_mask(base[r,c], shuffle_rotations[r,c])
    base_shuffled = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            base_shuffled[r, c] = rotate_mask(int(base[r, c]), int(shuffle_rotations[r, c]))

    puzzle = Puzzle(
        h=h, w=w,
        base=base_shuffled,
        source=source,
        wrap=wrap,
        allow_cycles=allow_cycles,
    )

    # 4. Nghiệm gốc: xoay ngược lại shuffle_rotations
    # Nếu ô được xoay k lần khi xáo, phải xoay (4-k)%4 lần để hoàn nguyên
    solution_rotations = np.array(
        [[(4 - int(shuffle_rotations[r, c])) % 4 for c in range(w)]
         for r in range(h)],
        dtype=np.int8,
    )

    return puzzle, solution_rotations


def make_solution_state(puzzle: Puzzle, solution_rotations: np.ndarray) -> State:
    """Tạo State đã giải từ puzzle và solution_rotations."""
    return State(puzzle=puzzle, rotations=solution_rotations.copy())


def make_initial_state(puzzle: Puzzle) -> State:
    """Tạo State ban đầu (tất cả xoay = 0 → đây là trạng thái xáo trộn)."""
    return State.from_puzzle(puzzle)
