"""
Pipes – Core: Engine (Logic thuần)
====================================
Không phụ thuộc UI hay bất kỳ framework bên ngoài.
Tất cả hàm nhận State và trả về kết quả, không sửa State tại chỗ
(trừ `rotate` trả về State mới).
"""

from __future__ import annotations

from collections import deque
from typing import Set, Tuple

import numpy as np

from core.puzzle import (
    N, E, S, W, DIRS, OPPOSITE, DR, DC,
    rotate_mask, State, Puzzle,
)


# ---------------------------------------------------------------------------
# Xoay ô
# ---------------------------------------------------------------------------

def rotate(state: State, r: int, c: int, k: int = 1) -> State:
    """Trả về State mới với ô (r,c) đã xoay thêm k lần (CW 90°)."""
    new_rot = state.rotations.copy()
    new_rot[r, c] = (int(new_rot[r, c]) + k) % 4
    return State(puzzle=state.puzzle, rotations=new_rot)


# ---------------------------------------------------------------------------
# Kiểm tra kết nối giữa hai ô kề nhau
# ---------------------------------------------------------------------------

def are_connected(state: State, r1: int, c1: int, r2: int, c2: int) -> bool:
    """Kiểm tra hai ô kề (r1,c1) và (r2,c2) có cổng hướng vào nhau không.

    Chỉ xét hai ô kề trực tiếp (1 bước), không kiểm tra kề xa.
    """
    h, w = state.puzzle.h, state.puzzle.w

    # Xác định hướng từ (r1,c1) → (r2,c2)
    dr, dc = r2 - r1, c2 - c1
    direction_map = {(-1, 0): N, (0, 1): E, (1, 0): S, (0, -1): W}
    if (dr, dc) not in direction_map:
        raise ValueError(f"({r1},{c1}) và ({r2},{c2}) không kề nhau.")

    dir_fwd = direction_map[(dr, dc)]
    dir_bwd = OPPOSITE[dir_fwd]

    mask1 = state.current_mask(r1, c1)
    mask2 = state.current_mask(r2, c2)
    return bool(mask1 & dir_fwd) and bool(mask2 & dir_bwd)


def get_connected_neighbors(state: State, r: int, c: int) -> list[Tuple[int, int]]:
    """Trả về danh sách ô kề nối được với (r,c)."""
    h, w = state.puzzle.h, state.puzzle.w
    puzzle = state.puzzle
    result = []
    mask = state.current_mask(r, c)
    for d in DIRS:
        if not (mask & d):
            continue
        nr, nc = r + DR[d], c + DC[d]
        # Kiểm tra biên
        if not (0 <= nr < h and 0 <= nc < w):
            continue  # wrap=False: bỏ qua
        # Ô kề có cổng ngược lại không?
        nb_mask = state.current_mask(nr, nc)
        if nb_mask & OPPOSITE[d]:
            result.append((nr, nc))
    return result


# ---------------------------------------------------------------------------
# Flood fill (BFS từ nguồn)
# ---------------------------------------------------------------------------

def flood_fill(state: State) -> Set[Tuple[int, int]]:
    """BFS từ source dọc theo các kết nối hợp lệ.

    Trả về tập các ô đã nối với nguồn (bao gồm bản thân nguồn).
    """
    source = state.puzzle.source
    visited: Set[Tuple[int, int]] = set()
    queue: deque[Tuple[int, int]] = deque([source])
    visited.add(source)

    while queue:
        r, c = queue.popleft()
        for nr, nc in get_connected_neighbors(state, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited


# ---------------------------------------------------------------------------
# Kiểm tra lời giải
# ---------------------------------------------------------------------------

def is_solved(state: State) -> bool:
    """Puzzle giải được khi:
    1. Mọi ô đều nối với nguồn (flood fill bao phủ toàn bộ).
    2. Không có cổng hở (mỗi cổng phải được kết nối với ô kề tương ứng).
    3. Không có vòng (nếu allow_cycles=False) – kiểm tra bằng BFS tree.

    Với cây spanning tree, flood fill = toàn bộ ô và số cạnh = H*W-1 là đủ.
    """
    puzzle = state.puzzle
    h, w = puzzle.h, puzzle.w
    total = h * w

    # --- Flood fill ---
    connected = flood_fill(state)
    if len(connected) != total:
        return False

    # --- Kiểm tra cổng hở ---
    for r in range(h):
        for c in range(w):
            if count_open_ports(state, r, c) > 0:
                return False

    # --- Kiểm tra vòng (nếu cần) ---
    if not puzzle.allow_cycles:
        # Với cây spanning: số cạnh hợp lệ phải đúng bằng total-1
        edges = _count_edges(state)
        if edges != total - 1:
            return False

    return True


def _count_edges(state: State) -> int:
    """Đếm số cạnh nối thực sự trong lưới (mỗi cặp ô kề chỉ tính 1 lần)."""
    h, w = state.puzzle.h, state.puzzle.w
    count = 0
    # Chỉ kiểm tra hướng S và E để tránh đếm đôi
    for r in range(h):
        for c in range(w):
            mask = state.current_mask(r, c)
            if (mask & S) and r + 1 < h:
                if state.current_mask(r + 1, c) & N:
                    count += 1
            if (mask & E) and c + 1 < w:
                if state.current_mask(r, c + 1) & W:
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Đếm vi phạm (heuristic cho solver và reward cho RL)
# ---------------------------------------------------------------------------

def count_open_ports(state: State, r: int, c: int) -> int:
    """Số cổng của ô (r,c) không được kết nối với ô kề tương ứng."""
    h, w = state.puzzle.h, state.puzzle.w
    puzzle = state.puzzle
    mask = state.current_mask(r, c)
    open_count = 0

    for d in DIRS:
        if not (mask & d):
            continue
        nr, nc = r + DR[d], c + DC[d]
        # Ra biên (wrap=False) → hở
        if not (0 <= nr < h and 0 <= nc < w):
            open_count += 1
            continue
        # Ô kề không có cổng đối diện → hở
        nb_mask = state.current_mask(nr, nc)
        if not (nb_mask & OPPOSITE[d]):
            open_count += 1

    return open_count


def count_violations(state: State) -> dict:
    """Tổng hợp các vi phạm trong state hiện tại.

    Returns
    -------
    dict với các key:
        open_ports  : tổng số cổng hở
        disconnected: số ô không nối được với nguồn
        cycles      : số chu trình (0 nếu allow_cycles=True hay không kiểm tra)
        total       : tổng vi phạm (số ô chưa giải)
    """
    h, w = state.puzzle.h, state.puzzle.w

    # Cổng hở
    open_ports = sum(
        count_open_ports(state, r, c)
        for r in range(h)
        for c in range(w)
    )

    # Ô không nối nguồn
    connected = flood_fill(state)
    disconnected = h * w - len(connected)

    # Vòng lặp
    cycles = 0
    if not state.puzzle.allow_cycles:
        edges = _count_edges(state)
        # Số cạnh trong cây = nodes - 1; thêm 1 cạnh → 1 vòng
        expected_edges = len(connected) - 1
        cycles = max(0, edges - expected_edges)

    total = open_ports + disconnected + cycles
    return {
        "open_ports": open_ports,
        "disconnected": disconnected,
        "cycles": cycles,
        "total": total,
    }
