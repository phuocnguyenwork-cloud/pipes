"""
Pipes – Core: Puzzle Definition
================================
Biểu diễn lõi của puzzle theo bitmask 4-bit:
    N = 1 (0001)
    E = 2 (0010)
    S = 4 (0100)
    W = 8 (1000)

Mỗi ô lưu base_mask (bất biến) và rotation hiện tại (0..3).
Puzzle là frozen dataclass – immutable, hashable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hằng số hướng
# ---------------------------------------------------------------------------
N = 1   # North  0001
E = 2   # East   0010
S = 4   # South  0100
W = 8   # West   1000

DIRS = (N, E, S, W)
DIR_NAMES = {N: "N", E: "E", S: "S", W: "W"}

# Vector dịch chuyển (dr, dc) theo hướng
DR = {N: -1, E: 0, S: 1, W: 0}
DC = {N: 0,  E: 1, S: 0, W: -1}

# Hướng ngược lại
OPPOSITE = {N: S, S: N, E: W, W: E}


# ---------------------------------------------------------------------------
# Tiện ích bitmask
# ---------------------------------------------------------------------------

def rotate_mask(mask: int, k: int = 1) -> int:
    """Xoay mask 90° CW k lần: N→E→S→W→N.

    Công thức dịch bit vòng: ((m << 1) | (m >> 3)) & 0xF
    """
    k = k % 4
    for _ in range(k):
        mask = ((mask << 1) | (mask >> 3)) & 0xF
    return mask


def mask_to_dirs(mask: int) -> list[int]:
    """Trả về danh sách các hướng có trong mask."""
    return [d for d in DIRS if mask & d]


def cell_type(mask: int) -> str:
    """Loại ô theo số cổng."""
    n = bin(mask).count("1")
    if n == 0:
        return "empty"
    if n == 1:
        return "dead_end"
    if n == 2:
        dirs = mask_to_dirs(mask)
        # Thẳng: N-S hoặc E-W
        if set(dirs) in ({N, S}, {E, W}):
            return "straight"
        return "elbow"
    if n == 3:
        return "tee"
    return "cross"  # 4 cổng (plan hiện tại không dùng)


# ---------------------------------------------------------------------------
# Dataclass Puzzle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Puzzle:
    """Mô tả cấu trúc của một puzzle pipes.

    Attributes
    ----------
    h, w        : kích thước lưới
    base        : mảng H×W uint8, base mask mỗi ô (hướng trong nghiệm gốc)
    source      : vị trí ô nguồn (r, c)
    wrap        : cho phép ống nối qua biên (luôn False theo plan)
    allow_cycles: cho phép vòng lặp (luôn False theo plan)
    """
    h: int
    w: int
    base: np.ndarray          # shape (h, w), dtype uint8, base mask
    source: Tuple[int, int]
    wrap: bool = False
    allow_cycles: bool = False

    def __post_init__(self):
        # numpy array không hashable nên frozen=True không thể dùng __hash__ mặc định.
        # Ta dùng object.__setattr__ để bỏ qua frozen cho trường _base_bytes.
        object.__setattr__(self, "_base_bytes", self.base.tobytes())

    def __hash__(self):
        return hash((self.h, self.w, self._base_bytes, self.source, self.wrap))

    def __eq__(self, other):
        if not isinstance(other, Puzzle):
            return NotImplemented
        return (self.h == other.h and self.w == other.w
                and self.source == other.source
                and self.wrap == other.wrap
                and np.array_equal(self.base, other.base))

    def cell_mask(self, r: int, c: int) -> int:
        """Base mask của ô (r, c)."""
        return int(self.base[r, c])


# ---------------------------------------------------------------------------
# State – trạng thái hiện tại của puzzle (mutable)
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Trạng thái động của puzzle: lưu số lần xoay mỗi ô.

    Attributes
    ----------
    puzzle   : Puzzle gốc (bất biến)
    rotations: mảng H×W int8, số lần xoay (0..3) mỗi ô
    """
    puzzle: Puzzle
    rotations: np.ndarray     # shape (h, w), dtype int8

    def current_mask(self, r: int, c: int) -> int:
        """Mask hiện tại = base_mask xoay thêm rotations[r,c] lần."""
        return rotate_mask(self.puzzle.cell_mask(r, c), int(self.rotations[r, c]))

    def current_grid(self) -> np.ndarray:
        """Trả về mảng H×W chứa mask hiện tại của toàn bộ ô."""
        h, w = self.puzzle.h, self.puzzle.w
        grid = np.zeros((h, w), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                grid[r, c] = self.current_mask(r, c)
        return grid

    def copy(self) -> "State":
        return State(puzzle=self.puzzle, rotations=self.rotations.copy())

    def __hash__(self) -> int:
        return hash((self.puzzle, self.rotations.tobytes()))

    def __eq__(self, other) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.puzzle == other.puzzle and np.array_equal(self.rotations, other.rotations)

    def __lt__(self, other) -> bool:
        # Hỗ trợ PriorityQueue/heapq khi có hai node cùng giá trị f(n)
        return False

    @classmethod
    def from_puzzle(cls, puzzle: Puzzle, rotations: np.ndarray | None = None) -> "State":
        """Tạo State từ Puzzle. Nếu không có rotations thì mặc định 0."""
        if rotations is None:
            rotations = np.zeros((puzzle.h, puzzle.w), dtype=np.int8)
        return cls(puzzle=puzzle, rotations=rotations.copy())
