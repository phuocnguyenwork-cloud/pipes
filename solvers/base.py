"""
Pipes – Solvers: Base Interface & Registry
==========================================
Mọi solver phải kế thừa Solver và đăng ký qua @register("tên").
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from core.puzzle import Puzzle
from core.engine import is_solved, count_violations
from generator.tree_gen import make_initial_state


# ---------------------------------------------------------------------------
# Budget – giới hạn tài nguyên
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Giới hạn tài nguyên cho solver.

    Attributes
    ----------
    max_time    : thời gian tối đa (giây); None = không giới hạn
    max_nodes   : số node/iteration tối đa; None = không giới hạn
    max_steps   : số bước action tối đa; None = không giới hạn
    """
    max_time: float | None = None
    max_nodes: int | None = None
    max_steps: int | None = None

    def timer(self) -> "_Timer":
        return _Timer(self.max_time)


class _Timer:
    def __init__(self, max_time: float | None):
        self._start = time.perf_counter()
        self._max = max_time

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def expired(self) -> bool:
        if self._max is None:
            return False
        return self.elapsed() >= self._max


# ---------------------------------------------------------------------------
# TraceStep – Vết từng bước suy luận của solver
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """Mô tả một bước suy luận/thử sai của thuật toán.

    Attributes
    ----------
    r, c            : tọa độ ô đang được chọn duyệt
    rotation        : số lần xoay 90° CW hiện tại (0..3)
    action          : loại hành động: "TRY", "PRUNE", "FORWARD", "BACKTRACK", "SOLVED"
    reason          : lý do cụ thể (tại sao chọn hướng này, tại sao vi phạm, tại sao quay lui)
    grid_rotations  : snapshot ma trận xoay tại bước này để hiển thị trực quan
    """
    r: int
    c: int
    rotation: int
    action: str
    reason: str
    grid_rotations: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# SolveResult
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    """Kết quả trả về từ solver.

    Attributes
    ----------
    solved      : True nếu solver tìm được nghiệm
    rotations   : mảng H×W int8 với số lần xoay cuối cùng (nghiệm chuẩn)
    moves       : danh sách move dẫn đến nghiệm (r, c, k)
    trace       : toàn bộ quá trình tìm kiếm/thử sai/quay lui kèm lý giải
    stats       : dict thống kê (nodes, time_s, iterations, …)
    """
    solved: bool
    rotations: np.ndarray
    moves: List[Tuple] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lớp cơ sở Solver
# ---------------------------------------------------------------------------

class Solver(ABC):
    """Interface chung cho mọi solver.

    Để thêm solver mới:
    1. Kế thừa lớp này.
    2. Implement `solve()`.
    3. Đăng ký bằng `@register("tên")`.
    """
    name: str = "base"

    @abstractmethod
    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult:
        """Giải puzzle.

        Parameters
        ----------
        puzzle : Puzzle cần giải
        budget : giới hạn tài nguyên

        Returns
        -------
        SolveResult
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[Solver]] = {}


def register(name: str) -> Callable[[Type[Solver]], Type[Solver]]:
    """Decorator đăng ký solver vào registry.

    Ví dụ::

        @register("backtrack")
        class BacktrackSolver(Solver):
            name = "backtrack"
            ...
    """
    def decorator(cls: Type[Solver]) -> Type[Solver]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_solver(name: str) -> Solver:
    """Lấy instance của solver theo tên."""
    if name not in _REGISTRY:
        raise KeyError(f"Solver '{name}' chưa được đăng ký. Có: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_solvers() -> List[str]:
    """Danh sách tên các solver đã đăng ký."""
    return list(_REGISTRY.keys())
