"""
Pipes – Gym Environment
========================
Bọc engine theo kiểu Gymnasium để dùng cho RL và tìm kiếm.

- action_space: Discrete(H*W) – action=r*W+c xoay ô (r,c) 1 lần CW
- observation_space: Box uint8, shape (H, W, 5):
      [:, :, 0..3] = one-hot 4 hướng của mask hiện tại
      [:, :, 4]    = 1 nếu ô nối với nguồn (flood fill)
- reward: delta_violations + bonus khi giải xong
- info: violations dict, solved bool, steps int
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_BASE = gym.Env
except ImportError:
    # Fallback nếu chưa cài gymnasium: vẫn hoạt động như một class thường
    _GYM_BASE = object
    spaces = None  # type: ignore

from core.puzzle import Puzzle, State
from core.engine import rotate, count_violations, is_solved, flood_fill
from generator.tree_gen import generate_puzzle, make_initial_state


class PipesEnv(_GYM_BASE):
    """Môi trường Pipes tương thích Gymnasium.

    Parameters
    ----------
    h, w    : kích thước lưới
    seed    : seed puzzle (None = ngẫu nhiên)
    puzzle  : dùng Puzzle có sẵn thay vì sinh mới (override h, w, seed)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        h: int = 5,
        w: int = 5,
        seed: int | None = None,
        puzzle: Puzzle | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self._init_seed = seed
        self._fixed_puzzle = puzzle
        self.render_mode = render_mode

        # Spaces (chỉ khởi tạo nếu có gymnasium)
        if spaces is not None:
            self.action_space = spaces.Discrete(h * w)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(h, w, 5), dtype=np.uint8
            )

        self._state: State | None = None
        self._puzzle: Puzzle | None = None
        self._solution_rotations: np.ndarray | None = None
        self._steps = 0
        self._prev_violations = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset môi trường, sinh puzzle mới nếu không có puzzle cố định."""
        use_seed = seed if seed is not None else self._init_seed

        if self._fixed_puzzle is not None:
            self._puzzle = self._fixed_puzzle
            self._solution_rotations = None
            from core.puzzle import State
            self._state = State.from_puzzle(self._puzzle)
        else:
            self._puzzle, self._solution_rotations = generate_puzzle(
                h=self.h, w=self.w, seed=use_seed
            )
            self._state = make_initial_state(self._puzzle)

        self._steps = 0
        viol = count_violations(self._state)
        self._prev_violations = viol["total"]

        obs = self._get_obs()
        info = self._get_info(viol)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Thực hiện action: xoay ô action = r*W + c một lần CW."""
        assert self._state is not None, "Gọi reset() trước step()."

        r, c = divmod(int(action), self.w)
        self._state = rotate(self._state, r, c)
        self._steps += 1

        viol = count_violations(self._state)
        curr_violations = viol["total"]

        # Reward: cải thiện violations + bonus giải xong
        reward = float(self._prev_violations - curr_violations)
        solved = is_solved(self._state)
        if solved:
            reward += 10.0

        self._prev_violations = curr_violations

        obs = self._get_obs()
        info = self._get_info(viol, solved=solved)
        terminated = solved
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None

    # ------------------------------------------------------------------
    # State access (cho solver dạng tìm kiếm)
    # ------------------------------------------------------------------

    def get_state(self) -> State:
        assert self._state is not None
        return self._state.copy()

    def set_state(self, state: State) -> None:
        self._state = state.copy()
        viol = count_violations(self._state)
        self._prev_violations = viol["total"]

    def clone(self) -> "PipesEnv":
        """Trả về bản sao môi trường với cùng state."""
        new_env = PipesEnv(
            h=self.h, w=self.w,
            seed=self._init_seed,
            puzzle=self._fixed_puzzle,
            render_mode=self.render_mode,
        )
        new_env._puzzle = self._puzzle
        new_env._solution_rotations = self._solution_rotations
        new_env._state = self._state.copy() if self._state else None
        new_env._steps = self._steps
        new_env._prev_violations = self._prev_violations
        return new_env

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Observation: H×W×5 uint8."""
        h, w = self.h, self.w
        obs = np.zeros((h, w, 5), dtype=np.uint8)
        connected = flood_fill(self._state)
        for r in range(h):
            for c in range(w):
                mask = self._state.current_mask(r, c)
                # 4 kênh one-hot hướng: bit 0=N,1=E,2=S,3=W
                for i, d in enumerate([1, 2, 4, 8]):
                    obs[r, c, i] = 1 if (mask & d) else 0
                obs[r, c, 4] = 1 if (r, c) in connected else 0
        return obs

    def _get_info(self, viol: dict | None = None, solved: bool | None = None) -> dict:
        if viol is None:
            viol = count_violations(self._state)
        if solved is None:
            solved = is_solved(self._state)
        return {
            "violations": viol,
            "solved": solved,
            "steps": self._steps,
        }

    def _render_ansi(self) -> str:
        """Vẽ lưới bằng ký tự ASCII."""
        h, w = self.h, self.w
        connected = flood_fill(self._state)
        source = self._state.puzzle.source

        # Ký tự biểu diễn mask
        _MASK_CHAR = {
            0b0000: "  ",
            0b0001: "╵ ",   # N
            0b0010: "╶─",   # E
            0b0011: "└─",   # N+E
            0b0100: "╷ ",   # S
            0b0101: "│ ",   # N+S
            0b0110: "┌─",   # E+S
            0b0111: "├─",   # N+E+S
            0b1000: "─╴",   # W
            0b1001: "┘ ",   # N+W
            0b1010: "──",   # E+W
            0b1011: "┴─",   # N+E+W
            0b1100: "┐ ",   # S+W
            0b1101: "┤ ",   # N+S+W
            0b1110: "┬─",   # E+S+W
            0b1111: "┼─",   # all
        }

        lines = []
        for r in range(h):
            row = ""
            for c in range(w):
                mask = self._state.current_mask(r, c)
                ch = _MASK_CHAR.get(mask, "??")
                if (r, c) == source:
                    ch = "★ "
                elif (r, c) in connected:
                    ch = f"\033[92m{ch}\033[0m"  # green
                row += ch
            lines.append(row)
        return "\n".join(lines)
