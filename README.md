# Pipes Puzzle Environment

Môi trường trò chơi **Pipes (FreeNet/NetWalk)** với kiến trúc tách biệt engine – generator – UI – solver, thiết kế để tích hợp và so sánh các thuật toán giải.

## Cấu trúc dự án

```
pipes/
├── core/
│   ├── puzzle.py      # Biểu diễn bitmask, Puzzle, State
│   └── engine.py      # Logic thuần: rotate, flood_fill, is_solved, count_violations
├── generator/
│   └── tree_gen.py    # Sinh puzzle từ cây bao trùm ngẫu nhiên
├── env/
│   └── gym_env.py     # Môi trường Gymnasium (cho RL và tìm kiếm)
├── solvers/
│   ├── base.py        # Interface Solver, Budget, SolveResult, Registry
│   └── backtracking.py # Solver baseline: backtracking + local consistency
├── ui/
│   └── pygame_app.py  # Giao diện Pygame (chơi tay + replay solver)
├── tests/
│   └── test_core.py   # Unit tests
├── main.py            # Entry point UI
├── requirements.txt
└── pyproject.toml
```

## Cài đặt

```bash
pip install -r requirements.txt
# hoặc
pip install -e ".[dev,rl]"
```

## Chạy

```bash
# Chơi puzzle 5×5
python main.py

# Tùy chỉnh kích thước và seed
python main.py --rows 7 --cols 7 --seed 42

# Chạy unit tests
pytest
```

## Điều khiển UI & Trực quan hóa suy luận

| Phím | Chức năng |
|------|-----------|
| Chuột trái | Chơi thủ công: Xoay ô CW 90° |
| `T` | **Xem vết suy luận AI (Trace Replay)**: Hiển thị từng bước thử sai, lý do vi phạm (PRUNE), đào sâu (FORWARD) và quay lui (BACKTRACK) trên bảng điều khiển |
| `S` | **Xem nghiệm (Solution Replay)**: Chạy solver và hiển thị chuỗi bước dẫn thẳng tới lời giải |
| `SPACE` | Tạm dừng / Tiếp tục phát tự động |
| `←` / `→` | Lùi lại / Tiến tới 1 bước suy luận thủ công |
| `↑` / `↓` | Tăng / Giảm tốc độ chạy replay |
| `N` / `R` | Tạo màn mới / Đặt lại trạng thái ban đầu |
| `ESC` | Thoát ứng dụng |

## Biểu diễn dữ liệu

Mỗi ô là bitmask 4-bit: **N=1, E=2, S=4, W=8**.

```python
# Xoay 90° CW
rotate_mask(mask, k=1)  # ((m << 1) | (m >> 3)) & 0xF

# Puzzle là frozen dataclass
puzzle = Puzzle(h=5, w=5, base=..., source=(0,0))

# State = Puzzle + rotations array
state = State(puzzle=puzzle, rotations=np.zeros((5,5), dtype=np.int8))
```

## Thêm Solver mới

```python
from solvers.base import Solver, SolveResult, Budget, register

@register("my_solver")
class MySolver(Solver):
    name = "my_solver"

    def solve(self, puzzle, budget):
        # ... thuật toán của bạn ...
        return SolveResult(solved=True, rotations=rot, moves=moves, stats={})
```

## Thiết kế (từ plan.md)

```
[Generator] → [Puzzle/State] ← [Engine] ← [Solver]
                                   ↑
                    [Gym Env]   [Pygame UI]
```

- **Engine** = pure logic, không phụ thuộc UI
- **Solver** giao tiếp qua interface `Solver.solve(puzzle, budget) → SolveResult`
- **Renderer** chỉ đọc State để vẽ
- **Quyết định cố định**: không wrap, không vòng lặp, không ô chữ thập, có seed

## Lộ trình

| Giai đoạn | Trạng thái |
|-----------|-----------|
| 1 – Core engine (bitmask, flood fill, is_solved) | ✅ Xong |
| 2 – Generator (cây bao trùm, shuffle, seed) | ✅ Xong |
| 3 – UI tối thiểu (Pygame + replay) | ✅ Xong |
| 4 – Framework solver + benchmark | 🔲 Chưa |
| 5 – Các thuật toán (CSP, SAT, SA, GA, RL) | 🔲 Chưa |
| 6 – So sánh và phân tích | 🔲 Chưa |
