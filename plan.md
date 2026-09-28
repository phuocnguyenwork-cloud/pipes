# Pipes (FreeNet / Net / NetWalk)

## Kế hoạch xây dựng môi trường trò chơi có thể tích hợp các thuật toán giải

---

## 1. Nguyên tắc thiết kế

Cần tách **lõi trò chơi (engine)** khỏi **giao diện** và **thuật toán**, để thuật toán nào cũng có thể chạy mà không cần UI:

```
[Generator] → [Puzzle/State] ← [Engine (luật)] ← [Solver (thuật toán)]
                                     ↑
                       [Env wrapper (Gym)]  [Renderer (UI)]
```

- **Engine** là mã thuần (pure logic), không phụ thuộc UI, chạy nhanh và dễ test.
- **Solver** chỉ giao tiếp qua một interface chung.
- **Renderer** chỉ đọc state để vẽ, kể cả khi phát lại lời giải của solver.

---

## 2. Chọn công nghệ

| Thành phần | Đề xuất | Lý do |
|---|---|---|
| Engine + Solver | **Python** (NumPy) | Dễ thử nghiệm thuật toán, có sẵn OR-Tools, PySAT, PyTorch |
| Giao diện | **Pygame** hoặc web (HTML/JS) | Pygame gọn; web dễ chia sẻ |
| Tối ưu tốc độ (nếu cần) | Numba hoặc Rust/C++ | Khi chạy simulated annealing / RL với hàng triệu bước |

---

## 3. Biểu diễn dữ liệu (quyết định quan trọng nhất)

Mỗi ô là một **bitmask 4 bit**: N=1, E=2, S=4, W=8.

- Xoay 90° chiều kim đồng hồ là dịch bit vòng: `((m << 1) | (m >> 3)) & 0xF`.
- Loại ô suy ra từ số bit: 1 bit là đầu cụt, 2 bit là thẳng hoặc cong, 3 bit là chữ T, 4 bit là chữ thập.
- Bảng là mảng H×W kiểu `uint8`, sao chép rẻ, băm được và so sánh nhanh, rất hợp với backtracking, BFS/A* và làm input cho mạng nơ-ron.
- Lưu **hướng gốc** của từng ô (cố định) và **số lần xoay hiện tại** (0–3). Khi đó không gian trạng thái chính là vector số lần xoay.

```python
@dataclass(frozen=True)
class Puzzle:
    h: int; w: int
    base: np.ndarray      # base mask of each cell
    source: tuple[int,int]
    wrap: bool = False
    allow_cycles: bool = False
```

---

## 4. Các module cần xây

### 4.1 Engine (`core/`)

- `rotate(state, r, c, k=1)`: xoay một ô.
- `neighbors_connected(state)`: kiểm tra hai ô kề có cổng hướng vào nhau không.
- `flood_fill(state, source)`: BFS tìm các ô đã nối với nguồn.
- `is_solved(state)`: kiểm tra mọi ô nối nguồn, không đầu hở, không vòng (nếu bật).
- `count_violations(state)`: số cổng hở, số ô chưa nối, số chu trình. Hàm này quan trọng cho heuristic và reward.

### 4.2 Generator (`generator/`)

1. Sinh **cây bao trùm ngẫu nhiên** trên lưới (randomized DFS/Prim/Kruskal), với nguồn là gốc.
2. Từ cây, suy ra mask của mỗi ô theo các cạnh nối với nó.
3. **Xáo trộn**: xoay mỗi ô ngẫu nhiên 0–3 lần.
4. Tùy chọn: kiểm tra **tính duy nhất của nghiệm** bằng solver đếm nghiệm, chấp nhận hoặc sinh lại. Có tham số độ khó (kích thước, tỷ lệ ô T, wrap, seed).
5. Luôn nhận `seed` để tái lập kết quả.

### 4.3 Môi trường (`env/`)

Bọc theo kiểu Gymnasium để dùng được cho cả RL và tìm kiếm:

- `reset(seed)`, `step(action)`, `render()`.
- **Action** là `(r, c)` xoay ô đó 90°. Có thể mở rộng thành `(r, c, k)`.
- **Observation** là mảng one-hot H×W×4 hoặc mask, kèm số lần xoay.
- **Reward** là thay đổi của `-violations` cộng thưởng lớn khi giải xong.
- Cung cấp `get_state()`, `set_state()`, `clone()` cho các solver dạng tìm kiếm.

### 4.4 Interface Solver (`solvers/base.py`)

```python
class Solver(ABC):
    name: str
    @abstractmethod
    def solve(self, puzzle: Puzzle, budget: Budget) -> SolveResult: ...

@dataclass
class SolveResult:
    solved: bool
    rotations: np.ndarray      # rotation count per cell
    moves: list[tuple]         # move sequence (for replay)
    stats: dict                # nodes, time, iterations...
```

Thêm solver mới chỉ cần kế thừa lớp này và đăng ký vào registry (`@register("sa")`). Không phải sửa engine hay UI.

### 4.5 Benchmark (`bench/`)

- Chạy N solver × M puzzle (nhiều kích thước, seed), ghi CSV/JSON.
- Chỉ số: tỷ lệ giải được, thời gian, số node/iteration, số lần xoay, bộ nhớ.
- Có timeout/budget thống nhất để so sánh công bằng.
- Vẽ biểu đồ bằng matplotlib.

### 4.6 Giao diện (`ui/`)

- Chơi thủ công: bấm để xoay, tô sáng ô đã nối nguồn, khóa ô.
- Chế độ **replay**: chọn solver, chạy và xem từng bước (có điều chỉnh tốc độ).
- Có thể hiển thị metric trực tiếp (violations giảm theo thời gian).

---

## 5. Cấu trúc thư mục

```
pipes/
├── core/         # puzzle.py, engine.py, state utils
├── generator/    # tree_gen.py, difficulty.py
├── env/          # gym_env.py
├── solvers/      # base.py, registry.py, backtracking.py, ...
├── bench/        # runner.py, metrics.py, plots.py
├── ui/           # pygame_app.py
├── tests/
└── configs/      # yaml cho benchmark
```

---

## 6. Lộ trình theo giai đoạn

| Giai đoạn | Nội dung |
|---|---|
| **1 – Nền tảng** | Biểu diễn bitmask, xoay, flood fill, `is_solved`, `count_violations`. Viết unit test cho từng loại ô và trường hợp wrap. |
| **2 – Generator** | Sinh cây ngẫu nhiên, xáo trộn, seed hóa. Kiểm tra mọi puzzle sinh ra đều giải được (giữ lại nghiệm gốc để đối chiếu). |
| **3 – Giao diện tối thiểu** | Chơi thủ công và replay lời giải. Có giao diện sớm giúp phát hiện lỗi luật rất nhanh. |
| **4 – Framework solver + benchmark** | Interface, registry, runner, đo thời gian. Cài **một solver baseline** (brute force cho lưới nhỏ) để kiểm chứng pipeline chạy thông suốt. |
| **5 – Các thuật toán** | Từ dễ đến khó, xem bảng bên dưới. |
| **6 – So sánh và phân tích** | Chạy benchmark quy mô lớn, vẽ biểu đồ theo kích thước lưới, độ khó, wrap hay không. |

### Giai đoạn 5 – Các nhóm thuật toán

| Nhóm | Thuật toán | Ghi chú |
|---|---|---|
| **Suy luận / CSP** | Backtracking + lan truyền ràng buộc (như Sudoku), AC-3, forward checking | Nền tảng, thường rất mạnh cho bài này |
| **Ràng buộc chuẩn** | SAT (PySAT), CP-SAT (OR-Tools), ILP | Mã hóa mỗi ô có 4 biến hướng xoay; ràng buộc ở cạnh giữa hai ô kề |
| **Tìm kiếm cục bộ** | Hill climbing, simulated annealing, tabu search | Hàm mục tiêu là violations |
| **Tiến hóa** | Genetic algorithm (nhiễm sắc thể là vector xoay) | Cần crossover theo vùng |
| **Tìm kiếm đồ thị** | BFS / A* / IDA* trên không gian trạng thái | Chỉ thực tế với lưới nhỏ |
| **Học tăng cường** | DQN / PPO qua Gym env, có thể dùng CNN | Thử nghiệm nghiên cứu, khó hơn nhiều |

---

## 7. Các quyết định nên chốt sớm

1. **Wrap hay không** Không.
2. **Có cho phép vòng lặp không** Không.
3. **Có ô chữ thập không.** Không.
4. **Tái lập kết quả**: mọi thành phần ngẫu nhiên phải nhận seed.

---
<!-- 
## 8. Kiểm thử

- **Unit test:** xoay 4 lần trở về ban đầu, flood fill, phát hiện vòng, biên bảng.
- **Property test:** mọi puzzle từ generator phải có nghiệm và solver baseline giải được.
- **Test hồi quy:** lưu vài puzzle cố định cùng nghiệm đúng.
- **Test hiệu năng:** đo số `step()` mỗi giây, vì đây là nút thắt cho SA và RL. -->

---

## 9. Gợi ý bắt đầu

Bắt đầu bằng **Giai đoạn 1 + 2 + 3** cho lưới 5×5 không wrap, cho phép cây, chưa có chữ thập. chưa viết solver.
