以下是针对 `Case2Solver` 的完整分类讨论结构、计算公式及结果筛选流程的总结：

---

### **一、分类讨论结构**
#### **1. 外层循环结构**
- **遍历3组基边**（每组基边含2个三角形顶点 + 1个对顶点）：
  ```python
  for base_idx in range(3):  # 基边索引 (0,1,2)
      p1, p2 = base_idx, (base_idx + 1) % 3  # 基边的两个顶点
      opp_idx = (base_idx + 2) % 3           # 对顶点索引
  ```

- **遍历矩形的4条边**（`bottom`/`top`/`left`/`right`）：
  ```python
  for rect_edge in ['bottom', 'top', 'left', 'right']:
  ```

#### **2. 基边与矩形边对齐方式**
- **水平基边**（`bottom`/`top`）：
  - 基边顶点需严格对齐矩形的 `y=0`（`bottom`）或 `y=n`（`top`）。
  - 对顶点需对齐相反边（`top`/`bottom`）。
  
- **垂直基边**（`left`/`right`）：
  - 基边顶点需严格对齐矩形的 `x=0`（`left`）或 `x=m`（`right`）。
  - 对顶点需对齐相反边（`right`/`left`）。

---

### **二、核心计算公式**
#### **1. 求解旋转角 `phi` 的方程**
- **水平基边方程**（`bottom`/`top`）：
  ```
  t_opp*sin(phi + θ_opp) - t_p1*sin(phi + θ_p1) = target
  （target = n 若基边为 bottom，-n 若基边为 top）
  ```
  化简为通用形式：
  ```
  A*sin(phi) + B*cos(phi) = target
  ```
  其中：
  ```
  A = t_opp*cos(θ_opp) - t_p1*cos(θ_p1)
  B = t_opp*sin(θ_opp) - t_p1*sin(θ_p1)
  ```

- **垂直基边方程**（`left`/`right`）：
  ```
  t_opp*cos(phi + θ_opp) - t_p1*cos(phi + θ_p1) = target
  （target = m 若基边为 left，-m 若基边为 right）
  ```
  化简为通用形式：
  ```
  A*cos(phi) - B*sin(phi) = target
  ```
  （`A`、`B` 同水平基边）

#### **2. 方程解的判定**
- **计算范数**：
  ```
  norm = sqrt(A² + B²)
  ```
  - **无解条件**：`norm < |target|` 或 `norm ≈ 0`（退化情况）。
  - **唯一解条件**：`norm ≈ |target|`：
    - 水平基边：`phi = atan2(B, A)`（或调整符号）。
    - 垂直基边：`phi = atan2(B, -A)`（或调整符号）。
  - **双解条件**：`norm > |target|`：
    - 水平基边：
      ```
      phi1 = asin(target/norm) - alpha
      phi2 = π - asin(target/norm) - alpha
      （alpha = atan2(B, A)）
      ```
    - 垂直基边：
      ```
      phi1 = acos(target/norm) - alpha
      phi2 = -acos(target/norm) - alpha
      ```

#### **3. 中心坐标 `(xO, yO)` 计算**
- **水平基边**：
  - `yO = target - t_p1*sin(phi + θ_p1)`（`target = 0` 或 `n`）。
  - `xO` 范围约束：
    ```
    x_min = max(-x_p1, -x_p2)
    x_max = min(m - x_p1, m - x_p2)
    xO = (x_min + x_max) / 2
    ```
    （`x_pi = t_pi*cos(phi + θ_pi)`）

- **垂直基边**：
  - `xO = target - t_p1*cos(phi + θ_p1)`（`target = 0` 或 `m`）。
  - `yO` 范围约束：
    ```
    y_min = max(-y_p1, -y_p2)
    y_max = min(n - y_p1, n - y_p2)
    yO = (y_min + y_max) / 2
    ```
    （`y_pi = t_pi*sin(phi + θ_pi)`）

---

### **三、结果筛选与验证**
#### **1. 初步筛选条件**
- **坐标有效性**：
  - `x_min ≤ x_max`（水平基边）或 `y_min ≤ y_max`（垂直基边）。
  - 排除 `NaN` 值（无解情况）。

#### **2. 严格验证条件**
1. **基边顶点对齐**：
   - 水平基边：`y_p1 ≈ y_p2 ≈ 0`（`bottom`）或 `≈ n`（`top`）。
   - 垂直基边：`x_p1 ≈ x_p2 ≈ 0`（`left`）或 `≈ m`（`right`）。

2. **对顶点对齐**：
   - 水平基边：`y_opp ≈ n`（若基边为 `bottom`）或 `≈ 0`（若基边为 `top`）。
   - 垂直基边：`x_opp ≈ m`（若基边为 `left`）或 `≈ 0`（若基边为 `right`）。

3. **所有顶点在矩形内**：
   - 检查 `0 ≤ x_i ≤ m` 且 `0 ≤ y_i ≤ n`（允许 `±TOL` 误差）。

#### **3. 最终解合并**
- 通过验证的解 `(xO, yO, phi)` 存入 `solutions` 列表。
- 去重处理（若不同基边组合产生相同解）。

---

### **四、流程总结**
```plaintext
1. 遍历所有基边组合 (3组)
   └─ 遍历矩形4条边 (bottom/top/left/right)
      ├─ 根据基边类型（水平/垂直）选择方程形式
      ├─ 计算 phi 候选解（0/1/2个）
      ├─ 对每个 phi 候选：
      │   ├─ 计算 (xO, yO)
      │   ├─ 检查坐标范围有效性
      │   └─ 验证基边、对顶点、全部顶点位置
      └─ 通过验证的解加入结果集
2. 输出所有有效解
```

--- 

### **关键点说明**
- **数值稳定性**：通过 `TOL` 处理浮点误差（默认 `1e-6`）。
- **对称性处理**：双解 `phi1` 和 `phi2` 可能对应镜像对称的三角形布局。
- **几何约束**：基边必须完全贴合矩形边，对顶点必须贴合对边。

此流程确保所有可能的有效解被枚举并验证，适用于任意三角形参数和矩形尺寸。