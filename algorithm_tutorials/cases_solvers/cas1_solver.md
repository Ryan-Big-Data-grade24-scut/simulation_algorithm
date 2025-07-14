根据您的代码和需求，我将完整总结Case1Solver的分类讨论结构、计算公式和验证流程。以下是系统化的总结：

---

### **一、分类讨论结构**
#### **1. 三级嵌套循环**
```python
for base_idx in range(3):           # 遍历三角形的三条边作为基边
    for rect_edge in edges:         # 尝试矩形的四条边（上/下/左/右）
        for adj_edge in adj_edges:   # 尝试当前矩形边的两条邻边
            solve_phi() → validate() → verify()
```

#### **2. 基边类型分支**
| 基边类型       | 基底约束          | 邻边选择          | 方程类型          |
|----------------|-------------------|-------------------|-------------------|
| 水平边（bottom/top） | y = 0 或 y = n    | left/right        | 正弦方程          |
| 垂直边（left/right） | x = 0 或 x = m    | bottom/top        | 余弦方程          |

---

### **二、核心计算公式**
#### **1. 水平基边（bottom/top）**
**约束方程**：
\[
t_1 \sin(\phi+\theta_1) = t_2 \sin(\phi+\theta_2)
\]
**展开形式**：
\[
A \sin\phi + B \cos\phi = 0 \quad \text{其中} \quad 
\begin{cases} 
A = t_1 \cos\theta_1 - t_2 \cos\theta_2 \\
B = t_1 \sin\theta_1 - t_2 \sin\theta_2 
\end{cases}
\]

**解的情况**：
- **退化**：`A=B=0` → 无解
- **A≈0**：直接解 \(\phi = \text{sign}(-B) \cdot \frac{\pi}{2}\)
- **常规**：\(\phi = \text{atan2}(-B, A)\)

#### **2. 垂直基边（left/right）**
**约束方程**：
\[
t_1 \cos(\phi+\theta_1) = t_2 \cos(\phi+\theta_2)
\]
**展开形式**：
\[
A \cos\phi - B \sin\phi = 0 \quad \text{其中} \quad 
\begin{cases} 
A = t_1 \sin\theta_1 - t_2 \sin\theta_2 \\
B = t_1 \cos\theta_1 - t_2 \cos\theta_2 
\end{cases}
\]

**解的情况**：
- **退化**：`A=B=0` → 无解
- **A≈0**：直接解 \(\phi = \begin{cases} 0 & \text{if } B>0 \\ \pi & \text{otherwise} \end{cases}\)
- **常规**：\(\phi = \text{atan2}(B, -A)\)

---

### **三、验证流程**
#### **1. 候选解验证（_validate_phi_candidates）**
```python
for phi in candidates:
    # 1. 基边对齐检查
    if 水平基边:
        检查 P1/P2 的 y 坐标差 < TOL
        检查 P3 的 y 方向位置（上/下）
    else:
        检查 P1/P2 的 x 坐标差 < TOL
        检查 P3 的 x 方向位置（左/右）
    
    # 2. 邻边约束检查
    if adj_edge == 'left':
        检查 P3.x ≤ min(P1.x, P2.x) + TOL
    elif adj_edge == 'right':
        检查 P3.x ≥ max(P1.x, P2.x) - TOL
    # ...（其他邻边类似）
    
    # 3. 平移可行性
    计算所有顶点的 x/y 极值
    检查 max_x - min_x ≤ m + TOL
    检查 max_y - min_y ≤ n + TOL
```

#### **2. 最终验证（_verify_solution）**
```python
# 1. 顶点边界检查
for P in [P0, P1, P2]:
    assert -TOL ≤ P.x ≤ m + TOL
    assert -TOL ≤ P.y ≤ n + TOL

# 2. 基边严格对齐
if rect_edge == 'bottom':
    assert abs(P1.y) < TOL and abs(P2.y) < TOL
elif rect_edge == 'top':
    assert abs(P1.y - n) < TOL and abs(P2.y - n) < TOL
# ...（其他边类似）

# 3. 邻边严格对齐
if adj_edge == 'left':
    assert abs(P3.x) < TOL
elif adj_edge == 'right':
    assert abs(P3.x - m) < TOL
# ...（其他邻边类似）
```

---

### **四、关键数学推导**
#### **1. 水平基边示例（bottom + left）**
**几何约束**：
1. \( y_{P1} = y_{P2} = 0 \)
2. \( x_{P3} = 0 \)

**方程推导**：
\[
\begin{cases}
y_O = -t_1 \sin(\phi+\theta_1) \\
x_O = -t_3 \cos(\phi+\theta_3)
\end{cases}
\]

**验证条件**：
- \( y_{P1} = y_O + t_1 \sin(\phi+\theta_1) = 0 \)
- \( x_{P3} = x_O + t_3 \cos(\phi+\theta_3) = 0 \)

#### **2. 垂直基边示例（left + top）**
**几何约束**：
1. \( x_{P1} = x_{P2} = 0 \)
2. \( y_{P3} = n \)

**方程推导**：
\[
\begin{cases}
x_O = -t_1 \cos(\phi+\theta_1) \\
y_O = n - t_3 \sin(\phi+\theta_3)
\end{cases}
\]

---

### **五、调试辅助设计**
#### **1. 日志分级**
```python
[MATH] 记录所有关键公式计算
[COMPARE] 记录带容差的比较结果
[VALIDATION] 分步骤验证结果
```

#### **2. 可视化验证**
```python
Solution 1:
O = (2.000000, 2.000000)
phi = 1.570796 rad (90.00°)
P0 = (2.00, 4.00)  ← 对齐top边
P1 = (0.00, 4.00)  ← 对齐top边
P2 = (0.00, 2.00)  ← 对齐left边
```

---

### **六、边界情况处理**
| 情况                | 处理方式                     |
|---------------------|-----------------------------|
| 退化方程（A=B=0）   | 立即返回空列表               |
| 数值不稳定（A≈0）   | 直接解析解代替数值计算       |
| 顶点越界            | 严格容差检查（TOL=1e-6）     |

这个设计通过三级循环覆盖所有几何可能性，通过分情况推导确保公式正确性，并通过多级验证保证解的可靠性。实际运行时的详细日志可精准定位任何计算异常。