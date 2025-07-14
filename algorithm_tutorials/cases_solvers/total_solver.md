### 情况2和情况3的解决方案设计

#### 一、情况2：三角形一边在矩形底边，对角在对边

**几何特征**：
- 三角形的一条边（P0P1）完全贴合在矩形底边（y=0）
- 对角顶点P2贴合在矩形顶边（y=n）
- 另外两个顶点P0和P1在底边上自由移动

**判断结构**：
```python
for base_idx in range(3):  # 遍历三条边作为基边
    if 基边可以水平对齐底边:
        for rect_edge in ['bottom']:  # 固定底边
            for opp_edge in ['top']:   # 固定对边
                solve_case2()
```

**核心公式推导**：
1. 基边约束（P0P1在y=0）：
   \[
   y_O + t_i \sin(\phi+\theta_i) = 0 \quad (i=0,1)
   \]
   ⇒ 需要满足 \( \sin(\phi+\theta_0) = \sin(\phi+\theta_1) \)

2. 对角约束（P2在y=n）：
   \[
   y_O + t_2 \sin(\phi+\theta_2) = n
   \]
   ⇒ 联立得：
   \[
   t_2 \sin(\phi+\theta_2) - t_0 \sin(\phi+\theta_0) = n
   \]

**解方程方法**：
1. 将方程展开为 \( A\sin\phi + B\cos\phi = n \) 的形式
2. 使用辅助角公式求解 \(\phi\)

**验证流程**：
1. 检查P0和P1的y坐标是否为0
2. 检查P2的y坐标是否为n
3. 检查所有顶点x坐标在[0,m]范围内

---

#### 二、情况3：三个顶点分别在三条不同边上

**几何特征**：
- 三个顶点P0、P1、P2分别位于：
  - 两条邻边 + 一条对边（如左+右+顶）
  - 或三条非平行边（如左+右+底）

**判断结构**：
```python
for edge_comb in 所有三条边的有效组合:
    # 例如 ['left', 'right', 'top'] 或 ['left', 'bottom', 'top']
    assign_vertices_to_edges()
    solve_case3()
```

**核心公式推导**：
对于顶点分配方案：
- P0在左边（x=0）：\( x_O + t_0 \cos(\phi+\theta_0) = 0 \)
- P1在右边（x=m）：\( x_O + t_1 \cos(\phi+\theta_1) = m \)
- P2在顶边（y=n）：\( y_O + t_2 \sin(\phi+\theta_2) = n \)

联立方程：
1. 从P0和P1的方程消去xO：
   \[
   t_1 \cos(\phi+\theta_1) - t_0 \cos(\phi+\theta_0) = m
   \]
2. 与P2方程联立求解

**解方程方法**：
1. 将三角方程展开为多项式形式
2. 可能需要数值解法（如牛顿迭代法）

**验证流程**：
1. 检查每个顶点是否在指定边
2. 检查三角形完整性（边长不变）
3. 检查第四个边不冲突（如不要求顶点在第四条边）

---

### 三、代码结构设计（伪代码）

#### 情况2求解器
```python
class Case2Solver:
    def solve(self):
        solutions = []
        for base_idx in range(3):
            p1, p2 = base_idx, (base_idx+1)%3
            opp_idx = (base_idx+2)%3
            
            # 只处理水平基边+顶边对边的情况
            if self._can_be_horizontal(p1, p2):
                phi_candidates = self._solve_case2_phi(p1, p2, opp_idx)
                for phi in phi_candidates:
                    xO, yO = self._compute_case2_position(phi, p1, opp_idx)
                    if self._verify_case2(xO, yO, phi, p1, p2, opp_idx):
                        solutions.append((xO, yO, phi))
        return solutions

    def _solve_case2_phi(self, p1, p2, opp_idx):
        """解情况2的角度方程"""
        # 展开为 A*sin(phi) + B*cos(phi) = n 的形式
        A = (self.t[opp_idx]*sin(self.theta[opp_idx]) - 
             self.t[p1]*sin(self.theta[p1]))
        B = (self.t[opp_idx]*cos(self.theta[opp_idx]) - 
             self.t[p1]*cos(self.theta[p1]))
        # 使用辅助角公式求解...
        return phis

    def _compute_case2_position(self, phi, p1, opp_idx):
        yO = -self.t[p1] * sin(phi + self.theta[p1])
        xO = ... # 根据P0或P1的x坐标约束计算
        return xO, yO
```

#### 情况3求解器
```python
class Case3Solver:
    def solve(self):
        solutions = []
        for edge_assignment in self._generate_edge_assignments():
            try:
                phi_candidates = self._solve_case3_phi(edge_assignment)
                for phi in phi_candidates:
                    xO, yO = self._compute_case3_position(phi, edge_assignment)
                    if self._verify_case3(xO, yO, phi, edge_assignment):
                        solutions.append((xO, yO, phi))
            except UnsolvableCase:
                continue
        return solutions

    def _solve_case3_phi(self, edge_assignment):
        """解三个顶点约束的联立方程"""
        # 建立方程组
        eq1 = ... # 根据第一个顶点的边约束
        eq2 = ... # 第二个顶点
        # 可能需要数值解法
        return numerical_solve([eq1, eq2, eq3])
```

---

### 四、验证与调试建议

1. **可视化检查**：
   ```python
   def plot_solution(xO, yO, phi):
       # 绘制矩形和三角形位置
       plt.plot([xO + t_i*cos(phi+theta_i) for i in 0,1,2,0], 
                [yO + t_i*sin(phi+theta_i) for i in 0,1,2,0])
   ```

2. **特殊测试用例**：
   - 等腰直角三角形：容易验证位置计算是否正确
   - 退化情况：如零面积三角形

3. **容差处理**：
   ```python
   TOL = 1e-6
   assert abs(calculated_edge - target) < TOL
   ```

---

### 五、数学推导示例（情况2）

**具体方程推导**：
对于P0在(0,0), P1在(x,0), P2在(x1,n)的情况：
1. 从P0: \( y_O = -t_0 \sin(\phi+\theta_0) \)
2. 从P2: \( -t_0 \sin(\phi+\theta_0) + t_2 \sin(\phi+\theta_2) = n \)
3. 展开后：
   \[
   [t_2 \sin\theta_2 - t_0 \sin\theta_0] \cos\phi + 
   [t_2 \cos\theta_2 - t_0 \cos\theta_0] \sin\phi = n
   \]
   可表示为 \( A\cos\phi + B\sin\phi = n \)，其解为：
   \[
   \phi = \arcsin\left(\frac{n}{\sqrt{A^2+B^2}}\right) - \arctan\left(\frac{A}{B}\right)
   \]

---

### 六、边界情况处理

| 情况 | 处理方法 |
|------|----------|
| 无解（如n过大） | 返回空列表 |
| 多解 | 返回所有有效解 |
| 数值不稳定 | 增加牛顿迭代法的迭代次数 |

这个设计通过扩展情况1的架构，针对新的几何约束建立了相应的数学方程和验证流程。实际实现时需要特别注意：
1. 情况3可能需要更复杂的方程求解方法
2. 顶点到边的分配组合需要系统化生成
3. 验证阶段要检查所有几何约束同时满足