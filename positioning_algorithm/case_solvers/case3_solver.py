import numpy as np
from math import sin, cos, atan2, asin, sqrt, pi

class Case3Solver:
    def __init__(self, t, theta, m, n):
        """
        t: 三角形顶点到中心的距离列表 [t0, t1, t2]
        theta: 三角形顶点角度列表 [θ0, θ1, θ2]（弧度）
        m: 矩形宽度
        n: 矩形高度
        """
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.TOL = 1e-6
    
    def solve(self):
        """处理所有情况3的组合"""
        solutions = []
        
        # 所有可能的边组合（顶点索引分配）
        edge_combinations = [
            # (p0_edge, p1_edge, p2_edge)
            ('left', 'right', 'top'),
            ('left', 'right', 'bottom'),
            ('left', 'top', 'bottom'),
            ('right', 'top', 'bottom')
        ]
        
        for p0_edge, p1_edge, p2_edge in edge_combinations:
            # 尝试所有顶点排列组合（6种排列）
            for p0, p1, p2 in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                phi_candidates = self._solve_edge_combo(
                    p0, p1, p2, p0_edge, p1_edge, p2_edge)
                
                for phi in phi_candidates:
                    xO, yO = self._compute_position(phi, p0, p1, p2, 
                                                  p0_edge, p1_edge, p2_edge)
                    if self._verify_solution(xO, yO, phi, p0_edge, p1_edge, p2_edge):
                        solutions.append((xO, yO, phi))
        
        return solutions
    
    def _solve_edge_combo(self, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """解特定边组合的方程"""
        if {p0_edge, p1_edge} == {'left', 'right'}:
            # 处理left+right+top/bottom的情况
            A = self.t[p1]*cos(self.theta[p1]) - self.t[p0]*cos(self.theta[p0])
            B = -(self.t[p1]*sin(self.theta[p1]) - self.t[p0]*sin(self.theta[p0]))
            C = self.m
        elif {p0_edge, p1_edge} == {'left', 'top'} and p2_edge == 'bottom':
            # 处理left+top+bottom的情况（通过p0和p2）
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p0]*sin(self.theta[p0])
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p0]*cos(self.theta[p0])
            C = self.n
        elif {p0_edge, p1_edge} == {'right', 'top'} and p2_edge == 'bottom':
            # 处理right+top+bottom的情况（通过p0和p2）
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p0]*sin(self.theta[p0])
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p0]*cos(self.theta[p0])
            C = self.n
        else:
            return []
        
        # 解 A*cos(phi) + B*sin(phi) = C
        norm = sqrt(A*A + B*B)
        if abs(C) > norm + self.TOL:
            return []  # 无解
        
        alpha = atan2(A, B)
        phi1 = asin(C / norm) - alpha
        phi2 = pi - asin(C / norm) - alpha
        
        # 返回在[-pi, pi]范围内的解
        solutions = []
        for phi in [phi1, phi2]:
            if phi < -pi:
                phi += 2*pi
            elif phi > pi:
                phi -= 2*pi
            solutions.append(phi)
        
        return solutions
    
    def _compute_position(self, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """计算中心点坐标"""
        xO, yO = 0, 0
        
        # 根据第一个顶点的约束计算中心
        if p0_edge == 'left':
            xO = -self.t[p0] * cos(phi + self.theta[p0])
        elif p0_edge == 'right':
            xO = self.m - self.t[p0] * cos(phi + self.theta[p0])
        elif p0_edge == 'top':
            yO = self.n - self.t[p0] * sin(phi + self.theta[p0])
        elif p0_edge == 'bottom':
            yO = -self.t[p0] * sin(phi + self.theta[p0])
        
        # 根据第二个顶点的约束修正（优先使用x约束）
        if p1_edge == 'left':
            xO = -self.t[p1] * cos(phi + self.theta[p1])
        elif p1_edge == 'right':
            xO = self.m - self.t[p1] * cos(phi + self.theta[p1])
        elif p1_edge in ['top', 'bottom'] and p0_edge in ['left', 'right']:
            # 如果第一个顶点用了x约束，这里用y约束
            if p1_edge == 'top':
                yO = self.n - self.t[p1] * sin(phi + self.theta[p1])
            else:
                yO = -self.t[p1] * sin(phi + self.theta[p1])
        
        return xO, yO
    
    def _verify_solution(self, xO, yO, phi, p0_edge, p1_edge, p2_edge):
        """验证所有顶点是否在指定边且不越界"""
        # 计算三个顶点的坐标
        points = []
        for i in range(3):
            x = xO + self.t[i] * cos(phi + self.theta[i])
            y = yO + self.t[i] * sin(phi + self.theta[i])
            points.append((x, y))
        
        # 检查每个顶点是否在指定边
        edge_checks = {
            'left': lambda x, y: abs(x) < self.TOL and 0 <= y <= self.n,
            'right': lambda x, y: abs(x - self.m) < self.TOL and 0 <= y <= self.n,
            'top': lambda x, y: abs(y - self.n) < self.TOL and 0 <= x <= self.m,
            'bottom': lambda x, y: abs(y) < self.TOL and 0 <= x <= self.m
        }
        
        edges = [p0_edge, p1_edge, p2_edge]
        for (x, y), edge in zip(points, edges):
            if not edge_checks[edge](x, y):
                return False
        
        # 检查第四个边没有顶点
        all_edges = {'left', 'right', 'top', 'bottom'}
        used_edges = set(edges)
        free_edge = list(all_edges - used_edges)[0]
        
        for x, y in points:
            if edge_checks[free_edge](x, y):
                return False  # 有顶点在禁止的边上
        
        # 检查三角形完整性（边长不变）
        # 这里可以添加边长验证（可选）
        
        return True

# 示例用法
if __name__ == "__main__":
    # 示例三角形（等腰直角三角形）
    t = [1, 1, np.sqrt(2)]
    theta = [-np.pi/4, np.pi/4, 3*np.pi/4]
    m, n = 4, 3  # 矩形尺寸
    
    solver = Case3Solver(t, theta, m, n)
    solutions = solver.solve()
    
    print(f"找到 {len(solutions)} 个解:")
    for i, (xO, yO, phi) in enumerate(solutions, 1):
        print(f"解 {i}: 中心({xO:.2f}, {yO:.2f}), 旋转角 {np.degrees(phi):.2f}°")