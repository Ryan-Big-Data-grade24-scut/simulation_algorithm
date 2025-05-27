import numpy as np
from scipy.optimize import fsolve

def calculate_xform(v1, v2, v3, m, n):
    # 解算xform的位置和朝向，基于情况一和情况二
    # 尝试所有向量组合和情况，返回第一个找到的解
    vectors = [v1, v2, v3]
    pairs = [(0, 1), (0, 2), (1, 2)]  # 所有向量对索引
    
    for i, j in pairs:
        # 情况一：向量i和j在同一边，假设在底边y=0
        # 定义方程组
        def equations_case1(params):
            phi, x0, y0 = params
            ri, thetai = vectors[i]
            rj, thetaj = vectors[j]
            rk, thetak = vectors[3 - i - j]  # 第三个向量
            
            # 向量i和j的终点在底边y=0
            eq1 = y0 + ri * np.sin(phi + thetai)
            eq2 = y0 + rj * np.sin(phi + thetaj)
            # 第三个向量假设在顶边y=n
            eq3 = y0 + rk * np.sin(phi + thetak) - n
            return [eq1, eq2, eq3]
        
        # 初始猜测：phi=0，x0和y0在中心
        initial_guess = [0, m/2, n/2]
        sol, info, ier, msg = fsolve(equations_case1, initial_guess, full_output=True)
        if ier == 1:
            phi, x0, y0 = sol
            # 检查边界条件
            if 0 <= x0 <= m and 0 <= y0 <= n:
                # 检查向量i和j的x坐标是否在0到m之间
                xi = x0 + vectors[i][0] * np.cos(phi + vectors[i][1])
                xj = x0 + vectors[j][0] * np.cos(phi + vectors[j][1])
                if 0 <= xi <= m and 0 <= xj <= m:
                    # 检查第三个向量是否在顶边
                    yk = y0 + vectors[3-i-j][0] * np.sin(phi + vectors[3-i-j][1])
                    if np.isclose(yk, n, atol=1e-3):
                        xk = x0 + vectors[3-i-j][0] * np.cos(phi + vectors[3-i-j][1])
                        if 0 <= xk <= m:
                            return (x0, y0, phi % (2*np.pi))
        
        # 情况二：向量i和j在相邻两边，如左边和底边
        def equations_case2(params):
            phi, x0, y0 = params
            ri, thetai = vectors[i]
            rj, thetaj = vectors[j]
            rk, thetak = vectors[3 - i - j]
            
            # 假设向量i在左边x=0，向量j在底边y=0
            eq1 = x0 + ri * np.cos(phi + thetai)  # 应等于0
            eq2 = y0 + rj * np.sin(phi + thetaj)  # 应等于0
            # 第三个向量假设在右边x=m
            eq3 = x0 + rk * np.cos(phi + thetak) - m
            return [eq1, eq2, eq3]
        
        sol, info, ier, msg = fsolve(equations_case2, initial_guess, full_output=True)
        if ier == 1:
            phi, x0, y0 = sol
            if 0 <= x0 <= m and 0 <= y0 <= n:
                # 检查向量i的y坐标是否在0到n之间
                yi = y0 + vectors[i][0] * np.sin(phi + vectors[i][1])
                # 检查向量j的x坐标是否在0到m之间
                xj = x0 + vectors[j][0] * np.cos(phi + vectors[j][1])
                if 0 <= yi <= n and 0 <= xj <= m:
                    # 检查第三个向量是否在右边
                    yk = y0 + vectors[3-i-j][0] * np.sin(phi + vectors[3-i-j][1])
                    if 0 <= yk <= n:
                        return (x0, y0, phi % (2*np.pi))
    
    # 如果未找到解，尝试其他边组合或抛出错误
    raise ValueError("No solution found under given constraints.")


if __name__ == '__main__':

    # 示例使用
    v1 = (2.0, np.radians(225))  # r, theta（相对于xform的角度）
    v2 = (3.0, np.radians(315))
    v3 = (1.5, np.radians(45))
    m, n = 10, 8

    try:
        x, y, phi = calculate_xform(v1, v2, v3, m, n)
        print(f"Solution found: x={x:.2f}, y={y:.2f}, phi={np.degrees(phi):.2f}°")
    except ValueError as e:
        print(e)


