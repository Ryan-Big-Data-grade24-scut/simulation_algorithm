import math
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging
import time

class BaseSolverConfig:
    """统一配置类"""
    def __init__(self, 
                 tol: float = 1e-6,
                 log_enabled: bool = True,
                 log_file: str = "solver.log",
                 log_level: str = "INFO"):
        self.tol = tol
        self.log_enabled = log_enabled
        self.log_file = log_file
        self.log_level = getattr(logging, log_level.upper())

class BaseSolver(ABC):
    """抽象基类（满足全部5点需求）"""
    def __init__(self, t: List[float], theta: List[float], 
                 m: float, n: float, config: BaseSolverConfig):
        self._validate_inputs(t, theta, m, n)
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.config = config
        self._setup_logging()
        

    def _setup_logging(self):
        """需求4：统一日志开关和格式"""
        self.logger = logging.getLogger(self.__class__.__name__)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        if self.config.log_enabled:
            handler = logging.FileHandler(
                self.config.log_file, 
                mode='w', 
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.log_level)

    def refresh_log(self):
        """清空日志文件并重新初始化日志系统"""
        if self.config.log_enabled:
            # 关闭并移除所有现有处理器
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
            
            # 重新创建文件（覆盖模式）
            open(self.config.log_file, 'w').close()
            
            # 重新初始化日志
            self._setup_logging()
            self.logger.info(f"Log refreshed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _validate_inputs(self, t, theta, m, n):
        """需求2：多级try的集中校验"""
        assert len(t) == 3, "t must have 3 elements"
        assert len(theta) == 3, "theta must have 3 elements"
        assert m > 0 and n > 0, "m and n must be positive"

    @abstractmethod
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """需求5：统一解格式 ((xmin,xmax), (ymin,ymax), phi)"""
        pass

    def _log_math(self, expr: str, result):
        """需求3：[MATH]级别日志"""
        self.logger.info(f"[MATH] {expr} = {self._format_result(result)}")

    def _log_compare(self, a: float, b: float):
        """需求3：[COMPARE]级别日志"""
        diff = abs(a - b)
        result = diff <= self.config.tol
        self.logger.info(
            f"[COMPARE] {a:.6f} ≈ {b:.6f}? "
            f"diff={diff:.2e}, {'PASS' if result else 'FAIL'}"
        )
        return result

    def _log_validation(self, step: str, is_passed: bool):
        """需求3：[VALIDATION]级别日志"""
        status = "PASSED" if is_passed else "FAILED"
        self.logger.info(f"[VALIDATION] {step.ljust(20)} {status}")

    def _format_result(self, result):
        """统一格式化输出"""
        if isinstance(result, tuple):
            return f"({', '.join(f'{x:.6f}' for x in result)})"
        return f"{result:.6f}"

    def _print_solution(self, solution, index: int):
        """需求3：可视化解决方案"""
        (x_range, y_range, phi) = solution
        self.logger.info(f"\nSolution {index}:")
        self.logger.info(f"  O_x ∈ {x_range[0]:.6f}~{x_range[1]:.6f}")
        self.logger.info(f"  O_y ∈ {y_range[0]:.6f}~{y_range[1]:.6f}")
        self.logger.info(f"  φ = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
        
        # 计算顶点位置示例
        for i in range(3):
            x = (x_range[0] + x_range[1])/2 + self.t[i] * math.cos(phi + self.theta[i])
            y = (y_range[0] + y_range[1])/2 + self.t[i] * math.sin(phi + self.theta[i])
            self.logger.info(f"  P{i} = ({x:.2f}, {y:.2f})")

def main():
    # 测试配置
    config = BaseSolverConfig(
        tol=1e-6,
        log_enabled=True,
        log_file="test_solver.log",
        log_level="DEBUG"
    )
    
    # 测试数据
    t = [1.0, 2.0, 3.0]
    theta = [0.0, math.pi/2, math.pi]
    m = 1.0
    n = 2.0
    
    # 创建测试用的具体求解器类
    class TestSolver(BaseSolver):
        def solve(self):
            # 实现抽象方法，返回测试解
            self._log_math("2 + 2", 4)
            self._log_compare(1.000001, 1.0)
            self._log_validation("Test validation", True)
            
            # 返回测试解
            solution = [
                ((0.0, 1.0), (0.0, 2.0), math.pi/4),
                ((1.0, 2.0), (1.0, 3.0), math.pi/3)
            ]
            
            for i, sol in enumerate(solution):
                self._print_solution(sol, i+1)
            
            return solution
    
    # 测试实例化
    try:
        solver = TestSolver(t, theta, m, n, config)
        print("Solver instantiated successfully.")
        
        # 测试求解
        solutions = solver.solve()
        print(f"Found {len(solutions)} solutions:")
        for i, sol in enumerate(solutions):
            print(f"Solution {i+1}: x_range={sol[0]}, y_range={sol[1]}, phi={sol[2]:.4f}")
            
        # 测试日志功能
        solver.logger.info("This is an info message")
        solver.logger.debug("This is a debug message")
        
        # 测试输入验证
        try:
            bad_solver = TestSolver([1,2], theta, m, n, config)
        except AssertionError as e:
            print(f"Input validation caught error: {e}")
            
        try:
            bad_solver = TestSolver(t, theta, -1, n, config)
        except AssertionError as e:
            print(f"Input validation caught error: {e}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)
        
    print("All tests completed successfully.")

if __name__ == "__main__":
    main()