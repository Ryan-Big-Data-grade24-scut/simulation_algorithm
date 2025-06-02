#./positioning_algorithm/pose_calculator.py
import numpy as np
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple
from .case_solvers.BaseSolver import BaseSolverConfig
from .case_solvers.case1_solver import Case1Solver
from .case_solvers.case2_solver import Case2Solver
from .case_solvers.case3_solver import Case3Solver
from itertools import combinations

class PoseCalculator():
    def __init__(self, robot, m: float, n: float):
        """
        位置计算类（非线程版本）
        
        参数:
            robot: 机器人对象
            m, n: 边界尺寸
        """
        self.robot = robot
        self.m = m
        self.n = n
        self.result_list = []

        # 初始化三个求解器（参数将在calculate_pose中更新）
        self.config1 = BaseSolverConfig(tol=1e-3, log_enabled=True, log_file="logs/solver1.log", log_level="INFO") 
        self.case1_solver = Case1Solver([1,1,1], [0,0,0], m, n, config=self.config1)
        self.config2 = BaseSolverConfig(tol=1e-3, log_enabled=True, log_file="logs/solver2.log", log_level="INFO")    
        self.case2_solver = Case2Solver([1,1,1], [0,0,0], m, n, config=self.config2)
        self.config3 = BaseSolverConfig(tol=1e-3, log_enabled=True, log_file="logs/solver3.log", log_level="INFO")
        self.case3_solver = Case3Solver([1,1,1], [0,0,0], m, n, config=self.config3)

    def get_sensor_data(self) -> Optional[np.ndarray]:
        """从机器人获取传感器数据"""
        last_distances = self.robot.current_distance_result
        if last_distances is None or len(last_distances) < 3:
            return None
        return np.array(last_distances[:3])
    
    def save_results(self, Q: Queue, v):
        """保存计算结果到队列"""
        resl_lst = []
        while not Q.empty():
            P, phi = Q.get()
            result = {
                'P': P,
                'phi': phi,
                'v': v
            }
            #print(result)
            resl_lst.append(result)
        self.result_list = resl_lst
    
    def calculate_pose(self, t: np.ndarray) -> List[Tuple[Queue, np.ndarray]]:
        """核心位姿计算算法，处理所有可能的3向量组合"""
        # 从激光配置中提取参数
        r, delta, theta = self._get_laser_params()
        
        # 计算所有v向量（机器人坐标系）
        all_v = self._calculate_v_vectors(r, delta, theta, t)
        
        # 获取所有可能的3向量组合
        v_combinations = list(combinations(range(len(all_v)), 3))
        
        results = []
        
        for idx_comb in v_combinations:
            # 获取当前组合的3个v向量
            v_comb = [all_v[i] for i in idx_comb]
            v_comb = np.array(v_comb)
            
            # 计算当前组合的t和theta
            t1, theta1 = self.compute_v_to_t_theta(v_comb)
            
            # 更新求解器参数（使用当前组合的激光参数）
            r_comb = [r[i] for i in idx_comb]
            delta_comb = [delta[i] for i in idx_comb]
            theta_comb = [theta[i] for i in idx_comb]
            #self._update_solvers(r_comb, delta_comb, theta_comb)
            
            # 使用三个求解器计算解
            xforms = Queue()
            
            # 情况1：边在底边，对角在邻边
            #"""
            self.case1_solver.t = t1
            self.case1_solver.theta = theta1
            solutions_case1 = self.case1_solver.solve()
            for sol in solutions_case1:
                x1, x2 = sol[0]
                y1, y2 = sol[1]
                phi = sol[2]
                xforms.put((((x1+x2)/2, (y1+y2)/2), phi))  # (P, phi)
            #"""
            
            # 情况2：边在底边，对角在对边
            #"""
            self.case2_solver.t = t1
            self.case2_solver.theta = theta1
            solutions_case2 = self.case2_solver.solve()
            for sol in solutions_case2:
                x1, x2 = sol[0]
                y1, y2 = sol[1]
                phi = sol[2]
                xforms.put((((x1+x2)/2, (y1+y2)/2), phi))  # (P, phi)
            #"""
            
            # 情况3：三个顶点在三条边上
            #"""
            self.case3_solver.t = t1
            self.case3_solver.theta = theta1
            solutions_case3 = self.case3_solver.solve()
            for sol in solutions_case3:
                x1, x2 = sol[0]
                y1, y2 = sol[1]
                phi = sol[2]
                xforms.put((((x1+x2)/2, (y1+y2)/2), phi))  # (P, phi)
            #"""

            results.append((xforms, v_comb))
        return xforms, all_v
    
    def _get_laser_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从机器人配置提取激光参数"""
        r = np.zeros(3)
        delta = np.zeros(3)
        theta = np.zeros(3)
        
        for i in range(3):
            if i >= len(self.robot.laser_configs):
                break
            (rel_r, rel_angle), laser_angle = self.robot.laser_configs[i]
            r[i] = rel_r
            delta[i] = rel_angle
            theta[i] = laser_angle
            
        return r, delta, theta
    
    def _calculate_v_vectors(self, r, delta, theta, t) -> np.ndarray:
        """计算从机器人中心到碰撞点的向量v_i"""
        v = np.zeros((3, 2))
        for i in range(3):
            # 使用向量加法计算碰撞点坐标
            x = r[i]*np.cos(delta[i]) + t[i]*np.cos(theta[i])
            y = r[i]*np.sin(delta[i]) + t[i]*np.sin(theta[i])
            v[i] = np.array([x, y])
        return v
    #"""
    def _update_solvers(self, r, delta, theta):
        #\"""更新三个求解器的参数\"""
        # 计算t和theta参数（相对于参考方向）
        # 这里假设参考方向为x轴正向
        t = [np.linalg.norm(v) for v in r]
        theta_rel = [theta[i] - delta[i] for i in range(3)]
        
        # 更新求解器
        self.case1_solver = Case1Solver(t, theta_rel, self.m, self.n, config=self.config1)
        self.case2_solver = Case2Solver(t, theta_rel, self.m, self.n, config=self.config2)
        self.case3_solver = Case3Solver(t, theta_rel, self.m, self.n, config=self.config3)
    #"""
    def normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [-π, π] 范围内"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def compute_v_to_t_theta(self, v:np.ndarray) :
        t = []
        theta = []
        for i in range(3):
            vi = v[i]
            t.append(np.linalg.norm(vi))
            the = np.arctan2(vi[1], vi[0])
            if the < 0:
                the += 2 * np.pi
            theta.append(the)
        return t, theta
    
    def stop(self):
        """停止线程"""
        self.running = False