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
    
    def calculate_pose(self, t: list, theta: list):
        """
        计算位姿，t和theta长度均为3
        """
        self.case1_solver.t = t
        self.case1_solver.theta = theta
        self.case2_solver.t = t
        self.case2_solver.theta = theta
        self.case3_solver.t = t
        self.case3_solver.theta = theta

        results = []
        results.extend(self.case1_solver.solve())
        results.extend(self.case2_solver.solve())
        results.extend(self.case3_solver.solve())
        return results
    
    def filter_solutions(self, solutions: List[Dict], tol: float = None) -> List[Dict]:
        """
        根据机器人当前位姿筛选解
        
        参数:
            solutions: 待筛选的解列表
            tol: 容忍度(可选)，如果为None则使用类初始化时的tol
            
        返回:
            筛选后的解列表
        """
        if tol is None:
            tol = self.tol
            
        filtered = Queue()
        robot_x = self.robot.x
        robot_y = self.robot.y
        robot_phi = self.robot.phi
        #print(solutions)
        #print(1)
        for sol in solutions:
            #print(sol[0], sol[1], sol[2])
            (xmin, xmax), (ymin, ymax), phi = sol[0], sol[1], sol[2]
            
            # 检查x坐标
            x_ok = (robot_x >= xmin - tol) and (robot_x <= xmax + tol)
            # 检查y坐标
            y_ok = (robot_y >= ymin - tol) and (robot_y <= ymax + tol)
            # 检查角度(考虑角度周期性)
            phi_diff = abs((robot_phi - phi + np.pi) % (2*np.pi) - np.pi)
            phi_ok = phi_diff <= tol
            
            if x_ok and y_ok and phi_ok:
                x = xmin
                y = ymin
                if not xmin==xmax:
                    x = robot_x
                if not ymin==ymax:
                    y = robot_y
                filtered.put(((x, y), phi))
        print(filtered)        
        return filtered    
    
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
    """
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
    """
    
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
    
    def get_t_theta_from_sensor_data(self, t_sensor: np.ndarray, r: np.ndarray, delta: np.ndarray, theta: np.ndarray):
        """
        根据传感器数据和激光参数，计算每束激光的 t 和 theta
        返回：t_list, theta_list
        """
        t_list = []
        theta_list = []
        for i in range(len(t_sensor)):
            # 计算碰撞点向量
            x = r[i]*np.cos(delta[i]) + t_sensor[i]*np.cos(theta[i])
            y = r[i]*np.sin(delta[i]) + t_sensor[i]*np.sin(theta[i])
            t_val = np.linalg.norm([x, y])
            theta_val = np.arctan2(y, x)
            if theta_val < 0:
                theta_val += 2 * np.pi
            t_list.append(t_val)
            theta_list.append(theta_val)
        return t_list, theta_list