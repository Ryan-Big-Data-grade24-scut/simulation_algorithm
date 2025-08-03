import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Tuple, Optional
from positioning_algorithm.batch_pose_calculator import PoseSolver
from .robot import VirtualRobot


class LaserLengthVisualizer:
    def __init__(self, robot: VirtualRobot, laser_lengths: Optional[List[float]] = None, 
                 solver: Optional[PoseSolver] = None, enable_solver: bool = False,
                 m: float = 20, n: float = 10):
        """
        激光长度可视化器
        
        Args:
            robot: 虚拟机器人实例
            laser_lengths: 指定的激光长度列表，如果为None则使用扫描结果
            solver: 位姿求解器，当enable_solver为True时使用
            enable_solver: 是否启用求解器显示解算结果
            m, n: 场地尺寸
        """
        self.robot = robot
        self.laser_lengths = laser_lengths
        self.solver = solver
        self.enable_solver = enable_solver
        self.m = m
        self.n = n

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # 初始化图形元素
        self.robot_artist = None
        self.laser_artists = []
        self.hit_markers = []
        self.arrow_artists = []
        self.solution_artists = []
        self.solution_lasers = []
        
        # 绘制场地边界
        self._draw_boundaries()
        
        # 创建控制面板
        self._create_sliders()

        # 初始绘制（包括求解）
        initial_laser_data = self._get_laser_data()
        initial_solutions = self._solve_if_enabled(initial_laser_data)
        self._update_display(
            true_pose=(self.robot.x, self.robot.y, self.robot.phi),
            laser_data=initial_laser_data,
            solutions=initial_solutions
        )

    def _create_sliders(self):
        """创建X/Y/Phi滑动条"""
        ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_y = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        self.slider_x = Slider(ax_x, 'X', 0, self.m, valinit=self.robot.x)
        self.slider_y = Slider(ax_y, 'Y', 0, self.n, valinit=self.robot.y)
        self.slider_phi = Slider(ax_phi, 'Phi', -np.pi, np.pi, valinit=self.robot.phi)
        
        # 绑定统一回调
        for slider in [self.slider_x, self.slider_y, self.slider_phi]:
            slider.on_changed(self._on_slider_changed)

    def _get_laser_data(self):
        """获取激光数据"""
        if self.laser_lengths is None:
            distances, angles = self.robot.scan()
        else:
            # 使用指定长度，但需要计算角度
            _, angles = self.robot.scan()
            distances = np.array(self.laser_lengths)
        return distances, angles
    
    def _solve_if_enabled(self, laser_data):
        """如果启用求解器则进行求解"""
        if self.enable_solver and self.solver is not None:
            distances, _ = laser_data
            
            # 检查求解器类型
            if hasattr(self.solver, 'perpendicular_solver'):
                # 双求解器
                try:
                    solutions = self.solver.solve(distances, self.robot.phi)
                    return solutions  # 已经是 [(x, y, phi), ...] 格式
                except Exception as e:
                    print(f"双求解器解算失败: {e}")
                    return None
            elif hasattr(self.solver, 'localize'):
                # 垂直距离定位器
                try:
                    solutions = self.solver.localize(distances, self.robot.phi)
                    return solutions  # 已经是 [(x, y, phi), ...] 格式
                except Exception as e:
                    print(f"垂直距离定位器解算失败: {e}")
                    return None
            else:
                # 原来的角度位姿计算器
                try:
                    solutions = self.solver.solve(distances, self.robot.phi)
                    return [(sol[0], sol[1], self.robot.phi) for sol in solutions]
                except Exception as e:
                    print(f"角度位姿计算器解算失败: {e}")
                    return None
        return None

    def _on_slider_changed(self, val):
        """滑动条回调：更新机器人位姿和显示"""
        # 更新机器人位姿
        self.robot.update_pose(
            x=self.slider_x.val,
            y=self.slider_y.val,
            phi=self.slider_phi.val
        )
        
        # 获取激光数据
        laser_data = self._get_laser_data()
        
        # 求解（如果启用）
        solutions = self._solve_if_enabled(laser_data)
        
        # 更新显示
        self._update_display(
            true_pose=(self.robot.x, self.robot.y, self.robot.phi),
            laser_data=laser_data,
            solutions=solutions
        )

    def _update_display(self, true_pose=None, laser_data=None, solutions=None):
        """更新显示（激光束和可选的解算结果）"""
        # 清除所有旧图形
        for artist in [self.robot_artist] + self.laser_artists + self.hit_markers + \
                     self.arrow_artists + self.solution_artists + self.solution_lasers:
            if artist is not None:
                artist.remove()

        # 初始化图形元素
        self.robot_artist = None
        self.laser_artists = []
        self.hit_markers = []
        self.arrow_artists = []
        self.solution_artists = []
        self.solution_lasers = []

        # 绘制机器人
        self._draw_robot(true_pose)

        # 绘制激光束
        if laser_data:
            self._draw_lasers(true_pose, laser_data)
        
        # 绘制解算结果（如果有）
        if solutions:
            self._draw_solutions(true_pose, solutions, laser_data)

        self.fig.canvas.draw_idle()

    def _draw_robot(self, pose):
        """绘制机器人本体（红色）"""
        x, y, phi = pose
        self.robot_artist = self.ax.plot(x, y, 'ro', markersize=10)[0]
        
        # 绘制朝向箭头
        arrow = self.ax.arrow(x, y, 
                            0.8*np.cos(phi), 0.8*np.sin(phi),
                            head_width=0.3, fc='r', ec='r')
        self.arrow_artists.append(arrow)

    def _draw_lasers(self, pose, laser_data):
        """绘制激光束（蓝色实线+碰撞点）"""
        distances, angles = laser_data
        self.laser_artists = []
        self.hit_markers = []
        
        for i, ((rel_r, rel_angle), laser_angle) in enumerate(self.robot.laser_configs):
            # 计算激光头位置
            laser_x = pose[0] + rel_r * np.cos(pose[2] + rel_angle)
            laser_y = pose[1] + rel_r * np.sin(pose[2] + rel_angle)
            
            # 获取对应激光数据
            dist = distances[i]
            angle = angles[i]
            
            # 绘制激光束
            end_x = laser_x + dist * np.cos(angle)
            end_y = laser_y + dist * np.sin(angle)
            line = self.ax.plot([laser_x, end_x], [laser_y, end_y], 'b-', linewidth=2)[0]
            self.laser_artists.append(line)
            
            # 绘制碰撞点（红色圆点）
            marker = self.ax.plot(end_x, end_y, 'ro', markersize=6)[0]
            self.hit_markers.append(marker)

    def _draw_solutions(self, pose, solutions, laser_data):
        """绘制解算结果（绿色+模拟激光束）"""
        self.solution_artists = []
        self.solution_lasers = []
        for (x, y, sol_phi) in solutions:
            sol_x = x
            sol_y = y
            
            # 解算位置点
            dot = self.ax.plot(sol_x, sol_y, 'go', markersize=8, alpha=0.7)[0]
            self.solution_artists.append(dot)
            
            # 解算朝向箭头
            arrow = self.ax.arrow(sol_x, sol_y,
                               0.5*np.cos(sol_phi), 0.5*np.sin(sol_phi),
                               head_width=0.2, fc='g', ec='g', alpha=0.7)
            self.arrow_artists.append(arrow)
            
            # 绘制解算激光束（浅绿色虚线）
            if laser_data:
                distances, _ = laser_data
                for i, ((rel_r, rel_angle), laser_angle) in enumerate(self.robot.laser_configs):
                    laser_x = sol_x + rel_r * np.cos(sol_phi + rel_angle)
                    laser_y = sol_y + rel_r * np.sin(sol_phi + rel_angle)
                    
                    dist = distances[i]
                    angle = sol_phi + laser_angle
                    
                    line = self.ax.plot(
                        [laser_x, laser_x + dist*np.cos(angle)],
                        [laser_y, laser_y + dist*np.sin(angle)],
                        'g-', alpha=0.5
                    )[0]
                    self.solution_lasers.append(line)

    def _draw_boundaries(self):
        """绘制场地边界"""
        for (p1, p2) in self.robot.boundary_lines:
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
        
        self.ax.set_xlim(-self.m/5, self.m*6/5)
        self.ax.set_ylim(-self.n/5, self.n*6/5)
        self.ax.grid(True)
        self.ax.set_aspect('equal')


class RobotVisualizer:
    def __init__(self, robot: VirtualRobot, solver: PoseSolver, m: float = 20, n: float = 10):
        """
        初始化包含：
        - 创建matplotlib图形
        - 设置滑动条（x/y/phi）
        - 绑定回调函数
        """
        self.robot = robot
        self.solver = solver
        self.m = m
        self.n = n

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # 初始化图形元素
        self.robot_artist = None
        self.laser_artists = []
        self.solution_artists = []
        self.solution_lasers = []  # 解算结果的激光束
        self.hit_markers = []     # 激光碰撞点标记
        self.arrow_artists = []    # 所有箭头对象
        
        # 绘制场地边界
        self._draw_boundaries()
        
        # 创建控制面板
        self._create_sliders()

        # 初始绘制
        self._update_display(true_pose=(self.robot.x, self.robot.y, self.robot.phi))            

    def _create_sliders(self):
        """创建X/Y/Phi滑动条"""
        ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_y = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        self.slider_x = Slider(ax_x, 'X', 0, self.m, valinit=self.robot.x)
        self.slider_y = Slider(ax_y, 'Y', 0, self.n, valinit=self.robot.y)
        self.slider_phi = Slider(ax_phi, 'Phi', -np.pi, np.pi, valinit=self.robot.phi)
        
        # 绑定统一回调
        for slider in [self.slider_x, self.slider_y, self.slider_phi]:
            slider.on_changed(self._on_slider_changed)

    def _on_slider_changed(self, val):
        """
        完全对应图片中的回调流程：
        1. 更新机器人位姿
        2. 获取扫描结果
        3. 调用算法模块
        4. 获取解算结果
        5. 重新绘制
        """
        # 1. 更新机器人位姿
        self.robot.update_pose(
            x=self.slider_x.val,
            y=self.slider_y.val,
            phi=self.slider_phi.val
        )
        
        # 2. 获取扫描结果
        distances, angles = self.robot.scan()
        
        # 3. 调用算法模块
        if hasattr(self.solver, 'perpendicular_solver'):
            # 双求解器
            solutions = self.solver.solve(distances, self.robot.phi)
        elif hasattr(self.solver, 'localize'):
            # 垂直距离定位器
            solutions = self.solver.localize(distances, self.robot.phi)
        else:
            # 原来的角度位姿计算器
            solutions = self.solver.solve(distances, self.robot.phi)
            solutions = [(sol[0], sol[1], self.robot.phi) for sol in solutions]  # (x_range, y_range, phi)
        
        # 4-5. 更新显示
        self._update_display(
            true_pose=(self.robot.x, self.robot.y, self.robot.phi),
            solutions=solutions,
            laser_data=(distances, angles)
        )

    def _update_display(self, true_pose = None, solutions = None, laser_data = None):
        """更新显示（完全匹配图片中的回调流程）"""
        # 清除所有旧图形
        for artist in [self.robot_artist] + self.laser_artists + \
                     self.solution_artists + self.solution_lasers + \
                     self.hit_markers + self.arrow_artists:
            if artist is not None:
                artist.remove()

        # 初始化图形元素
        self.robot_artist = None
        self.laser_artists = []
        self.solution_artists = []
        self.solution_lasers = []  # 解算结果的激光束
        self.hit_markers = []     # 激光碰撞点标记
        self.arrow_artists = []    # 所有箭头对象 

        # 绘制机器人核心元素
        self._draw_robot(true_pose)

        # 绘制真实激光束和碰撞点
        if laser_data:
            self._draw_real_lasers(true_pose, laser_data)
        
        # 绘制解算结果
        if solutions:
            self._draw_solutions(true_pose, solutions, laser_data)

        self.fig.canvas.draw_idle()

    def _draw_robot(self, pose):
        """绘制机器人本体（红色）"""
        x, y, phi = pose
        self.robot_artist = self.ax.plot(x, y, 'ro', markersize=10)[0]
        
        # 绘制朝向箭头（添加到箭头列表）
        arrow = self.ax.arrow(x, y, 
                            0.8*np.cos(phi), 0.8*np.sin(phi),
                            head_width=0.3, fc='r', ec='r')
        self.arrow_artists.append(arrow)

    def _draw_real_lasers(self, pose, laser_data):
        """绘制真实激光束（蓝色虚线+碰撞点）"""
        distances, angles = laser_data
        self.laser_artists = []
        self.hit_markers = []
        for (rel_r, rel_angle), laser_angle in self.robot.laser_configs:
            # 计算激光头位置
            laser_x = pose[0] + rel_r * np.cos(pose[2] + rel_angle)
            laser_y = pose[1] + rel_r * np.sin(pose[2] + rel_angle)
            
            # 获取对应激光数据
            idx = self.robot.laser_configs.index(((rel_r, rel_angle), laser_angle))
            dist = distances[idx]
            angle = angles[idx]
            
            # 绘制激光束
            end_x = laser_x + dist * np.cos(angle)
            end_y = laser_y + dist * np.sin(angle)
            line = self.ax.plot([laser_x, end_x], [laser_y, end_y], 'b--', alpha=0.7)[0]
            self.laser_artists.append(line)
            
            # 绘制碰撞点（红色×）
            if dist < 20:  # 有效碰撞
                marker = self.ax.plot(end_x, end_y, 'rx', markersize=8)[0]
                self.hit_markers.append(marker)

    def _draw_solutions(self, pose, solutions, laser_data):
        """绘制解算结果（绿色+模拟激光束）"""
        self.solution_artists = []
        self.solution_lasers = []
        for (x, y, sol_phi) in solutions:
            sol_x = x
            sol_y = y
            
            # 解算位置点
            dot = self.ax.plot(sol_x, sol_y, 'go', markersize=8, alpha=0.7)[0]
            self.solution_artists.append(dot)
            
            # 解算朝向箭头
            arrow = self.ax.arrow(sol_x, sol_y,
                               0.5*np.cos(sol_phi), 0.5*np.sin(sol_phi),
                               head_width=0.2, fc='g', ec='g', alpha=0.7)
            self.arrow_artists.append(arrow)
            
            # 绘制解算激光束（浅绿色虚线）
            if laser_data:
                distances, _ = laser_data
                for (rel_r, rel_angle), laser_angle in self.robot.laser_configs:
                    laser_x = sol_x + rel_r * np.cos(sol_phi + rel_angle)
                    laser_y = sol_y + rel_r * np.sin(sol_phi + rel_angle)
                    
                    idx = self.robot.laser_configs.index(((rel_r, rel_angle), laser_angle))
                    dist = distances[idx]
                    angle = sol_phi + laser_angle
                    
                    line = self.ax.plot(
                        [laser_x, laser_x + dist*np.cos(angle)],
                        [laser_y, laser_y + dist*np.sin(angle)],
                        'g-', alpha=0.5
                    )[0]
                    self.solution_lasers.append(line)

    def _draw_boundaries(self):
        """绘制场地边界（根据VirtualRobot的默认设置）"""
        for (p1, p2) in self.robot.boundary_lines:
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
        
        self.ax.set_xlim(-self.m/5, self.m*6/5)
        self.ax.set_ylim(-self.n/5, self.n*6/5)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
