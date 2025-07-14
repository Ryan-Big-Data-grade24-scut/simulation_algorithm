import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Tuple
from positioning_algorithm.pose_calculator import PoseSolver
from .robot import VirtualRobot


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
        
        # 绘制场地边界
        self._draw_boundaries()
        
        # 创建控制面板
        self._create_sliders()
        
        # 初始绘制
        self._update_display(true_pose=(self.robot.x, self.robot.y, self.robot.phi))        
    


    def _update_display(self, 
                      true_pose=None,
                      solutions=None,
                      laser_data=None):
        """更新所有可视化元素"""
        # 清除旧图形
        for artist in [self.robot_artist] + self.laser_artists + self.solution_artists:
            if artist is not None:
                artist.remove()
        
        # 绘制机器人
        x, y, phi = true_pose
        self.robot_artist = self.ax.plot(x, y, 'ro', markersize=10)[0]
        
        # 绘制朝向箭头
        arrow_len = 0.8
        dx, dy = arrow_len * np.cos(phi), arrow_len * np.sin(phi)
        self.ax.arrow(x, y, dx, dy, head_width=0.3, fc='r', ec='r')

        if laser_data:
            # 绘制激光束
            distances, angles = laser_data
            self.laser_artists = []
            for dist, angle in zip(distances, angles):
                end_x = x + dist * np.cos(angle)
                end_y = y + dist * np.sin(angle)
                line = self.ax.plot([x, end_x], [y, end_y], 'b--', alpha=0.7)[0]
                self.laser_artists.append(line)
        
        # 绘制解算结果（如果存在）
        if solutions:
            self.solution_artists = []
            for (x_range, y_range, sol_phi) in solutions:
                sol_x = sum(x_range) / 2
                sol_y = sum(y_range) / 2
                dot = self.ax.plot(sol_x, sol_y, 'go', markersize=8, alpha=0.7)[0]
                arrow = self.ax.arrow(sol_x, sol_y, 
                                    0.5*np.cos(sol_phi), 0.5*np.sin(sol_phi),
                                    head_width=0.2, fc='g', ec='g', alpha=0.7)
                self.solution_artists.extend([dot, arrow])
        
        self.fig.canvas.draw_idle()

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
        solutions = self.solver.solve(distances, angles)
        
        # 4-5. 更新显示
        self._update_display(
            true_pose=(self.robot.x, self.robot.y, self.robot.phi),
            solutions=solutions,
            laser_data=(distances, angles)
        )

    def _draw_boundaries(self):
        """绘制场地边界（根据VirtualRobot的默认设置）"""
        for (p1, p2) in self.robot.boundary_lines:
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2)
        
        self.ax.set_xlim(0, self.m)
        self.ax.set_ylim(0, self.n)
        self.ax.grid(True)
        self.ax.set_aspect('equal')

    def _draw_robot(self, x: float, y: float, phi: float):
        """绘制机器人（红色）"""
    
    def _draw_solutions(self, solutions: List):
        """绘制解算结果（绿色）"""
    
    def _draw_lasers(self, distances: np.ndarray, angles: np.ndarray):
        """绘制激光线（蓝色虚线）"""