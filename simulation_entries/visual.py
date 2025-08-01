import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox
from typing import List, Tuple
from positioning_algorithm.batch_pose_calculator import PoseSolver
from .robot import VirtualRobot


class RobotVisualizer:
    def __init__(self, robot: VirtualRobot, solver: PoseSolver, m: float = 20, n: float = 10):
        """
        初始化包含：
        - 创建matplotlib图形
        - 设置滑动条（x/y/phi）
        - 添加手动输入框和激光扫描模式选择
        - 绑定回调函数
        """
        self.robot = robot
        self.solver = solver
        self.m = m
        self.n = n

        # 创建图形（为控件预留更多空间）
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.4, right=0.75)
        
        # 初始化图形元素
        self.robot_artist = None
        self.laser_artists = []
        self.solution_artists = []
        self.solution_lasers = []  # 解算结果的激光束
        self.hit_markers = []     # 激光碰撞点标记
        self.arrow_artists = []    # 所有箭头对象
        
        # 激光扫描模式和手动数据
        self.scan_mode = 'auto'  # 'auto' 或 'manual'
        self.manual_distances = None
        
        # 绘制场地边界
        self._draw_boundaries()
        
        # 创建控制面板
        self._create_controls()

        # 初始绘制
        self._update_display(true_pose=(self.robot.x, self.robot.y, self.robot.phi))            

    def _create_controls(self):
        """创建所有控制组件：滑动条、输入框、模式选择"""
        # 位姿滑动条
        ax_x = plt.axes([0.25, 0.25, 0.4, 0.03])
        ax_y = plt.axes([0.25, 0.20, 0.4, 0.03])
        ax_phi = plt.axes([0.25, 0.15, 0.4, 0.03])
        
        self.slider_x = Slider(ax_x, 'X', 0, self.m, valinit=self.robot.x)
        self.slider_y = Slider(ax_y, 'Y', 0, self.n, valinit=self.robot.y)
        self.slider_phi = Slider(ax_phi, 'Phi', -np.pi, np.pi, valinit=self.robot.phi)
        
        # 绑定滑动条回调
        for slider in [self.slider_x, self.slider_y, self.slider_phi]:
            slider.on_changed(self._on_slider_changed)
        
        # 手动输入框
        ax_input_x = plt.axes([0.25, 0.10, 0.1, 0.03])
        ax_input_y = plt.axes([0.4, 0.10, 0.1, 0.03])
        ax_input_phi = plt.axes([0.55, 0.10, 0.1, 0.03])
        
        self.input_x = TextBox(ax_input_x, 'X:', initial=str(self.robot.x))
        self.input_y = TextBox(ax_input_y, 'Y:', initial=str(self.robot.y))
        self.input_phi = TextBox(ax_input_phi, 'Phi:', initial=str(self.robot.phi))
        
        # 绑定输入框回调
        self.input_x.on_submit(self._on_manual_input)
        self.input_y.on_submit(self._on_manual_input)
        self.input_phi.on_submit(self._on_manual_input)
        
        # 激光扫描模式选择
        ax_mode = plt.axes([0.78, 0.4, 0.15, 0.15])
        self.mode_radio = RadioButtons(ax_mode, ('Auto Scan', 'Manual Input'))
        self.mode_radio.on_clicked(self._on_mode_changed)
        
        # 激光数据手动输入区域
        self._create_laser_inputs()
    
    def _create_laser_inputs(self):
        """创建激光数据手动输入框"""
        self.laser_inputs = []
        num_lasers = len(self.robot.laser_configs)
        
        for i in range(num_lasers):
            ax_laser = plt.axes([0.78, 0.3 - i*0.05, 0.15, 0.03])
            laser_input = TextBox(ax_laser, f'Laser{i+1}:', initial='0.0')
            laser_input.on_submit(self._on_laser_input_changed)
            self.laser_inputs.append(laser_input)
    
    def _on_mode_changed(self, label):
        """激光扫描模式切换回调"""
        if label == 'Auto Scan':
            self.scan_mode = 'auto'
        else:
            self.scan_mode = 'manual'
        self._trigger_update()
    
    def _on_laser_input_changed(self, text):
        """激光数据输入回调"""
        if self.scan_mode == 'manual':
            try:
                self.manual_distances = []
                for laser_input in self.laser_inputs:
                    dist = float(laser_input.text)
                    self.manual_distances.append(dist)
                self.manual_distances = np.array(self.manual_distances)
                self._trigger_update()
            except ValueError:
                print("Please enter valid numbers")
    
    def _on_manual_input(self, text):
        """手动输入位姿回调"""
        try:
            x = float(self.input_x.text)
            y = float(self.input_y.text)
            phi = float(self.input_phi.text)
            
            # 更新滑动条位置（不触发回调）
            self.slider_x.set_val(x)
            self.slider_y.set_val(y)
            self.slider_phi.set_val(phi)
            
            # 更新机器人位姿并触发更新
            self.robot.update_pose(x, y, phi)
            self._trigger_update()
            
        except ValueError:
            print("Please enter valid numbers")

    def _create_sliders(self):
        """创建X/Y/Phi滑动条（保留原有方法用于兼容）"""
        self._create_controls()

    def _on_slider_changed(self, val):
        """
        滑动条回调流程：
        1. 更新机器人位姿
        2. 同步输入框显示
        3. 获取扫描结果（根据模式）
        4. 调用算法模块
        5. 获取解算结果
        6. 重新绘制
        """
        # 1. 更新机器人位姿
        self.robot.update_pose(
            x=self.slider_x.val,
            y=self.slider_y.val,
            phi=self.slider_phi.val
        )
        
        # 2. 同步输入框显示
        self.input_x.set_val(str(round(self.slider_x.val, 2)))
        self.input_y.set_val(str(round(self.slider_y.val, 2)))
        self.input_phi.set_val(str(round(self.slider_phi.val, 3)))
        
        # 3-6. 触发更新
        self._trigger_update()
    
    def _trigger_update(self):
        """统一的更新触发函数"""
        # 2. 获取扫描结果（根据模式）
        if self.scan_mode == 'auto':
            distances, angles = self.robot.scan()
        else:  # manual
            if self.manual_distances is not None:
                distances = self.manual_distances
                # 计算对应的角度
                angles = []
                for (rel_r, rel_angle), laser_angle in self.robot.laser_configs:
                    current_angle = self.robot.phi + laser_angle
                    angles.append(current_angle)
                angles = np.array(angles)
            else:
                # 如果没有手动输入数据，使用自动扫描
                distances, angles = self.robot.scan()
        
        # 3. 调用算法模块
        solutions = self.solver.solve(distances)
        
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

        # 更新标题
        self._update_title()

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
        for (x_range, y_range, sol_phi) in solutions:
            sol_x = sum(x_range) / 2
            sol_y = sum(y_range) / 2
            
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
        
        self.ax.set_xlim(-self.m*0.2, self.m*1.2)
        self.ax.set_ylim(-self.n*0.2, self.n*1.2)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # 添加标题显示当前模式
        self.ax.set_title('Laser Positioning Simulation - Current Mode: Auto Scan')
    
    def _update_title(self):
        """更新标题显示当前扫描模式"""
        mode_text = "Auto Scan" if self.scan_mode == 'auto' else "Manual Input"
        self.ax.set_title(f'Laser Positioning Simulation - Current Mode: {mode_text}')
    
    def show(self):
        """显示可视化界面"""
        plt.show()
