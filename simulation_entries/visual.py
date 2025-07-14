#./simulation_entries/visual.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection

class MultiRobotVisualizer:
    def __init__(self, robot_scanners=None, m=10, n=10):
        """
        多机器人可视化类
        
        参数:
            robot_scanners: RobotLaserScanner实例列表
        """

        self.m = m
        self.n = n
        self.robots = robot_scanners if robot_scanners else []
        
        # 创建图形界面
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, left=0.25)
        
        # 图形元素存储
        self.robot_elements = []  # 存储每个机器人的图形元素
        self.current_robot_idx = 0  # 当前控制的机器人索引
        
        # 创建控制面板
        self.create_controls()
        
        # 初始绘制
        self.draw_environment()
        #self.update_display()
    
    def create_controls(self):
        """创建控制面板"""
        # 机器人选择按钮
        self.ax_prev = plt.axes([0.05, 0.15, 0.1, 0.05])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_robot)
        
        self.ax_next = plt.axes([0.05, 0.05, 0.1, 0.05])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.next_robot)
        
        # 机器人控制滑动条
        self.ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_x = Slider(self.ax_x, 'X', 0, self.m, valinit=5)
        
        self.ax_y = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.slider_y = Slider(self.ax_y, 'Y', 0, self.n, valinit=5)
        
        self.ax_phi = plt.axes([0.25, 0.05, 0.65, 0.03])
        self.slider_phi = Slider(self.ax_phi, 'Phi', -np.pi, np.pi, valinit=np.pi/4)
        
        # 绑定事件
        self.slider_x.on_changed(self.update_current_robot)
        self.slider_y.on_changed(self.update_current_robot)
        self.slider_phi.on_changed(self.update_current_robot)
    
    def prev_robot(self, event):
        """切换到上一个机器人"""
        if len(self.robots) > 1:
            self.current_robot_idx = (self.current_robot_idx - 1) % len(self.robots)
            self.update_sliders()
            self.update_display()
    
    def next_robot(self, event):
        """切换到下一个机器人"""
        if len(self.robots) > 1:
            self.current_robot_idx = (self.current_robot_idx + 1) % len(self.robots)
            self.update_sliders()
            self.update_display()
    
    def update_sliders(self):
        """更新滑动条值为当前机器人的状态"""
        if self.robots:
            robot = self.robots[self.current_robot_idx]
            for i in self.robots:
                print(i.x, i.y)
            self.slider_x.set_val(robot.x)
            self.slider_y.set_val(robot.y)
            self.slider_phi.set_val(robot.phi)
    
    def update_current_robot(self, val):
        """更新当前机器人的状态（不再调用update_display）"""
        if self.robots:
            robot = self.robots[self.current_robot_idx]
            robot.update_pose(
                self.slider_x.val,
                self.slider_y.val,
                self.slider_phi.val
            )
    
    def draw_environment(self):
        """绘制静态环境"""
        if self.robots:
            # 使用第一个机器人的边界配置（假设所有机器人共享同一环境）
            for line in self.robots[0].boundary_lines:
                self.ax.plot([line[0][0], line[1][0]], 
                           [line[0][1], line[1][1]], 
                           'k-', linewidth=2)
        
        self.ax.set_xlim(-self.m, self.m)
        self.ax.set_ylim(-self.n, self.n)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Multi-Robot Laser Scanner Simulation')
    
    def update_display(self):
        """更新所有可视化元素"""
        # 清除所有现有元素
        for elements in self.robot_elements:
            self.remove_robot_elements(elements)
        self.robot_elements = []
        
        # 为每个机器人创建图形元素
        for i, robot in enumerate(self.robots):
            elements = {
                'dots': [],
                'arrows': None,
                'laser_lines': None,
                'collision_points': None,
                'distance_text': None
            }
            
            # 绘制机器人
            color = 'r' if i == self.current_robot_idx else 'm'
            dot = self.ax.plot(robot.x, robot.y, f'{color}o', markersize=10)[0]
            elements['dots'].append(dot)
            
            # 绘制朝向箭头
            arrow_length = 0.8
            dx = arrow_length * np.cos(robot.phi)
            dy = arrow_length * np.sin(robot.phi)
            arrow = self.ax.arrow(
                robot.x, robot.y, dx, dy, 
                head_width=0.3, head_length=0.4, 
                fc=color, ec=color)
            elements['arrows'] = arrow
            
            # 获取扫描结果
            scan_results = robot.scan()
            
            # 准备可视化数据
            laser_segments = []
            collision_coords = []
            distance_info = []
            
            for j, result in enumerate(scan_results):
                # 激光束可视化
                if result['hit_point']:
                    laser_segments.append([result['position'], result['hit_point']])
                    collision_coords.append(result['hit_point'])
                    dist_str = f"R{i+1}L{j+1}: {result['distance']:.2f}"
                    distance_info.append(dist_str)
                else:
                    max_dist = 15
                    end_x = result['position'][0] + max_dist * np.cos(result['angle'])
                    end_y = result['position'][1] + max_dist * np.sin(result['angle'])
                    laser_segments.append([result['position'], (end_x, end_y)])
                    dist_str = f"R{i+1}L{j+1}: ∞"
                    distance_info.append(dist_str)
                
                # 绘制激光头
                head_dot = self.ax.plot(
                    result['position'][0], result['position'][1], 
                    'bo', markersize=6)[0]
                elements['dots'].append(head_dot)
            
            # 绘制激光束
            if laser_segments:
                lc = LineCollection(
                    laser_segments, 
                    colors='b', 
                    linewidths=1, 
                    linestyles='dashed')
                self.ax.add_collection(lc)
                elements['laser_lines'] = lc
            
            # 绘制碰撞点
            if collision_coords:
                x, y = zip(*collision_coords)
                scatter = self.ax.scatter(
                    x, y, c='r', s=50, marker='x', zorder=5)
                elements['collision_points'] = scatter
            
            # 显示距离信息
            if i == self.current_robot_idx:
                text = self.ax.text(
                    0.02, 0.98, '\n'.join(distance_info),
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))
                elements['distance_text'] = text
            
            self.robot_elements.append(elements)
        
        self.fig.canvas.draw_idle()
    
    def remove_robot_elements(self, elements):
        """安全移除单个机器人的图形元素"""
        # 移除点
        for dot in elements.get('dots', []):
            try:
                dot.remove()
            except (ValueError, AttributeError):
                pass
        
        # 移除其他元素
        for key in ['arrows', 'laser_lines', 'collision_points', 'distance_text']:
            artist = elements.get(key)
            if artist is not None:
                try:
                    if key == 'laser_lines':
                        artist.remove()
                    else:
                        artist.remove()
                except (ValueError, AttributeError):
                    pass
    
    def add_robot(self, robot_scanner):
        """添加新机器人"""
        self.robots.append(robot_scanner)
        if len(self.robots) == 1:  # 如果是第一个机器人，更新滑动条
            self.update_sliders()
        self.update_display()

    def draw_robot_state(self, robot_data):
        """
        绘制机器人状态
        参数:
            robot_data: prepare_robot_data()返回的数据结构
        """
        # 清除旧的机器人元素
        for elements in self.robot_elements:
            self.remove_robot_elements(elements)
        self.robot_elements = []
        
        for i, data in enumerate(robot_data):
            elements = {
                'dots': [],
                'arrows': None,
                'laser_lines': None,
                'collision_points': None,
                'distance_text': None
            }
            
            # 绘制机器人
            color = 'r' if i == self.current_robot_idx else 'm'
            dot = self.ax.plot(
                data['Point'][0], data['Point'][1],
                f'{color}o', markersize=10)[0]
            elements['dots'].append(dot)
            
            # 绘制朝向箭头
            arrow_length = 0.8
            dx = arrow_length * np.cos(data['Direction'])
            dy = arrow_length * np.sin(data['Direction'])
            arrow = self.ax.arrow(
                data['Point'][0], data['Point'][1],
                dx, dy,
                head_width=0.3, head_length=0.4,
                fc=color, ec=color)
            elements['arrows'] = arrow
            
            # 绘制激光相关
            laser_segments = []
            collision_coords = []
            distance_info = []
            
            for j, laser in enumerate(data['laser']):
                if laser['hit_point']:
                    laser_segments.append([laser['position'], laser['hit_point']])
                    collision_coords.append(laser['hit_point'])
                    dist_str = f"R{i+1}L{j+1}: {laser['distance']:.2f}"
                else:
                    max_dist = 15
                    end_x = laser['position'][0] + max_dist * np.cos(laser['angle'])
                    end_y = laser['position'][1] + max_dist * np.sin(laser['angle'])
                    laser_segments.append([laser['position'], (end_x, end_y)])
                    dist_str = f"R{i+1}L{j+1}: ∞"
                distance_info.append(dist_str)
                
                # 激光头
                head_dot = self.ax.plot(
                    laser['position'][0], laser['position'][1],
                    'bo', markersize=6)[0]
                elements['dots'].append(head_dot)
            
            # 绘制激光束
            if laser_segments:
                lc = LineCollection(
                    laser_segments,
                    colors='b', linewidths=1, linestyles='dashed')
                self.ax.add_collection(lc)
                elements['laser_lines'] = lc
            
            # 绘制碰撞点
            if collision_coords:
                x, y = zip(*collision_coords)
                scatter = self.ax.scatter(
                    x, y, c='r', s=50, marker='x', zorder=5)
                elements['collision_points'] = scatter
            
            # 显示距离信息
            if i == self.current_robot_idx:
                text = self.ax.text(
                    0.02, 0.98, '\n'.join(distance_info),
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))
                elements['distance_text'] = text
            
            self.robot_elements.append(elements)
        
        self.fig.canvas.draw_idle()

    def draw_calculator_state(self, calculator_data):
        """
        绘制计算器状态（支持多解算结果）
        参数:
            calculator_data: 包含所有解算结果的嵌套列表
        """
        # 清除旧的解算元素
        if hasattr(self, 'calculator_elements'):
            for elements in self.calculator_elements:
                self.remove_calculator_elements(elements)
        else:
            self.calculator_elements = []
        
        for i, calculator_data_1 in enumerate(calculator_data):
            if calculator_data_1 is None:
                continue
                
            elements = {
                'estimate_dots': [],
                'estimate_arrows': [],
                'estimate_lines': []
            }
            
            # 为每个解算结果绘制图形
            for data in calculator_data_1:
                # 绘制解算位置点
                est_dot = self.ax.plot(
                    data['P'][0], data['P'][1],
                    'go', markersize=8, alpha=0.7)[0]
                elements['estimate_dots'].append(est_dot)
                
                # 绘制解算朝向箭头
                arrow_length = 0.6
                dx = arrow_length * np.cos(data['phi'])
                dy = arrow_length * np.sin(data['phi'])
                est_arrow = self.ax.arrow(
                    data['P'][0], data['P'][1],
                    dx, dy,
                    head_width=0.2, head_length=0.3,
                    fc='g', ec='g', alpha=0.7)
                elements['estimate_arrows'].append(est_arrow)
                """
                # 绘制解算向量
                vector_segments = []
                for vec in data['v']:
                    # 坐标转换
                    rotated_x = vec[0] * np.cos(data['phi']) - vec[1] * np.sin(data['phi'])
                    rotated_y = vec[0] * np.sin(data['phi']) + vec[1] * np.cos(data['phi'])
                    global_vec = (rotated_x + data['P'][0], rotated_y + data['P'][1])
                    
                    vector_segments.append([data['P'], global_vec])
                
                if vector_segments:
                    vc = LineCollection(
                        vector_segments,
                        colors='g', linewidths=1.5, alpha=0.5)
                    self.ax.add_collection(vc)
                    elements['estimate_lines'].append(vc)
                """
            self.calculator_elements.append(elements)
        
        self.fig.canvas.draw_idle()

    def remove_calculator_elements(self, elements):
        """移除解算图形元素（支持多结果）"""
        for dot in elements.get('estimate_dots', []):
            try:
                dot.remove()
            except (ValueError, AttributeError):
                pass
                
        for arrow in elements.get('estimate_arrows', []):
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
                
        for line_collection in elements.get('estimate_lines', []):
            try:
                line_collection.remove()
            except (ValueError, AttributeError):
                pass