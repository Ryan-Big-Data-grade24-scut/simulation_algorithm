from simulation_entries.robot import VirtualRobot 
from simulation_entries.visual import  RobotVisualizer
from positioning_algorithm.pose_calculator import PoseSolver
from matplotlib import pyplot as plt
import numpy as np

# 多束激光测试：
lsr_cfg = [
    ((0.5, np.pi/2), np.pi/2),   # 左前传感器
    ((0.5, np.pi), np.pi),        # 正前传感器
    ((0.5, -np.pi/2), -np.pi/2), 
    ((0.5, 0), 0)
]

# 初始化（使用默认参数）
robot = VirtualRobot(laser_configs=lsr_cfg)
solver = PoseSolver(20, 10, robot.laser_configs)
viz = RobotVisualizer(robot, solver)

# 显示界面
plt.show()