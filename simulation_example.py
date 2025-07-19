
import os
import glob
from simulation_entries.robot import VirtualRobot 
from simulation_entries.visual import  RobotVisualizer
from positioning_algorithm.pose_calculator import PoseSolver
from matplotlib import pyplot as plt
import numpy as np

# 清空 logs 文件夹
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
for f in glob.glob(os.path.join(log_dir, '*')):
    if os.path.isfile(f):
        os.remove(f)

# 四束激光测试：
lsr_cfg = [
    ((0.5, 0), 0),                      # 0°
    ((0.5, np.pi/2), np.pi/2),          # 90°
    ((0.5, np.pi), np.pi),              # 180°
    ((0.5, -np.pi/2), -np.pi/2)       # 270°
]

# 你可以根据实际环境决定是否启用ROS日志
enable_ros_logging = False
ros_logger = None

# 初始化（使用默认参数）
robot = VirtualRobot(laser_configs=lsr_cfg)
solver = PoseSolver(
    20, 10, robot.laser_configs,
    config=None,
    ros_logger=ros_logger
)
viz = RobotVisualizer(robot, solver)

# 显示界面
plt.show()