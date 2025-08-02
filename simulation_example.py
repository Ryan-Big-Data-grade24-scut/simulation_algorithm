import os
import glob
from simulation_entries.robot import VirtualRobot 
from simulation_entries.visual import  RobotVisualizer
from positioning_algorithm.angle_pose_calculator import PoseSolver
from matplotlib import pyplot as plt
import numpy as np

# 清空 logs 文件夹
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
for f in glob.glob(os.path.join(log_dir, '*')):
    if os.path.isfile(f):
        os.remove(f)

# 5束激光测试——均匀72°分布：
lsr_cfg = [
    ((0.36453, 1.55456), 1.25663),    # 极坐标(r,θ), 波束角度
    ((0.36453, -1.55456), -1.25663),  # 极坐标(r,θ), 波束角度
    ((0.37645, 0.47385), 0.0),        # 极坐标(r,θ), 波束角度
    ((0.37537, 2.40907), 2.51327),     # 极坐标(r,θ), 波束角度
    ((0.37537, -2.42530), -2.51327)   # 极坐标(r,θ), 波束角度
]
# 你可以根据实际环境决定是否启用ROS日志
enable_ros_logging = False
ros_logger = None

m, n = 10, 8

# 初始化（使用默认参数）
robot = VirtualRobot(phi=np.pi, m=m, n=n, laser_configs=lsr_cfg)
solver = PoseSolver(
    m, n, robot.laser_configs,
    config=None,
    Rcl_logger=ros_logger
)
viz = RobotVisualizer(robot, solver, m=m, n=n)

# 显示界面
plt.show()