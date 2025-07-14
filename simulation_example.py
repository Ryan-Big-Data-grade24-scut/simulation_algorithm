from simulation_entries.robot import VirtualRobot 
from simulation_entries.visual import  RobotVisualizer
from positioning_algorithm.pose_calculator import PoseSolver
from matplotlib import pyplot as plt

# 初始化（使用默认参数）
robot = VirtualRobot()
solver = PoseSolver()
viz = RobotVisualizer(robot, solver)

# 显示界面
plt.show()