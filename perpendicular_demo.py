import os
import glob
from simulation_entries.robot import VirtualRobot 
from simulation_entries.visual import RobotVisualizer, LaserLengthVisualizer
from positioning_algorithm.angle_pose_calculator import PoseSolver
from positioning_algorithm.perpendicular_localizer import PerpendicularDistanceLocalizer, create_calibration_data, create_calibration_data_from_robot
from matplotlib import pyplot as plt
import numpy as np

def parse_laser_lengths(laser_string):
    """
    从字符串中解析激光长度
    
    Args:
        laser_string: 激光长度字符串，如 "[1.359 6.482 3.695 5.742 6.973]"
    
    Returns:
        list: 激光长度列表
    """
    # 移除方括号和多余空格
    clean_string = laser_string.strip("[]")
    # 按空格分割并转换为浮点数
    lengths = [float(x) for x in clean_string.split()]
    return lengths

def parse_pose_string(pose_string):
    """
    从字符串中解析位姿参数
    
    Args:
        pose_string: 位姿字符串，如 "x=7.847, y=0.488, angle=-3.102"
    
    Returns:
        dict: 包含x, y, phi的字典
    """
    import re
    
    # 使用正则表达式提取数值
    x_match = re.search(r'x\s*=\s*([-+]?\d*\.?\d+)', pose_string)
    y_match = re.search(r'y\s*=\s*([-+]?\d*\.?\d+)', pose_string)
    angle_match = re.search(r'angle\s*=\s*([-+]?\d*\.?\d+)', pose_string)
    
    if not all([x_match, y_match, angle_match]):
        raise ValueError(f"无法从字符串中解析位姿参数: {pose_string}")
    
    return {
        'x': float(x_match.group(1)),
        'y': float(y_match.group(1)),
        'phi': float(angle_match.group(1))  # angle对应phi
    }

# 清空 logs 文件夹
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
for f in glob.glob(os.path.join(log_dir, '*')):
    if os.path.isfile(f):
        os.remove(f)

# 默认激光长度字符串
default_laser_lengths_str = "[0.122 7.476 7.538 0.865 1.923]"

# 默认位姿字符串（直接粘贴你的数据格式）
default_pose_str = "x=7.871, y=0.498, angle=-3.142"

# 配置选项
USE_DEFAULT_LASER_LENGTHS = False  # 是否使用默认激光长度
USE_DEFAULT_POSE = False          # 是否使用默认位姿
ENABLE_SOLVER = True             # 是否启用求解器
USE_PERPENDICULAR_LOCALIZER = True  # 是否使用垂直距离定位器

# 5束激光测试——均匀72°分布：
lsr_cfg = [
    ((0.36889, 1.56782), np.pi/2),    # 极坐标(r,θ), 波束角度
    ((0.36889, -1.56782), -np.pi/2),  # 极坐标(r,θ), 波束角度
    ((0.37645, 0.47385), 0.0),        # 极坐标(r,θ), 波束角度
    ((0.37537, 2.40907), np.pi),     # 极坐标(r,θ), 波束角度
    ((0.37537, -2.42530), -np.pi)   # 极坐标(r,θ), 波束角度
]
# 你可以根据实际环境决定是否启用ROS日志
enable_ros_logging = False
ros_logger = None

m, n = 10, 8

# 根据配置确定初始位姿
if USE_DEFAULT_POSE:
    default_pose = parse_pose_string(default_pose_str)
    init_x, init_y, init_phi = default_pose['x'], default_pose['y'], default_pose['phi']
    print(f"解析位姿字符串: {default_pose_str}")
    print(f"得到位姿: x={init_x:.3f}, y={init_y:.3f}, angle={init_phi:.3f}弧度")
else:
    init_x, init_y, init_phi = 5.0, 5.0, 0.0  # 备用位姿
    print(f"使用备用位姿: x={init_x}, y={init_y}, angle={init_phi}")

# 初始化机器人
robot = VirtualRobot(x=init_x, y=init_y, phi=init_phi, m=m, n=n, laser_configs=lsr_cfg)

# 根据配置确定激光长度
if USE_DEFAULT_LASER_LENGTHS:
    laser_lengths = parse_laser_lengths(default_laser_lengths_str)
    print(f"使用默认激光长度: {laser_lengths}")
else:
    laser_lengths = None
    print("使用扫描结果作为激光长度")

# 根据配置创建求解器
solver = None
if ENABLE_SOLVER:
    if USE_PERPENDICULAR_LOCALIZER:
        # 使用简化版垂直距离定位器
        solver = PerpendicularDistanceLocalizer(robot, tolerance=0.1)
        print("简化版垂直距离定位器已启用并完成标定")
    else:
        # 使用原来的角度位姿计算器
        solver = PoseSolver(
            m, n, robot.laser_configs,
            Rcl_logger=ros_logger
        )
        print("角度位姿求解器已启用")

# 选择可视化模式
use_original_visualizer = False  # 设置为True使用原始的RobotVisualizer

if use_original_visualizer:
    # 使用原来的可视化器（仅当启用求解器时）
    if not ENABLE_SOLVER:
        print("警告：原始可视化器需要求解器，自动启用求解器")
        solver = PoseSolver(m, n, robot.laser_configs, config=None, Rcl_logger=ros_logger)
    viz = RobotVisualizer(robot, solver, m=m, n=n)
else:
    # 使用新的激光长度可视化器
    viz = LaserLengthVisualizer(
        robot=robot, 
        laser_lengths=laser_lengths, 
        solver=solver,
        enable_solver=ENABLE_SOLVER,
        m=m, 
        n=n
    )

# 显示界面
plt.show()