#./simulation_example.py
from simulation_entries.robot import RobotLaserScanner
from simulation_entries.visual import MultiRobotVisualizer
import matplotlib.pyplot as plt
from simulation_thread import RobotController

def main():
    # 初始化参数
    m, n = 10.0, 8.0  # 边界尺寸
    # 创建计算线程和显示线程
    # 创建机器人实例
    
    try:
        # 启动所有线程
        robot1 = RobotLaserScanner(x=2, y=3)
        controller = RobotController(robots=[robot1], m=20, n=10)
        # 创建可视化器并关联
        visualizer = MultiRobotVisualizer(robot_scanners=[robot1], m=20, n=10)
        controller.visualizer = visualizer
        controller.start()
        plt.show()
    except KeyboardInterrupt:
        print("\n正在停止程序...")
    finally:
        # 安全停止所有线程
        controller.stop()
        controller.join()
        
        print("程序已完全停止")
    
    
    
    # 创建仿真线程
    #simulator = SimulationThread(robot, calculator, display, move_interval=0.2)
    """
    
    """

if __name__ == "__main__":
    main()