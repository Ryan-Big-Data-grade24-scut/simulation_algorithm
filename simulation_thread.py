#./simulation_thread.py
import threading
from positioning_algorithm.pose_calculator import PoseCalculator
import time

class RobotController(threading.Thread):
    def __init__(self, robots, m: float, n: float, update_interval: float = 0.1):
        """
        机器人控制线程
        
        参数:
            robots: 机器人对象列表
            m, n: 边界尺寸
            update_interval: 更新间隔(秒)
        """
        super().__init__()
        self.robots = robots
        self.pose_calculators = [PoseCalculator(robot, m, n) for robot in robots]
        self.update_interval = update_interval
        self.running = True
        self.daemon = True

    def run(self):
        """线程主循环"""
        while self.running:
            try:
                self.perform_scanning()
                self.perform_localization()
                
                # 新增可视化更新
                if hasattr(self, 'visualizer'):
                    # 分别准备数据
                    robot_data = self.prepare_robot_data()
                    calculator_data = self.prepare_calculator_data()
                    
                    # 分别调用可视化函数
                    self.visualizer.draw_robot_state(robot_data)
                    self.visualizer.draw_calculator_state(calculator_data)
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"控制线程错误: {e}")

    def perform_scanning(self):
        """执行所有机器人的扫描逻辑"""
        for robot in self.robots:
            robot.scan()

    def perform_localization(self):
        """执行位姿计算流程"""
        for calculator in self.pose_calculators:
            # 1. 获取数据
            t = calculator.get_sensor_data()
            if t is None:
                continue
            
            # 2. 计算位姿
            Q, v = calculator.calculate_pose(t)
            
            # 3. 保存结果
            calculator.save_results(Q, v)
    def prepare_robot_data(self):
        """准备机器人状态数据"""
        robot_data = []
        for robot in self.robots:
            robot_data.append({
                'Point': (robot.x, robot.y),
                'Direction': robot.phi,
                'laser': robot.current_scan_result
            })
        return robot_data

    def prepare_calculator_data(self):
        """准备计算器状态数据"""
        calculator_data = []
        for calculator in self.pose_calculators:
            if calculator.result_list:
                #print(len(calculator.result_list))
                calculator_data.append(calculator.result_list)  # 取最新结果
            else:
                calculator_data.append(None)
        #print(calculator_data)
        return calculator_data

    

    def stop(self):
        """停止线程"""
        self.running = False

def main():
    from simulation_entries.robot import RobotLaserScanner
    robot = RobotLaserScanner()
    rc = RobotController([robot], 10, 10)
    rc.run()

if __name__ == '__main__':
    main()