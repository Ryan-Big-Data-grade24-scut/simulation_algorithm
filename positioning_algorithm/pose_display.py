import threading
import time
from queue import Queue
from typing import Tuple, Optional

class PoseDisplay(threading.Thread):
    def __init__(self, calculator, update_interval: float = 0.5):
        """
        命令行结果显示线程
        
        参数:
            calculator: 位置计算线程
            update_interval: 显示间隔(秒)
        """
        super().__init__()
        self.calculator = calculator
        self.update_interval = update_interval
        self.running = True
        self.daemon = True
        
    def run(self):
        """线程主循环"""
        print("=== 机器人定位系统启动 ===")
        print("格式: [X坐标] [Y坐标] [朝向(度)]")
        
        while self.running:
            try:
                # 获取最新结果
                latest_result = None
                while not self.calculator.result_queue.empty():
                    latest_result = self.calculator.result_queue.get()
                
                if latest_result:
                    P, phi_deg = latest_result
                    # 清空上一行并移动光标到行首
                    print("\r", end="")
                    # 显示最新结果
                    print(f"位置: {P[0]:.2f}, {P[1]:.2f} | 朝向: {phi_deg:.2f}°", end="", flush=True)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"\n显示错误: {e}")
        
        print("\n=== 显示线程已停止 ===")
    
    def stop(self):
        """停止线程"""
        self.running = False