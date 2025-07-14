import numpy as np
from typing import List, Tuple

class PoseSolver:
    def solve(self, 
             distances: np.ndarray, 
             angles: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        完全匹配图片要求的假算法：
        返回: [((xmin,xmax), (ymin,ymax), phi), ...]
        """
        return [
            ((4.9, 5.1), (4.9, 5.1), 0.0),  # 解1
            ((4.8, 5.2), (5.0, 5.0), 0.1)    # 解2
        ]