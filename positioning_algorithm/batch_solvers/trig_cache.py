"""
三角函数预计算缓存单例
为了节省内存和计算资源，在整个项目中维护唯一的三角函数缓存对象
"""
import numpy as np
from typing import Dict, Optional
import logging

class TrigonometryCache:
    """三角函数预计算缓存单例类"""
    
    _instance: Optional['TrigonometryCache'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not TrigonometryCache._initialized:
            self.logger = logging.getLogger(__name__)
            self.cache_data: Dict = {}
            self.phi_candidates: Optional[np.ndarray] = None
            self.sin_phi_candidates: Optional[np.ndarray] = None
            self.cos_phi_candidates: Optional[np.ndarray] = None
            TrigonometryCache._initialized = True
    
    def initialize_cache(self, combinations: np.ndarray, phi_resolution: float = 0.5):
        """
        初始化三角函数缓存
        
        Args:
            combinations: 激光组合数组 (N, 3, 2) - [t, theta]
            phi_resolution: phi角度分辨率（度）
        """
        self.logger.info("初始化三角函数缓存...")
        
        # 生成phi候选值
        phi_step = np.deg2rad(phi_resolution)
        self.phi_candidates = np.arange(0, 2*np.pi, phi_step)
        
        # 预计算phi的三角函数
        self.sin_phi_candidates = np.sin(self.phi_candidates)
        self.cos_phi_candidates = np.cos(self.phi_candidates)
        
        # 预计算组合中theta的三角函数
        theta_values = combinations[:, :, 1]  # 提取所有theta值
        unique_thetas = np.unique(theta_values.flatten())
        
        self.cache_data['theta_values'] = theta_values
        self.cache_data['sin_theta'] = np.sin(theta_values)
        self.cache_data['cos_theta'] = np.cos(theta_values)
        self.cache_data['unique_thetas'] = unique_thetas
        self.cache_data['sin_unique_thetas'] = np.sin(unique_thetas)
        self.cache_data['cos_unique_thetas'] = np.cos(unique_thetas)
        
        self.logger.info(f"缓存初始化完成: {len(self.phi_candidates)} 个phi值, "
                        f"{len(unique_thetas)} 个唯一theta值")
    
    def update_combinations(self, new_combinations: np.ndarray):
        """
        更新组合，只重新计算变化的theta值 - 避免每次solve都计算720个角度
        
        Args:
            new_combinations: 新的组合数组 (N, 3, 2) - [t, theta]
        """
        if len(new_combinations) == 0:
            return
            
        if 'theta_values' not in self.cache_data:
            # 如果缓存未初始化，直接初始化
            self.initialize_cache(new_combinations)
            return
        
        new_thetas = new_combinations[:, :, 1]  # 提取新的theta值
        new_unique_thetas = np.unique(new_thetas.flatten())
        old_unique_thetas = self.cache_data['unique_thetas']
        
        # 检查是否有新的theta值需要计算
        new_theta_mask = ~np.isin(new_unique_thetas, old_unique_thetas)
        if np.any(new_theta_mask):
            # 有新的theta值，更新缓存
            all_unique_thetas = np.unique(np.concatenate([old_unique_thetas, new_unique_thetas]))
            self.cache_data['unique_thetas'] = all_unique_thetas
            self.cache_data['sin_unique_thetas'] = np.sin(all_unique_thetas)
            self.cache_data['cos_unique_thetas'] = np.cos(all_unique_thetas)
            
            self.logger.debug(f"更新theta缓存: 新增 {np.sum(new_theta_mask)} 个theta值")
        else:
            self.logger.debug("无新theta值，跳过三角函数计算")
        
        # 更新当前组合的theta数据
        self.cache_data['theta_values'] = new_thetas
        self.cache_data['sin_theta'] = np.sin(new_thetas)
        self.cache_data['cos_theta'] = np.cos(new_thetas)
    
    def get_phi_candidates(self) -> np.ndarray:
        """获取phi候选值"""
        return self.phi_candidates
    
    def get_phi_trigonometry(self) -> tuple:
        """获取phi的三角函数值"""
        return self.sin_phi_candidates, self.cos_phi_candidates
    
    def get_theta_trigonometry(self) -> tuple:
        """获取当前组合theta的三角函数值"""
        return self.cache_data['sin_theta'], self.cache_data['cos_theta']
    
    def get_cache_info(self) -> dict:
        """获取缓存信息用于调试"""
        if not self.cache_data:
            return {"status": "未初始化"}
        
        return {
            "phi_count": len(self.phi_candidates) if self.phi_candidates is not None else 0,
            "unique_theta_count": len(self.cache_data.get('unique_thetas', [])),
            "current_combinations_shape": self.cache_data.get('theta_values', np.array([])).shape,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算缓存内存使用量（MB）"""
        total_elements = 0
        
        if self.phi_candidates is not None:
            total_elements += len(self.phi_candidates) * 3  # phi, sin_phi, cos_phi
        
        for key, value in self.cache_data.items():
            if isinstance(value, np.ndarray):
                total_elements += value.size
        
        # 假设每个float64元素8字节
        return (total_elements * 8) / (1024 * 1024)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_data.clear()
        self.phi_candidates = None
        self.sin_phi_candidates = None
        self.cos_phi_candidates = None
        self.logger.info("三角函数缓存已清空")

# 全局单例实例
trig_cache = TrigonometryCache()
