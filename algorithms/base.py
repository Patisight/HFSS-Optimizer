"""
优化算法基类
定义统一的优化器接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, config: Dict):
        """
        初始化优化器
        
        Args:
            config: 配置字典，包含 variables, objectives 等
        """
        self.config = config
        self.variables = config.get('variables', [])
        self.objectives = config.get('objectives', [])
        
        # 统计信息
        self.evaluation_count = 0
        self.real_evaluation_count = 0
        
        # 缓存
        self._cache = {}
    
    @abstractmethod
    def run(self, evaluator) -> List[Dict]:
        """
        运行优化
        
        Args:
            evaluator: 目标评估器
            
        Returns:
            Pareto 前沿解列表
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        pass
    
    def _params_to_key(self, params: np.ndarray) -> tuple:
        """将参数转换为缓存键"""
        return tuple(round(p, 4) for p in params)
    
    def _check_cache(self, params: np.ndarray) -> Optional[Tuple]:
        """检查缓存"""
        key = self._params_to_key(params)
        return self._cache.get(key)
    
    def _add_to_cache(self, params: np.ndarray, result: Tuple):
        """添加到缓存"""
        key = self._params_to_key(params)
        self._cache[key] = result
    
    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Pareto 支配判断
        
        Args:
            a: 解 a 的目标值向量
            b: 解 b 的目标值向量
            
        Returns:
            a 是否支配 b
        """
        return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))
    
    def fast_non_dominated_sort(self, population: List, objectives: List) -> List[List[int]]:
        """
        快速非支配排序
        
        Args:
            population: 种群
            objectives: 目标值列表
            
        Returns:
            分层的 fronts 列表
        """
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self.dominates(objectives[p], objectives[q]):
                    dominated_solutions[p].append(q)
                elif self.dominates(objectives[q], objectives[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def crowding_distance(self, front: List[int], objectives: List) -> List[float]:
        """
        计算拥挤距离
        
        Args:
            front: 前沿索引列表
            objectives: 目标值列表
            
        Returns:
            拥挤距离列表
        """
        n = len(front)
        if n == 0:
            return []
        
        n_obj = len(objectives[0])
        distances = [0.0] * n
        
        for obj_idx in range(n_obj):
            sorted_indices = sorted(range(n), key=lambda i: objectives[front[i]][obj_idx])
            obj_values = [objectives[front[i]][obj_idx] for i in sorted_indices]
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = obj_values[-1] - obj_values[0]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range
        
        return distances
    
    def get_bounds(self) -> np.ndarray:
        """获取变量边界"""
        return np.array([v['bounds'] for v in self.variables])
    
    def clip_to_bounds(self, params: np.ndarray) -> np.ndarray:
        """将参数裁剪到边界内"""
        bounds = self.get_bounds()
        return np.clip(params, bounds[:, 0], bounds[:, 1])
    
    def format_param(self, value: float, var_index: int) -> float:
        """
        根据变量配置格式化参数值（控制小数点位数）
        
        Args:
            value: 参数值
            var_index: 变量索引
        
        Returns:
            格式化后的参数值
        """
        if var_index < len(self.variables):
            precision = self.variables[var_index].get('precision', 4)
            return round(value, precision)
        return round(value, 4)
    
    def format_params(self, params: np.ndarray) -> np.ndarray:
        """
        格式化所有参数值
        
        Args:
            params: 参数向量
        
        Returns:
            格式化后的参数向量
        """
        return np.array([self.format_param(p, i) for i, p in enumerate(params)])