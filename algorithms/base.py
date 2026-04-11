"""
优化算法基类
定义统一的优化器接口
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constraint import VariableConstraint


class BaseOptimizer(ABC):
    """优化器基类"""

    def __init__(self, config: Dict):
        """
        初始化优化器

        Args:
            config: 配置字典，包含 variables, objectives 等
        """
        self.config = config
        self.variables = config.get("variables", [])
        self.objectives = config.get("objectives", [])

        self.stop_when_goal_met = config.get("stop_when_goal_met", False)
        self.n_solutions_to_stop = config.get("n_solutions_to_stop", 3)

        self.evaluation_count = 0
        self.real_evaluation_count = 0

        self._cache = {}

        self.constraint_mgr = VariableConstraint(self.variables)
        self._has_formulas = self.constraint_mgr.has_formulas()

        if self._has_formulas:
            logger.info(f"[CONSTRAINT] Found formula bounds in variables")
            logger.info(f"[CONSTRAINT] Evaluation order: {self.constraint_mgr.eval_order}")
            dep_vars = self.constraint_mgr.get_dependent_vars()
            if dep_vars:
                logger.info(f"[CONSTRAINT] Dependent variables: {dep_vars}")

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

            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_range = obj_values[-1] - obj_values[0]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range

        return distances

    def get_bounds(self) -> np.ndarray:
        """获取变量边界（静态，不考虑公式）"""
        bounds = []
        for v in self.variables:
            lb, ub = v["bounds"]
            if isinstance(lb, str):
                lb = 0.0
            if isinstance(ub, str):
                ub = 1.0
            bounds.append([lb, ub])
        return np.array(bounds, dtype=float)

    def get_static_bounds(self) -> np.ndarray:
        """获取静态边界（用于算法初始化，公式边界取默认值）"""
        return self.get_bounds()

    def clip_to_bounds(self, params: np.ndarray) -> np.ndarray:
        """将参数裁剪到边界内（支持公式边界）"""
        if not self._has_formulas:
            bounds = self.get_static_bounds()
            return np.clip(params, bounds[:, 0], bounds[:, 1])

        params_dict = self._params_to_dict(params)
        repaired = self.constraint_mgr.repair_params(params_dict)
        return self._dict_to_params(repaired)

    def check_constraints(self, params: np.ndarray) -> Tuple[bool, str]:
        """检查参数是否满足约束"""
        if not self._has_formulas:
            return True, ""
        params_dict = self._params_to_dict(params)
        return self.constraint_mgr.check_constraints(params_dict)

    def get_penalty_objectives(self, n_obj: int = None) -> np.ndarray:
        """获取惩罚目标值（违反约束时返回）"""
        if n_obj is None:
            n_obj = len(self.objectives)

        penalty = np.zeros(n_obj)
        for i, obj in enumerate(self.objectives):
            target = obj.get("target", "minimize")
            if target == "minimize":
                penalty[i] = 999.0
            else:
                penalty[i] = -999.0
        return penalty

    def is_penalty_value(self, y: np.ndarray) -> bool:
        """检查目标值是否为惩罚值或异常值

        Args:
            y: 目标值数组

        Returns:
            True 如果是惩罚值或异常值
        """
        PENALTY_VALUE = 999.0
        SIMULATION_FAILURE_VALUE = 1000.0
        ABNORMAL_THRESHOLD = 100.0

        y_flat = y.flatten()
        for val in y_flat:
            if abs(abs(val) - PENALTY_VALUE) < 1e-6:
                return True
            if abs(val - SIMULATION_FAILURE_VALUE) < 1e-6:
                return True
            if abs(val) > ABNORMAL_THRESHOLD:
                return True
        return False

    def _params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """参数数组转字典"""
        return {v["name"]: params[i] for i, v in enumerate(self.variables)}

    def _dict_to_params(self, params_dict: Dict[str, float]) -> np.ndarray:
        """参数字典转数组"""
        return np.array([params_dict.get(v["name"], 0.0) for v in self.variables])

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
            precision = self.variables[var_index].get("precision", 4)
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

    def count_solutions_meeting_goals(self, results_list: List) -> int:
        """
        统计有多少个解的所有目标都达到了

        Args:
            results_list: 评估结果列表，每个元素是 {obj_name: ObjectiveResult} 字典

        Returns:
            达标的解数量
        """
        count = 0
        for results in results_list:
            if results is None:
                continue
            all_met = True
            for name, res in results.items():
                if isinstance(res, dict):
                    goal_met = res.get("goal_met")
                else:
                    goal_met = res.goal_met
                if goal_met is not True:
                    all_met = False
                    break
            if all_met:
                count += 1
        return count

    def should_stop_early(self, results_list: List) -> bool:
        """
        判断是否应该早停

        Args:
            results_list: 当前种群所有个体的评估结果列表

        Returns:
            是否应该停止
        """
        if not self.stop_when_goal_met:
            return False
        count = self.count_solutions_meeting_goals(results_list)
        return count >= self.n_solutions_to_stop

    def check_objectives_meet_goals(self, obj_values: np.ndarray) -> bool:
        """
        检查一组目标值是否全部达到目标（用于MOPSO等算法）

        注意：对于 maximize 目标，obj_values 存储的是 -actual（负值）

        Args:
            obj_values: 目标值数组

        Returns:
            是否全部达标
        """
        all_met = True
        for i, obj_config in enumerate(self.objectives):
            if i >= len(obj_values):
                return False
            val = obj_values[i]
            target = obj_config.get("target", "minimize")
            goal = obj_config.get("goal")

            if goal is None:
                continue

            if target == "minimize":
                if val > goal:
                    all_met = False
            elif target == "maximize":
                # val 是 -actual，所以要检查 -val >= goal，即 actual >= goal
                if -val < goal:
                    all_met = False

        return all_met

    def count_objectives_meeting_goals_from_arrays(self, objectives_array: List[np.ndarray]) -> int:
        """
        统计有多少个解的所有目标都达到了（基于目标值数组）

        Args:
            objectives_array: 目标值数组列表

        Returns:
            达标的解数量
        """
        count = 0
        for idx, obj_values in enumerate(objectives_array):
            if self.check_objectives_meet_goals(obj_values):
                count += 1
                # 打印达标的解
                obj_str = ", ".join([f"{v:.4f}" for v in obj_values])
                logger.info(f"  [Goal met] Solution {idx+1}: [{obj_str}]")
        return count
