"""
约束感知参数管理器
统一处理变量间的公式约束关系
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constraint import VariableConstraint


class ConstrainedParameterManager:
    """
    约束感知参数管理器
    
    统一管理带公式约束的变量，确保任何参数设置都自动满足所有约束。
    用于：自检、随机采样、优化算法等场景。
    """
    
    def __init__(self, variables: List[Dict]):
        """
        初始化
        
        Args:
            variables: 变量配置列表
        """
        self.variables = variables
        self.constraint = VariableConstraint(variables)
        self.current_values: Dict[str, float] = {}
        self._initialize_values()
    
    def _initialize_values(self):
        """初始化所有变量值（使用边界中点）"""
        for name in self.constraint.eval_order:
            bounds = self.constraint.resolve_bounds(self.current_values)
            lb, ub = bounds.get(name, (0, 1))
            if lb <= ub:
                self.current_values[name] = (lb + ub) / 2
            else:
                self.current_values[name] = lb
    
    def get_value(self, name: str) -> Optional[float]:
        """获取变量值"""
        return self.current_values.get(name)
    
    def get_all_values(self) -> Dict[str, float]:
        """获取所有变量值"""
        return self.current_values.copy()
    
    def set_value(self, name: str, value: float, auto_adjust: bool = True) -> Tuple[bool, str]:
        """
        设置单个变量值
        
        Args:
            name: 变量名
            value: 新值
            auto_adjust: 是否自动调整依赖变量
        
        Returns:
            (是否成功, 消息)
        """
        if name not in self.current_values:
            return False, f"未知变量: {name}"
        
        # 设置新值
        old_value = self.current_values.get(name)
        self.current_values[name] = value
        
        # 自动调整依赖变量
        adjusted = []
        if auto_adjust:
            adjusted = self._adjust_dependent_vars(name)
        
        # 检查所有约束
        valid, msg = self.constraint.check_constraints(self.current_values)
        
        if not valid:
            # 约束不满足，恢复旧值
            if old_value is not None:
                self.current_values[name] = old_value
            # 恢复被调整的变量
            for adj_name, adj_old in adjusted:
                self.current_values[adj_name] = adj_old
            return False, msg
        
        return True, ""
    
    def set_values(self, values: Dict[str, float], auto_adjust: bool = True) -> Tuple[bool, str]:
        """
        批量设置变量值
        
        Args:
            values: 变量值字典
            auto_adjust: 是否自动调整依赖变量
        
        Returns:
            (是否成功, 消息)
        """
        old_values = self.current_values.copy()
        self.current_values.update(values)
        
        if auto_adjust:
            self._adjust_all_dependent_vars()
        
        valid, msg = self.constraint.check_constraints(self.current_values)
        
        if not valid:
            self.current_values = old_values
            return False, msg
        
        return True, ""
    
    def set_to_boundary(self, name: str, which: str = 'min') -> Tuple[bool, float, List[str]]:
        """
        将变量设置到边界值，自动调整依赖变量
        
        Args:
            name: 变量名
            which: 'min' 或 'max'
        
        Returns:
            (是否成功, 实际设置的值, 被调整的变量列表)
        """
        if name not in self.current_values:
            return False, 0, []
        
        bounds = self.constraint.resolve_bounds(self.current_values)
        lb, ub = bounds.get(name, (0, 1))
        
        target_value = lb if which == 'min' else ub
        
        old_values = self.current_values.copy()
        self.current_values[name] = target_value
        
        # 调整依赖变量
        adjusted_names = self._adjust_dependent_vars(name)
        adjusted_names = [n for n, _ in adjusted_names]
        
        # 检查约束
        valid, msg = self.constraint.check_constraints(self.current_values)
        
        if not valid:
            self.current_values = old_values
            return False, target_value, []
        
        return True, self.current_values[name], adjusted_names
    
    def _adjust_dependent_vars(self, changed_var: str) -> List[Tuple[str, float]]:
        """
        调整依赖于 changed_var 的变量
        
        Returns:
            [(变量名, 旧值), ...] 被调整的变量列表
        """
        adjusted = []
        
        for var_name in self.constraint.eval_order:
            if var_name == changed_var:
                continue
            
            deps = self.constraint.dependencies.get(var_name, set())
            if changed_var not in deps:
                continue
            
            if var_name not in self.current_values:
                continue
            
            old_value = self.current_values[var_name]
            
            # 计算新的边界
            bounds = self.constraint.resolve_bounds(self.current_values)
            lb, ub = bounds.get(var_name, (0, 1))
            
            # 调整到边界内
            new_value = max(lb, min(old_value, ub))
            
            if abs(new_value - old_value) > 1e-9:
                self.current_values[var_name] = new_value
                adjusted.append((var_name, old_value))
        
        # 递归调整（处理链式依赖）
        for adj_name, _ in adjusted:
            sub_adjusted = self._adjust_dependent_vars(adj_name)
            adjusted.extend(sub_adjusted)
        
        return adjusted
    
    def _adjust_all_dependent_vars(self):
        """调整所有依赖变量"""
        for name in self.constraint.eval_order:
            if name in self.current_values:
                bounds = self.constraint.resolve_bounds(self.current_values)
                lb, ub = bounds.get(name, (0, 1))
                value = self.current_values[name]
                self.current_values[name] = max(lb, min(value, ub))
    
    def generate_random_params(self) -> Dict[str, float]:
        """
        生成满足约束的随机参数
        
        按依赖顺序生成，确保公式边界正确计算
        """
        params = {}
        
        for name in self.constraint.eval_order:
            bounds = self.constraint.resolve_bounds(params)
            lb, ub = bounds.get(name, (0, 1))
            
            if lb <= ub:
                params[name] = np.random.uniform(lb, ub)
            else:
                params[name] = (lb + ub) / 2
        
        self.current_values = params
        return params
    
    def validate_current(self) -> Tuple[bool, str]:
        """验证当前参数是否满足所有约束"""
        return self.constraint.check_constraints(self.current_values)
    
    def get_bounds_for_var(self, name: str) -> Tuple[float, float]:
        """获取变量的当前有效边界"""
        bounds = self.constraint.resolve_bounds(self.current_values)
        return bounds.get(name, (0, 1))
    
    def get_all_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取所有变量的当前有效边界"""
        return self.constraint.resolve_bounds(self.current_values)
    
    def has_formulas(self) -> bool:
        """是否有公式边界"""
        return self.constraint.has_formulas()
    
    def get_dependency_info(self) -> Dict[str, Dict]:
        """获取依赖信息"""
        return self.constraint.get_dependency_info()
    
    def get_eval_order(self) -> List[str]:
        """获取变量计算顺序"""
        return self.constraint.eval_order
    
    def test_boundary_feasibility(self, name: str, which: str = 'min') -> Tuple[bool, float, str]:
        """
        测试变量边界是否可行
        
        对于有公式约束的变量，会自动调整相关变量以最大化/最小化边界可达性
        
        Args:
            name: 变量名
            which: 'min' 或 'max'
        
        Returns:
            (是否可行, 实际可达值, 信息)
        """
        self._initialize_values()
        
        var = next((v for v in self.variables if v['name'] == name), None)
        if not var:
            return False, 0, "变量不存在"
        
        # 优化所有相关变量，使目标边界可达性最大化
        self._optimize_vars_for_boundary(name, which)
        
        # 计算实际边界
        resolved_bounds = self.constraint.resolve_bounds(self.current_values)
        lb, ub = resolved_bounds.get(name, (0, 1))
        target = lb if which == 'min' else ub
        
        # 设置目标值
        self.current_values[name] = target
        
        # 调整受影响的变量
        self._adjust_all_dependent_vars()
        
        # 检查约束
        valid, msg = self.constraint.check_constraints(self.current_values)
        
        if valid:
            actual = self.current_values[name]
            return True, actual, ""
        
        # 约束不满足，搜索最接近的可行值
        self._initialize_values()
        feasible_value = self._find_feasible_boundary(name, which)
        
        if feasible_value is not None:
            return False, feasible_value, f"无法达到理论边界 {target:.4f}，最接近可行值 {feasible_value:.4f}"
        else:
            return False, 0, f"约束冲突: {msg}"
    
    def _optimize_vars_for_boundary(self, name: str, which: str):
        """
        优化所有相关变量，使目标变量的边界可达性最大化
        
        两种情况：
        1. 目标变量有公式边界 -> 优化公式中涉及的变量
        2. 其他变量的公式边界约束了目标变量 -> 优化那些公式中涉及的变量
        """
        var = next((v for v in self.variables if v['name'] == name), None)
        if not var:
            return
        
        bounds = var.get('bounds', [0, 1])
        has_formula_bound = (which == 'min' and isinstance(bounds[0], str)) or \
                           (which == 'max' and isinstance(bounds[1], str))
        
        # 收集所有需要优化的变量
        vars_to_optimize = set()
        
        # 情况1：目标变量有公式边界
        if has_formula_bound:
            deps = self.constraint.dependencies.get(name, set())
            vars_to_optimize.update(deps)
        
        # 情况2：其他变量的公式边界约束了目标变量
        for other_var in self.variables:
            other_name = other_var['name']
            if other_name == name:
                continue
            
            other_bounds = other_var.get('bounds', [0, 1])
            lb_formula = other_bounds[0] if isinstance(other_bounds[0], str) else None
            ub_formula = other_bounds[1] if isinstance(other_bounds[1], str) else None
            
            # 检查公式是否包含目标变量
            for formula in [lb_formula, ub_formula]:
                if formula and name in formula:
                    # 这个变量的公式边界涉及目标变量
                    other_deps = self.constraint.dependencies.get(other_name, set())
                    vars_to_optimize.update(other_deps)
        
        # 按拓扑顺序优化每个变量
        for var_name in self.constraint.eval_order:
            if var_name not in vars_to_optimize:
                continue
            
            var_config = next((v for v in self.variables if v['name'] == var_name), None)
            if not var_config:
                continue
            
            var_bounds = var_config.get('bounds', [0, 1])
            if isinstance(var_bounds[0], str) or isinstance(var_bounds[1], str):
                continue  # 本身有公式边界，跳过
            
            # 尝试不同值，选择使目标变量边界最优的
            test_values = [var_bounds[0], var_bounds[1]]
            best_value = self.current_values[var_name]
            best_target_bound = None
            
            for test_val in test_values:
                self.current_values[var_name] = test_val
                resolved = self.constraint.resolve_bounds(self.current_values)
                target_lb, target_ub = resolved.get(name, (0, 1))
                target_bound = target_lb if which == 'min' else target_ub
                
                if best_target_bound is None:
                    best_target_bound = target_bound
                    best_value = test_val
                elif which == 'min':
                    if target_bound < best_target_bound:
                        best_target_bound = target_bound
                        best_value = test_val
                else:
                    if target_bound > best_target_bound:
                        best_target_bound = target_bound
                        best_value = test_val
            
            self.current_values[var_name] = best_value
    
    def _optimize_deps_for_boundary(self, name: str, which: str):
        """
        优化依赖变量，使目标变量的边界可达性最大化
        
        例如：Lp >= Lm + Lm2 + 2
        要让 Lm2 最大，需要 Lm 最小
        """
        deps = self.constraint.dependencies.get(name, set())
        if not deps:
            return
        
        # 分析公式，确定依赖变量应该取最大值还是最小值
        var = next((v for v in self.variables if v['name'] == name), None)
        if not var:
            return
        
        bounds = var.get('bounds', [0, 1])
        formula = bounds[0] if which == 'min' else bounds[1]
        
        if not isinstance(formula, str):
            return
        
        # 简化处理：对于 min 边界公式，让依赖变量取值使公式结果最小
        # 对于 max 边界公式，让依赖变量取值使公式结果最大
        
        # 策略：对每个依赖变量，尝试将其设置到使目标最优的位置
        dep_vars = list(deps)
        
        for dep_name in dep_vars:
            dep_var = next((v for v in self.variables if v['name'] == dep_name), None)
            if not dep_var:
                continue
            
            dep_bounds = dep_var.get('bounds', [0, 1])
            if isinstance(dep_bounds[0], str) or isinstance(dep_bounds[1], str):
                # 依赖变量本身有公式边界，跳过
                continue
            
            # 尝试两种边界值，选择使目标变量边界更优的那个
            test_values = [dep_bounds[0], dep_bounds[1], (dep_bounds[0] + dep_bounds[1]) / 2]
            best_value = test_values[2]
            best_target_bound = None
            
            for test_val in test_values:
                self.current_values[dep_name] = test_val
                resolved = self.constraint.resolve_bounds(self.current_values)
                target_lb, target_ub = resolved.get(name, (0, 1))
                target_bound = target_lb if which == 'min' else target_ub
                
                if best_target_bound is None:
                    best_target_bound = target_bound
                    best_value = test_val
                elif which == 'min':
                    # 希望 min 边界尽可能小
                    if target_bound < best_target_bound:
                        best_target_bound = target_bound
                        best_value = test_val
                else:
                    # 希望 max 边界尽可能大
                    if target_bound > best_target_bound:
                        best_target_bound = target_bound
                        best_value = test_val
            
            self.current_values[dep_name] = best_value
    
    def _find_feasible_boundary(self, name: str, which: str) -> Optional[float]:
        """二分搜索找到最接近边界的可行值"""
        bounds = self.constraint.resolve_bounds(self.current_values)
        lb, ub = bounds.get(name, (0, 1))
        
        if lb > ub:
            return None
        
        if which == 'min':
            # 从 lb 开始向上搜索
            low, high = lb, ub
            best = None
            for _ in range(20):  # 二分搜索20次
                mid = (low + high) / 2
                self.current_values[name] = mid
                adjusted = self._adjust_dependent_vars(name)
                valid, _ = self.constraint.check_constraints(self.current_values)
                if valid:
                    best = mid
                    high = mid  # 继续向下找更小的可行值
                else:
                    low = mid
            return best
        else:
            # 从 ub 开始向下搜索
            low, high = lb, ub
            best = None
            for _ in range(20):
                mid = (low + high) / 2
                self.current_values[name] = mid
                adjusted = self._adjust_dependent_vars(name)
                valid, _ = self.constraint.check_constraints(self.current_values)
                if valid:
                    best = mid
                    low = mid  # 继续向上找更大的可行值
                else:
                    high = mid
            return best
    
    def get_batch_values_for_boundary(self, name: str, which: str = 'min') -> Optional[Dict[str, float]]:
        """
        获取设置变量到边界时的所有变量值（批量设置用）
        
        返回满足约束的完整参数集，或 None 如果不可行
        """
        self._initialize_values()
        
        success, actual_value, msg = self.test_boundary_feasibility(name, which)
        
        if success or actual_value > 0:
            return self.get_all_values()
        
        return None
