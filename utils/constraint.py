"""
变量约束模块
支持公式边界和约束检查
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FormulaEvaluator:
    """公式求值器，支持基本运算和数学函数"""

    ALLOWED_NAMES = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        self._compiled_cache = {}

    def eval(self, formula: str, variables: Dict[str, float]) -> float:
        """
        求值公式

        Args:
            formula: 公式字符串，如 "(Wc-Wm2)/2+0.01"
            variables: 变量值字典，如 {"Wc": 4.5, "Wm2": 0.8}

        Returns:
            计算结果
        """
        if not isinstance(formula, str):
            return float(formula)

        formula = formula.strip()
        if not formula:
            return 0.0

        # 确保所有变量值都是float类型
        numeric_vars = {}
        for k, v in variables.items():
            if isinstance(v, str):
                try:
                    numeric_vars[k] = float(v)
                except ValueError:
                    numeric_vars[k] = 0.0
            else:
                numeric_vars[k] = float(v)

        cache_key = (formula, tuple(sorted(numeric_vars.keys())))
        if cache_key in self._compiled_cache:
            code = self._compiled_cache[cache_key]
        else:
            code = self._compile(formula, numeric_vars)
            self._compiled_cache[cache_key] = code

        try:
            return float(eval(code, {"__builtins__": {}}, {**self.ALLOWED_NAMES, **numeric_vars}))
        except Exception as e:
            raise ValueError(f"Formula error '{formula}': {e}")

    def _compile(self, formula: str, variables: Dict[str, float]) -> str:
        """编译公式，替换变量名为安全表达式"""
        expr = formula

        for var_name in sorted(variables.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(var_name) + r"\b"
            expr = re.sub(pattern, var_name, expr)

        return expr

    def extract_variables(self, formula: str) -> List[str]:
        """提取公式中引用的变量名"""
        if not isinstance(formula, str):
            return []

        names = set()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", formula):
            name = match.group(1)
            if name not in self.ALLOWED_NAMES:
                names.add(name)

        return sorted(list(names))

    def is_formula(self, value) -> bool:
        """判断值是否为公式"""
        return isinstance(value, str) and value.strip() != ""

    def resolve_value(self, value, variables: Dict[str, float]) -> float:
        """解析值（数字直接返回，公式则求值）"""
        if self.is_formula(value):
            return self.eval(value, variables)
        return float(value)


class VariableConstraint:
    """变量约束管理器"""

    def __init__(self, variables_config: List[Dict]):
        """
        初始化约束管理器

        Args:
            variables_config: 变量配置列表
        """
        self.variables = variables_config
        self.evaluator = FormulaEvaluator()
        self._build_dependency_graph()

    def _build_dependency_graph(self):
        """构建变量依赖图，确定计算顺序"""
        self.dependencies = {}

        for var in self.variables:
            name = var["name"]
            deps = set()

            lb, ub = var.get("bounds", [0, 1])
            if self.evaluator.is_formula(lb):
                deps.update(self.evaluator.extract_variables(lb))
            if self.evaluator.is_formula(ub):
                deps.update(self.evaluator.extract_variables(ub))

            self.dependencies[name] = deps

        self._compute_order()

    def _compute_order(self):
        """拓扑排序确定变量生成顺序"""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies.get(name, set()):
                visit(dep)
            order.append(name)

        for var in self.variables:
            visit(var["name"])

        self.eval_order = order

    def get_independent_vars(self) -> List[str]:
        """获取无依赖的变量（可自由生成）"""
        return [name for name in self.eval_order if not self.dependencies.get(name, set())]

    def get_dependent_vars(self) -> List[str]:
        """获取有依赖的变量"""
        return [name for name in self.eval_order if self.dependencies.get(name, set())]

    def resolve_bounds(self, params_dict: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        动态计算所有变量的边界

        Args:
            params_dict: 已生成变量的值

        Returns:
            {变量名: (下限, 上限)}
        """
        bounds = {}

        for name in self.eval_order:
            var = next((v for v in self.variables if v["name"] == name), None)
            if not var:
                continue

            lb, ub = var.get("bounds", [0, 1])

            # 检查公式依赖的变量是否都有值
            deps = self.dependencies.get(name, set())
            missing_deps = [d for d in deps if d not in params_dict]

            if missing_deps:
                # 依赖变量还没有值，使用默认边界
                if isinstance(lb, str):
                    lb = 0.5
                if isinstance(ub, str):
                    ub = 1.5
                bounds[name] = (float(lb), float(ub))
            else:
                # 可以计算实际边界
                resolved_lb = self.evaluator.resolve_value(lb, params_dict)
                resolved_ub = self.evaluator.resolve_value(ub, params_dict)
                bounds[name] = (resolved_lb, resolved_ub)

        return bounds

    def check_constraints(self, params_dict: Dict[str, float]) -> Tuple[bool, str]:
        """
        检查变量值是否满足约束

        Args:
            params_dict: 变量值

        Returns:
            (是否满足, 错误信息)
        """
        for name in self.eval_order:
            if name not in params_dict:
                continue

            var = next((v for v in self.variables if v["name"] == name), None)
            if not var:
                continue

            value = params_dict[name]
            # 确保value是float类型
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue  # 如果转换失败，跳过此变量

            lb, ub = var.get("bounds", [0, 1])

            resolved_lb = self.evaluator.resolve_value(lb, params_dict)
            resolved_ub = self.evaluator.resolve_value(ub, params_dict)

            if resolved_lb > resolved_ub:
                return False, f"{name}: lower bound ({resolved_lb:.4f}) > upper bound ({resolved_ub:.4f})"

            if value < resolved_lb:
                return False, f"{name}: {value:.4f} < lower bound ({resolved_lb:.4f})"

            if value > resolved_ub:
                return False, f"{name}: {value:.4f} > upper bound ({resolved_ub:.4f})"

        return True, ""

    def repair_params(self, params_dict: Dict[str, float]) -> Dict[str, float]:
        """
        修复违反约束的参数

        Args:
            params_dict: 原始参数

        Returns:
            修复后的参数
        """
        repaired = params_dict.copy()

        for name in self.eval_order:
            if name not in repaired:
                continue

            var = next((v for v in self.variables if v["name"] == name), None)
            if not var:
                continue

            value = repaired[name]
            # 确保value是float类型
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    value = 0.0

            lb, ub = var.get("bounds", [0, 1])

            resolved_lb = self.evaluator.resolve_value(lb, repaired)
            resolved_ub = self.evaluator.resolve_value(ub, repaired)

            if resolved_lb > resolved_ub:
                repaired[name] = (resolved_lb + resolved_ub) / 2
            else:
                repaired[name] = max(resolved_lb, min(value, resolved_ub))

        return repaired

    def has_formulas(self) -> bool:
        """检查是否有公式边界"""
        for var in self.variables:
            lb, ub = var.get("bounds", [0, 1])
            if self.evaluator.is_formula(lb) or self.evaluator.is_formula(ub):
                return True
        return False

    def get_dependency_info(self) -> Dict[str, Dict]:
        """获取依赖信息（用于GUI显示）"""
        info = {}
        for var in self.variables:
            name = var["name"]
            lb, ub = var.get("bounds", [0, 1])

            deps = set()
            if self.evaluator.is_formula(lb):
                deps.update(self.evaluator.extract_variables(lb))
            if self.evaluator.is_formula(ub):
                deps.update(self.evaluator.extract_variables(ub))

            info[name] = {
                "has_formula": bool(deps),
                "dependencies": sorted(list(deps)),
                "lb": lb,
                "ub": ub,
            }

        return info
