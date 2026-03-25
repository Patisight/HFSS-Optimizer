"""
目标评估器模块
支持多种目标类型的评估：S参数、增益、阻抗等
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ObjectiveResult:
    """目标评估结果"""
    name: str
    value: float  # 目标函数值（用于优化）
    actual_value: float  # 实际物理值
    goal_met: Optional[bool] = None  # 是否达标
    uncertainty: float = 0.0  # 不确定性（代理模型）
    is_surrogate: bool = False  # 是否来自代理模型


class ObjectiveEvaluator:
    """目标评估器"""
    
    def __init__(self, objectives_config: List[Dict], hfss_controller, output_dir: str = None):
        """
        初始化评估器
        
        Args:
            objectives_config: 目标配置列表
            hfss_controller: HFSS 控制器实例
            output_dir: 输出目录，用于保存每次仿真数据
        """
        self.objectives = objectives_config
        self.hfss = hfss_controller
        self._s_data_cache = None
        
        # 仿真数据保存
        self.output_dir = output_dir
        self.eval_count = 0
        
        # 创建保存文件
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.eval_file = os.path.join(output_dir, 'evaluations.jsonl')
            # 清空或创建文件
            with open(self.eval_file, 'w') as f:
                pass  # 创建空文件
            print(f"[OK] Evaluation log: {self.eval_file}")
        else:
            self.eval_file = None
    
    def set_output_dir(self, output_dir: str):
        """设置输出目录"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.eval_file = os.path.join(output_dir, 'evaluations.jsonl')
        with open(self.eval_file, 'w') as f:
            pass
        print(f"[OK] Evaluation log: {self.eval_file}")
    
    def _save_evaluation(self, params: np.ndarray, results: Dict[str, ObjectiveResult]):
        """保存每次评估的详细数据"""
        if not self.eval_file:
            return
        
        self.eval_count += 1
        
        # 构建保存数据
        data = {
            'eval_id': self.eval_count,
            'timestamp': datetime.now().isoformat(),
            'parameters': params.tolist() if hasattr(params, 'tolist') else list(params),
            'objectives': {}
        }
        
        # 添加每个目标的详细信息
        for name, result in results.items():
            # 确保 goal_met 是 Python 原生类型（bool/None），不是 numpy.bool_
            goal_met = result.goal_met
            if goal_met is not None and hasattr(goal_met, 'item'):
                goal_met = goal_met.item()
            
            data['objectives'][name] = {
                'value': float(result.value),
                'actual_value': float(result.actual_value),
                'goal_met': goal_met
            }
        
        # 追加到文件
        with open(self.eval_file, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"[SAVED] Eval #{self.eval_count} -> {self.eval_file}")
    
    def clear_cache(self):
        """清除缓存"""
        self._s_data_cache = None
    
    def evaluate_all(self, params: np.ndarray) -> Tuple[List[float], Dict[str, ObjectiveResult]]:
        """
        评估所有目标
        
        Args:
            params: 参数向量
            
        Returns:
            (目标值列表, 结果字典)
        """
        obj_values = []
        results = {}
        
        for obj in self.objectives:
            result = self._evaluate_single(obj, params)
            obj_values.append(result.value)
            results[result.name] = result
        
        # 保存每次评估的数据
        self._save_evaluation(params, results)
        
        return obj_values, results
    
    def _evaluate_single(self, obj: Dict, params: np.ndarray) -> ObjectiveResult:
        """评估单个目标"""
        obj_type = obj['type']
        name = obj.get('name', obj_type)
        
        try:
            if obj_type in ['s_mag', 's_phase', 's_db']:
                value, actual = self._evaluate_s_parameter(obj)
            elif obj_type == 'gain':
                value, actual = self._evaluate_gain(obj)
            elif obj_type == 'peak_gain':
                value, actual = self._evaluate_peak_gain(obj)
            elif obj_type == 'z_real':
                value, actual = self._evaluate_z_real(obj)
            elif obj_type == 'z_imag':
                value, actual = self._evaluate_z_imag(obj)
            else:
                value, actual = 1000.0, 1000.0
            
            goal_met = self._check_goal(obj, value)
            
            return ObjectiveResult(
                name=name,
                value=value,
                actual_value=actual,
                goal_met=goal_met,
            )
            
        except Exception as e:
            print(f"[WARN] Evaluate {name}: {e}")
            return ObjectiveResult(
                name=name,
                value=1000.0,
                actual_value=1000.0,
                goal_met=False,
            )
    
    def _evaluate_s_parameter(self, obj: Dict) -> Tuple[float, float]:
        """评估 S 参数"""
        # 获取 S 参数数据
        if self._s_data_cache is None:
            # 确保 port 是元组（可哈希）
            ports = []
            for o in self.objectives:  # 注意：使用 o 而不是 obj，避免覆盖参数
                if o['type'] in ['s_mag', 's_phase', 's_db']:
                    port = o.get('port', (1, 1))
                    # 如果 port 是列表，转换为元组
                    if isinstance(port, list):
                        port = tuple(port)
                    ports.append(port)
            ports = list(set(ports))
            self._s_data_cache = self.hfss.get_s_parameters(ports)
        
        if self._s_data_cache is None:
            return 1000.0, 1000.0
        
        port = obj.get('port', (1, 1))
        # 如果 port 是列表，转换为元组
        if isinstance(port, list):
            port = tuple(port)
        
        if port not in self._s_data_cache['ports']:
            return 1000.0, 1000.0
        
        freq = self._s_data_cache['freq']
        port_data = self._s_data_cache['ports'][port]
        
        # 确定频率范围
        if 'freq_range' in obj and obj['freq_range']:
            f_min, f_max = obj['freq_range']
            # 如果是列表，转换为元组
            if isinstance(f_min, list):
                f_min, f_max = f_min[0], f_min[1]
            mask = (freq >= f_min) & (freq <= f_max)
        elif 'freq' in obj and obj['freq']:
            idx = np.argmin(np.abs(freq - obj['freq']))
            mask = np.zeros(len(freq), dtype=bool)
            mask[idx] = True
        else:
            mask = np.ones(len(freq), dtype=bool)
        
        # 获取值
        obj_type = obj['type']
        if obj_type == 's_mag':
            values = port_data['mag'][mask]
        elif obj_type == 's_phase':
            values = port_data['phase'][mask]
        else:  # s_db
            values = port_data['db'][mask]
        
        if len(values) == 0:
            return 1000.0, 1000.0
        
        # 获取什么就是什么，不做任何转换
        # HFSS 的 dB(S(1,1)) 返回什么值就用什么值
        
        # 约束类型
        constraint = obj.get('constraint', 'value')
        if constraint == 'max':
            actual = np.max(values)
        elif constraint == 'min':
            actual = np.min(values)
        elif constraint == 'mean':
            actual = np.mean(values)
        else:
            actual = values[0]
        
        # 转换为目标函数值
        value = self._actual_to_objective(actual, obj)
        
        return value, actual
    
    def _evaluate_gain(self, obj: Dict) -> Tuple[float, float]:
        """评估增益"""
        freq = obj.get('freq', 5.9)
        actual = self.hfss.get_gain(freq)
        
        if actual is None:
            # 增益获取失败，返回惩罚值
            print(f"[WARN] Gain not available, using penalty value")
            return 1000.0, 1000.0
        
        value = self._actual_to_objective(actual, obj)
        return value, actual
    
    def _evaluate_peak_gain(self, obj: Dict) -> Tuple[float, float]:
        """
        评估 PeakGain - 使用 antenna_parameters 报告
        
        注意: 对于 Interpolating Sweep，只能在 Setup 频率点获取 PeakGain
        如果目标频率不在 Setup 中，会自动更新 Setup 频率
        """
        target_freq = obj.get('freq', 4.0)  # 目标频率 (GHz)
        
        # 检查并确保 Setup 频率匹配
        current_freq = self.hfss.get_setup_frequency()
        
        if abs(current_freq - target_freq) > 0.01:  # 10 MHz tolerance
            print(f"[INFO] Setup frequency ({current_freq:.2f} GHz) != Target ({target_freq:.2f} GHz)")
            print(f"[INFO] Updating Setup frequency to {target_freq:.2f} GHz...")
            
            # 更新 Setup 频率
            if self.hfss.ensure_setup_frequency(target_freq):
                # 清除缓存，需要重新仿真
                self._s_data_cache = None
                print(f"[OK] Setup frequency updated - re-simulation required")
            else:
                print(f"[WARN] Failed to update Setup frequency")
        
        # 获取增益
        actual = self.hfss.get_gain(target_freq)
        
        if actual is None:
            print(f"[WARN] PeakGain not available, using penalty value")
            return 1000.0, 1000.0
        
        value = self._actual_to_objective(actual, obj)
        
        return value, actual
    
    def _evaluate_z_real(self, obj: Dict) -> Tuple[float, float]:
        """评估阻抗实部"""
        z_data = self.hfss.get_z_parameters()
        if z_data is None:
            return 1000.0, 1000.0
        
        freq = z_data['freq']
        z_real = z_data['z_real']
        
        target_freq = obj.get('freq', 5.9)
        idx = np.argmin(np.abs(freq - target_freq))
        actual = z_real[idx]
        
        # 目标是接近某值
        target_val = obj.get('value', 50.0)
        tolerance = obj.get('tolerance', 10.0)
        
        diff = abs(actual - target_val)
        value = max(0, diff - tolerance)
        
        return value, actual
    
    def _evaluate_z_imag(self, obj: Dict) -> Tuple[float, float]:
        """评估阻抗虚部"""
        z_data = self.hfss.get_z_parameters()
        if z_data is None:
            return 1000.0, 1000.0
        
        freq = z_data['freq']
        z_imag = z_data['z_imag']
        
        target_freq = obj.get('freq', 5.9)
        idx = np.argmin(np.abs(freq - target_freq))
        actual = z_imag[idx]
        
        target_val = obj.get('value', 0.0)
        tolerance = obj.get('tolerance', 5.0)
        
        diff = abs(actual - target_val)
        value = max(0, diff - tolerance)
        
        return value, actual
    
    def _actual_to_objective(self, actual: float, obj: Dict) -> float:
        """将实际值转换为目标函数值"""
        target = obj.get('target', 'minimize')
        
        if target == 'minimize':
            return actual
        elif target == 'maximize':
            return -actual
        elif target == 'range':
            r_min, r_max = obj['range']
            if r_min <= actual <= r_max:
                return 0
            else:
                return min(abs(actual - r_min), abs(actual - r_max))
        elif target == 'target':
            return actual
        
        return actual
    
    def _check_goal(self, obj: Dict, obj_value: float) -> Optional[bool]:
        """检查是否达到目标"""
        if 'goal' not in obj:
            return None
        
        goal = obj['goal']
        target = obj.get('target', 'minimize')
        
        if target == 'minimize':
            return obj_value <= goal
        elif target == 'maximize':
            return -obj_value >= goal
        elif target == 'range':
            return obj_value == 0
        
        return None


def format_results(results: Dict[str, ObjectiveResult], 
                   objectives_config: List[Dict]) -> str:
    """
    格式化结果为字符串
    
    Args:
        results: 评估结果字典
        objectives_config: 目标配置
        
    Returns:
        格式化的字符串
    """
    parts = []
    for obj in objectives_config:
        name = obj.get('name', obj['type'])
        if name in results:
            res = results[name]
            s = '✅' if res.goal_met else '❌' if res.goal_met is not None else ''
            parts.append(f"{name}={res.actual_value:.2f}{s}")
    return ', '.join(parts)