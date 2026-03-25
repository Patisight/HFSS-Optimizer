"""
多目标粒子群优化算法 (MOPSO)

基于 Pareto 支配的多目标粒子群优化，适合：
- 多维多变量优化
- 多目标同时优化
- 需要较快收敛的场景

特点：
1. 粒子群协作搜索，收敛快
2. 外部档案存储 Pareto 前沿
3. 拥挤距离保持多样性
4. 可选代理模型加速（减少仿真次数）

参考文献：
- Coello Coello, C.A., et al. (2004). "Handling multiple objectives with particle swarm optimization"
- Reyes-Sierra, M., & Coello Coello, C.A. (2006). "Multi-Objective Particle Swarm Optimizers"
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import random
import json
import os
import sys
from .base import BaseOptimizer

# 导入代理模型模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.surrogate import SurrogateManager


class MOPSO(BaseOptimizer):
    """
    多目标粒子群优化器 (MOPSO)
    
    参数：
    - population_size: 种群大小，建议变量数×4~10
    - n_generations: 迭代代数，建议20-50
    - w: 惯性权重，建议0.4-0.9
    - c1: 认知学习因子，建议1.5-2.0
    - c2: 社会学习因子，建议1.5-2.0
    - use_surrogate: 是否使用代理模型加速
    - surrogate_type: 代理模型类型 ('gp' 或 'rf')
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # PSO 参数
        self.population_size = config.get('population_size', 50)
        self.n_generations = config.get('n_generations', 30)
        self.w = config.get('inertia_weight', 0.5)  # 惯性权重
        self.c1 = config.get('c1', 1.5)  # 认知学习因子
        self.c2 = config.get('c2', 1.5)  # 社会学习因子
        
        # 自适应惯性权重
        self.w_max = config.get('w_max', 0.9)
        self.w_min = config.get('w_min', 0.4)
        
        # 代理模型设置
        self.use_surrogate = config.get('use_surrogate', False)
        self.surrogate_type = config.get('surrogate_type', 'gp')
        self.surrogate_min_samples = config.get('surrogate_min_samples', 5)
        self.surrogate_threshold = config.get('surrogate_threshold', 1.0)
        self.surrogate_manager = None
        self.surrogate_eval_count = 0  # 代理模型评估次数
        self.real_eval_count = 0  # 真实仿真次数
        
        # 加载历史评估数据
        self.load_evaluations_path = config.get('load_evaluations', None)
        self.loaded_evaluations = []  # 加载的历史数据
        
        # 外部档案（Pareto 前沿）
        self.archive = []
        self.max_archive_size = config.get('max_archive_size', 100)
        
        # 变量边界（在 run 中初始化）
        self.bounds = None
        
        # 粒子
        self.particles = []
        self.velocities = []
        self.pbest = []  # 个体最优
        self.pbest_objectives = []
        
        # 统计
        self.evaluation_count = 0
        self.real_evaluation_count = 0
        
        # 回调
        self.callback = None
    
    def run(self, evaluator, callback: Callable = None) -> List[Dict]:
        """运行 MOPSO 优化"""
        self.callback = callback
        self.evaluator = evaluator  # 保存引用，用于写入历史数据
        
        # 初始化变量和目标数量
        self.n_variables = len(self.variables)
        self.n_objectives = len(self.objectives)
        
        n_vars = self.n_variables
        n_objs = self.n_objectives
        
        # 变量边界
        self.bounds = self.get_bounds()
        
        print(f"\n{'='*60}")
        print("MULTI-OBJECTIVE PARTICLE SWARM OPTIMIZATION (MOPSO)")
        print(f"{'='*60}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Inertia weight: {self.w_min}-{self.w_max} (adaptive)")
        print(f"Learning factors: c1={self.c1}, c2={self.c2}")
        print(f"Surrogate: {'Enabled (' + self.surrogate_type + ')' if self.use_surrogate else 'Disabled'}")
        if self.load_evaluations_path:
            print(f"Load evaluations: {self.load_evaluations_path}")
        print(f"{'='*60}")
        
        # 加载历史评估数据
        n_loaded = self._load_historical_data()
        
        # 初始化代理模型管理器
        if self.use_surrogate:
            # 使用用户配置的最少样本数，或默认为种群大小
            min_samples = self.surrogate_min_samples if self.surrogate_min_samples > 0 else self.population_size
            self.surrogate_manager = SurrogateManager(
                n_objectives=self.n_objectives,
                model_type=self.surrogate_type,
                min_samples=min_samples
            )
            print(f"[INFO] Surrogate model initialized: {self.surrogate_type}")
            print(f"[INFO] Min samples to train: {min_samples}")
            print(f"[INFO] Uncertainty threshold: {self.surrogate_threshold}")
            
            # 用历史数据训练代理模型
            if self.loaded_evaluations:
                for eval_data in self.loaded_evaluations:
                    self.surrogate_manager.add_sample(eval_data['params'], eval_data['objectives'])
                print(f"[INFO] Surrogate model trained with {len(self.loaded_evaluations)} historical samples")
                
                # 更新真实仿真计数（历史数据也算真实仿真）
                self.real_evaluation_count = len(self.loaded_evaluations)
        
        # 初始化粒子群
        self._initialize_particles(n_vars)
        
        # 初始评估（强制真实仿真以建立代理模型训练数据）
        print(f"\n[Initialization] Evaluating {self.population_size} particles...")
        for i, particle in enumerate(self.particles):
            y, is_real = self._evaluate(particle, evaluator, force_real=True)
            self.pbest[i] = particle.copy()
            self.pbest_objectives[i] = y.copy()
            
            # 更新外部档案（只用真实仿真结果）
            self._update_archive(particle, y)
            
            # 回调更新图表（只用真实仿真结果）
            if self.callback:
                self.callback(i, self.population_size, particle, y, 'initial')
        
        # 迭代优化
        for gen in range(self.n_generations):
            # 自适应惯性权重
            w = self.w_max - (self.w_max - self.w_min) * gen / self.n_generations
            
            print(f"\n[Generation {gen + 1}/{self.n_generations}]")
            print(f"  Archive size: {len(self.archive)}, Inertia: {w:.3f}")
            
            for i in range(self.population_size):
                # 从档案中选择全局最优
                gbest = self._select_gbest()
                
                # 更新速度
                r1, r2 = np.random.random(n_vars), np.random.random(n_vars)
                self.velocities[i] = (w * self.velocities[i] + 
                                      self.c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                      self.c2 * r2 * (gbest - self.particles[i]))
                
                # 限制速度
                max_velocity = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
                self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
                
                # 更新位置
                self.particles[i] = self.particles[i] + self.velocities[i]
                self.particles[i] = self.clip_to_bounds(self.particles[i])
            
            # 评估新位置
            for i in range(self.population_size):
                y, is_real = self._evaluate(self.particles[i], evaluator)
                
                # PSO 迭代：总是用返回的目标值更新个体最优
                if self._dominates(y, self.pbest_objectives[i]):
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_objectives[i] = y.copy()
                elif np.random.random() < 0.5:  # 非支配解有一定概率更新
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_objectives[i] = y.copy()
                
                # 更新外部档案（标记数据来源）
                self._update_archive(self.particles[i], y, is_predicted=not is_real)
                
                # 回调更新图表（只用真实仿真结果）
                if is_real and self.callback:
                    self.callback(gen, self.n_generations, self.particles[i], y, 'iteration')
        
        # 返回 Pareto 前沿
        return self._get_pareto_solutions()
    
    def _load_historical_data(self) -> int:
        """
        加载历史评估数据
        
        从 evaluations.jsonl 文件加载之前的仿真结果，用于：
        1. 初始化代理模型训练数据
        2. 初始化 Pareto 档案
        3. 避免重复评估已知的参数点
        
        Returns:
            加载的数据点数量
        """
        if not self.load_evaluations_path:
            return 0
        
        if not os.path.exists(self.load_evaluations_path):
            print(f"[WARN] Evaluations file not found: {self.load_evaluations_path}")
            return 0
        
        loaded_count = 0
        self.loaded_evaluations = []
        
        try:
            with open(self.load_evaluations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # 提取参数
                        params = np.array(data['parameters'])
                        
                        # 提取目标值
                        objectives_data = data.get('objectives', {})
                        if isinstance(objectives_data, dict):
                            # 字典格式：{name: {value: ..., actual_value: ...}}
                            obj_names = list(objectives_data.keys())
                            obj_values = [objectives_data[name]['value'] for name in obj_names]
                            objectives = np.array(obj_values)
                        elif isinstance(objectives_data, list):
                            # 列表格式
                            objectives = np.array(objectives_data)
                        else:
                            continue
                        
                        # 检查数据有效性
                        if len(params) != self.n_variables:
                            print(f"[WARN] Skip record: param count mismatch ({len(params)} vs {self.n_variables})")
                            continue
                        
                        if len(objectives) != self.n_objectives:
                            print(f"[WARN] Skip record: objective count mismatch ({len(objectives)} vs {self.n_objectives})")
                            continue
                        
                        # 存储加载的数据
                        self.loaded_evaluations.append({
                            'params': params,
                            'objectives': objectives
                        })
                        
                        # 添加到缓存
                        cache_key = tuple(round(p, 4) for p in params)
                        self._cache[cache_key] = objectives.copy()
                        
                        loaded_count += 1
                        
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        print(f"[WARN] Skip invalid line: {e}")
                        continue
            
            print(f"[OK] Loaded {loaded_count} historical evaluations from: {self.load_evaluations_path}")
            
            # 用历史数据初始化 Pareto 档案
            if self.loaded_evaluations:
                self._init_archive_from_history()
                
                # 把历史数据写入新的 evaluations 文件
                self._write_history_to_output()
            
            return loaded_count
            
        except Exception as e:
            print(f"[ERROR] Failed to load evaluations: {e}")
            return 0
    
    def _init_archive_from_history(self):
        """从历史数据初始化 Pareto 档案"""
        for eval_data in self.loaded_evaluations:
            x = eval_data['params']
            y = eval_data['objectives']
            self._update_archive(x, y)
        
        print(f"[OK] Initialized Pareto archive with {len(self.archive)} solutions from history")
    
    def _write_history_to_output(self):
        """把历史数据写入新的输出目录的 evaluations 文件"""
        if not self.evaluator or not self.evaluator.eval_file:
            return
        
        if not self.load_evaluations_path:
            return
        
        try:
            # 直接复制原始文件内容到新的 evaluations 文件
            with open(self.load_evaluations_path, 'r', encoding='utf-8') as src:
                with open(self.evaluator.eval_file, 'a', encoding='utf-8') as dst:
                    for line in src:
                        line = line.strip()
                        if line:
                            dst.write(line + '\n')
            
            # 更新 evaluator 的计数
            self.evaluator.eval_count = len(self.loaded_evaluations)
            print(f"[OK] Historical data written to: {self.evaluator.eval_file}")
            
        except Exception as e:
            print(f"[WARN] Failed to write historical data: {e}")
    
    def _initialize_particles(self, n_vars: int):
        """初始化粒子群"""
        self.particles = []
        self.velocities = []
        self.pbest = []
        self.pbest_objectives = []
        
        # 拉丁超立方采样
        for i in range(self.population_size):
            particle = np.array([
                np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])
                for j in range(n_vars)
            ])
            self.particles.append(particle)
            
            velocity = np.array([
                np.random.uniform(-0.1, 0.1) * (self.bounds[j, 1] - self.bounds[j, 0])
                for j in range(n_vars)
            ])
            self.velocities.append(velocity)
            
            self.pbest.append(particle.copy())
            self.pbest_objectives.append(None)
    
    def _evaluate(self, x: np.ndarray, evaluator, force_real: bool = False) -> Tuple[np.ndarray, bool]:
        """
        评估粒子
        
        简化逻辑：
        - 不确定性 ≥ 阈值 → 真实仿真，用真实值迭代
        - 不确定性 < 阈值 → 跳过仿真，用预测值迭代
        
        Args:
            x: 参数向量
            evaluator: 评估器
            force_real: 是否强制真实仿真
        
        Returns:
            (目标值向量, 是否为真实仿真结果)
        """
        self.evaluation_count += 1
        
        # 检查缓存（只缓存真实仿真结果）
        cached = self._check_cache(x)
        if cached is not None:
            return cached, True
        
        # 判断是否使用代理模型
        if self.use_surrogate and self.surrogate_manager and not force_real:
            # 初始阶段（前 min_samples 次）用真实仿真
            min_samples = self.surrogate_manager.min_samples_to_train
            if self.real_evaluation_count >= min_samples:
                # 获取代理预测和不确定性
                y_pred, y_std = self.surrogate_manager.predict(x.reshape(1, -1), return_std=True)
                y_pred = y_pred.flatten()
                y_std = y_std.flatten()
                
                # 计算相对不确定性（更直观）
                # 公式：y_std / |y_pred|
                # 表示预测的标准差占预测值大小的比例
                normalized_uncertainty = np.mean(y_std / (np.abs(y_pred) + 1e-8))
                
                # 不确定性低于阈值 → 使用预测值（跳过真实仿真）
                if normalized_uncertainty < self.surrogate_threshold:
                    self.surrogate_eval_count += 1
                    print(f"  [Surrogate] Eval #{self.evaluation_count} (uncertainty: {normalized_uncertainty:.3f} < {self.surrogate_threshold}, prediction)")
                    return y_pred, False
        
        # 真实仿真（不确定性高 或 未启用代理 或 初始阶段）
        y = self._real_evaluate(x, evaluator)
        self.real_evaluation_count += 1
        
        # 更新代理模型
        if self.use_surrogate and self.surrogate_manager:
            self.surrogate_manager.add_sample(x, y)
        
        # 缓存结果
        self._add_to_cache(x, y)
        
        print(f"  [Real] Eval #{self.evaluation_count}")
        
        return y, True
    
    def _real_evaluate(self, x: np.ndarray, evaluator) -> np.ndarray:
        """真实仿真评估"""
        # 格式化参数（根据变量配置的 precision）
        x_formatted = self.format_params(x)
        
        # 设置变量并运行仿真
        for j, var in enumerate(self.variables):
            evaluator.hfss.set_variable(var['name'], x_formatted[j], var.get('unit', 'mm'))
        
        if not evaluator.hfss.analyze(force=True):
            # 仿真失败，返回惩罚值
            return np.array([1e6] * self.n_objectives)
        
        # 清除缓存
        evaluator.clear_cache()
        
        # 获取目标值
        result = evaluator.evaluate_all(x)
        if result is None:
            return np.array([1e6] * self.n_objectives)
        
        # evaluate_all 返回 (obj_values, results_dict)
        if isinstance(result, tuple) and len(result) == 2:
            obj_values, _ = result
            y = np.array(obj_values, dtype=float)
        elif isinstance(result, dict):
            y = np.array(list(result.values()), dtype=float)
        elif isinstance(result, (list, tuple)):
            y = np.array(result, dtype=float)
        elif isinstance(result, (int, float)):
            y = np.array([float(result)])
        else:
            print(f"[WARN] Unknown result type: {type(result)}")
            return np.array([1e6] * self.n_objectives)
        
        # 确保是一维数组
        if y.ndim > 1:
            y = y.flatten()
        
        return y
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Pareto 支配判断（最小化）"""
        if b is None:
            return True
        return np.all(a <= b) and np.any(a < b)
    
    def _update_archive(self, x: np.ndarray, y: np.ndarray, is_predicted: bool = False):
        """
        更新外部档案
        
        Args:
            x: 参数向量
            y: 目标值向量
            is_predicted: 是否为代理模型预测值
        """
        # 检查是否被档案中的解支配
        dominated = False
        to_remove = []
        
        for i, sol in enumerate(self.archive):
            if self._dominates(np.array(sol['objectives']), y):
                dominated = True
                break
            if self._dominates(y, np.array(sol['objectives'])):
                to_remove.append(i)
        
        if dominated:
            return
        
        # 移除被支配的解
        for i in sorted(to_remove, reverse=True):
            self.archive.pop(i)
        
        # 添加新解
        self.archive.append({
            'parameters': x.tolist(),
            'objectives': y.tolist(),
            'is_predicted': is_predicted
        })
        
        # 档案大小限制
        if len(self.archive) > self.max_archive_size:
            self._prune_archive()
    
    def _prune_archive(self):
        """使用拥挤距离裁剪档案"""
        if len(self.archive) <= self.max_archive_size:
            return
        
        # 计算拥挤距离
        objectives = np.array([sol['objectives'] for sol in self.archive])
        distances = self._crowding_distance(objectives)
        
        # 保留拥挤距离大的解
        indices = np.argsort(distances)[::-1][:self.max_archive_size]
        self.archive = [self.archive[i] for i in indices]
    
    def _crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """计算拥挤距离"""
        n = len(objectives)
        if n == 0:
            return np.array([])
        
        distances = np.zeros(n)
        
        for m in range(objectives.shape[1]):
            sorted_idx = np.argsort(objectives[:, m])
            
            # 边界点距离设为无穷大
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # 计算中间点距离
            obj_range = objectives[sorted_idx[-1], m] - objectives[sorted_idx[0], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        objectives[sorted_idx[i + 1], m] - objectives[sorted_idx[i - 1], m]
                    ) / obj_range
        
        return distances
    
    def _select_gbest(self) -> np.ndarray:
        """从档案中选择全局最优"""
        if not self.archive:
            return self.particles[0]
        
        # 如果只有一个解，直接返回
        if len(self.archive) == 1:
            return np.array(self.archive[0]['parameters'])
        
        # 使用轮盘赌选择，拥挤距离大的解被选中概率高
        objectives = np.array([sol['objectives'] for sol in self.archive])
        distances = self._crowding_distance(objectives)
        
        # 处理无穷大和 NaN
        distances = np.nan_to_num(distances, nan=1.0, posinf=1e6, neginf=0.0)
        
        # 确保所有距离都是非负的
        distances = np.maximum(distances, 0.0)
        
        # 归一化为概率（确保和为 1）
        total = distances.sum()
        if total <= 0:
            probs = np.ones(len(distances)) / len(distances)  # 等概率
        else:
            probs = distances / total
        
        # 再次确保概率和为 1
        probs = probs / probs.sum()
        
        selected_idx = np.random.choice(len(self.archive), p=probs)
        
        return np.array(self.archive[selected_idx]['parameters'])
    
    def _get_pareto_solutions(self) -> List[Dict]:
        """获取 Pareto 前沿解"""
        return self.archive.copy()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        best_objectives = None
        if self.archive:
            objectives = np.array([sol['objectives'] for sol in self.archive])
            best_objectives = objectives.min(axis=0).tolist()
        
        stats = {
            'n_evaluations': self.evaluation_count,
            'real_evaluations': self.real_evaluation_count,
            'pareto_size': len(self.archive),
            'best_objectives': best_objectives or [],
        }
        
        # 添加加载的历史数据统计
        if self.loaded_evaluations:
            stats['loaded_evaluations'] = len(self.loaded_evaluations)
        
        # 添加代理模型统计（仅显示使用情况，不声称节省时间）
        if self.use_surrogate and self.surrogate_eval_count > 0:
            stats['surrogate_predictions'] = self.surrogate_eval_count
        
        return stats