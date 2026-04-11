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
from loguru import logger
from .base import BaseOptimizer

# 导入代理模型模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.surrogate import SurrogateManager, IncrementalSurrogateManager, GPflowSVSManager


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
        
        # 双线架构配置
        self.dual_line_mode = config.get('dual_line_mode', False)
        self.shared_dir = config.get('shared_dir', './shared_data')
        
        # 读取代理模型配置（兼容新旧格式）
        surrogate_config = config.get('surrogate_config', {})
        
        # 通用参数
        self.surrogate_min_samples = surrogate_config.get('min_samples', config.get('surrogate_min_samples', 5))
        self.surrogate_threshold = surrogate_config.get('uncertainty_threshold', config.get('surrogate_threshold', 1.0))
        
        # 模型专属参数
        self.surrogate_model_params = surrogate_config.get('model_params', {})
        
        # 全量模型重训练间隔（仅单线模式的 gp, rf 有效）
        self.retrain_interval = self.surrogate_model_params.get('retrain_interval', 0)
        
        self.surrogate_manager = None
        self.surrogate_eval_count = 0  # 代理模型评估次数
        self.real_eval_count = 0  # 真实仿真次数
        self._last_retrain_count = 0  # 上次训练时的样本数
        
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
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-OBJECTIVE PARTICLE SWARM OPTIMIZATION (MOPSO)")
        logger.info(f"{'='*60}")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Generations: {self.n_generations}")
        logger.info(f"Inertia weight: {self.w_min}-{self.w_max} (adaptive)")
        logger.info(f"Learning factors: c1={self.c1}, c2={self.c2}")
        logger.info(f"Surrogate: {'Enabled (' + self.surrogate_type + ')' if self.use_surrogate else 'Disabled'}")
        if self.load_evaluations_path:
            logger.info(f"Load evaluations: {self.load_evaluations_path}")
        logger.info(f"{'='*60}")
        
        # 加载历史评估数据
        n_loaded = self._load_historical_data()
        
        # 初始化代理模型管理器
        if self.use_surrogate:
            # 使用用户配置的最少样本数，或默认为种群大小
            min_samples = self.surrogate_min_samples if self.surrogate_min_samples > 0 else self.population_size
            
            # 从配置中读取模型专属参数
            model_params = self.surrogate_model_params
            
            # 双线模式：使用支持热替换的管理器
            if self.dual_line_mode:
                from core.surrogate_hotswap import DualLineSurrogateManager
                
                self.surrogate_manager = DualLineSurrogateManager(
                    n_objectives=self.n_objectives,
                    model_type=self.surrogate_type,
                    shared_dir=self.shared_dir,
                    min_samples=min_samples,
                    **model_params
                )
                
                logger.info(f"[INFO] Dual-line mode enabled: {self.surrogate_type}")
                logger.info(f"[INFO] Shared directory: {self.shared_dir}")
                
                # 尝试从共享内存加载已有模型
                if self.surrogate_manager.initialize_from_shared_memory():
                    logger.info(f"[INFO] Loaded existing model from shared memory")
                
            elif self.surrogate_type == 'incremental':
                # RFF+SGD 增量学习
                n_features = model_params.get('n_features', 100)
                gamma = model_params.get('gamma', 0.1)
                self.surrogate_manager = IncrementalSurrogateManager(
                    n_objectives=self.n_objectives,
                    min_samples=min_samples,
                    n_features=n_features,
                    gamma=gamma
                )
                logger.info(f"[INFO] Surrogate: incremental (n_features={n_features}, gamma={gamma})")
                
            elif self.surrogate_type == 'gpflow_svgp':
                # GPflow 稀疏变分高斯过程
                n_inducing = model_params.get('n_inducing', 100)
                kernel_type = model_params.get('kernel_type', 'matern52')
                self.surrogate_manager = GPflowSVSManager(
                    n_objectives=self.n_objectives,
                    min_samples=min_samples,
                    n_inducing=n_inducing,
                    kernel_type=kernel_type
                )
                logger.info(f"[INFO] Surrogate: gpflow_svgp (n_inducing={n_inducing}, kernel={kernel_type})")
                
            elif self.surrogate_type == 'rf':
                # 随机森林
                n_estimators = model_params.get('n_estimators', 100)
                self.retrain_interval = model_params.get('retrain_interval', self.retrain_interval)
                self.surrogate_manager = SurrogateManager(
                    n_objectives=self.n_objectives,
                    model_type='rf',
                    min_samples=min_samples,
                    n_estimators=n_estimators
                )
                logger.info(f"[INFO] Surrogate: RF (n_estimators={n_estimators}, retrain_interval={self.retrain_interval})")
                
            else:
                # 默认 GP
                self.retrain_interval = model_params.get('retrain_interval', self.retrain_interval)
                self.surrogate_manager = SurrogateManager(
                    n_objectives=self.n_objectives,
                    model_type='gp',
                    min_samples=min_samples
                )
                logger.info(f"[INFO] Surrogate: GP (retrain_interval={self.retrain_interval})")
            
            logger.info(f"[INFO] Min samples: {min_samples}, Threshold: {self.surrogate_threshold}")
            
            # 用历史数据训练代理模型
            if self.loaded_evaluations:
                logger.info(f"[INFO] Loading {len(self.loaded_evaluations)} historical samples into surrogate model...")
                
                # 打印历史数据的目标值范围
                all_objectives = np.array([e['objectives'] for e in self.loaded_evaluations])
                logger.info(f"[INFO] Historical objective ranges:")
                for i, obj_config in enumerate(self.objectives):
                    if i < all_objectives.shape[1]:
                        logger.info(f"  {obj_config.get('name')}: min={all_objectives[:, i].min():.2f}, max={all_objectives[:, i].max():.2f}, std={all_objectives[:, i].std():.2f}")
                
                # 批量添加样本（不立即训练）
                for eval_data in self.loaded_evaluations:
                    self.surrogate_manager.add_sample(eval_data['params'], eval_data['objectives'])
                
                # 用所有历史数据重新训练模型
                self.surrogate_manager.retrain_all()
                self._last_retrain_count = len(self.loaded_evaluations)
                
                # 验证代理模型训练成功
                if self.surrogate_manager.surrogate.is_trained:
                    logger.success(f"[OK] Surrogate model trained with all {len(self.loaded_evaluations)} samples!")
                    
                    # 测试预测
                    test_x = self.loaded_evaluations[0]['params'].reshape(1, -1)
                    test_y_pred, test_y_std = self.surrogate_manager.predict(test_x, return_std=True)
                    test_y_true = self.loaded_evaluations[0]['objectives']
                    logger.info(f"[INFO] Validation prediction on first sample:")
                    logger.info(f"  True values: S11={test_y_true[0]:.2f}, PG={-test_y_true[1]:.2f} dB")
                    logger.info(f"  Predicted: S11={test_y_pred.flatten()[0]:.2f}, PG={-test_y_pred.flatten()[1]:.2f} dB")
                    logger.info(f"  Uncertainty: {test_y_std.flatten()}")
                    
                    # 计算预测误差
                    pred_error = np.abs(test_y_pred.flatten() - test_y_true)
                    logger.info(f"  Prediction error: S11={pred_error[0]:.2f}, PG={pred_error[1]:.2f}")
                else:
                    logger.warning(f"[WARN] Surrogate model training failed!")
                
                # 更新真实仿真计数（历史数据也算真实仿真）
                self.real_evaluation_count = len(self.loaded_evaluations)
        
        # 初始化粒子群
        self._initialize_particles(n_vars)
        
        # 初始评估
        # 如果有历史数据且代理模型已训练，可以使用代理模型预测
        use_surrogate_for_init = (
            self.use_surrogate and 
            self.surrogate_manager and 
            self.surrogate_manager.surrogate.is_trained and
            len(self.loaded_evaluations) >= self.surrogate_manager.min_samples_to_train
        )
        
        if use_surrogate_for_init:
            logger.info(f"\n[Initialization] Evaluating {self.population_size} particles (surrogate-assisted)...")
        else:
            logger.info(f"\n[Initialization] Evaluating {self.population_size} particles (real simulation)...")
        
        n_real_evals = 0
        n_surrogate_evals = 0
        
        for i, particle in enumerate(self.particles):
            while True:
                try:
                    force_real = not use_surrogate_for_init
                    y, is_real = self._evaluate(particle, evaluator, force_real=force_real)
                    break  # 成功，跳出循环
                except RuntimeError as e:
                    logger.info(f"  [ERROR] Evaluation failed (HFSS error): {e}")
                    logger.info(f"  [INFO] Will retry after reconnection...")
                    import time
                    time.sleep(10)
                    continue  # 重试
            
            if is_real:
                n_real_evals += 1
            else:
                n_surrogate_evals += 1
            
            if not self.is_penalty_value(y):
                self.pbest[i] = particle.copy()
                self.pbest_objectives[i] = y.copy()
                self._update_archive(particle, y, is_predicted=not is_real)
            
            if self.callback:
                if not self.is_penalty_value(y):
                    surrogate_preds = self._last_surrogate_pred if not is_real else None
                    if is_real and self._last_surrogate_pred is not None:
                        surrogate_preds = self._last_surrogate_pred
                    is_surrogate = not is_real
                    self.callback(i, self.population_size, particle, y, 'initial', surrogate_preds, is_surrogate)
        
        logger.info(f"[INFO] Initial evaluation: {n_real_evals} real, {n_surrogate_evals} surrogate")
        logger.info(f"[INFO] Archive size after initialization: {len(self.archive)}")
        
        # 初始化后早停检查
        if self.stop_when_goal_met:
            archive_objectives = [np.array(sol['objectives']) for sol in self.archive]
            if len(archive_objectives) > 0:
                goals_count = self.count_objectives_meeting_goals_from_arrays(archive_objectives)
                logger.info(f"[INFO] Goals check: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                if goals_count >= self.n_solutions_to_stop:
                    logger.info(f"\n[INFO] Early stop after initialization: {goals_count} solutions meet goals")
                    return self._get_pareto_solutions()
        
        # 迭代优化
        for gen in range(self.n_generations):
            # 自适应惯性权重
            w = self.w_max - (self.w_max - self.w_min) * gen / self.n_generations
            
            logger.info(f"\n[Generation {gen + 1}/{self.n_generations}]")
            logger.info(f"  Archive size: {len(self.archive)}, Inertia: {w:.3f}")
            
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
                while True:
                    try:
                        y, is_real = self._evaluate(self.particles[i], evaluator)
                        break  # 成功，跳出循环
                    except RuntimeError as e:
                        logger.info(f"  [ERROR] Evaluation failed (HFSS error): {e}")
                        logger.info(f"  [INFO] Will retry after reconnection...")
                        import time
                        time.sleep(10)
                        continue  # 重试
                
                if not self.is_penalty_value(y):
                    if self._dominates(y, self.pbest_objectives[i]):
                        self.pbest[i] = self.particles[i].copy()
                        self.pbest_objectives[i] = y.copy()
                    elif np.random.random() < 0.5:
                        self.pbest[i] = self.particles[i].copy()
                        self.pbest_objectives[i] = y.copy()
                    
                    self._update_archive(self.particles[i], y, is_predicted=not is_real)
                
                if self.callback:
                    if not self.is_penalty_value(y):
                        surrogate_preds = self._last_surrogate_pred if not is_real else None
                        if is_real and self._last_surrogate_pred is not None:
                            surrogate_preds = self._last_surrogate_pred
                        is_surrogate = not is_real
                        self.callback(gen, self.n_generations, self.particles[i], y, 'iteration', surrogate_preds, is_surrogate)

            # 早停检查（统计所有解，包括代理预测）
            if self.stop_when_goal_met:
                archive_objectives = [np.array(sol['objectives']) for sol in self.archive]
                if len(archive_objectives) > 0:
                    goals_count = self.count_objectives_meeting_goals_from_arrays(archive_objectives)
                    if goals_count >= self.n_solutions_to_stop:
                        logger.info(f"\n[INFO] Early stop: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                        break
        
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
            logger.warning(f"[WARN] Evaluations file not found: {self.load_evaluations_path}")
            return 0
        
        loaded_count = 0
        skipped_invalid = 0  # 异常值计数
        skipped_mismatch = 0  # 维度不匹配计数
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
                        
                        # 提取目标值 - 按照当前配置的目标顺序对齐！
                        objectives_data = data.get('objectives', {})
                        if isinstance(objectives_data, dict):
                            # 字典格式：{name: {value: ..., actual_value: ...}}
                            # 重要：按照当前配置的目标顺序提取值
                            obj_values = []
                            for obj_config in self.objectives:
                                obj_name = obj_config.get('name')
                                if obj_name in objectives_data:
                                    obj_values.append(objectives_data[obj_name]['value'])
                                else:
                                    # 目标名称不匹配，跳过此记录
                                    logger.warning(f"[WARN] Objective '{obj_name}' not found in history data")
                                    obj_values = None
                                    break
                            
                            if obj_values is None:
                                continue
                            objectives = np.array(obj_values)
                            
                        elif isinstance(objectives_data, list):
                            # 列表格式 - 假设顺序一致
                            objectives = np.array(objectives_data)
                        else:
                            continue
                        
                        # 检查数据有效性
                        if len(params) != self.n_variables:
                            logger.warning(f"[WARN] Skip record: param count mismatch ({len(params)} vs {self.n_variables})")
                            continue
                        
                        if len(objectives) != self.n_objectives:
                            logger.warning(f"[WARN] Skip record: objective count mismatch ({len(objectives)} vs {self.n_objectives})")
                            continue
                        
                        # 过滤异常值（仿真失败时目标值通常设为大正数）
                        INVALID_THRESHOLD = 100  # 超过此值视为异常
                        if np.any(np.abs(objectives) > INVALID_THRESHOLD):
                            skipped_invalid += 1
                            continue
                        
                        # 检查是否包含NaN或Inf
                        if np.any(np.isnan(objectives)) or np.any(np.isinf(objectives)):
                            skipped_invalid += 1
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
                        logger.warning(f"[WARN] Skip invalid line: {e}")
                        continue
            
            logger.success(f"[OK] Loaded {loaded_count} historical evaluations from: {self.load_evaluations_path}")
            if skipped_invalid > 0:
                logger.info(f"[INFO] Filtered {skipped_invalid} records with abnormal values (|obj| > 100)")
            
            # 用历史数据初始化 Pareto 档案
            if self.loaded_evaluations:
                self._init_archive_from_history()
                
                # 把历史数据写入新的 evaluations 文件
                self._write_history_to_output()
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load evaluations: {e}")
            return 0
    
    def _init_archive_from_history(self):
        """从历史数据初始化 Pareto 档案"""
        for eval_data in self.loaded_evaluations:
            x = eval_data['params']
            y = eval_data['objectives']
            self._update_archive(x, y)
        
        logger.success(f"[OK] Initialized Pareto archive with {len(self.archive)} solutions from history")
    
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
            
            # 更新 evaluator 的计数（累加历史数据数量）
            self.evaluator.eval_count += len(self.loaded_evaluations)
            logger.success(f"[OK] Historical data written to: {self.evaluator.eval_file}")
            
        except Exception as e:
            logger.warning(f"[WARN] Failed to write historical data: {e}")
    
    def _initialize_particles(self, n_vars: int):
        """初始化粒子群
        
        如果有历史数据，优先使用历史最优解附近采样
        """
        self.particles = []
        self.velocities = []
        self.pbest = []
        self.pbest_objectives = []
        
        # 如果有历史数据，从中选择一些好的解作为初始粒子
        n_from_history = 0
        if self.loaded_evaluations and len(self.loaded_evaluations) > 0:
            # 从历史数据中选择部分解作为初始粒子
            n_from_history = min(len(self.loaded_evaluations) // 2, self.population_size // 2)
            
            # 按第一个目标排序（假设是最小化）
            sorted_history = sorted(self.loaded_evaluations, key=lambda x: x['objectives'][0])
            
            for i in range(n_from_history):
                particle = sorted_history[i]['params'].copy()
                # 添加小扰动
                perturbation = np.array([
                    np.random.uniform(-0.05, 0.05) * (self.bounds[j, 1] - self.bounds[j, 0])
                    for j in range(n_vars)
                ])
                particle = np.clip(particle + perturbation, self.bounds[:, 0], self.bounds[:, 1])
                self.particles.append(particle)
                
                velocity = np.array([
                    np.random.uniform(-0.1, 0.1) * (self.bounds[j, 1] - self.bounds[j, 0])
                    for j in range(n_vars)
                ])
                self.velocities.append(velocity)
                
                self.pbest.append(particle.copy())
                self.pbest_objectives.append(None)
            
            logger.info(f"[INFO] Initialized {n_from_history} particles from historical data")
        
        # 剩余粒子使用拉丁超立方采样
        for i in range(n_from_history, self.population_size):
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
        
        安全策略：
        - min_samples_to_train 不低于 5，防止训练数据过少导致严重外推
        - 每次真实仿真后都更新代理模型
        - 不确定性 ≥ 阈值 → 真实仿真
        - 不确定性 < 阈值 → 使用预测值
        - 确保每代至少有一定比例的真实仿真
        
        Args:
            x: 参数向量
            evaluator: 评估器
            force_real: 是否强制真实仿真
        
        Returns:
            (目标值向量, 是否为真实仿真结果)
        """
        self.evaluation_count += 1
        self._last_surrogate_pred = None  # 记录最近一次代理预测
        
        # 安全检查：确保最小训练样本数不低于 5
        if self.use_surrogate and self.surrogate_manager:
            if self.surrogate_manager.min_samples_to_train < 5:
                logger.warning(f"[WARN] surrogate_min_samples={self.surrogate_manager.min_samples_to_train} is too low, raising to 5")
                self.surrogate_manager.min_samples_to_train = 5
        
        # 检查缓存（只缓存真实仿真结果）
        cached = self._check_cache(x)
        if cached is not None:
            return cached, True
        
        # 如果代理模型已训练，先用代理模型预测（用于对比图）
        if self.use_surrogate and self.surrogate_manager and self.surrogate_manager.surrogate.is_trained:
            try:
                y_pred_check, _ = self.surrogate_manager.predict(x.reshape(1, -1), return_std=False)
                self._last_surrogate_pred = y_pred_check.flatten()
            except Exception:
                pass
        
        # 判断是否使用代理模型
        if self.use_surrogate and self.surrogate_manager and not force_real:
            # 初始阶段（前 min_samples 次）用真实仿真
            min_samples = self.surrogate_manager.min_samples_to_train
            # 只有模型已训练后才考虑使用代理模型
            if self.real_evaluation_count >= min_samples and self.surrogate_manager.surrogate.is_trained:
                # 获取代理预测和不确定性
                y_pred, y_std = self.surrogate_manager.predict(x.reshape(1, -1), return_std=True)
                y_pred = y_pred.flatten()
                y_std = y_std.flatten()
                
                # 预测值范围钳制：限制在训练数据范围内，防止严重外推
                if self.surrogate_manager.y_samples:
                    y_arr = np.array(self.surrogate_manager.y_samples)
                    y_min = y_arr.min(axis=0)
                    y_max = y_arr.max(axis=0)
                    y_pred = np.clip(y_pred, y_min, y_max)
                    # 超出训练数据范围的点视为高不确定性
                    out_of_range = np.any((y_pred <= y_min) | (y_pred >= y_max))
                
                # 计算相对不确定性
                # 使用训练数据标准差归一化（比 y_pred 归一化更稳定）
                if self.surrogate_manager.y_samples:
                    y_arr = np.array(self.surrogate_manager.y_samples)
                    y_std_train = np.std(y_arr, axis=0)
                    # 归一化不确定性 = 预测标准差 / 训练数据标准差
                    if np.all(y_std_train > 1e-8):
                        normalized_uncertainty = np.mean(y_std / y_std_train)
                    else:
                        # 回退到基于预测值大小的归一化
                        pred_magnitude = np.mean(np.abs(y_pred))
                        if pred_magnitude > 1e-8:
                            normalized_uncertainty = np.mean(y_std) / pred_magnitude
                        else:
                            normalized_uncertainty = float('inf')
                else:
                    pred_magnitude = np.mean(np.abs(y_pred))
                    if pred_magnitude > 1e-8:
                        normalized_uncertainty = np.mean(y_std) / pred_magnitude
                    else:
                        normalized_uncertainty = float('inf')
                
                # 调试输出：打印预测值和不确定性
                logger.info(f"  [Surrogate Debug] y_pred={y_pred}, y_std={y_std}, uncertainty={normalized_uncertainty:.3f}")
                
                # 额外安全检查：如果真实仿真次数占比太低，强制做真实仿真
                total_evals = self.real_evaluation_count + self.surrogate_eval_count
                if total_evals > 0:
                    real_ratio = self.real_evaluation_count / total_evals
                    # 确保至少 30% 的评估是真实仿真
                    if real_ratio < 0.3:
                        normalized_uncertainty = float('inf')
                        logger.info(f"  [Safety] Real eval ratio {real_ratio:.1%} too low, forcing real simulation")
                
                # 不确定性低于阈值 → 使用预测值（跳过真实仿真）
                if normalized_uncertainty < self.surrogate_threshold:
                    self.surrogate_eval_count += 1
                    self._last_surrogate_pred = y_pred  # 记录代理预测值
                    logger.info(f"  [Surrogate] Eval #{self.evaluation_count} (uncertainty: {normalized_uncertainty:.3f} < {self.surrogate_threshold}, prediction)")
                    return y_pred, False
        
        # 真实仿真（不确定性高 或 未启用代理 或 初始阶段）
        y = self._real_evaluate(x, evaluator)
        self.real_evaluation_count += 1
        
        # 更新代理模型
        if self.use_surrogate and self.surrogate_manager:
            # 双线模式：只写入共享内存，不训练
            if self.dual_line_mode:
                self.surrogate_manager.add_sample(x, y, is_real=True)
            else:
                # 单线模式：训练模型
                self.surrogate_manager.add_sample(x, y)
                
                # GP/RF/GPflow模型：定期全量重训练
                if self.retrain_interval > 0 and self.surrogate_type in ['gp', 'rf', 'gpflow_svgp']:
                    current_sample_count = len(self.surrogate_manager.X_samples)
                    samples_since_retrain = current_sample_count - self._last_retrain_count
                    
                    if samples_since_retrain >= self.retrain_interval:
                        logger.info(f"  [Retraining] Full retrain triggered (interval={self.retrain_interval}, new samples={samples_since_retrain})")
                        self.surrogate_manager.retrain_all()
                        self._last_retrain_count = current_sample_count
        
        # 缓存结果
        self._add_to_cache(x, y)
        
        logger.info(f"  [Real] Eval #{self.evaluation_count}")
        
        return y, True
    
    def _real_evaluate(self, x: np.ndarray, evaluator) -> np.ndarray:
        """真实仿真评估"""
        # 格式化参数
        x_formatted = self.format_params(x)
        
        # 检查变量约束
        if self._has_formulas:
            params_dict = {v['name']: x_formatted[i] for i, v in enumerate(self.variables)}
            valid, msg = self.constraint_mgr.check_constraints(params_dict)
            if not valid:
                logger.info(f"  [CONSTRAINT VIOLATION] {msg} -> returning penalty")
                return self.get_penalty_objectives()
        
        # 设置变量并运行仿真 - 持续重试直到成功
        while True:
            try:
                for j, var in enumerate(self.variables):
                    evaluator.hfss.set_variable(var['name'], x_formatted[j], var.get('unit', 'mm'))
                break  # 成功设置变量，跳出循环
            except RuntimeError as e:
                logger.info(f"  [ERROR] Failed to set variable: {e}")
                logger.info(f"  [INFO] HFSS disconnected, waiting to reconnect...")
                import time
                time.sleep(10)  # 等待10秒
                continue  # 继续重试
        
        # 分析 - 持续重试直到成功
        while True:
            try:
                if evaluator.hfss.analyze(force=True):
                    break  # 分析成功，跳出循环
                # analyze() 返回 False 表示失败，等待后重试
                logger.info(f"  [WARN] Analysis returned False, retrying...")
            except Exception as e:
                logger.info(f"  [ERROR] Analysis failed: {e}")
            logger.info(f"  [INFO] HFSS disconnected, waiting to reconnect...")
            import time
            time.sleep(10)  # 等待10秒
            continue  # 继续重试
        
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
            logger.warning(f"[WARN] Unknown result type: {type(result)}")
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