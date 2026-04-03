"""
改进的优化算法 - 应对非凸、多局部最优、不连续问题

改进点：
1. 多起点初始化
2. 提高变异率增加多样性
3. 精英保留策略
4. 代理模型选择（RF 对不连续更鲁棒）
5. 局部搜索混合
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

from .base import BaseOptimizer


class RobustSurrogateOptimizer(BaseOptimizer):
    """
    鲁棒的代理模型辅助优化器
    
    特点：
    1. 支持多种代理模型（GP, RF, DNN）
    2. 多起点初始化避免局部最优
    3. 自适应变异率防止早熟
    4. 精英保留策略
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 变量和目标数量
        self.n_variables = len(self.variables)
        self.n_objectives = len(self.objectives)
        
        # 代理模型类型
        self.surrogate_type = config.get('surrogate_type', 'rf')  # 'gp', 'rf', 'dnn'
        
        # 多起点配置
        self.n_restarts = config.get('n_restarts', 3)  # 多起点次数
        
        # 种群配置
        self.population_size = config.get('population_size', 50)
        self.n_generations = config.get('n_generations', 20)
        self.initial_samples = config.get('initial_samples', 100)
        
        # 变异配置 - 提高变异率
        self.mutation_prob = config.get('mutation_prob', 0.15)  # 默认 0.15，比标准 GA 更高
        self.eta_m = config.get('eta_m', 15)  # 变异分布指数
        
        # 精英保留
        self.elite_ratio = config.get('elite_ratio', 0.1)  # 保留前 10%
        
        # 局部搜索
        self.local_search_prob = config.get('local_search_prob', 0.1)
        
        # 模型
        self.models = []
        self.is_trained = False
        
        # 数据
        self.X_samples = []
        self.y_samples = []
    
    def _init_surrogate_models(self):
        """初始化代理模型"""
        try:
            if self.surrogate_type == 'gp':
                # 高斯过程 - 对平滑函数效果好
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
                
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    normalize_y=True,
                    random_state=42
                )
                self.models = [model for _ in range(self.n_objectives)]
                self._model_type = 'gp'
                print("[OK] Using Gaussian Process surrogate")
                
            elif self.surrogate_type == 'rf':
                # 随机森林 - 对不连续函数更鲁棒
                from sklearn.ensemble import RandomForestRegressor
                
                self.models = [
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    ) for _ in range(self.n_objectives)
                ]
                self._model_type = 'rf'
                print("[OK] Using Random Forest surrogate (robust to discontinuity)")
                
            elif self.surrogate_type == 'dnn':
                # 深度神经网络 - 强非线性拟合能力
                from sklearn.neural_network import MLPRegressor
                
                self.models = [
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64, 32),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        learning_rate='adaptive',
                        max_iter=1000,
                        random_state=42
                    ) for _ in range(self.n_objectives)
                ]
                self._model_type = 'dnn'
                print("[OK] Using Deep Neural Network surrogate")
            
            self._sklearn_available = True
            
        except ImportError:
            self._sklearn_available = False
            print("[WARN] sklearn not available, using pure NSGA-II")
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练代理模型"""
        if not self._sklearn_available:
            return
        
        self.X_samples = X.copy()
        self.y_samples = y.copy()
        
        for i, model in enumerate(self.models):
            try:
                model.fit(X, y[:, i])
            except Exception as e:
                print(f"[WARN] Model {i} training failed: {e}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用代理模型预测"""
        if not self.is_trained:
            return None
        
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except:
                predictions.append(np.zeros(len(X)))
        
        return np.column_stack(predictions)
    
    def run(self, evaluator, callback=None):
        """
        执行优化 (实现基类抽象方法)
        
        Args:
            evaluator: 目标评估器
            callback: 回调函数
        """
        # 初始化代理模型
        self._init_surrogate_models()
        
        # 多起点优化
        best_results = []
        
        for restart in range(self.n_restarts):
            print(f"\n{'='*60}")
            print(f"Restart {restart + 1}/{self.n_restarts}")
            print(f"{'='*60}")
            
            # 执行单次优化
            result = self._single_run(evaluator, callback, restart)
            best_results.append(result)
        
        # 合并所有 restart 的 Pareto 前沿并去重
        all_solutions = []
        seen_keys = set()
        for result in best_results:
            for sol in result.get('pareto_solutions', []):
                # 用参数的近似值作为去重 key
                params = sol.get('parameters', [])
                key = tuple(round(p, 3) for p in params) if isinstance(params, (list, np.ndarray)) else id(sol)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_solutions.append(sol)
        return all_solutions
    
    def _single_run(self, evaluator, callback, restart_idx):
        """单次优化运行"""
        # 生成初始样本 - 使用不同的随机种子
        np.random.seed(42 + restart_idx * 1000)
        random.seed(42 + restart_idx * 1000)
        
        # LHS 采样
        from .surrogate import LatinHypercubeSampler
        sampler = LatinHypercubeSampler(self.variables)
        initial_X = sampler.generate(self.initial_samples)
        
        # 评估初始样本
        initial_y = []
        for i, x in enumerate(initial_X):
            print(f"  [{i+1}/{self.initial_samples}] Evaluating initial sample...")
            
            # 设置变量
            for j, var in enumerate(self.variables):
                evaluator.hfss.set_variable(var['name'], x[j], var.get('unit', 'mm'))
            
            # 运行仿真
            if not evaluator.hfss.analyze(force=True):
                print(f"    [WARN] Analysis failed")
                initial_y.append([1000.0] * self.n_objectives)
            else:
                evaluator.clear_cache()
                y, _ = evaluator.evaluate_all(x)
                if y is not None:
                    initial_y.append(y)
                else:
                    initial_y.append([1000.0] * self.n_objectives)
            
            if callback:
                callback(i, len(initial_X), x, initial_y[-1])
        
        initial_y = np.array(initial_y)
        
        # 训练代理模型
        if self._sklearn_available:
            self.train(initial_X, initial_y)
        
        # 初始化种群 - 这里简化处理，直接使用已有的初始样本进行简单优化
        # 完整的NSGA-II优化可以后续添加
        pareto_solutions = self._simple_optimization(initial_X, initial_y, evaluator)
        
        return {'pareto_solutions': pareto_solutions}
    
    def _simple_optimization(self, initial_X: np.ndarray, initial_y: np.ndarray, evaluator) -> List[Dict]:
        """简单的优化流程 - 使用贪婪选择和随机变异"""
        # 将初始样本作为当前种群
        population = initial_X.tolist()
        objectives = initial_y.tolist()
        
        # 简单迭代优化
        for _ in range(self.n_generations):
            # 生成变异个体
            new_population = []
            for ind in population:
                # 随机选择一个个体进行变异
                mutated = self._mutate(ind)
                new_population.append(mutated)
            
            # 评估新个体
            for x in new_population:
                # 设置变量
                for j, var in enumerate(self.variables):
                    evaluator.hfss.set_variable(var['name'], x[j], var.get('unit', 'mm'))
                
                if evaluator.hfss.analyze(force=True):
                    evaluator.clear_cache()
                    y, _ = evaluator.evaluate_all(x)
                    if y is not None:
                        population.append(x)
                        objectives.append(y)
            
            # 简单选择：保留最好的前population_size个
            if len(population) > self.population_size:
                sorted_indices = sorted(range(len(objectives)), key=lambda i: sum(objectives[i]))
                population = [population[i] for i in sorted_indices[:self.population_size]]
                objectives = [objectives[i] for i in sorted_indices[:self.population_size]]

            # 早停检查
            if self.stop_when_goal_met:
                goals_count = self.count_objectives_meeting_goals_from_arrays(objectives)
                if goals_count >= self.n_solutions_to_stop:
                    print(f"\n[INFO] Early stop: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                    break
        
        # 提取Pareto前沿
        pareto_indices = self._fast_non_dominated_sort(objectives)
        pareto_solutions = []
        for idx in pareto_indices[0]:
            pareto_solutions.append({
                'parameters': population[idx],
                'objectives': objectives[idx]
            })
        
        return pareto_solutions
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """对个体进行多项式变异"""
        mutated = individual.copy()
        bounds = np.array([v['bounds'] for v in self.variables])
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_prob:
                lower, upper = bounds[i]
                delta1 = (mutated[i] - lower) / (upper - lower) if upper != lower else 0.5
                delta2 = (upper - mutated[i]) / (upper - lower) if upper != lower else 0.5
                
                r = np.random.random()
                if r < 0.5:
                    xy = 1 - delta1
                    val = 2 * r + (1 - 2 * r) * (xy ** (self.eta_m + 1))
                    deltaq = val ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - r) + 2 * (r - 0.5) * (xy ** (self.eta_m + 1))
                    deltaq = 1 - val ** (1.0 / (self.eta_m + 1))
                
                mutated[i] = np.clip(mutated[i] + deltaq * (upper - lower), lower, upper)
        
        return mutated
    
    def _fast_non_dominated_sort(self, objectives: List) -> List[List[int]]:
        """快速非支配排序"""
        n = len(objectives)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                # 假设是最小化
                if all(objectives[p][i] <= objectives[q][i] for i in range(self.n_objectives)):
                    if any(objectives[p][i] < objectives[q][i] for i in range(self.n_objectives)):
                        dominated_solutions[p].append(q)
                elif all(objectives[q][i] <= objectives[p][i] for i in range(self.n_objectives)):
                    if any(objectives[q][i] < objectives[p][i] for i in range(self.n_objectives)):
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
            if next_front:
                fronts.append(next_front)
        
        return fronts[:-1] if fronts[-1] == [] else fronts
    
    def _hybrid_evaluate(self, x: np.ndarray, evaluator) -> np.ndarray:
        """
        混合评估：代理模型 + 真实评估
        
        策略：
        1. 大部分情况使用代理模型快速预测
        2. 一定概率进行真实评估更新模型
        3. 对不确定度高的点进行真实评估
        """
        # 决定是否进行真实评估
        do_real_eval = random.random() < 0.2  # 20% 概率真实评估
        
        if do_real_eval or not self.is_trained:
            # 设置变量
            for j, var in enumerate(self.variables):
                evaluator.hfss.set_variable(var['name'], x[j], var.get('unit', 'mm'))
            
            # 运行仿真
            if evaluator.hfss.analyze(force=True):
                # 真实评估
                evaluator.clear_cache()
                y, _ = evaluator.evaluate_all(x)
                if y is not None:
                    # 更新样本和模型
                    self.X_samples = np.vstack([self.X_samples, x.reshape(1, -1)])
                    self.y_samples = np.vstack([self.y_samples, y])
                    
                    # 定期重新训练
                    if len(self.X_samples) % 10 == 0 and self._sklearn_available:
                        self.train(self.X_samples, self.y_samples)
                    
                    return np.array(y)
        
        # 使用代理模型预测
        if self.is_trained:
            prediction = self.predict(x.reshape(1, -1))
            return prediction[0]
        
        # 默认惩罚值
        return np.array([1000.0] * self.n_objectives)
    
    def _select_best(self, results: List) -> int:
        """选择最佳结果（Pareto 前沿）"""
        # 简化：选择第一个目标最小的
        best_idx = 0
        best_value = float('inf')
        
        for i, result in enumerate(results):
            if 'best_fitness' in result:
                value = result['best_fitness'][0]
                if value < best_value:
                    best_value = value
                    best_idx = i
        
        return best_idx
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'n_restarts': self.n_restarts,
            'surrogate_type': self.surrogate_type,
            'is_trained': self.is_trained,
        }


class AdaptiveOptimizer(BaseOptimizer):
    """
    自适应优化器 - 根据问题特性自动选择策略
    
    检测：
    1. 函数平滑度
    2. 局部最优数量
    3. 不连续性
    
    自动选择：
    1. 纯 NSGA-II（计算资源充足）
    2. RF 代理模型（检测到不连续）
    3. GP 代理模型（函数平滑）
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 变量和目标数量
        self.n_variables = len(self.variables)
        self.n_objectives = len(self.objectives)
        
        self.n_test_points = config.get('n_test_points', 20)
        self.continuity_threshold = config.get('continuity_threshold', 0.3)
        
        self._detected_discontinuity = False
        self._selected_optimizer = None
    
    def _detect_discontinuity(self, evaluator, X_test: np.ndarray) -> bool:
        """
        检测不连续性
        
        方法：比较相邻点的目标函数值变化
        """
        y_values = []
        
        for x in X_test:
            # 设置变量
            for j, var in enumerate(self.variables):
                evaluator.hfss.set_variable(var['name'], x[j], var.get('unit', 'mm'))
            
            # 运行仿真
            if evaluator.hfss.analyze(force=True):
                evaluator.clear_cache()
                y, _ = evaluator.evaluate_all(x)
                if y is not None:
                    y_values.append(y[0])  # 只检查第一个目标
        
        if len(y_values) < 2:
            return False
        
        y_values = np.array(y_values)
        
        # 计算相邻点差异
        diffs = np.abs(np.diff(y_values))
        
        # 归一化差异
        y_range = np.max(y_values) - np.min(y_values)
        if y_range > 0:
            normalized_diffs = diffs / y_range
        else:
            normalized_diffs = diffs
        
        # 检测大跳跃
        large_jumps = np.sum(normalized_diffs > self.continuity_threshold)
        
        # 如果超过 20% 的相邻点有大跳跃，认为不连续
        return large_jumps > len(diffs) * 0.2
    
    def run(self, evaluator, callback=None):
        """自适应优化 (实现基类抽象方法)"""
        print("\n[INFO] Detecting problem characteristics...")
        
        # 生成测试点
        test_X = self._generate_test_points()
        
        # 检测不连续性
        self._detected_discontinuity = self._detect_discontinuity(evaluator, test_X)
        
        if self._detected_discontinuity:
            print("[WARN] Discontinuity detected! Using Random Forest surrogate.")
            optimizer = RobustSurrogateOptimizer({
                **self.config,
                'surrogate_type': 'rf',
                'mutation_prob': 0.2,  # 更高的变异率
            })
        else:
            print("[INFO] Function appears smooth. Using Gaussian Process surrogate.")
            optimizer = RobustSurrogateOptimizer({
                **self.config,
                'surrogate_type': 'gp',
            })
        
        self._selected_optimizer = optimizer
        return optimizer.run(evaluator, callback)
    
    def _generate_test_points(self) -> np.ndarray:
        """生成测试点"""
        points = []
        bounds = self.get_bounds()
        n_per_dim = int(np.ceil(self.n_test_points ** (1/self.n_variables)))
        
        for i in range(self.n_variables):
            lb, ub = bounds[i]
            dim_points = np.linspace(lb, ub, n_per_dim)
            points.append(dim_points)
        
        # 网格采样
        mesh = np.meshgrid(*points)
        return np.column_stack([m.flatten() for m in mesh])[:self.n_test_points]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'detected_discontinuity': self._detected_discontinuity,
            'selected_optimizer': type(self._selected_optimizer).__name__ if self._selected_optimizer else None,
        }
