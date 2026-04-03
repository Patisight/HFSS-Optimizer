"""
代理模型辅助优化算法
结合高斯过程代理模型和 NSGA-II
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

from .base import BaseOptimizer
from .nsga2 import NSGA2


class GPSurrogateModel:
    """高斯过程代理模型 (GP + StandardScaler)"""
    
    def __init__(self, n_variables: int, n_objectives: int):
        self.n_variables = n_variables
        self.n_objectives = n_objectives
        
        self.models = []
        self.scalers_X = []
        self.scalers_y = []
        
        self._init_models()
        
        self.is_trained = False
        self.training_X = []
        self.training_y = []
    
    def _init_models(self):
        """初始化模型"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            
            for _ in range(self.n_objectives):
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    normalize_y=True,
                    random_state=42
                )
                self.models.append(gp)
                self.scalers_X.append(StandardScaler())
                self.scalers_y.append(StandardScaler())
            
            self._sklearn_available = True
            
        except ImportError:
            self._sklearn_available = False
            print("[WARN] scikit-learn not available, surrogate disabled")
    
    def add_sample(self, x: np.ndarray, y: List[float]):
        """添加训练样本"""
        self.training_X.append(np.array(x))
        self.training_y.append(np.array(y))
    
    def train(self) -> bool:
        """训练代理模型"""
        if not self._sklearn_available or len(self.training_X) < 10:
            return False
        
        try:
            X = np.array(self.training_X)
            Y = np.array(self.training_y)
            
            for i in range(self.n_objectives):
                X_scaled = self.scalers_X[i].fit_transform(X)
                y_scaled = self.scalers_y[i].fit_transform(Y[:, i].reshape(-1, 1)).ravel()
                self.models[i].fit(X_scaled, y_scaled)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"[WARN] Train surrogate: {e}")
            return False
    
    def predict(self, x: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """预测"""
        if not self.is_trained:
            return None
        
        try:
            x = np.array(x).reshape(1, -1)
            predictions = []
            uncertainties = []
            
            for i in range(self.n_objectives):
                x_scaled = self.scalers_X[i].transform(x)
                y_scaled, std = self.models[i].predict(x_scaled, return_std=True)
                y = self.scalers_y[i].inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
                predictions.append(y)
                uncertainties.append(std)
            
            return np.array(predictions), np.array(uncertainties)
            
        except Exception:
            return None


class LatinHypercubeSampler:
    """拉丁超立方采样器"""
    
    def __init__(self, variables_config: List[Dict]):
        self.variables = variables_config
        self.n_vars = len(variables_config)
        self.bounds = np.array([v['bounds'] for v in variables_config])
    
    def generate(self, n_samples: int) -> np.ndarray:
        """生成样本"""
        try:
            from scipy.stats import qmc
            
            sampler = qmc.LatinHypercube(d=self.n_vars)
            samples = sampler.random(n=n_samples)
            samples_scaled = qmc.scale(samples, self.bounds[:, 0], self.bounds[:, 1])
            
            return samples_scaled
            
        except ImportError:
            # 回退到随机采样
            return np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(n_samples, self.n_vars)
            )
    
    def generate_with_corners(self, n_samples: int) -> np.ndarray:
        """生成样本 + 边界点"""
        lhs_samples = self.generate(n_samples)
        
        # 添加边界点
        n_corners = min(2 ** self.n_vars, 32)
        corner_samples = []
        
        for i in range(n_corners):
            corner = []
            for j in range(self.n_vars):
                if (i >> j) & 1:
                    corner.append(self.bounds[j, 1])
                else:
                    corner.append(self.bounds[j, 0])
            corner_samples.append(corner)
        
        return np.vstack([lhs_samples, np.array(corner_samples)])


class SurrogateAssistedNSGA2(NSGA2):
    """代理模型辅助的 NSGA-II"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 代理模型配置
        self.surrogate_enabled = config.get('surrogate_enabled', True)
        self.initial_samples = config.get('initial_samples', 50)
        self.min_real_evals = config.get('min_real_evals', 30)
        self.update_interval = config.get('update_interval', 10)
        
        # 代理模型
        self.surrogate = None
        self.lhs_sampler = None
        self.surrogate_eval_count = 0
    
    def run(self, evaluator) -> List[Dict]:
        """运行优化"""
        print("\n" + "=" * 60)
        print("SURROGATE-ASSISTED NSGA-II OPTIMIZATION")
        print(f"Initial samples: {self.initial_samples}")
        print(f"Surrogate: {'Enabled' if self.surrogate_enabled else 'Disabled'}")
        print("=" * 60)
        
        # 初始化代理模型
        if self.surrogate_enabled:
            self._init_surrogate()
        
        # Phase 1: 拉丁超立方采样
        population, objectives, all_results = self._lhs_initialization(evaluator)
        
        # 训练代理模型
        if self.surrogate and len(self.training_X) >= self.min_real_evals:
            self._train_surrogate()
        
        # Phase 2: NSGA-II 优化
        for gen in range(self.n_generations):
            print(f"\n[Generation {gen + 1}/{self.n_generations}]")
            
            fronts = self.fast_non_dominated_sort(population, objectives)
            offspring = self._generate_offspring_surrogate(population, objectives, fronts, evaluator)
            
            population, objectives, all_results = self._select_next_generation(
                population, objectives, all_results,
                offspring[0], offspring[1], offspring[2]
            )

            # 早停检查
            if self.should_stop_early(all_results):
                goals_count = self.count_solutions_meeting_goals(all_results)
                print(f"\n[INFO] Early stop: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                break
            
            current_fronts = self.fast_non_dominated_sort(population, objectives)
            if current_fronts and current_fronts[0]:
                print(f"  Pareto front: {len(current_fronts[0])} solutions")
        
        # Phase 3: 验证 Pareto 前沿
        fronts = self.fast_non_dominated_sort(population, objectives)
        pareto_params = self._validate_and_extract(fronts[0], population, all_results, evaluator)
        
        return pareto_params
    
    def get_statistics(self) -> Dict:
        stats = super().get_statistics()
        stats['surrogate_predictions'] = self.surrogate_eval_count
        stats['simulation_savings'] = (
            (self.surrogate_eval_count / max(self.evaluation_count, 1)) * 100
            if self.evaluation_count > 0 else 0
        )
        return stats
    
    def _init_surrogate(self):
        """初始化代理模型"""
        self.surrogate = GPSurrogateModel(
            len(self.variables),
            len(self.objectives)
        )
        self.lhs_sampler = LatinHypercubeSampler(self.variables)
        self.training_X = []
        self.training_y = []
        print(f"[OK] Surrogate model initialized")
    
    def _lhs_initialization(self, evaluator) -> Tuple:
        """拉丁超立方采样初始化"""
        print(f"\n[LHS] Generating {self.initial_samples} dispersed samples...")
        
        samples = self.lhs_sampler.generate_with_corners(self.initial_samples)
        
        population = []
        objectives = []
        all_results = []
        
        for i, sample in enumerate(samples):
            result = self._evaluate_with_cache(sample, evaluator)
            if result is not None:
                population.append(sample)
                objectives.append(result[0])
                all_results.append(result[1])
                
                # 添加到训练数据
                if self.surrogate:
                    self.training_X.append(sample)
                    self.training_y.append(result[0])
                
                self._print_result(i + 1, result[1])
        
        print(f"[LHS] Complete: {len(population)} valid samples")
        return population, objectives, all_results
    
    def _train_surrogate(self):
        """训练代理模型"""
        for x, y in zip(self.training_X, self.training_y):
            self.surrogate.add_sample(x, y)
        
        if self.surrogate.train():
            print(f"\n[Surrogate] Trained with {len(self.training_X)} samples")
    
    def _generate_offspring_surrogate(self, population, objectives, fronts, evaluator) -> Tuple:
        """生成子代（使用代理模型）"""
        offspring = []
        selected = self._selection(population, objectives, fronts)
        
        n_vars = len(self.variables)
        mut_prob = self.mutation_prob or (1.0 / n_vars)
        
        while len(offspring) < self.population_size:
            idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
            parent1, parent2 = selected[idx1], selected[idx2]
            
            if np.random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = self._mutation(child1, mut_prob)
            child2 = self._mutation(child2, mut_prob)
            
            offspring.extend([child1, child2])
        
        offspring = offspring[:self.population_size]
        
        # 评估子代
        offspring_objectives = []
        offspring_results = []
        
        for ind in offspring:
            # 尝试使用代理模型
            if self.surrogate and self.surrogate.is_trained:
                pred = self.surrogate.predict(ind)
                
                if pred is not None:
                    self.surrogate_eval_count += 1
                    offspring_objectives.append(pred[0])
                    offspring_results.append(self._create_surrogate_results(pred))
                    continue
            
            # 真实评估
            result = self._evaluate_with_cache(ind, evaluator)
            if result is not None:
                offspring_objectives.append(result[0])
                offspring_results.append(result[1])
                
                # 更新训练数据
                if self.surrogate:
                    self.training_X.append(ind)
                    self.training_y.append(result[0])
                    
                    # 定期更新模型
                    if len(self.training_X) % self.update_interval == 0:
                        self._train_surrogate()
            else:
                offspring_objectives.append([1000] * len(self.objectives))
                offspring_results.append(None)
        
        return offspring, offspring_objectives, offspring_results
    
    def _create_surrogate_results(self, pred: Tuple) -> Dict:
        """创建代理模型结果"""
        results = {}
        values, uncertainties = pred
        
        for i, obj in enumerate(self.objectives):
            name = obj.get('name', obj['type'])
            target = obj.get('target', 'minimize')
            actual = -values[i] if target == 'maximize' else values[i]
            
            # 使用简单的字典格式
            results[name] = {
                'value': values[i],
                'actual_value': actual,
                'goal_met': None,
                'uncertainty': uncertainties[i],
                'is_surrogate': True,
            }
        
        return results
    
    def _validate_and_extract(self, front, population, all_results, evaluator) -> List[Dict]:
        """验证并提取 Pareto 前沿"""
        pareto_params = []
        
        for idx in front:
            params = population[idx]
            results = all_results[idx]
            
            # 如果是代理模型结果，进行真实验证
            if results and any(
                res.get('is_surrogate', False) 
                for res in results.values() if isinstance(res, dict)
            ):
                print(f"  Validating solution with real simulation...")
                result = self._evaluate_with_cache(params, evaluator)
                if result is not None:
                    results = result[1]
            
            param_dict = {
                v['name']: round(params[i], 4)
                for i, v in enumerate(self.variables)
            }
            
            obj_dict = {}
            if results:
                for name, res in results.items():
                    # 兼容字典和对象格式
                    if isinstance(res, dict):
                        obj_dict[name] = {
                            'value': res.get('value', 0),
                            'actual_value': res.get('actual_value', 0),
                            'goal_met': res.get('goal_met'),
                        }
                    else:
                        obj_dict[name] = {
                            'value': res.value,
                            'actual_value': res.actual_value,
                            'goal_met': res.goal_met,
                        }
            
            pareto_params.append({
                'params': param_dict,
                'objectives': obj_dict,
            })
        
        return pareto_params