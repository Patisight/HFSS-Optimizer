"""
多目标贝叶斯优化 (MOBO) - 为昂贵黑箱优化设计

特点：
1. 最少仿真次数找到最优解
2. 平衡探索和利用
3. 支持多目标优化
4. 使用高斯过程代理模型 + 不确定度估计
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')
from loguru import logger

from .base import BaseOptimizer


class GaussianProcessSurrogate:
    """
    高斯过程代理模型 - 提供预测值和不确定度
    
    优点：
    1. 小样本拟合精度高
    2. 提供预测置信区间
    3. 支持贝叶斯优化的采集函数
    """
    
    def __init__(self, kernel='matern', nu=2.5, normalize_y=True):
        self.kernel = kernel
        self.nu = nu
        self.normalize_y = normalize_y
        self.model = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
            
            # 定义核函数
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=self.nu) + WhiteKernel(noise_level=0.01)
            
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=self.normalize_y,
                random_state=42
            )
            
            self.model.fit(X, y)
            self._is_fitted = True
            return True
            
        except ImportError:
            logger.warning(f"[WARN] sklearn not available")
            return False
    
    def predict(self, X: np.ndarray, return_std: bool = True):
        """预测，返回均值和标准差"""
        if not self._is_fitted:
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))
        
        if return_std:
            return self.model.predict(X, return_std=True)
        else:
            return self.model.predict(X, return_std=False)
    
    def get_posterior_samples(self, X: np.ndarray, n_samples: int = 10):
        """从后验分布采样"""
        if not self._is_fitted:
            return np.random.randn(n_samples, len(X))
        
        return self.model.sample_y(X, n_samples=n_samples, random_state=42)


class AcquisitionFunction:
    """
    采集函数 - 决定下一个评估点
    
    支持的采集函数：
    1. EI (Expected Improvement) - 期望改进
    2. UCB (Upper Confidence Bound) - 上置信界
    3. PI (Probability of Improvement) - 改进概率
    4. EHVI (Expected Hypervolume Improvement) - 多目标扩展
    """
    
    @staticmethod
    def expected_improvement(X, model, y_best, xi=0.01):
        """
        期望改进 (EI)

        EI = E[max(f(x) - f*, 0)]
        """
        from scipy.stats import norm

        mu, sigma = model.predict(X, return_std=True)

        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-10] = 0.0

        return ei
    
    @staticmethod
    def upper_confidence_bound(X, model, beta=2.0):
        """
        上置信界 (UCB)
        
        UCB = mu + beta * sigma
        
        beta 控制探索-利用平衡：
        - 较大 beta -> 更多探索
        - 较小 beta -> 更多利用
        """
        mu, sigma = model.predict(X, return_std=True)
        return mu + beta * sigma
    
    @staticmethod
    def probability_of_improvement(X, model, y_best, xi=0.01):
        """
        改进概率 (PI)
        
        PI = P(f(x) > f* + xi)
        """
        from scipy.stats import norm
        
        mu, sigma = model.predict(X, return_std=True)
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best - xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma < 1e-10] = 0.0
        
        return pi


class MultiObjectiveBayesianOptimizer(BaseOptimizer):
    """
    多目标贝叶斯优化器 (MOBO)
    
    专为昂贵黑箱多目标优化设计，完美匹配：
    - 多维多变量
    - 非凸、多局部最优
    - 不连续
    - 单次仿真昂贵
    
    算法流程：
    1. 拉丁超立方采样生成初始样本
    2. 训练高斯过程代理模型
    3. 使用采集函数选择下一个评估点
    4. 真实评估，更新模型
    5. 重复直到收敛
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 优化配置 - 支持多种键名
        self.n_initial = config.get('initial_samples', config.get('n_initial', 50))
        self.n_iterations = config.get('n_iterations', config.get('n_generations', 100))
        self.acquisition = config.get('acquisition', 'ehvi')  # ei, ucb, ehvi
        self.beta = config.get('beta', 2.0)  # UCB 参数
        
        # 模型
        self.models = []  # 每个目标一个模型
        
        # 数据
        self.X_observed = []
        self.y_observed = []
        
        # 帕累托前沿
        self.pareto_front = []
        
        # 回调
        self.callback = None
        
        # 变量数量和目标数量
        self.n_variables = len(self.variables)
        self.n_objectives = len(self.objectives)
        
        # 边界
        self.bounds = self.get_bounds()
    
    def run(self, evaluator, callback: Callable = None) -> List[Dict]:
        """
        运行优化 (实现基类抽象方法)
        
        Args:
            evaluator: 目标评估器
            callback: 回调函数，用于 GUI 更新
            
        Returns:
            Pareto 前沿解列表
        """
        self.callback = callback
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MULTI-OBJECTIVE BAYESIAN OPTIMIZATION (MOBO)")
        logger.info(f"Initial samples: {self.n_initial}")
        logger.info(f"Max iterations: {self.n_iterations}")
        logger.info(f"Acquisition: {self.acquisition.upper()}")
        logger.info(f"{'='*60}")
        
        X_init = []
        y_init = []
        
        for i in range(self.n_initial):
            while True:
                x = self._lhs_sampling(1).flatten()
                logger.info(f"  [{len(X_init)+1}/{self.n_initial}] Evaluating...")
                
                if self._has_formulas:
                    params_dict = {v['name']: x[j] for j, v in enumerate(self.variables)}
                    valid, msg = self.constraint_mgr.check_constraints(params_dict)
                    if not valid:
                        logger.info(f"  [CONSTRAINT VIOLATION] {msg} -> resampling")
                        continue
                
                for j, var in enumerate(self.variables):
                    evaluator.hfss.set_variable(var['name'], x[j], var.get('unit', 'mm'))
                
                # 分析 - 持续重试直到成功
                while True:
                    try:
                        if evaluator.hfss.analyze(force=True):
                            break
                        logger.info(f"    [WARN] Analysis returned False, retrying...")
                    except Exception as e:
                        logger.info(f"    [ERROR] Analysis failed: {e}")
                    logger.info(f"    [INFO] HFSS disconnected, waiting to reconnect...")
                    import time
                    time.sleep(10)
                    continue
                
                evaluator.clear_cache()
                
                # 评估 - 持续重试直到成功
                while True:
                    try:
                        result = evaluator.evaluate_all(x)
                        break
                    except RuntimeError as e:
                        logger.info(f"    [ERROR] Evaluation failed: {e}")
                        logger.info(f"    [INFO] HFSS disconnected, waiting to reconnect...")
                        import time
                        time.sleep(10)
                        continue
                
                if result is None:
                    logger.info(f"    [WARN] Evaluation failed -> resampling")
                    continue
                
                y = np.array(result[0]) if isinstance(result, tuple) else np.array(result)
                if self.is_penalty_value(y):
                    logger.info(f"    [WARN] Abnormal objective value -> resampling")
                    continue
                
                X_init.append(x)
                y_init.append(y.tolist())
                
                if self.callback:
                    self.callback(len(X_init)-1, self.n_initial, x, y, 'initial')
                break
        
        self.X_observed = np.array(X_init)
        self.y_observed = np.array(y_init)
        
        self._train_models()
        
        # 初始化后早停检查
        if self.stop_when_goal_met:
            goals_count = self.count_objectives_meeting_goals_from_arrays([self.y_observed[i] for i in range(len(self.y_observed))])
            logger.info(f"[INFO] Goals check: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
            if goals_count >= self.n_solutions_to_stop:
                logger.info(f"\n[INFO] Early stop after initialization: {goals_count} solutions meet goals")
                return self._get_pareto_solutions()
        
        # Step 4: 迭代优化
        for iteration in range(self.n_iterations):
            logger.info(f"\n[Iteration {iteration + 1}/{self.n_iterations}]")
            
            # 选择下一个评估点
            x_next = self._select_next_point()
            
            if self._has_formulas:
                params_dict = {v['name']: x_next[j] for j, v in enumerate(self.variables)}
                valid, msg = self.constraint_mgr.check_constraints(params_dict)
                if not valid:
                    logger.info(f"  [CONSTRAINT VIOLATION] {msg} -> penalty (ignored)")
                    continue
            
            for j, var in enumerate(self.variables):
                evaluator.hfss.set_variable(var['name'], x_next[j], var.get('unit', 'mm'))
            
            # 分析 - 持续重试直到成功
            while True:
                try:
                    if evaluator.hfss.analyze(force=True):
                        break
                    logger.info(f"  [WARN] Analysis returned False, retrying...")
                except Exception as e:
                    logger.info(f"  [ERROR] Analysis failed: {e}")
                logger.info(f"  [INFO] HFSS disconnected, waiting to reconnect...")
                import time
                time.sleep(10)
                continue
            
            evaluator.clear_cache()
            
            # 评估 - 持续重试直到成功
            while True:
                try:
                    result = evaluator.evaluate_all(x_next)
                    break
                except RuntimeError as e:
                    logger.info(f"  [ERROR] Evaluation failed: {e}")
                    logger.info(f"  [INFO] HFSS disconnected, waiting to reconnect...")
                    import time
                    time.sleep(10)
                    continue
            
            if result is None:
                logger.info(f"  [WARN] Evaluation failed")
                continue
            else:
                y_next = np.array(result[0]) if isinstance(result, tuple) else np.array(result)
            
            if self.is_penalty_value(y_next):
                logger.info(f"  [WARN] Abnormal objective value, skipping this point")
                continue
            
            self.X_observed = np.vstack([self.X_observed, x_next.reshape(1, -1)])
            self.y_observed = np.vstack([self.y_observed, y_next.reshape(1, -1)])
            
            self._train_models()
            
            self._update_pareto_front()

            # 早停检查
            if self.stop_when_goal_met:
                goals_count = self.count_objectives_meeting_goals_from_arrays([self.y_observed[i] for i in range(len(self.y_observed))])
                if goals_count >= self.n_solutions_to_stop:
                    logger.info(f"\n[INFO] Early stop: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                    break
            
            if self.callback:
                self.callback(iteration, self.n_iterations, x_next, y_next, 'iteration')
        
        # 返回帕累托前沿
        return self._get_pareto_solutions()
    
    def _lhs_sampling(self, n_samples: int) -> np.ndarray:
        """拉丁超立方采样"""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.n_variables)
            sample = sampler.random(n=n_samples)
            # 缩放到边界
            lower_bounds = self.bounds[:, 0]
            upper_bounds = self.bounds[:, 1]
            return qmc.scale(sample, lower_bounds, upper_bounds)
        except ImportError:
            # 回退到随机采样
            samples = np.random.rand(n_samples, self.n_variables)
            for i in range(self.n_variables):
                samples[:, i] = self.bounds[i, 0] + samples[:, i] * (self.bounds[i, 1] - self.bounds[i, 0])
            return samples
    
    def _train_models(self):
        """训练每个目标的高斯过程模型"""
        self.models = []
        
        for obj_idx in range(self.n_objectives):
            model = GaussianProcessSurrogate(kernel='matern', nu=2.5)
            model.fit(self.X_observed, self.y_observed[:, obj_idx])
            self.models.append(model)
    
    def _select_next_point(self) -> np.ndarray:
        """使用采集函数选择下一个评估点"""
        # 生成候选点
        n_candidates = 1000
        X_candidates = self._lhs_sampling(n_candidates)
        
        if self.acquisition == 'ehvi':
            # 多目标：使用 EHVI（期望超体积改进）
            scores = self._compute_ehvi(X_candidates)
        elif self.acquisition == 'ucb':
            # UCB：选择上置信界最大的点
            scores = self._compute_ucb(X_candidates)
        else:
            # EI：选择期望改进最大的点
            scores = self._compute_ei(X_candidates)
        
        # 选择最佳点
        best_idx = np.argmax(scores)
        return X_candidates[best_idx]
    
    def _compute_ei(self, X: np.ndarray) -> np.ndarray:
        """计算期望改进"""
        from scipy.stats import norm
        
        y_best = np.min(self.y_observed, axis=0)  # 假设是最小化
        ei_scores = np.ones(len(X))
        
        for obj_idx, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            imp = y_best[obj_idx] - mu
            Z = imp / (sigma + 1e-10)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei_scores *= ei  # 多目标乘积
        
        return ei_scores
    
    def _compute_ucb(self, X: np.ndarray) -> np.ndarray:
        """计算上置信界"""
        # 对于最小化问题，使用 LCB = mu - beta * sigma
        ucb_scores = np.ones(len(X))
        
        for obj_idx, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            lcb = mu - self.beta * sigma
            ucb_scores *= -lcb  # 取负转为最大化
        
        return ucb_scores
    
    def _compute_ehvi(self, X: np.ndarray) -> np.ndarray:
        """
        计算期望超体积改进 (EHVI)
        
        这是多目标贝叶斯优化的标准采集函数
        """
        # 简化实现：使用加权的 EI
        y_best = np.min(self.y_observed, axis=0)
        ehvi_scores = np.zeros(len(X))
        
        for obj_idx, model in enumerate(self.models):
            from scipy.stats import norm
            mu, sigma = model.predict(X, return_std=True)
            imp = y_best[obj_idx] - mu
            Z = imp / (sigma + 1e-10)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ehvi_scores += ei / self.n_objectives
        
        return ehvi_scores
    
    def _update_pareto_front(self):
        """更新帕累托前沿"""
        # 非支配排序
        is_pareto = np.ones(len(self.y_observed), dtype=bool)
        
        for i, yi in enumerate(self.y_observed):
            if is_pareto[i]:
                # 检查是否被支配
                for j, yj in enumerate(self.y_observed):
                    if i != j:
                        # yi 被 yj 支配？(最小化)
                        if np.all(yj <= yi) and np.any(yj < yi):
                            is_pareto[i] = False
                            break
        
        self.pareto_front = self.y_observed[is_pareto]
        self.pareto_params = self.X_observed[is_pareto]
    
    def _get_pareto_solutions(self) -> List[Dict]:
        """获取帕累托最优解"""
        solutions = []
        
        for i, (params, objectives) in enumerate(zip(self.pareto_params, self.pareto_front)):
            sol = {
                'id': i,
                'parameters': params.tolist(),
                'objectives': objectives.tolist(),
            }
            solutions.append(sol)
        
        return solutions
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'n_evaluations': len(self.X_observed),
            'pareto_size': len(self.pareto_front),
            'best_objectives': self.pareto_front.min(axis=0).tolist() if len(self.pareto_front) > 0 else [],
        }


# 辅助函数
def norm_cdf(x):
    """标准正态 CDF"""
    from scipy.stats import norm
    return norm.cdf(x)


def norm_pdf(x):
    """标准正态 PDF"""
    from scipy.stats import norm
    return norm.pdf(x)
