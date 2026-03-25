"""
代理模型模块
提供高斯过程（GP）和随机森林（RF）代理模型
可被不同优化算法复用
"""
import numpy as np
from typing import Optional, Tuple, List


class SurrogateModel:
    """代理模型基类"""
    
    def __init__(self, model_type: str = 'gp'):
        """
        初始化代理模型
        
        Args:
            model_type: 模型类型 'gp' (高斯过程) 或 'rf' (随机森林)
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.X_train = None
        self.y_train = None
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练代理模型
        
        Args:
            X: 输入参数 (n_samples, n_features)
            y: 目标值 (n_samples, n_objectives)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        if self.model_type == 'gp':
            self._train_gp(X, y)
        else:
            self._train_rf(X, y)
        
        self.is_trained = True
    
    def _train_gp(self, X: np.ndarray, y: np.ndarray):
        """训练高斯过程模型"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
            
            # 为每个目标训练一个 GP
            n_objectives = y.shape[1] if y.ndim > 1 else 1
            self.models = []
            
            for i in range(n_objectives):
                y_i = y[:, i] if y.ndim > 1 else y
                kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
                gp.fit(X, y_i)
                self.models.append(gp)
            
            self.model = self.models
            
        except ImportError:
            print("[WARN] sklearn not available, using simple interpolation")
            self.model = None
    
    def _train_rf(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            n_objectives = y.shape[1] if y.ndim > 1 else 1
            self.models = []
            
            for i in range(n_objectives):
                y_i = y[:, i] if y.ndim > 1 else y
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X, y_i)
                self.models.append(rf)
            
            self.model = self.models
            
        except ImportError:
            print("[WARN] sklearn not available, using simple interpolation")
            self.model = None
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测目标值
        
        Args:
            X: 输入参数 (n_samples, n_features)
            return_std: 是否返回不确定性
        
        Returns:
            mean: 预测均值 (n_samples, n_objectives)
            std: 不确定性 (n_samples, n_objectives) 或 None
        """
        if not self.is_trained or self.model is None:
            # 未训练，返回默认值
            n_samples = X.shape[0]
            n_objectives = self.y_train.shape[1] if self.y_train is not None and self.y_train.ndim > 1 else 1
            mean = np.zeros((n_samples, n_objectives))
            std = np.ones((n_samples, n_objectives)) if return_std else None
            return mean, std
        
        n_objectives = len(self.models)
        n_samples = X.shape[0]
        
        mean = np.zeros((n_samples, n_objectives))
        std = np.zeros((n_samples, n_objectives)) if return_std else None
        
        for i, model in enumerate(self.models):
            if self.model_type == 'gp':
                if return_std:
                    mean[:, i], std[:, i] = model.predict(X, return_std=True)
                else:
                    mean[:, i] = model.predict(X)
            else:
                mean[:, i] = model.predict(X)
                # RF 不提供不确定性估计，使用预测方差近似
                if return_std:
                    # 使用不同树的预测方差作为不确定性
                    predictions = np.array([tree.predict(X) for tree in model.estimators_])
                    std[:, i] = np.std(predictions, axis=0)
        
        return mean, std
    
    def add_sample(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        添加新样本并重新训练
        
        Args:
            X_new: 新输入参数
            y_new: 新目标值
        """
        if self.X_train is None:
            self.train(X_new, y_new)
        else:
            X = np.vstack([self.X_train, X_new.reshape(1, -1) if X_new.ndim == 1 else X_new])
            y = np.vstack([self.y_train, y_new.reshape(1, -1) if y_new.ndim == 1 else y_new])
            self.train(X, y)
    
    def expected_improvement(self, X: np.ndarray, y_best: float, objective_idx: int = 0) -> np.ndarray:
        """
        计算期望改进（Expected Improvement）
        
        Args:
            X: 候选点
            y_best: 当前最优值
            objective_idx: 目标索引
        
        Returns:
            EI 值
        """
        if not self.is_trained or self.model is None:
            return np.zeros(X.shape[0])
        
        mean, std = self.predict(X, return_std=True)
        mean = mean[:, objective_idx]
        std = std[:, objective_idx]
        
        # 避免除零
        std = np.maximum(std, 1e-8)
        
        # EI 公式（最小化问题）
        from scipy.stats import norm
        
        improvement = y_best - mean
        Z = improvement / std
        
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei = np.maximum(ei, 0)
        
        return ei


class SurrogateManager:
    """
    代理模型管理器
    管理多个代理模型，支持增量训练
    """
    
    def __init__(self, n_objectives: int, model_type: str = 'gp', min_samples: int = 3):
        """
        初始化管理器
        
        Args:
            n_objectives: 目标数量
            model_type: 模型类型
            min_samples: 最少训练样本数（默认 3）
        """
        self.n_objectives = n_objectives
        self.model_type = model_type
        self.surrogate = SurrogateModel(model_type)
        
        # 样本缓存
        self.X_samples = []
        self.y_samples = []
        self.min_samples_to_train = min_samples
    
    def add_sample(self, X: np.ndarray, y: np.ndarray):
        """添加样本"""
        self.X_samples.append(X.flatten() if hasattr(X, 'flatten') else list(X))
        self.y_samples.append(y.flatten() if hasattr(y, 'flatten') else list(y))
        
        # 样本足够时训练模型
        if len(self.X_samples) >= self.min_samples_to_train:
            if not self.surrogate.is_trained:
                print(f"[Surrogate] Training with {len(self.X_samples)} samples...")
            self._retrain()
            if len(self.X_samples) == self.min_samples_to_train:
                print(f"[Surrogate] Model trained and ready!")
    
    def _retrain(self):
        """重新训练模型"""
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        self.surrogate.train(X, y)
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测"""
        if not self.surrogate.is_trained:
            # 未训练，返回默认值
            n_samples = X.shape[0] if hasattr(X, 'shape') else 1
            mean = np.zeros((n_samples, self.n_objectives))
            std = np.ones((n_samples, self.n_objectives)) if return_std else None
            return mean, std
        
        return self.surrogate.predict(X, return_std)
    
    def should_use_surrogate(self, iteration: int, n_initial: int = 10) -> bool:
        """
        判断是否应该使用代理模型
        
        策略：
        1. 初始阶段用真实仿真（收集训练数据）
        2. 模型训练后，根据预测不确定性决定是否使用
        3. 不确定性低 = 模型对这一带比较熟悉 = 可以用代理模型
        4. 不确定性高 = 模型对这一带不熟悉 = 需要真实仿真
        
        Args:
            iteration: 当前迭代次数
            n_initial: 初始真实评估次数
        
        Returns:
            是否使用代理模型
        """
        # 初始阶段用真实评估
        if iteration < n_initial:
            return False
        
        # 模型未训练，用真实评估
        if not self.surrogate.is_trained:
            return False
        
        return True
    
    def should_use_for_point(self, X: np.ndarray, uncertainty_threshold: float = 0.5) -> Tuple[bool, float]:
        """
        判断是否应该对特定点使用代理模型
        
        基于预测不确定性：
        - 不确定性低：模型熟悉这个区域，可以用代理模型
        - 不确定性高：模型不熟悉这个区域，需要真实仿真
        
        Args:
            X: 待评估点
            uncertainty_threshold: 不确定性阈值
        
        Returns:
            (是否使用代理模型, 不确定性值)
        """
        if not self.surrogate.is_trained:
            return False, 1.0
        
        # 获取预测和不确定性
        _, std = self.surrogate.predict(X.reshape(1, -1), return_std=True)
        
        # 平均不确定性
        avg_uncertainty = np.mean(std)
        
        # 归一化不确定性（相对于训练数据的标准差）
        if self.y_samples:
            y_std = np.std(self.y_samples, axis=0)
            normalized_uncertainty = np.mean(avg_uncertainty / (y_std + 1e-8))
        else:
            normalized_uncertainty = avg_uncertainty
        
        # 不确定性低于阈值才用代理模型
        use_surrogate = normalized_uncertainty < uncertainty_threshold
        
        if use_surrogate:
            print(f"[Surrogate] Using surrogate (uncertainty={normalized_uncertainty:.3f} < {uncertainty_threshold})")
        else:
            print(f"[Surrogate] Using real simulation (uncertainty={normalized_uncertainty:.3f} >= {uncertainty_threshold})")
        
        return use_surrogate, normalized_uncertainty
    
    def get_training_progress(self) -> dict:
        """获取训练进度信息"""
        return {
            'n_samples': len(self.X_samples),
            'is_trained': self.surrogate.is_trained,
            'model_type': self.model_type
        }
    
    def get_model_quality(self) -> dict:
        """
        评估代理模型质量
        
        使用交叉验证或留一法评估预测误差
        """
        if not self.surrogate.is_trained or len(self.X_samples) < 5:
            return {'r2': None, 'mae': None, 'status': 'insufficient_data'}
        
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        
        try:
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import r2_score, mean_absolute_error
            
            # 简化：只评估第一个目标
            y_0 = y[:, 0] if y.ndim > 1 else y
            
            if self.model_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, WhiteKernel
                kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
                model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, max_depth=8)
            
            # 交叉验证预测
            y_pred = cross_val_predict(model, X, y_0, cv=min(5, len(X) // 2))
            
            r2 = r2_score(y_0, y_pred)
            mae = mean_absolute_error(y_0, y_pred)
            
            return {
                'r2': r2,
                'mae': mae,
                'status': 'good' if r2 > 0.5 else 'poor'
            }
        except Exception as e:
            return {'r2': None, 'mae': None, 'status': f'error: {e}'}
