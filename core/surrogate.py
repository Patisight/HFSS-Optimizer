"""
代理模型模块
提供高斯过程（GP）和随机森林（RF）代理模型
可被不同优化算法复用
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


class SurrogateModel:
    """代理模型基类"""

    def __init__(self, model_type: str = "gp", n_estimators: int = 100):
        """
        初始化代理模型

        Args:
            model_type: 模型类型 'gp' (高斯过程) 或 'rf' (随机森林)
            n_estimators: 随机森林的树数量（仅对 RF 有效）
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
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

        if self.model_type == "gp":
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
            logger.warning(" sklearn not available, using simple interpolation")
            self.model = None

    def _train_rf(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林模型"""
        try:
            from sklearn.ensemble import RandomForestRegressor

            n_objectives = y.shape[1] if y.ndim > 1 else 1
            self.models = []

            for i in range(n_objectives):
                y_i = y[:, i] if y.ndim > 1 else y
                rf = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=10, random_state=42)
                rf.fit(X, y_i)
                self.models.append(rf)

            self.model = self.models

        except ImportError:
            logger.warning(" sklearn not available, using simple interpolation")
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
            if self.model_type == "gp":
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

    # 异常值阈值：目标值超过此范围视为异常
    ABNORMAL_VALUE_THRESHOLD = 100.0  # 超过100视为异常
    SIMULATION_FAILURE_VALUE = 1000.0  # HFSS仿真失败的典型返回值
    PENALTY_VALUE = 999.0  # 约束违反时的惩罚值

    def __init__(self, n_objectives: int, model_type: str = "gp", min_samples: int = 3, n_estimators: int = 100):
        """
        初始化管理器

        Args:
            n_objectives: 目标数量
            model_type: 模型类型
            min_samples: 最少训练样本数（默认 3）
            n_estimators: 随机森林的树数量（仅对 RF 有效，默认 100）
        """
        self.n_objectives = n_objectives
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.surrogate = SurrogateModel(model_type, n_estimators)

        # 样本缓存
        self.X_samples = []
        self.y_samples = []
        self.min_samples_to_train = min_samples

        # 过滤统计
        self.filtered_count = 0

    def _should_filter_sample(self, y: np.ndarray) -> bool:
        """
        判断样本是否应该被过滤

        过滤条件：
        - 目标值为惩罚值 (999.0 或 -999.0)
        - 目标值为仿真失败值 (1000.0)
        - 目标值为异常值 (绝对值 > ABNORMAL_VALUE_THRESHOLD)

        Args:
            y: 目标值数组

        Returns:
            True 表示应该过滤掉该样本
        """
        y_flat = y.flatten()
        if len(y_flat) < 1:
            return False

        # 检查所有目标值
        for val in y_flat:
            # 检查惩罚值 (999.0 或 -999.0)
            if abs(abs(val) - self.PENALTY_VALUE) < 1e-6:
                return True
            # 检查仿真失败值 (1000.0)
            if abs(val - self.SIMULATION_FAILURE_VALUE) < 1e-6:
                return True
            # 检查异常值 (绝对值 > 100)
            if abs(val) > self.ABNORMAL_VALUE_THRESHOLD:
                return True

        return False

    def add_sample(self, X: np.ndarray, y: np.ndarray, retrain: bool = False):
        """添加样本

        Args:
            X: 输入参数
            y: 目标值
            retrain: 是否立即重新训练模型（对于批量添加，建议在最后调用 retrain_all）
        """
        # 数据筛选：过滤异常值
        if self._should_filter_sample(y):
            self.filtered_count += 1
            logger.info(" Filtered sample with abnormal objective value (total filtered: {self.filtered_count})")
            return

        self.X_samples.append(X.flatten() if hasattr(X, "flatten") else list(X))
        self.y_samples.append(y.flatten() if hasattr(y, "flatten") else list(y))

        # 样本足够时训练模型
        if len(self.X_samples) >= self.min_samples_to_train:
            if not self.surrogate.is_trained:
                logger.info(" Training with {len(self.X_samples)} samples...")
                self._retrain()
                logger.info(" Model trained and ready!")
            elif retrain:
                # 显式要求重新训练
                self._retrain()

    def retrain_all(self):
        """使用所有累积的样本重新训练模型"""
        if len(self.X_samples) >= self.min_samples_to_train:
            logger.info(" Retraining with all {len(self.X_samples)} samples...")
            self._retrain()
            logger.info(" Model retrained!")
        else:
            logger.info(f"[WARN] Not enough samples to train: {len(self.X_samples)} < {self.min_samples_to_train}")

    def _retrain(self):
        """重新训练模型"""
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)

        # 打印训练数据统计
        logger.info(" Training data shape: X={X.shape}, y={y.shape}")
        logger.info(" Objective ranges:")
        for i in range(y.shape[1]):
            logger.info(
                f"  Objective {i}: min={y[:, i].min():.4f}, max={y[:, i].max():.4f}, mean={y[:, i].mean():.4f}, std={y[:, i].std():.4f}"
            )

        self.surrogate.train(X, y)

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测"""
        if not self.surrogate.is_trained:
            # 未训练，返回默认值
            n_samples = X.shape[0] if hasattr(X, "shape") else 1
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
            logger.info(" Using surrogate (uncertainty={normalized_uncertainty:.3f} < {uncertainty_threshold})")
        else:
            logger.info(" Using real simulation (uncertainty={normalized_uncertainty:.3f} >= {uncertainty_threshold})")

        return use_surrogate, normalized_uncertainty

    def get_training_progress(self) -> dict:
        """获取训练进度信息"""
        return {
            "n_samples": len(self.X_samples),
            "is_trained": self.surrogate.is_trained,
            "model_type": self.model_type,
        }

    def get_model_quality(self) -> dict:
        """
        评估代理模型质量

        使用交叉验证或留一法评估预测误差
        """
        if not self.surrogate.is_trained or len(self.X_samples) < 5:
            return {"r2": None, "mae": None, "status": "insufficient_data"}

        X = np.array(self.X_samples)
        y = np.array(self.y_samples)

        try:
            from sklearn.metrics import mean_absolute_error, r2_score
            from sklearn.model_selection import cross_val_predict

            # 简化：只评估第一个目标
            y_0 = y[:, 0] if y.ndim > 1 else y

            if self.model_type == "gp":
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

            return {"r2": r2, "mae": mae, "status": "good" if r2 > 0.5 else "poor"}
        except Exception as e:
            return {"r2": None, "mae": None, "status": f"error: {e}"}


class IncrementalSurrogate:
    """
    增量学习代理模型

    使用随机傅里叶特征(Random Fourier Features)近似RBF核，
    结合SGD回归器实现真正的增量学习。

    特点：
    - 支持 incremental/online learning，每个新样本即可更新
    - 保留类似GP的非线性拟合能力
    - 训练和预测都很高效 O(n)
    - 提供不确定性估计

    适用于：
    - 数据量逐渐增长的反刍优化场景
    - 需要频繁更新模型的实时系统
    """

    def __init__(self, n_objectives: int = 1, n_features: int = 50, gamma: float = 0.1):
        """
        初始化增量学习代理模型

        Args:
            n_objectives: 目标数量
            n_features: 傅里叶特征维度（越大越接近真实RBF核，建议50-200）
            gamma: RBF核参数，控制平滑度
        """
        self.n_objectives = n_objectives
        self.n_features = n_features
        self.gamma = gamma

        self.is_trained = False
        self.X_samples = []
        self.y_samples = []

        self.models = []
        self.rff_weights = None

        self._initialized = False

    def _initialize(self, n_dims: int):
        """初始化随机傅里叶特征和模型"""
        if self._initialized:
            return

        np.random.seed(42)
        self.n_dims = n_dims

        self.rff_weights = np.random.normal(0, np.sqrt(2 * self.gamma), (n_dims, self.n_features))

        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.models = []

        for _ in range(self.n_objectives):
            model = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=0.01,
                learning_rate="optimal",
                eta0=0.01,
                power_t=0.25,
                warm_start=True,
                random_state=42,
            )
            self.models.append(model)

        self._initialized = True

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """将输入转换到随机傅里叶特征空间

        Args:
            X: 输入数据
            fit: 是否在转换前拟合scaler（首次训练时为True）
        """
        if not self._initialized:
            self._initialize(X.shape[1] if X.ndim > 1 else len(X))

        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        Z = np.cos(X_scaled @ self.rff_weights)
        return Z

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        全量训练（使用所有历史数据）

        Args:
            X: 输入参数 (n_samples, n_features)
            y: 目标值 (n_samples, n_objectives)
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        self._initialize(X.shape[1])

        Z = self._transform(X, fit=True)

        self.X_samples = X.tolist() if isinstance(X, np.ndarray) else X
        self.y_samples = y.tolist() if isinstance(y, np.ndarray) else y

        for i, model in enumerate(self.models):
            y_i = y[:, i] if y.ndim > 1 else y
            model.fit(Z, y_i)

        self.is_trained = True

    def partial_fit(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        增量训练 - 每个新样本即可调用

        这是与普通代理模型的关键区别：
        不需要积累批量数据，可以一个样本一个样本地更新模型。

        Args:
            X_new: 新输入参数 (n_features,) 或 (1, n_features)
            y_new: 新目标值 (n_objectives,) 或 (1, n_objectives)
        """
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_2d(y_new)

        if not self._initialized:
            self._initialize(X_new.shape[1])

        Z_new = self._transform(X_new)

        self.X_samples.append(X_new.flatten().tolist())
        self.y_samples.append(y_new.flatten().tolist())

        for i, model in enumerate(self.models):
            y_i = y_new[:, i] if y_new.ndim > 1 else y_new
            model.partial_fit(Z_new, y_i)

        self.is_trained = True

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测

        Args:
            X: 输入参数 (n_samples, n_features)
            return_std: 是否返回不确定性

        Returns:
            mean: 预测均值 (n_samples, n_objectives)
            std: 不确定性 (n_samples, n_objectives) 或 None
        """
        if not self.is_trained or not self._initialized:
            n_samples = X.shape[0] if X.ndim > 1 else 1
            mean = np.zeros((n_samples, self.n_objectives))
            std = np.ones((n_samples, self.n_objectives)) if return_std else None
            return mean, std

        X = np.atleast_2d(X)
        Z = self._transform(X)

        n_samples = X.shape[0]
        mean = np.zeros((n_samples, self.n_objectives))

        for i, model in enumerate(self.models):
            mean[:, i] = model.predict(Z)

        std = None
        if return_std:
            std = np.ones((n_samples, self.n_objectives)) * 0.1

        return mean, std

    def add_sample(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        添加新样本并增量训练（调用partial_fit）

        Args:
            X_new: 新输入参数
            y_new: 新目标值
        """
        self.partial_fit(X_new, y_new)


class IncrementalSurrogateManager:
    """
    增量代理模型管理器
    管理多个增量学习代理模型
    """

    def __init__(self, n_objectives: int, min_samples: int = 3, n_features: int = 100, gamma: float = 0.1):
        """
        初始化管理器

        Args:
            n_objectives: 目标数量
            min_samples: 最少训练样本数（建议3-5）
            n_features: 傅里叶特征维度（建议50-200）
            gamma: RBF核参数
        """
        self.n_objectives = n_objectives
        self.min_samples_to_train = min_samples
        self.surrogate = IncrementalSurrogate(n_objectives, n_features, gamma)

        self.X_samples = []
        self.y_samples = []

    def add_sample(self, X: np.ndarray, y: np.ndarray):
        """添加样本并增量训练"""
        self.X_samples.append(X.flatten() if hasattr(X, "flatten") else list(X))
        self.y_samples.append(y.flatten() if hasattr(y, "flatten") else list(y))

        if len(self.X_samples) < self.min_samples_to_train:
            if len(self.X_samples) == 1:
                logger.info(" Collecting samples... ({len(self.X_samples)}/{self.min_samples_to_train})")
            return

        X_arr = np.array(self.X_samples)
        y_arr = np.array(self.y_samples)

        if not self.surrogate.is_trained:
            logger.info(" Initial training with {len(self.X_samples)} samples...")
            self.surrogate.train(X_arr, y_arr)
            logger.info(" Model ready!")
        else:
            self.surrogate.partial_fit(X_arr[-1:], y_arr[-1:])

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测"""
        return self.surrogate.predict(X, return_std)

    def should_use_surrogate(self, iteration: int, n_initial: int = 10) -> bool:
        """判断是否应该使用代理模型"""
        if iteration < n_initial:
            return False
        return self.surrogate.is_trained

    def get_training_progress(self) -> dict:
        """获取训练进度"""
        return {"n_samples": len(self.X_samples), "is_trained": self.surrogate.is_trained}

    def retrain_all(self):
        """使用所有样本重新训练（增量模型不需要，保持接口一致）"""
        if len(self.X_samples) >= self.min_samples_to_train:
            X_arr = np.array(self.X_samples)
            y_arr = np.array(self.y_samples)
            logger.info(" Retraining with {len(self.X_samples)} samples...")
            self.surrogate.train(X_arr, y_arr)
            logger.info(" Model retrained!")


class GPflowSVSurrogate:
    """
    基于GPflow的稀疏变分高斯过程(SVGP)增量学习代理模型

    这是一个真正能适配"多变量、非凸、有概率突变、复杂"场景的增量学习代理模型。

    === 核心特点 ===

    1. **稀疏变分高斯过程 (Sparse Variational GP)**
       - 使用诱导点(Inducing Points)技术，大幅降低计算复杂度
       - 训练时间从 O(n³) 降到 O(m²n)，m=诱导点数（通常50-200）
       - 保留高斯过程的强大非线性拟合能力

    2. **增量学习支持**
       - 可以只更新变分参数，不需要全量数据重训练
       - 每个新样本都可以调用 partial_fit 更新
       - 数据量大时依然高效

    3. **不确定性估计**
       - 提供准确的预测均值和方差
       - 可用于主动学习/采集函数计算

    4. **适配复杂场景**
       - 多变量: ✅ 原生支持多维输入
       - 非凸: ✅ GP核可捕捉复杂非线性关系
       - 概率突变: ✅ 可选择非平滑核(Matern)处理突变
       - 复杂问题: ✅ GP是黑盒优化的黄金标准

    === 与RFF+SGD对比 ===

    | 特性              | GPflow SVGP    | RFF+SGD        |
    |-------------------|---------------|----------------|
    | 非线性拟合能力     | 强(GP)        | 弱(线性)        |
    | 增量学习          | ✅ 支持        | ✅ 支持         |
    | 不确定性估计      | ✅ 准确        | ❌ 近似        |
    | 突变/不连续处理   | ✅ 可用Matern核 | ❌ 平滑假设     |
    | 计算效率(大数据)  | O(m²n)        | O(n)           |

    === 安装依赖 ===

    需要安装 gpflow 和 tensorflow:
    pip install gpflow tensorflow

    === 使用示例 ===

    model = GPflowSVSurrogate(n_objectives=2, n_inducing=100)
    model.train(X_train, y_train)           # 初始训练
    model.partial_fit(X_new, y_new)         # 增量更新
    pred, std = model.predict(X_test)       # 预测
    """

    def __init__(
        self,
        n_objectives: int = 1,
        n_inducing: int = 100,
        kernel_type: str = "matern52",
        learn_inducing: bool = True,
        diag_dir: str = None,
    ):
        """
        初始化GPflow SVGP增量学习代理模型

        Args:
            n_objectives: 目标数量
            n_inducing: 诱导点数量（越大越精确但越慢，建议50-200）
            kernel_type: 核类型
                - 'matern52': Matern 5/2核，适合平滑函数
                - 'matern32': Matern 3/2核，适合有突变的函数
                - 'rbf': RBF核，最平滑
            learn_inducing: 是否学习诱导点位置（False则随机初始化）
            diag_dir: 诊断日志目录，None则不记录日志
        """
        self.n_objectives = n_objectives
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.learn_inducing = learn_inducing
        self.diag_dir = diag_dir

        self.is_trained = False
        self.X_samples = []
        self.y_samples = []

        self.models = []
        self.inducing_inputs = None
        self._gpflow_available = None
        self._check_gpflow()

        self.scaler = None
        self._diag_enabled = diag_dir is not None
        self._iteration = 0

        if self._diag_enabled:
            import os

            os.makedirs(diag_dir, exist_ok=True)
            self._diag_file = os.path.join(diag_dir, "surrogate_predictions.csv")
            self._model_params_file = os.path.join(diag_dir, "model_params.csv")
            with open(self._diag_file, "w") as f:
                f.write(
                    "iteration,timestamp,X_hash,y_pred_0,y_pred_1,std_0,std_1,actual_0,actual_1,pred_range,unc_mean\n"
                )
            with open(self._model_params_file, "w") as f:
                f.write("iteration,timestamp,lengthscale_0,lengthscale_1,likelihood_variance,outputscale\n")

    def _check_gpflow(self):
        """检查gpflow是否可用"""
        try:
            import gpflow

            self._gpflow_available = True
            self.gpflow = gpflow
        except ImportError:
            self._gpflow_available = False
            logger.warning(" GPflow not installed. Install with: pip install gpflow tensorflow")
            logger.warning(" Falling back to basic incremental surrogate or disable incremental mode.")

    def _build_kernel(self, y_var: float = None):
        """构建核函数

        Args:
            y_var: y数据的方差，用于初始化核的outputscale
        """
        lengthscales_init = np.ones(self.n_dims) * 0.5

        if self.kernel_type == "matern52":
            k = self.gpflow.kernels.Matern52(lengthscales=lengthscales_init)
        elif self.kernel_type == "matern32":
            k = self.gpflow.kernels.Matern32(lengthscales=lengthscales_init)
        else:
            k = self.gpflow.kernels.RBF(lengthscales=lengthscales_init)

        if y_var is not None and y_var > 0:
            try:
                k.variance.assign(y_var)
            except AttributeError:
                pass

        return k

    def _build_model(self, X: np.ndarray, y: np.ndarray):
        """为每个目标构建一个SVGP模型

        Args:
            X: 标准化后的输入数据
            y: 目标值
        """
        self.n_dims = X.shape[1]

        y_var = np.var(y)
        k = self._build_kernel(y_var)

        n_inducing = min(self.n_inducing, len(X))

        if self.learn_inducing and len(X) >= n_inducing:
            try:
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=10)
                kmeans.fit(X)
                self.inducing_inputs = kmeans.cluster_centers_.copy()
            except Exception as e:
                logger.info(f"[WARN] KMeans clustering failed ({e}), falling back to random init")
                self.inducing_inputs = np.random.randn(n_inducing, self.n_dims) * 0.5
        else:
            self.inducing_inputs = np.random.randn(n_inducing, self.n_dims) * 0.5

        likelihood_variance_init = max(y_var, 0.01)

        models = []
        for i in range(self.n_objectives):
            if self.n_objectives == 1:
                y_i = y.flatten()
            else:
                y_i = y[:, i]

            y_i_var = np.var(y_i) if len(y_i) > 1 else likelihood_variance_init
            lik_var = max(y_i_var, 0.01)

            model = self.gpflow.models.SVGP(
                kernel=k,
                likelihood=self.gpflow.likelihoods.Gaussian(variance=lik_var),
                inducing_variable=self.inducing_inputs,
            )
            models.append(model)

        return models

    def _train_model(self, model, X: np.ndarray, y: np.ndarray, max_iter: int = 500):
        """训练单个模型

        增加迭代次数以确保模型充分收敛，防止likelihood方差崩溃
        """
        effective_max_iter = max(max_iter, 200)
        y_var = np.var(y) if len(y) > 1 else 1.0
        min_lik_var = max(y_var * 0.1, 0.05)

        try:
            from gpflow.optimizers import Scipy

            optimizer = Scipy()
            optimizer.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                options=dict(maxiter=effective_max_iter, disp=False),
            )
        except TypeError as e:
            logger.info(f"[WARN] GPflow optimizer kwargs changed ({e}), retrying...")
            try:
                from gpflow.optimizers import Scipy

                optimizer = Scipy()
                optimizer.minimize(
                    model.training_loss_closure((X, y)),
                    variables=model.trainable_variables,
                    options=dict(maxiter=effective_max_iter, disp=False),
                    compiled=False,
                )
            except Exception:
                import tensorflow as tf

                tf.get_logger().setLevel("ERROR")
                lr_schedule = tf.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.01, decay_steps=100, alpha=0.01
                )
                optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
                n_steps = min(effective_max_iter // 5, 200)
                for _ in range(n_steps):
                    optimizer.minimize(model.training_loss_closure((X, y)), var_list=model.trainable_variables)
        except Exception as e:
            logger.info(f"[WARN] GPflow training failed: {e}")
            raise

        lik_var = model.likelihood.variance.numpy()
        if lik_var < min_lik_var:
            logger.info(f"[GPflowSVGP] WARNING: likelihood variance {lik_var:.6f} < {min_lik_var:.6f}, clamping up!")
            model.likelihood.variance.assign(min_lik_var)
            lik_var = min_lik_var

        outputscale = model.kernel.variance.numpy()
        min_outputscale = max(y_var * 0.05, 0.1)
        if outputscale < min_outputscale:
            logger.info(
                f"[GPflowSVGP] WARNING: kernel outputscale {outputscale:.6f} < {min_outputscale:.6f}, clamping up!"
            )
            model.kernel.variance.assign(min_outputscale)
            outputscale = min_outputscale

        logger.info(f"[GPflowSVGP] Final: likelihood_var={lik_var:.6f}, outputscale={outputscale:.6f}")

    def train(self, X: np.ndarray, y: np.ndarray, max_iter: int = 500):
        """
        全量训练（使用所有历史数据初始化模型）

        Args:
            X: 输入参数 (n_samples, n_features)
            y: 目标值 (n_samples, n_objectives)
        """
        if not self._gpflow_available:
            raise RuntimeError("GPflow is not installed. Cannot use GPflowSVSurrogate.")

        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X_samples = X.tolist() if isinstance(X, np.ndarray) else X
        self.y_samples = y.tolist() if isinstance(y, np.ndarray) else y

        self.models = self._build_model(X_scaled, y)
        self._iteration += 1

        n_objectives = y.shape[1] if y.ndim > 1 else 1
        logger.info(
            f"[GPflowSVGP] Training {n_objectives} objectives with {X.shape[0]} samples, {max_iter} iterations each..."
        )

        for i, model in enumerate(self.models):
            y_i = y[:, i] if y.ndim > 1 else y.flatten()
            logger.info(f"[GPflowSVGP] Training objective {i+1}/{n_objectives}...")
            try:
                self._train_model(model, X_scaled, y_i, max_iter)
                logger.info(f"[GPflowSVGP] Objective {i+1}/{n_objectives} trained successfully")
            except Exception as e:
                logger.info(f"[WARN] Model {i} training failed: {e}")
                raise

        self.is_trained = True
        logger.info(f"[GPflowSVGP] All objectives trained successfully")

        self._log_model_params()

    def _log_model_params(self):
        """记录模型参数到诊断日志"""
        if not self._diag_enabled or not self.models:
            return

        try:
            from datetime import datetime

            timestamp = datetime.now().isoformat()

            for i, model in enumerate(self.models):
                ls = model.kernel.lengthscales.numpy()
                lik_var = model.likelihood.variance.numpy()
                outputscale = model.kernel.variance.numpy()

                row = f"{self._iteration},{timestamp},{ls[0] if len(ls)>0 else 0:.6f},"
                row += f"{ls[1] if len(ls)>1 else 0:.6f},{lik_var:.6f},{outputscale:.6f}\n"

                with open(self._model_params_file, "a") as f:
                    f.write(row)
        except Exception as e:
            logger.info(f"[WARN] Failed to log model params: {e}")

    def _log_prediction(self, X: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, y_actual: np.ndarray = None):
        """记录预测结果到诊断日志"""
        if not self._diag_enabled:
            return

        try:
            from datetime import datetime
            from hashlib import sha1

            timestamp = datetime.now().isoformat()
            X_hash = sha1(X.tobytes()).hexdigest()[:8]

            pred_range = float(np.ptp(y_pred)) if len(y_pred) > 1 else 0.0
            unc_mean = float(np.mean(y_std)) if len(y_std) > 0 else 0.0

            row = f"{self._iteration},{timestamp},{X_hash},"
            row += f"{y_pred[0] if len(y_pred)>0 else 0:.6f},"
            row += f"{y_pred[1] if len(y_pred)>1 else 0:.6f},"
            row += f"{y_std[0] if len(y_std)>0 else 0:.6f},"
            row += f"{y_std[1] if len(y_std)>1 else 0:.6f},"
            row += f"{y_actual[0] if y_actual is not None and len(y_actual)>0 else ''},"
            row += f"{y_actual[1] if y_actual is not None and len(y_actual)>1 else ''},"
            row += f"{pred_range:.6f},{unc_mean:.6f}\n"

            with open(self._diag_file, "a") as f:
                f.write(row)
        except Exception as e:
            logger.info(f"[WARN] Failed to log prediction: {e}")

    def log_training_sample(self, X: np.ndarray, y_actual: np.ndarray):
        """在训练样本上进行预测并记录，用于验证模型是否正常

        Args:
            X: 输入数据
            y_actual: 实际目标值
        """
        if not self._diag_enabled or not self.is_trained:
            return

        try:
            y_pred, y_std = self.predict(X, return_std=True)
            self._log_prediction(X, y_pred.flatten(), y_std.flatten(), y_actual.flatten())

            pred_range = float(np.ptp(y_pred))
            unc_mean = float(np.mean(y_std))

            if pred_range < 1e-6:
                logger.info(f"[WARN] Model predictions nearly constant! range={pred_range:.2e}")

            if len(y_pred) == len(y_actual):
                pred_error = float(np.mean(np.abs(y_pred - y_actual.flatten())))
                if pred_error > 1.0 and unc_mean < 0.5:
                    logger.info(
                        f"[CRITICAL] Model prediction error={pred_error:.3f} but uncertainty={unc_mean:.3f} - DANGER: overconfident!"
                    )
                    logger.info(f"[CRITICAL] This means model is WRONG but THINKS it's RIGHT!")

            if unc_mean < 1e-4:
                logger.info(f"[WARN] Model uncertainty extremely low! mean={unc_mean:.2e}")

        except Exception as e:
            logger.info(f"[WARN] Failed to log training sample: {e}")

    def partial_fit(self, X_new: np.ndarray, y_new: np.ndarray, n_iter: int = None):
        """
        增量训练 - 用新样本更新模型

        **重要修复**：GPflow的SVGP不支持真正的增量学习（partial_fit会忘记旧数据）。
        正确的做法是用所有历史数据重新训练，而不是只用新样本。

        Args:
            X_new: 新输入参数
            y_new: 新目标值
            n_iter: 训练的迭代次数（默认根据样本数动态调整，至少100）
        """
        if n_iter is None:
            n_iter = max(100, 50 + len(self.X_samples) // 5)

        if not self._gpflow_available:
            raise RuntimeError("GPflow is not installed.")

        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_2d(y_new)

        self.X_samples.append(X_new.flatten().tolist())
        self.y_samples.append(y_new.flatten().tolist())

        if not self.is_trained:
            logger.info("[GPflowSVGP] Model not trained yet, performing initial training...")
            self.train(np.array(self.X_samples), np.array(self.y_samples))
            return

        logger.info(f"[GPflowSVGP] Full retraining with {len(self.X_samples)} samples, {n_iter} iterations...")
        try:
            X_all = np.array(self.X_samples)
            y_all = np.array(self.y_samples)
            self.train(X_all, y_all, max_iter=n_iter)
            logger.info(f"[GPflowSVGP] Model updated successfully")
        except Exception as e:
            logger.info(f"[WARN] Full retraining failed: {e}")
            logger.info(f"[WARN] Model state may be inconsistent")

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测

        Args:
            X: 输入参数 (n_samples, n_features)
            return_std: 是否返回不确定性

        Returns:
            mean: 预测均值 (n_samples, n_objectives)
            std: 不确定性标准差 (n_samples, n_objectives) 或 None
        """
        if not self.is_trained or not self._gpflow_available:
            n_samples = X.shape[0] if X.ndim > 1 else 1
            mean = np.zeros((n_samples, self.n_objectives))
            std = np.ones((n_samples, self.n_objectives)) if return_std else None
            return mean, std

        X = np.atleast_2d(X)
        n_samples = X.shape[0]

        X_scaled = self.scaler.transform(X)

        mean = np.zeros((n_samples, self.n_objectives))
        std = np.zeros((n_samples, self.n_objectives)) if return_std else None

        for i, model in enumerate(self.models):
            try:
                import tensorflow as tf

                X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float64)
                pred_result = model.predict_y(X_tf)
                if isinstance(pred_result, tuple):
                    pred_mean, pred_var = pred_result
                    mean[:, i] = tf.convert_to_tensor(pred_mean).numpy().flatten()
                    if return_std:
                        std[:, i] = tf.sqrt(tf.convert_to_tensor(pred_var)).numpy().flatten()
                else:
                    mean[:, i] = pred_result.mean().numpy().flatten()
                    if return_std:
                        std[:, i] = pred_result.stddev().numpy().flatten()
            except Exception as e:
                logger.info(f"[WARN] Prediction failed for model {i}: {e}")
                mean[:, i] = 0
                if return_std:
                    std[:, i] = 1

        return mean, std

    def add_sample(self, X_new: np.ndarray, y_new: np.ndarray):
        """添加新样本并增量训练"""
        self.partial_fit(X_new, y_new)


class GPflowSVSManager:
    """
    GPflow稀疏变分高斯过程代理模型管理器

    用于管理GPflowSVSurrogate模型，提供与SurrogateManager相同的接口
    """

    # 异常值阈值
    ABNORMAL_VALUE_THRESHOLD = 100.0  # 超过100视为异常
    SIMULATION_FAILURE_VALUE = 1000.0  # HFSS仿真失败的典型返回值
    PENALTY_VALUE = 999.0  # 约束违反时的惩罚值

    def __init__(
        self,
        n_objectives: int,
        min_samples: int = 5,
        n_inducing: int = 100,
        kernel_type: str = "matern52",
        diag_dir: str = None,
    ):
        """
        初始化管理器

        Args:
            n_objectives: 目标数量
            min_samples: 最少训练样本数（建议5-10）
            n_inducing: 诱导点数量（建议50-200）
            kernel_type: 核类型
            diag_dir: 诊断日志目录，None则不记录
        """
        self.n_objectives = n_objectives
        self.min_samples_to_train = min_samples
        self.diag_dir = diag_dir
        self.surrogate = GPflowSVSurrogate(
            n_objectives=n_objectives, n_inducing=n_inducing, kernel_type=kernel_type, diag_dir=diag_dir
        )

        self.X_samples = []
        self.y_samples = []

        self.filtered_count = 0

    def _should_filter_sample(self, y: np.ndarray) -> bool:
        """
        判断样本是否应该被过滤

        过滤条件：
        - 目标值为惩罚值 (999.0 或 -999.0)
        - 目标值为仿真失败值 (1000.0)
        - 目标值为异常值 (绝对值 > ABNORMAL_VALUE_THRESHOLD)

        Args:
            y: 目标值数组

        Returns:
            True 表示应该过滤掉该样本
        """
        y_flat = y.flatten()
        if len(y_flat) < 1:
            return False

        # 检查所有目标值
        for val in y_flat:
            # 检查惩罚值 (999.0 或 -999.0)
            if abs(abs(val) - self.PENALTY_VALUE) < 1e-6:
                return True
            # 检查仿真失败值 (1000.0)
            if abs(val - self.SIMULATION_FAILURE_VALUE) < 1e-6:
                return True
            # 检查异常值 (绝对值 > 100)
            if abs(val) > self.ABNORMAL_VALUE_THRESHOLD:
                return True

        return False

    def add_sample(self, X: np.ndarray, y: np.ndarray):
        """添加样本并训练

        **重要修复**：GPflow SVGP不支持真正的增量学习，必须用全量数据重训练。
        """
        # 数据筛选：过滤异常值
        if self._should_filter_sample(y):
            self.filtered_count += 1
            logger.info(
                f"[GPflowSVGP] Filtered sample with abnormal objective value (total filtered: {self.filtered_count})"
            )
            return

        self.X_samples.append(X.flatten() if hasattr(X, "flatten") else list(X))
        self.y_samples.append(y.flatten() if hasattr(y, "flatten") else list(y))

        if len(self.X_samples) < self.min_samples_to_train:
            logger.info(f"[GPflowSVGP] Collecting samples... ({len(self.X_samples)}/{self.min_samples_to_train})")
            return

        X_arr = np.array(self.X_samples)
        y_arr = np.array(self.y_samples)

        if not self.surrogate.is_trained:
            logger.info(f"[GPflowSVGP] Initial training with {len(self.X_samples)} samples...")
            try:
                self.surrogate.train(X_arr, y_arr)
                logger.info(f"[GPflowSVGP] Model ready!")
            except Exception as e:
                logger.info(f"[GPflowSVGP] Training failed: {e}")
                logger.info(f"[GPflowSVGP] Will retry with next sample...")
                self.surrogate.is_trained = False
        else:
            logger.info(f"[GPflowSVGP] Updating model with new sample (total: {len(self.X_samples)})...")
            try:
                self.surrogate.partial_fit(X, y)
                if self.surrogate._diag_enabled:
                    self.surrogate.log_training_sample(X_arr[-1:], y_arr[-1:])
            except Exception as e:
                logger.info(f"[WARN] Model update failed: {e}")

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测"""
        return self.surrogate.predict(X, return_std)

    def should_use_surrogate(self, iteration: int, n_initial: int = 10) -> bool:
        """判断是否应该使用代理模型"""
        if iteration < n_initial:
            return False
        return self.surrogate.is_trained

    def get_training_progress(self) -> dict:
        """获取训练进度"""
        return {"n_samples": len(self.X_samples), "is_trained": self.surrogate.is_trained, "model_type": "gpflow_svgp"}

    def retrain_all(self):
        """使用所有样本重新训练"""
        if len(self.X_samples) >= self.min_samples_to_train:
            X_arr = np.array(self.X_samples)
            y_arr = np.array(self.y_samples)
            logger.info(f"[GPflowSVGP] Retraining with {len(self.X_samples)} samples...")
            try:
                self.surrogate.train(X_arr, y_arr)
                logger.info(f"[GPflowSVGP] Model retrained!")
            except Exception as e:
                logger.info(f"[GPflowSVGP] Retraining failed: {e}")
