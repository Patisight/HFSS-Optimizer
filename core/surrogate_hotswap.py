"""
支持热替换的代理模型管理器

在原有SurrogateManager基础上，增加：
1. 模型热替换功能
2. 与SharedMemoryManager集成
3. 自动检查并加载新模型

使用方式：
    # 初始化
    manager = SurrogateManagerWithHotSwap(
        n_objectives=2,
        model_type='gp',
        shared_dir='./shared_data'
    )

    # 检查并执行热替换
    if manager.check_and_swap():
        logger.info(f"Model swapped!")

    # 使用当前模型预测
    mean, std = manager.predict(X, return_std=True)
"""

import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from core.shared_memory import HotSwapManager, SharedMemoryManager
from core.surrogate import GPflowSVSManager, SurrogateManager


class SurrogateManagerWithHotSwap:
    """
    支持热替换的代理模型管理器

    核心功能：
    1. 代理模型预测（不训练）
    2. 模型热替换（从共享内存加载新模型）
    3. 版本管理
    """

    def __init__(
        self,
        n_objectives: int,
        model_type: str = "gp",
        shared_dir: str = "./shared_data",
        min_samples: int = 5,
        n_estimators: int = 100,
        **model_params,
    ):
        """
        初始化支持热替换的代理模型管理器

        Args:
            n_objectives: 目标数量
            model_type: 模型类型 ('gp', 'rf', 'gpflow_svgp')
            shared_dir: 共享数据目录
            min_samples: 最少训练样本数（用于初始训练）
            n_estimators: 随机森林树数量（仅对RF有效）
            **model_params: 其他模型参数
        """
        self.n_objectives = n_objectives
        self.model_type = model_type
        self.shared_dir = shared_dir
        self.min_samples = min_samples
        self.n_estimators = n_estimators
        self.model_params = model_params

        # 创建内部管理器
        if model_type == "gpflow_svgp":
            n_inducing = model_params.get("n_inducing", 100)
            kernel_type = model_params.get("kernel_type", "matern52")
            self.surrogate = GPflowSVSManager(
                n_objectives=n_objectives, min_samples=min_samples, n_inducing=n_inducing, kernel_type=kernel_type
            )
        else:
            self.surrogate = SurrogateManager(
                n_objectives=n_objectives, model_type=model_type, min_samples=min_samples, n_estimators=n_estimators
            )

        # 共享内存管理器
        self.shared_memory = SharedMemoryManager(shared_dir)

        # 热替换管理器
        self.hot_swap_manager = HotSwapManager(self.shared_memory)

        # 模型锁（保证热替换原子性）
        self.model_lock = threading.Lock()

        # 统计
        self.n_swaps = 0
        self.swap_history = []

    def initialize_from_shared_memory(self) -> bool:
        """
        从共享内存初始化模型

        如果共享内存中有已训练的模型，加载它。

        Returns:
            是否成功加载
        """
        model_state = self.shared_memory.load_model_state()

        if model_state is None:
            logger.info(f"[HotSwap] No existing model found in shared memory")
            return False

        try:
            # 恢复模型状态
            self.surrogate.X_samples = model_state.get("X_samples", [])
            self.surrogate.y_samples = model_state.get("y_samples", [])
            self.surrogate.surrogate.is_trained = model_state.get("is_trained", False)

            # 恢复模型特定参数
            model_params = model_state.get("model_params", {})

            if "model" in model_params:
                self.surrogate.surrogate.model = model_params["model"]

            if "models" in model_params:
                self.surrogate.surrogate.models = model_params["models"]

            # 更新版本
            version_info = self.shared_memory.get_model_version()
            self.hot_swap_manager.current_version = version_info.get("version", 0)

            logger.info(
                f"[HotSwap] Model loaded from shared memory: version={self.hot_swap_manager.current_version}, "
                f"samples={len(self.surrogate.X_samples)}"
            )

            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to load model from shared memory: {e}")
            import traceback

            traceback.print_exc()
            return False

    def check_and_swap(self) -> bool:
        """
        检查并执行热替换

        Returns:
            是否执行了替换
        """
        with self.model_lock:
            # 检查是否有新模型
            if not self.shared_memory.has_new_model(self.hot_swap_manager.current_version):
                return False

            try:
                # 加载新模型状态
                model_state = self.shared_memory.load_model_state()

                if model_state is None:
                    return False

                # 备份当前模型（用于回滚）
                backup_X = self.surrogate.X_samples
                backup_y = self.surrogate.y_samples
                backup_is_trained = self.surrogate.surrogate.is_trained
                backup_model = getattr(self.surrogate.surrogate, "model", None)
                backup_models = getattr(self.surrogate.surrogate, "models", None)

                # 应用新模型状态
                self.surrogate.X_samples = model_state.get("X_samples", [])
                self.surrogate.y_samples = model_state.get("y_samples", [])
                self.surrogate.surrogate.is_trained = model_state.get("is_trained", False)

                model_params = model_state.get("model_params", {})

                if "model" in model_params:
                    self.surrogate.surrogate.model = model_params["model"]

                if "models" in model_params:
                    self.surrogate.surrogate.models = model_params["models"]

                # 更新版本
                old_version = self.hot_swap_manager.current_version
                version_info = self.shared_memory.get_model_version()
                self.hot_swap_manager.current_version = version_info.get("version", 0)

                # 记录历史
                from datetime import datetime

                self.swap_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "old_version": old_version,
                        "new_version": self.hot_swap_manager.current_version,
                        "n_samples": len(self.surrogate.X_samples),
                        "model_quality": version_info.get("model_quality"),
                    }
                )

                self.n_swaps += 1

                logger.info(
                    f"\n[HotSwap] ✅ Model swapped: v{old_version} -> v{self.hot_swap_manager.current_version} "
                    f"(samples={len(self.surrogate.X_samples)})"
                )

                return True

            except Exception as e:
                logger.error(f"[ERROR] Hot swap failed: {e}")
                import traceback

                traceback.print_exc()

                # 回滚
                self.surrogate.X_samples = backup_X
                self.surrogate.y_samples = backup_y
                self.surrogate.surrogate.is_trained = backup_is_trained

                if backup_model is not None:
                    self.surrogate.surrogate.model = backup_model

                if backup_models is not None:
                    self.surrogate.surrogate.models = backup_models

                logger.info(f"[HotSwap] ❌ Rolled back to version {self.hot_swap_manager.current_version}")

                return False

    def add_sample(self, X: np.ndarray, y: np.ndarray):
        """
        添加样本（用于初始训练）

        注意：在双线架构中，优化线通常不调用此方法。
        此方法仅用于初始训练阶段。

        Args:
            X: 输入参数
            y: 目标值
        """
        with self.model_lock:
            self.surrogate.add_sample(X, y)

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
        with self.model_lock:
            return self.surrogate.predict(X, return_std)

    def should_use_surrogate(self, iteration: int, n_initial: int = 10) -> bool:
        """
        判断是否应该使用代理模型

        Args:
            iteration: 当前迭代次数
            n_initial: 初始真实评估次数

        Returns:
            是否使用代理模型
        """
        return self.surrogate.should_use_surrogate(iteration, n_initial)

    def get_training_progress(self) -> dict:
        """获取训练进度信息"""
        return {
            "n_samples": len(self.surrogate.X_samples),
            "is_trained": self.surrogate.surrogate.is_trained,
            "model_type": self.model_type,
            "model_version": self.hot_swap_manager.current_version,
            "n_swaps": self.n_swaps,
        }

    def get_model_quality(self) -> dict:
        """获取模型质量"""
        return self.surrogate.get_model_quality()

    def get_current_version(self) -> int:
        """获取当前模型版本"""
        return self.hot_swap_manager.current_version

    def get_swap_history(self) -> list:
        """获取热替换历史"""
        return self.swap_history

    def is_trained(self) -> bool:
        """模型是否已训练"""
        return self.surrogate.surrogate.is_trained

    def get_n_samples(self) -> int:
        """获取训练样本数"""
        return len(self.surrogate.X_samples)


class DualLineSurrogateManager:
    """
    双线架构代理模型管理器

    专门为双线架构设计，提供：
    1. 初始训练支持（优化线首次运行）
    2. 模型热替换支持（训练线训练完成后）
    3. 评估数据写入共享内存
    """

    def __init__(
        self,
        n_objectives: int,
        model_type: str = "gp",
        shared_dir: str = "./shared_data",
        min_samples: int = 5,
        n_estimators: int = 100,
        **model_params,
    ):
        """
        初始化双线架构代理模型管理器

        Args:
            n_objectives: 目标数量
            model_type: 模型类型
            shared_dir: 共享数据目录
            min_samples: 最少训练样本数
            n_estimators: 随机森林树数量
            **model_params: 其他模型参数
        """
        self.n_objectives = n_objectives
        self.model_type = model_type
        self.shared_dir = shared_dir

        # 内部管理器（支持热替换）
        self.manager = SurrogateManagerWithHotSwap(
            n_objectives=n_objectives,
            model_type=model_type,
            shared_dir=shared_dir,
            min_samples=min_samples,
            n_estimators=n_estimators,
            **model_params,
        )

        # 共享内存管理器
        self.shared_memory = SharedMemoryManager(shared_dir)

        # 初始训练标志
        self.initial_training_done = False
        self.initial_samples_needed = min_samples

    def add_sample(self, X: np.ndarray, y: np.ndarray, is_real: bool = True):
        """
        添加样本

        在双线架构中：
        - 初始阶段：优化线训练初始模型
        - 后续阶段：只写入共享内存，不训练

        Args:
            X: 输入参数
            y: 目标值
            is_real: 是否真实仿真
        """
        # 写入共享内存
        from datetime import datetime

        eval_data = {
            "eval_id": self.shared_memory.get_evaluation_count() + 1,
            "timestamp": datetime.now().isoformat(),
            "parameters": X.flatten().tolist() if hasattr(X, "flatten") else list(X),
            "objectives": y.flatten().tolist() if hasattr(y, "flatten") else list(y),
            "is_real": is_real,
        }
        self.shared_memory.append_evaluation(eval_data)

        # 初始训练阶段
        if not self.initial_training_done:
            self.manager.add_sample(X, y)

            # 检查是否完成初始训练
            if self.manager.get_n_samples() >= self.initial_samples_needed:
                if self.manager.is_trained():
                    self.initial_training_done = True
                    logger.info(f"[DualLine] Initial training completed with {self.manager.get_n_samples()} samples")

        # 后续阶段：检查热替换
        if self.initial_training_done:
            self.manager.check_and_swap()

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测目标值

        Args:
            X: 输入参数
            return_std: 是否返回不确定性

        Returns:
            mean: 预测均值
            std: 不确定性
        """
        # 先检查热替换
        self.manager.check_and_swap()

        return self.manager.predict(X, return_std)

    def should_use_surrogate(self, iteration: int, n_initial: int = 10) -> bool:
        """判断是否应该使用代理模型"""
        return self.manager.should_use_surrogate(iteration, n_initial)

    def get_training_progress(self) -> dict:
        """获取训练进度"""
        progress = self.manager.get_training_progress()
        progress["initial_training_done"] = self.initial_training_done
        return progress

    def get_model_quality(self) -> dict:
        """获取模型质量"""
        return self.manager.get_model_quality()

    def get_current_version(self) -> int:
        """获取当前模型版本"""
        return self.manager.get_current_version()

    def get_swap_history(self) -> list:
        """获取热替换历史"""
        return self.manager.get_swap_history()

    def is_trained(self) -> bool:
        """模型是否已训练"""
        return self.manager.is_trained()

    def get_n_samples(self) -> int:
        """获取训练样本数"""
        return self.manager.get_n_samples()

    def check_for_model_update(self) -> bool:
        """
        检查是否有模型更新

        Returns:
            是否有新模型
        """
        return self.manager.check_and_swap()
