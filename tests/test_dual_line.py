"""
双线架构单元测试

测试核心组件：
1. SharedMemoryManager - 共享内存管理器
2. HotSwapManager - 热替换管理器
3. SurrogateManagerWithHotSwap - 支持热替换的代理模型管理器
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.shared_memory import HotSwapManager, SharedMemoryManager
from core.surrogate_hotswap import SurrogateManagerWithHotSwap


class TestSharedMemoryManager(unittest.TestCase):
    """测试SharedMemoryManager"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.shared_memory = SharedMemoryManager(self.test_dir)

    def tearDown(self):
        """测试后清理"""
        self.shared_memory.cleanup()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.exists(self.shared_memory.control_file))
        self.assertTrue(os.path.exists(self.shared_memory.status_file))

    def test_append_and_get_evaluations(self):
        """测试评估数据的追加和读取"""
        # 追加数据
        eval_data = {"eval_id": 1, "parameters": [1.0, 2.0, 3.0], "objectives": [0.5, -10.0], "is_real": True}

        self.shared_memory.append_evaluation(eval_data)

        # 读取数据
        evals = self.shared_memory.get_all_evaluations()

        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["eval_id"], 1)
        self.assertEqual(evals[0]["parameters"], [1.0, 2.0, 3.0])

    def test_get_new_evaluations(self):
        """测试增量读取评估数据"""
        # 追加多条数据
        for i in range(5):
            self.shared_memory.append_evaluation(
                {"eval_id": i + 1, "parameters": [float(i), float(i + 1)], "objectives": [float(i)], "is_real": True}
            )

        # 第一次读取
        new_evals, count = self.shared_memory.get_new_evaluations(0)
        self.assertEqual(len(new_evals), 5)
        self.assertEqual(count, 5)

        # 再追加2条
        for i in range(5, 7):
            self.shared_memory.append_evaluation(
                {"eval_id": i + 1, "parameters": [float(i), float(i + 1)], "objectives": [float(i)], "is_real": True}
            )

        # 增量读取
        new_evals, count = self.shared_memory.get_new_evaluations(5)
        self.assertEqual(len(new_evals), 2)
        self.assertEqual(count, 7)

    def test_save_and_load_model_state(self):
        """测试模型状态的保存和加载"""
        # 保存模型状态
        model_state = {
            "model_type": "gp",
            "n_objectives": 2,
            "X_samples": [[1.0, 2.0], [3.0, 4.0]],
            "y_samples": [[0.5, -10.0], [0.6, -12.0]],
            "is_trained": True,
        }

        self.shared_memory.save_model_state(model_state, n_samples=2, model_quality={"r2": 0.85})

        # 加载模型状态
        loaded_state = self.shared_memory.load_model_state()

        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state["model_type"], "gp")
        self.assertEqual(len(loaded_state["X_samples"]), 2)

        # 检查版本
        version_info = self.shared_memory.get_model_version()
        self.assertEqual(version_info["version"], 1)
        self.assertEqual(version_info["n_samples"], 2)

    def test_control_signals(self):
        """测试控制信号"""
        # 发送优化线信号
        self.shared_memory.send_optimizer_signal("running", {"iteration": 10})

        # 读取信号
        control = self.shared_memory.read_control_signals()
        self.assertEqual(control["optimizer_signal"], "running")
        self.assertEqual(control["optimizer_data"]["iteration"], 10)

        # 发送训练线信号
        self.shared_memory.send_trainer_signal("training")

        control = self.shared_memory.read_control_signals()
        self.assertEqual(control["trainer_signal"], "training")

    def test_status_update(self):
        """测试状态更新"""
        # 更新优化线状态
        self.shared_memory.update_optimizer_status(
            {"status": "running", "iteration": 5, "n_real_evals": 20, "n_surrogate_evals": 50, "model_version": 2}
        )

        # 更新训练线状态
        self.shared_memory.update_trainer_status(
            {"status": "model_ready", "n_samples": 30, "model_version": 2, "model_quality": {"r2": 0.88, "mae": 0.12}}
        )

        # 获取完整状态
        status = self.shared_memory.get_full_status()

        self.assertEqual(status["optimizer"]["status"], "running")
        self.assertEqual(status["optimizer"]["iteration"], 5)
        self.assertEqual(status["trainer"]["status"], "model_ready")
        self.assertEqual(status["trainer"]["n_samples"], 30)


class TestHotSwapManager(unittest.TestCase):
    """测试HotSwapManager"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.shared_memory = SharedMemoryManager(self.test_dir)
        self.hot_swap_manager = HotSwapManager(self.shared_memory)

    def tearDown(self):
        """测试后清理"""
        self.shared_memory.cleanup()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_has_new_model(self):
        """测试新模型检测"""
        # 初始状态：没有新模型
        self.assertFalse(self.shared_memory.has_new_model(0))

        # 保存一个模型
        model_state = {"model_type": "gp", "X_samples": [[1.0, 2.0]], "y_samples": [[0.5, -10.0]], "is_trained": True}
        self.shared_memory.save_model_state(model_state, n_samples=1)

        # 现在有新模型
        self.assertTrue(self.shared_memory.has_new_model(0))
        self.assertFalse(self.shared_memory.has_new_model(1))


class TestSurrogateManagerWithHotSwap(unittest.TestCase):
    """测试SurrogateManagerWithHotSwap"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.manager = SurrogateManagerWithHotSwap(
            n_objectives=2, model_type="gp", shared_dir=self.test_dir, min_samples=3
        )

    def tearDown(self):
        """测试后清理"""
        self.manager.shared_memory.cleanup()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initial_training(self):
        """测试初始训练"""
        # 添加样本
        for i in range(5):
            X = np.array([float(i), float(i + 1)])
            y = np.array([float(i) * 0.5, -float(i) * 2.0])
            self.manager.add_sample(X, y)

        # 检查是否已训练
        self.assertTrue(self.manager.is_trained())
        self.assertEqual(self.manager.get_n_samples(), 5)

    def test_hot_swap(self):
        """测试热替换"""
        # 初始训练
        for i in range(5):
            X = np.array([float(i), float(i + 1)])
            y = np.array([float(i) * 0.5, -float(i) * 2.0])
            self.manager.add_sample(X, y)

        initial_version = self.manager.get_current_version()

        # 模拟训练线保存新模型
        new_model_state = {
            "model_type": "gp",
            "n_objectives": 2,
            "X_samples": [[float(i), float(i + 1)] for i in range(10)],
            "y_samples": [[float(i) * 0.5, -float(i) * 2.0] for i in range(10)],
            "is_trained": True,
        }
        self.manager.shared_memory.save_model_state(new_model_state, n_samples=10)

        # 检查并执行热替换
        swapped = self.manager.check_and_swap()

        self.assertTrue(swapped)
        self.assertEqual(self.manager.get_n_samples(), 10)
        self.assertGreater(self.manager.get_current_version(), initial_version)

    def test_predict(self):
        """测试预测功能"""
        # 训练
        for i in range(5):
            X = np.array([float(i), float(i + 1)])
            y = np.array([float(i) * 0.5, -float(i) * 2.0])
            self.manager.add_sample(X, y)

        # 预测
        X_test = np.array([[2.5, 3.5]])
        mean, std = self.manager.predict(X_test, return_std=True)

        self.assertEqual(mean.shape, (1, 2))
        self.assertEqual(std.shape, (1, 2))


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_optimizer_trainer_communication(self):
        """测试优化线和训练线通信"""
        # 创建共享内存
        shared_memory = SharedMemoryManager(self.test_dir)

        # 模拟优化线写入评估数据
        for i in range(10):
            shared_memory.append_evaluation(
                {
                    "eval_id": i + 1,
                    "parameters": [float(i), float(i + 1)],
                    "objectives": [float(i) * 0.5, -float(i) * 2.0],
                    "is_real": True,
                }
            )

        # 模拟训练线读取数据
        evals = shared_memory.get_all_evaluations()
        self.assertEqual(len(evals), 10)

        # 模拟训练线保存模型
        model_state = {
            "model_type": "gp",
            "X_samples": [e["parameters"] for e in evals],
            "y_samples": [e["objectives"] for e in evals],
            "is_trained": True,
        }
        shared_memory.save_model_state(model_state, n_samples=10)

        # 模拟优化线加载模型
        loaded_state = shared_memory.load_model_state()
        self.assertIsNotNone(loaded_state)
        self.assertEqual(len(loaded_state["X_samples"]), 10)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSharedMemoryManager))
    suite.addTests(loader.loadTestsFromTestCase(TestHotSwapManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSurrogateManagerWithHotSwap))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
