"""
训练线进程模块

独立的训练进程，负责：
1. 监听评估数据变化
2. 使用所有历史数据全量训练代理模型
3. 保存模型状态到共享内存
4. 发送模型就绪信号

运行模式：
- 作为独立进程运行
- 通过SharedMemoryManager与优化线通信
- 支持多种代理模型类型（GP, RF, GPflow-SVGP）
"""

import os
import sys
import time
import json
import signal
import argparse
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.shared_memory import SharedMemoryManager
from core.surrogate import SurrogateManager, GPflowSVSManager


class TrainerProcess:
    """
    训练线进程
    
    独立运行，负责代理模型的全量训练。
    """
    
    def __init__(self, config: Dict):
        """
        初始化训练线进程
        
        Args:
            config: 配置字典，包含：
                - shared_dir: 共享数据目录
                - model_type: 代理模型类型 ('gp', 'rf', 'gpflow_svgp')
                - n_objectives: 目标数量
                - min_samples: 最少训练样本数
                - retrain_interval: 重训练间隔（新样本数）
                - model_params: 模型特定参数
        """
        self.config = config
        
        # 共享内存
        self.shared_dir = config.get('shared_dir', './shared_data')
        self.shared_memory = SharedMemoryManager(self.shared_dir)
        
        # 模型配置
        self.model_type = config.get('model_type', 'gp')
        self.n_objectives = config.get('n_objectives', 1)
        self.min_samples = config.get('min_samples', 5)
        # 至少积累多少新样本才触发训练（避免训练过于频繁）
        self.min_new_samples_to_train = config.get('min_new_samples_to_train', 5)
        self.model_params = config.get('model_params', {})
        
        # 训练状态
        self.running = False
        self.last_train_count = 0
        self.current_version = 0
        self.n_trains = 0
        
        # 代理模型管理器
        self.surrogate_manager = None
        
        # 性能统计
        self.train_times = []
        self.model_qualities = []
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"\n[Trainer] Received signal {signum}, shutting down...")
        self.running = False
    
    def _should_filter_sample(self, obj_values: list) -> bool:
        """
        判断样本是否应该被过滤
        
        过滤条件：
        - 仅检查 S11（第一个目标值）是否为 1000.0
        - S11=1000.0 表示 HFSS 仿真失败
        
        Args:
            obj_values: 目标值列表，obj_values[0] 为 S11
            
        Returns:
            True 表示应该过滤掉该样本
        """
        if len(obj_values) < 1:
            return False
        
        # 只检查 S11（第一个目标值）是否为 1000.0
        s11_value = obj_values[0]
        if abs(s11_value - 1000.0) < 1e-6:
            return True
        
        return False
    
    def initialize(self):
        """
        初始化训练线
        
        Returns:
            是否初始化成功
        """
        logger.info(f"\n" + "="*60)
        logger.info(f"TRAINER PROCESS INITIALIZATION")
        logger.info(f"="*60)
        logger.info(f"Shared directory: {self.shared_dir}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Objectives: {self.n_objectives}")
        logger.info(f"Min samples: {self.min_samples}")
        logger.info(f"Min new samples to train: {self.min_new_samples_to_train}")
        logger.info(f"="*60)
        
        # 初始化代理模型管理器
        try:
            if self.model_type == 'gpflow_svgp':
                n_inducing = self.model_params.get('n_inducing', 100)
                kernel_type = self.model_params.get('kernel_type', 'matern52')
                self.surrogate_manager = GPflowSVSManager(
                    n_objectives=self.n_objectives,
                    min_samples=self.min_samples,
                    n_inducing=n_inducing,
                    kernel_type=kernel_type
                )
                logger.info(f"[Trainer] GPflow-SVGP initialized (n_inducing={n_inducing}, kernel={kernel_type})")
                
            elif self.model_type == 'rf':
                n_estimators = self.model_params.get('n_estimators', 100)
                self.surrogate_manager = SurrogateManager(
                    n_objectives=self.n_objectives,
                    model_type='rf',
                    min_samples=self.min_samples,
                    n_estimators=n_estimators
                )
                logger.info(f"[Trainer] Random Forest initialized (n_estimators={n_estimators})")
                
            else:  # gp
                self.surrogate_manager = SurrogateManager(
                    n_objectives=self.n_objectives,
                    model_type='gp',
                    min_samples=self.min_samples
                )
                logger.info(f"[Trainer] Gaussian Process initialized")
            
            # 更新状态
            self.shared_memory.update_trainer_status({
                'status': 'ready',
                'n_samples': 0,
                'model_version': 0,
                'model_quality': None
            })
            
            # 发送就绪信号
            self.shared_memory.send_trainer_signal('ready')
            
            logger.info(f"[Trainer] Initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.shared_memory.update_trainer_status({
                'status': 'error',
                'n_samples': 0,
                'model_version': 0,
                'model_quality': None,
                'error': str(e)
            })
            
            return False
    
    def run(self):
        """
        运行训练线主循环
        
        监听评估数据变化，触发训练。
        """
        self.running = True
        logger.info(f"\n[Trainer] Starting main loop...")
        
        # 检查是否有历史数据
        existing_evals = self.shared_memory.get_all_evaluations()
        if existing_evals:
            logger.info(f"[Trainer] Found {len(existing_evals)} existing evaluations")
            self._train_with_new_data(existing_evals)
        
        # 主循环
        poll_interval = 2.0  # 轮询间隔（秒）
        last_check_time = time.time()
        
        while self.running:
            try:
                # 检查控制信号
                control = self.shared_memory.read_control_signals()
                optimizer_signal = control.get('optimizer_signal', 'ready')
                
                # 如果优化线停止，训练线也停止
                if optimizer_signal == 'stopped':
                    logger.info(f"[Trainer] Optimizer stopped, shutting down...")
                    break
                
                # 检查新数据
                current_time = time.time()
                if current_time - last_check_time >= poll_interval:
                    new_evals, current_count = self.shared_memory.get_new_evaluations(self.last_train_count)
                    
                    if new_evals:
                        logger.info(f"\n[Trainer] Detected {len(new_evals)} new evaluations (total: {current_count})")
                        
                        # 更新状态
                        self.shared_memory.update_trainer_status({
                            'status': 'collecting',
                            'n_samples': current_count,
                            'model_version': self.current_version,
                            'model_quality': None
                        })
                        
                        # 检查是否需要训练
                        n_new = current_count - self.last_train_count
                        if n_new >= self.min_new_samples_to_train:
                            logger.info(f"[Trainer] Triggering training ({n_new} new samples >= {self.min_new_samples_to_train})")
                            self._train_with_new_data(new_evals)
                        else:
                            logger.info(f"[Trainer] Waiting for more data ({n_new}/{self.min_new_samples_to_train} new samples)")
                    
                    last_check_time = current_time
                
                # 更新状态
                self.shared_memory.update_trainer_status({
                    'status': 'idle' if self.surrogate_manager.surrogate.is_trained else 'waiting_data',
                    'n_samples': self.shared_memory.get_evaluation_count(),
                    'model_version': self.current_version,
                    'model_quality': self.model_qualities[-1] if self.model_qualities else None
                })
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"[ERROR] Main loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5.0)
        
        logger.info(f"\n[Trainer] Main loop ended")
        self._cleanup()
    
    def _train_with_new_data(self, new_evals: List[Dict]):
        """
        使用新数据训练模型
        
        Args:
            new_evals: 新评估数据列表
        """
        start_time = time.time()
        
        # 更新状态
        self.shared_memory.update_trainer_status({
            'status': 'training',
            'n_samples': self.shared_memory.get_evaluation_count(),
            'model_version': self.current_version,
            'model_quality': None
        })
        self.shared_memory.send_trainer_signal('training')
        
        try:
            # 获取所有评估数据
            all_evals = self.shared_memory.get_all_evaluations()
            n_samples = len(all_evals)
            
            logger.info(f"\n[Trainer] Training with {n_samples} samples...")
            
            # 提取参数和目标值
            X_list = []
            y_list = []
            filtered_count = 0
            
            for eval_data in all_evals:
                params = eval_data.get('parameters', [])
                objectives = eval_data.get('objectives', {})
                
                # 转换目标值为列表
                if isinstance(objectives, dict):
                    # 按目标名称排序
                    obj_values = [objectives.get(f'obj_{i}', 0.0) for i in range(self.n_objectives)]
                elif isinstance(objectives, list):
                    obj_values = objectives
                else:
                    logger.warning(f"[WARN] Invalid objectives format: {objectives}")
                    continue
                
                # 数据筛选：过滤掉目标值异常的数据
                # S11 最大值为 1000.0 表示仿真失败或数据异常
                if self._should_filter_sample(obj_values):
                    filtered_count += 1
                    continue
                
                X_list.append(params)
                y_list.append(obj_values)
            
            if len(X_list) < self.min_samples:
                logger.info(f"[Trainer] Not enough samples: {len(X_list)} < {self.min_samples}")
                return
            
            # 显示过滤统计
            if filtered_count > 0:
                logger.info(f"[Trainer] Filtered {filtered_count} samples with abnormal objective values (e.g., S11=1000.0)")
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            logger.info(f"[Trainer] Training data shape: X={X.shape}, y={y.shape}")
            logger.info(f"[Trainer] Objective ranges:")
            for i in range(y.shape[1]):
                logger.info(f"  Objective {i}: min={y[:, i].min():.4f}, max={y[:, i].max():.4f}, mean={y[:, i].mean():.4f}")
            
            # 训练模型（全量训练）
            self.surrogate_manager.X_samples = []  # 清空旧数据
            self.surrogate_manager.y_samples = []
            
            for i in range(len(X)):
                self.surrogate_manager.add_sample(X[i], y[i])
            
            # 强制重训练（确保使用所有数据）
            self.surrogate_manager.retrain_all()
            
            # 评估模型质量
            model_quality = self._evaluate_model_quality()
            
            # 保存模型状态
            self._save_model_state(model_quality)
            
            # 更新统计
            train_time = time.time() - start_time
            self.train_times.append(train_time)
            self.model_qualities.append(model_quality)
            self.last_train_count = n_samples
            self.n_trains += 1
            
            logger.info(f"[Trainer] Training completed in {train_time:.2f}s")
            logger.info(f"[Trainer] Model quality: {model_quality}")
            
            # 更新状态
            self.shared_memory.update_trainer_status({
                'status': 'model_ready',
                'n_samples': n_samples,
                'model_version': self.current_version,
                'model_quality': model_quality
            })
            self.shared_memory.send_trainer_signal('model_ready', {
                'version': self.current_version,
                'n_samples': n_samples,
                'quality': model_quality
            })
            
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.shared_memory.update_trainer_status({
                'status': 'error',
                'n_samples': self.shared_memory.get_evaluation_count(),
                'model_version': self.current_version,
                'model_quality': None,
                'error': str(e)
            })
            self.shared_memory.send_trainer_signal('error', {'error': str(e)})
    
    def _evaluate_model_quality(self) -> Dict:
        """
        评估模型质量
        
        Returns:
            质量指标字典
        """
        try:
            quality = self.surrogate_manager.get_model_quality()
            logger.info(f"[Trainer] Model quality: {quality}")
            return quality
        except Exception as e:
            logger.warning(f"[WARN] Failed to evaluate model quality: {e}")
            return {'r2': None, 'mae': None, 'status': 'error'}
    
    def _save_model_state(self, model_quality: Dict):
        """
        保存模型状态到共享内存
        
        Args:
            model_quality: 模型质量指标
        """
        try:
            # 提取模型状态
            surrogate = self.surrogate_manager.surrogate
            
            model_state = {
                'model_type': self.model_type,
                'n_objectives': self.n_objectives,
                'X_samples': surrogate.X_samples,
                'y_samples': surrogate.y_samples,
                'is_trained': surrogate.is_trained,
                'model_params': {}
            }
            
            # 保存模型特定参数
            if hasattr(surrogate, 'model'):
                model_state['model_params']['model'] = surrogate.model
            
            if hasattr(surrogate, 'models'):
                model_state['model_params']['models'] = surrogate.models
            
            if hasattr(surrogate, 'n_estimators'):
                model_state['model_params']['n_estimators'] = surrogate.n_estimators
            
            if hasattr(surrogate, 'n_inducing'):
                model_state['model_params']['n_inducing'] = surrogate.n_inducing
            
            if hasattr(surrogate, 'kernel_type'):
                model_state['model_params']['kernel_type'] = surrogate.kernel_type
            
            # 保存到共享内存
            n_samples = len(surrogate.X_samples)
            self.shared_memory.save_model_state(model_state, n_samples, model_quality)
            
            # 更新版本
            version_info = self.shared_memory.get_model_version()
            self.current_version = version_info.get('version', 0)
            
            logger.info(f"[Trainer] Model state saved: version={self.current_version}, samples={n_samples}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save model state: {e}")
            import traceback
            traceback.print_exc()
    
    def _cleanup(self):
        """清理资源"""
        logger.info(f"\n[Trainer] Cleaning up...")
        
        # 保存最终统计
        stats = {
            'n_trains': self.n_trains,
            'total_samples': self.shared_memory.get_evaluation_count(),
            'train_times': self.train_times,
            'model_qualities': self.model_qualities,
            'final_version': self.current_version
        }
        
        stats_file = os.path.join(self.shared_dir, 'trainer_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Trainer] Stats saved to {stats_file}")
        
        # 清理共享内存
        self.shared_memory.cleanup()
        
        logger.info(f"[Trainer] Cleanup completed")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Trainer Process for HFSS Optimizer')
    parser.add_argument('--config', type=str, required=True, help='Configuration JSON file')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建并运行训练线
    trainer = TrainerProcess(config)
    
    if trainer.initialize():
        trainer.run()
    else:
        logger.error(f"[ERROR] Initialization failed, exiting")
        sys.exit(1)


if __name__ == '__main__':
    main()
