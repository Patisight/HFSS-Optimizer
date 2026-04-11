#!/usr/bin/env python
"""
HFSS 天线优化程序 - 双线架构主入口

双线架构：
- 优化线：运行优化算法，使用代理模型预测
- 训练线：独立训练代理模型，热替换到优化线

使用方法:
    python run_dual_line.py --algorithm mopso
    python run_dual_line.py --config my_config.py
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading
import traceback
from datetime import datetime
from loguru import logger

# 日志配置
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dual_line_optimizer_{time}.log", rotation="10 MB", retention="7 days", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入模块
from core import HFSSController, ObjectiveEvaluator, format_results
from core.shared_memory import SharedMemoryManager
from utils import OptimizationVisualizer
from config import get_default_config, validate_config


class DualLineOrchestrator:
    """
    双线架构协调器
    
    负责：
    1. 启动训练线进程
    2. 运行优化线
    3. 监控双线状态
    4. 清理资源
    """
    
    def __init__(self, config: dict):
        """
        初始化协调器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.shared_dir = config.get('shared_dir', './shared_data')
        
        # 共享内存管理器
        self.shared_memory = SharedMemoryManager(self.shared_dir)
        
        # 进程
        self.trainer_process = None
        self.optimizer_thread = None
        
        # 状态
        self.running = False
        self.trainer_config_file = None
    
    def start_trainer_process(self):
        """
        启动训练线进程
        
        Returns:
            是否成功启动
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING TRAINER PROCESS")
        logger.info("="*60)
        
        # 创建训练线配置
        trainer_config = {
            'shared_dir': self.shared_dir,
            'model_type': self.config['surrogate_config'].get('type', 'gp'),
            'n_objectives': len(self.config['objectives']),
            'min_samples': self.config['surrogate_config'].get('min_samples', 5),
            # 至少积累多少新样本才触发训练
            'min_new_samples_to_train': self.config['surrogate_config'].get('min_new_samples_to_train', 5),
            'model_params': self.config['surrogate_config'].get('model_params', {})
        }
        
        # 保存配置文件
        self.trainer_config_file = os.path.join(self.shared_dir, 'trainer_config.json')
        os.makedirs(self.shared_dir, exist_ok=True)
        
        with open(self.trainer_config_file, 'w', encoding='utf-8') as f:
            json.dump(trainer_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Orchestrator] Trainer config saved: {self.trainer_config_file}")
        
        # 启动训练线进程
        trainer_script = os.path.join(PROJECT_ROOT, 'core', 'trainer_process.py')
        
        try:
            self.trainer_process = subprocess.Popen(
                [sys.executable, trainer_script, '--config', self.trainer_config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            logger.info(f"[Orchestrator] Trainer process started (PID: {self.trainer_process.pid})")
            
            # 启动输出监控线程
            self._start_output_monitor()
            
            # 等待训练线就绪
            logger.info("[Orchestrator] Waiting for trainer to be ready...")
            if self.shared_memory.wait_for_trainer_signal('ready', timeout=30.0):
                logger.info("[Orchestrator] Trainer is ready!")
                return True
            else:
                logger.error(" Trainer failed to initialize within 30 seconds")
                return False
                
        except Exception as e:
            logger.info(f"[ERROR] Failed to start trainer process: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_output_monitor(self):
        """启动输出监控线程"""
        def monitor_stdout():
            for line in iter(self.trainer_process.stdout.readline, ''):
                if line:
                    logger.info(f"[Trainer] {line.rstrip()}")
        
        def monitor_stderr():
            for line in iter(self.trainer_process.stderr.readline, ''):
                if line:
                    logger.info(f"[Trainer ERROR] {line.rstrip()}")
        
        stdout_thread = threading.Thread(target=monitor_stdout, daemon=True)
        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
    
    def run_optimizer(self):
        """
        运行优化线
        
        Returns:
            优化结果
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING OPTIMIZER LINE")
        logger.info("="*60)
        
        # 更新配置，启用双线模式
        config = self.config.copy()
        config['dual_line_mode'] = True
        config['shared_dir'] = self.shared_dir
        
        # 运行优化（调用原有的run_optimization函数）
        from run import run_optimization
        result = run_optimization(config, config.get('algorithm', 'mopso'))
        
        return result
    
    def stop_trainer_process(self):
        """停止训练线进程"""
        if self.trainer_process is None:
            return
        
        logger.info("\n[Orchestrator] Stopping trainer process...")
        
        # 发送停止信号
        self.shared_memory.send_optimizer_signal('stopped')
        
        # 等待进程结束
        try:
            self.trainer_process.wait(timeout=10.0)
            logger.info(f"[Orchestrator] Trainer process stopped (exit code: {self.trainer_process.returncode})")
        except subprocess.TimeoutExpired:
            logger.info("[Orchestrator] Trainer process did not stop gracefully, terminating...")
            self.trainer_process.terminate()
            self.trainer_process.wait(timeout=5.0)
        
        self.trainer_process = None
    
    def monitor_status(self):
        """监控双线状态（调试用）"""
        logger.info("\n" + "="*60)
        logger.info("DUAL-LINE STATUS MONITOR")
        logger.info("="*60)
        
        while self.running:
            status = self.shared_memory.get_full_status()
            
            optimizer_status = status.get('optimizer', {})
            trainer_status = status.get('trainer', {})
            
            logger.info(f"\n[Optimizer] Status: {optimizer_status.get('status', 'unknown')}")
            logger.info(f"  Iteration: {optimizer_status.get('iteration', 0)}")
            logger.info(f"  Real evals: {optimizer_status.get('n_real_evals', 0)}")
            logger.info(f"  Surrogate evals: {optimizer_status.get('n_surrogate_evals', 0)}")
            logger.info(f"  Model version: {optimizer_status.get('model_version', 0)}")
            
            logger.info(f"\n[Trainer] Status: {trainer_status.get('status', 'unknown')}")
            logger.info(f"  Samples: {trainer_status.get('n_samples', 0)}")
            logger.info(f"  Model version: {trainer_status.get('model_version', 0)}")
            logger.info(f"  Model quality: {trainer_status.get('model_quality', {})}")
            
            time.sleep(5.0)
    
    def cleanup(self):
        """清理资源"""
        logger.info("\n[Orchestrator] Cleaning up...")
        
        # 停止训练线
        self.stop_trainer_process()
        
        # 清理共享内存
        self.shared_memory.cleanup()
        
        # 删除临时配置文件
        if self.trainer_config_file and os.path.exists(self.trainer_config_file):
            os.remove(self.trainer_config_file)
        
        logger.info("[Orchestrator] Cleanup completed")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='HFSS Optimizer with Dual-Line Architecture')
    parser.add_argument('--algorithm', type=str, default='mopso', help='Optimization algorithm')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--monitor', action='store_true', help='Enable status monitoring')
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # 验证配置
    if not validate_config(config):
        logger.error(" Invalid configuration")
        sys.exit(1)
    
    # 创建协调器
    orchestrator = DualLineOrchestrator(config)
    
    try:
        # 启动训练线
        if not orchestrator.start_trainer_process():
            logger.error(" Failed to start trainer process")
            sys.exit(1)
        
        # 启动状态监控（可选）
        if args.monitor:
            monitor_thread = threading.Thread(target=orchestrator.monitor_status, daemon=True)
            monitor_thread.start()
        
        # 运行优化线
        result = orchestrator.run_optimizer()
        
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*60)
        
        # 打印结果
        if result:
            logger.info(f"Pareto front size: {len(result)}")
            logger.info(f"Best objectives:")
            for i, sol in enumerate(result[:5]):
                logger.info(f"  Solution {i+1}: {sol.get('objectives', {})}")
        
    except KeyboardInterrupt:
        logger.info("\n[INFO] Interrupted by user")
    except Exception as e:
        logger.info(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        orchestrator.cleanup()


if __name__ == '__main__':
    main()
