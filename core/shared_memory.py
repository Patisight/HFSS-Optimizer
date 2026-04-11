"""
共享内存管理器模块

实现进程间通信机制，支持优化线和训练线之间的数据交换。

核心功能：
1. 评估数据管理（evaluations.jsonl）- 优化线写入，训练线读取
2. 模型状态管理（model_state.pkl）- 训练线写入，优化线读取
3. 控制信号机制（control.json）- 双向通信
4. 状态信息管理（status.json）- 双向状态同步

数据结构：
shared_data/
├── evaluations.jsonl       # 所有评估数据（追加写入）
├── model_state.pkl         # 当前可用模型状态
├── model_version.json      # 模型版本信息
├── control.json            # 控制信号
└── status.json             # 状态信息
"""

import os
import json
import pickle
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from loguru import logger


class SharedMemoryManager:
    """
    共享内存管理器
    
    基于文件系统实现进程间通信，简单可靠，跨平台。
    使用文件锁保证数据一致性。
    """
    
    def __init__(self, shared_dir: str = './shared_data'):
        """
        初始化共享内存管理器
        
        Args:
            shared_dir: 共享数据目录路径
        """
        self.shared_dir = shared_dir
        self._ensure_dir()
        
        # 文件路径
        self.evaluations_file = os.path.join(shared_dir, 'evaluations.jsonl')
        self.model_state_file = os.path.join(shared_dir, 'model_state.pkl')
        self.model_version_file = os.path.join(shared_dir, 'model_version.json')
        self.control_file = os.path.join(shared_dir, 'control.json')
        self.status_file = os.path.join(shared_dir, 'status.json')
        
        # 文件锁
        self._eval_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._status_lock = threading.Lock()
        
        # 初始化文件
        self._init_files()
        
        # 评估数据缓存（避免频繁读取文件）
        self._eval_cache = None
        self._eval_cache_time = 0
        self._eval_cache_ttl = 1.0  # 缓存有效期（秒）
    
    def _ensure_dir(self):
        """确保共享数据目录存在"""
        os.makedirs(self.shared_dir, exist_ok=True)
    
    def _init_files(self):
        """初始化文件（如果不存在）"""
        # 初始化控制信号
        if not os.path.exists(self.control_file):
            self._write_json(self.control_file, {
                'optimizer_signal': 'ready',
                'trainer_signal': 'ready',
                'timestamp': datetime.now().isoformat()
            })
        
        # 初始化状态信息
        if not os.path.exists(self.status_file):
            self._write_json(self.status_file, {
                'optimizer': {
                    'status': 'idle',
                    'iteration': 0,
                    'n_real_evals': 0,
                    'n_surrogate_evals': 0,
                    'model_version': 0
                },
                'trainer': {
                    'status': 'idle',
                    'n_samples': 0,
                    'model_version': 0,
                    'model_quality': None
                },
                'timestamp': datetime.now().isoformat()
            })
        
        # 初始化模型版本
        if not os.path.exists(self.model_version_file):
            self._write_json(self.model_version_file, {
                'version': 0,
                'n_samples': 0,
                'timestamp': datetime.now().isoformat()
            })
    
    # ==================== 评估数据管理 ====================
    
    def append_evaluation(self, eval_data: Dict):
        """
        追加评估数据（优化线调用）
        
        Args:
            eval_data: 评估数据字典，包含：
                - eval_id: 评估ID
                - timestamp: 时间戳
                - parameters: 参数值
                - objectives: 目标值
                - is_real: 是否真实仿真
        """
        with self._eval_lock:
            # 添加时间戳（如果没有）
            if 'timestamp' not in eval_data:
                eval_data['timestamp'] = datetime.now().isoformat()
            
            # 追加写入
            with open(self.evaluations_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(eval_data, ensure_ascii=False) + '\n')
            
            # 清除缓存
            self._eval_cache = None
    
    def append_evaluations_batch(self, evals: List[Dict]):
        """
        批量追加评估数据
        
        Args:
            evals: 评估数据列表
        """
        with self._eval_lock:
            with open(self.evaluations_file, 'a', encoding='utf-8') as f:
                for eval_data in evals:
                    if 'timestamp' not in eval_data:
                        eval_data['timestamp'] = datetime.now().isoformat()
                    f.write(json.dumps(eval_data, ensure_ascii=False) + '\n')
            
            self._eval_cache = None
    
    def get_all_evaluations(self, use_cache: bool = True) -> List[Dict]:
        """
        获取所有评估数据（训练线调用）
        
        Args:
            use_cache: 是否使用缓存
        
        Returns:
            评估数据列表
        """
        # 检查缓存
        if use_cache and self._eval_cache is not None:
            if time.time() - self._eval_cache_time < self._eval_cache_ttl:
                return self._eval_cache
        
        with self._eval_lock:
            if not os.path.exists(self.evaluations_file):
                return []
            
            evals = []
            with open(self.evaluations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        evals.append(json.loads(line))
            
            # 更新缓存
            self._eval_cache = evals
            self._eval_cache_time = time.time()
            
            return evals
    
    def get_new_evaluations(self, last_count: int) -> Tuple[List[Dict], int]:
        """
        获取新增的评估数据（增量读取）
        
        Args:
            last_count: 上次读取的记录数
        
        Returns:
            (新增评估数据列表, 当前总记录数)
        """
        all_evals = self.get_all_evaluations()
        current_count = len(all_evals)
        
        if current_count <= last_count:
            return [], current_count
        
        new_evals = all_evals[last_count:]
        return new_evals, current_count
    
    def get_evaluation_count(self) -> int:
        """获取评估数据总数"""
        return len(self.get_all_evaluations())
    
    def clear_evaluations(self):
        """清除所有评估数据（谨慎使用）"""
        with self._eval_lock:
            if os.path.exists(self.evaluations_file):
                os.remove(self.evaluations_file)
            self._eval_cache = None
    
    # ==================== 模型状态管理 ====================
    
    def save_model_state(self, model_state: Dict, n_samples: int, model_quality: Optional[Dict] = None):
        """
        保存模型状态（训练线调用）
        
        Args:
            model_state: 模型状态字典，包含：
                - model_type: 模型类型
                - model_params: 模型参数
                - X_samples: 训练样本X
                - y_samples: 训练样本y
                - 其他模型特定参数
            n_samples: 训练样本数
            model_quality: 模型质量指标（可选）
        """
        with self._model_lock:
            # 保存模型状态
            with open(self.model_state_file, 'wb') as f:
                pickle.dump(model_state, f)
            
            # 更新模型版本
            current_version = self._read_json(self.model_version_file).get('version', 0)
            new_version = current_version + 1
            
            self._write_json(self.model_version_file, {
                'version': new_version,
                'n_samples': n_samples,
                'model_quality': model_quality,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"[SharedMemory] Model state saved: version={new_version}, samples={n_samples}")
    
    def load_model_state(self) -> Optional[Dict]:
        """
        加载模型状态（优化线调用）
        
        Returns:
            模型状态字典，如果不存在则返回None
        """
        with self._model_lock:
            if not os.path.exists(self.model_state_file):
                return None
            
            try:
                with open(self.model_state_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.info(f"[WARN] Failed to load model state: {e}")
                return None
    
    def get_model_version(self) -> Dict:
        """
        获取模型版本信息
        
        Returns:
            版本信息字典
        """
        return self._read_json(self.model_version_file)
    
    def has_new_model(self, current_version: int) -> bool:
        """
        检查是否有新模型可用
        
        Args:
            current_version: 当前模型版本
        
        Returns:
            是否有新模型
        """
        latest_version = self._read_json(self.model_version_file).get('version', 0)
        return latest_version > current_version
    
    # ==================== 控制信号机制 ====================
    
    def send_optimizer_signal(self, signal: str, data: Optional[Dict] = None):
        """
        发送优化线信号（优化线调用）
        
        Args:
            signal: 信号类型
                - 'ready': 准备就绪
                - 'running': 正在运行
                - 'paused': 已暂停
                - 'stopped': 已停止
                - 'request_model': 请求新模型
            data: 附加数据（可选）
        """
        with self._control_lock:
            control = self._read_json(self.control_file)
            control['optimizer_signal'] = signal
            control['optimizer_data'] = data
            control['timestamp'] = datetime.now().isoformat()
            self._write_json(self.control_file, control)
    
    def send_trainer_signal(self, signal: str, data: Optional[Dict] = None):
        """
        发送训练线信号（训练线调用）
        
        Args:
            signal: 信号类型
                - 'ready': 准备就绪
                - 'training': 正在训练
                - 'model_ready': 模型已就绪
                - 'error': 发生错误
            data: 附加数据（可选）
        """
        with self._control_lock:
            control = self._read_json(self.control_file)
            control['trainer_signal'] = signal
            control['trainer_data'] = data
            control['timestamp'] = datetime.now().isoformat()
            self._write_json(self.control_file, control)
    
    def read_control_signals(self) -> Dict:
        """
        读取所有控制信号
        
        Returns:
            控制信号字典
        """
        with self._control_lock:
            return self._read_json(self.control_file)
    
    def wait_for_trainer_signal(self, expected_signal: str, timeout: float = 60.0) -> bool:
        """
        等待训练线信号（优化线调用）
        
        Args:
            expected_signal: 期望的信号
            timeout: 超时时间（秒）
        
        Returns:
            是否收到期望信号
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            control = self.read_control_signals()
            if control.get('trainer_signal') == expected_signal:
                return True
            time.sleep(0.1)
        return False
    
    def wait_for_optimizer_signal(self, expected_signal: str, timeout: float = 60.0) -> bool:
        """
        等待优化线信号（训练线调用）
        
        Args:
            expected_signal: 期望的信号
            timeout: 超时时间（秒）
        
        Returns:
            是否收到期望信号
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            control = self.read_control_signals()
            if control.get('optimizer_signal') == expected_signal:
                return True
            time.sleep(0.1)
        return False
    
    # ==================== 状态信息管理 ====================
    
    def update_optimizer_status(self, status: Dict):
        """
        更新优化线状态（优化线调用）
        
        Args:
            status: 状态字典，包含：
                - status: 状态（idle/running/paused/stopped）
                - iteration: 当前迭代
                - n_real_evals: 真实仿真次数
                - n_surrogate_evals: 代理模型评估次数
                - model_version: 当前模型版本
        """
        with self._status_lock:
            full_status = self._read_json(self.status_file)
            full_status['optimizer'] = status
            full_status['timestamp'] = datetime.now().isoformat()
            self._write_json(self.status_file, full_status)
    
    def update_trainer_status(self, status: Dict):
        """
        更新训练线状态（训练线调用）
        
        Args:
            status: 状态字典，包含：
                - status: 状态（idle/training/model_ready/error）
                - n_samples: 训练样本数
                - model_version: 当前模型版本
                - model_quality: 模型质量指标
        """
        with self._status_lock:
            full_status = self._read_json(self.status_file)
            full_status['trainer'] = status
            full_status['timestamp'] = datetime.now().isoformat()
            self._write_json(self.status_file, full_status)
    
    def get_full_status(self) -> Dict:
        """
        获取完整状态信息
        
        Returns:
            状态信息字典
        """
        with self._status_lock:
            return self._read_json(self.status_file)
    
    def get_optimizer_status(self) -> Dict:
        """获取优化线状态"""
        return self.get_full_status().get('optimizer', {})
    
    def get_trainer_status(self) -> Dict:
        """获取训练线状态"""
        return self.get_full_status().get('trainer', {})
    
    # ==================== 辅助方法 ====================
    
    def _read_json(self, filepath: str) -> Dict:
        """读取JSON文件"""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.info(f"[WARN] Failed to read {filepath}: {e}")
            return {}
    
    def _write_json(self, filepath: str, data: Dict):
        """写入JSON文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.info(f"[WARN] Failed to write {filepath}: {e}")
    
    def cleanup(self):
        """清理资源"""
        # 清除缓存
        self._eval_cache = None
        logger.info("[SharedMemory] Cleanup completed")


class HotSwapManager:
    """
    模型热替换管理器
    
    管理代理模型的热替换，确保替换过程原子性。
    """
    
    def __init__(self, shared_memory: SharedMemoryManager):
        """
        初始化热替换管理器
        
        Args:
            shared_memory: 共享内存管理器实例
        """
        self.shared_memory = shared_memory
        
        # 当前模型
        self.current_model = None
        self.current_version = 0
        
        # 备份模型（用于回滚）
        self.backup_model = None
        self.backup_version = 0
        
        # 热替换锁
        self._swap_lock = threading.Lock()
        
        # 热替换历史
        self.swap_history = []
    
    def load_initial_model(self, model_class, model_config: Dict) -> bool:
        """
        加载初始模型
        
        Args:
            model_class: 模型类
            model_config: 模型配置
        
        Returns:
            是否成功加载
        """
        model_state = self.shared_memory.load_model_state()
        
        if model_state is None:
            logger.info("[HotSwap] No initial model found, will use default")
            return False
        
        try:
            # 创建模型实例
            self.current_model = model_class(**model_config)
            
            # 恢复模型状态
            self._restore_model_state(model_state)
            
            # 更新版本
            version_info = self.shared_memory.get_model_version()
            self.current_version = version_info.get('version', 0)
            
            logger.info(f"[HotSwap] Initial model loaded: version={self.current_version}")
            return True
            
        except Exception as e:
            logger.info(f"[WARN] Failed to load initial model: {e}")
            return False
    
    def check_and_swap(self, model_class, model_config: Dict) -> bool:
        """
        检查并执行热替换（优化线调用）
        
        Args:
            model_class: 模型类
            model_config: 模型配置
        
        Returns:
            是否执行了替换
        """
        if not self.shared_memory.has_new_model(self.current_version):
            return False
        
        with self._swap_lock:
            try:
                # 加载新模型状态
                model_state = self.shared_memory.load_model_state()
                if model_state is None:
                    return False
                
                # 备份当前模型
                self.backup_model = self.current_model
                self.backup_version = self.current_version
                
                # 创建新模型
                new_model = model_class(**model_config)
                
                # 恢复新模型状态
                self._restore_model_state(model_state, new_model)
                
                # 替换
                self.current_model = new_model
                version_info = self.shared_memory.get_model_version()
                self.current_version = version_info.get('version', 0)
                
                # 记录历史
                self.swap_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'old_version': self.backup_version,
                    'new_version': self.current_version,
                    'n_samples': version_info.get('n_samples', 0),
                    'model_quality': version_info.get('model_quality')
                })
                
                logger.info(f"[HotSwap] Model swapped: v{self.backup_version} -> v{self.current_version}")
                
                # 清理备份
                self.backup_model = None
                
                return True
                
            except Exception as e:
                logger.info(f"[ERROR] Hot swap failed: {e}")
                # 回滚
                if self.backup_model is not None:
                    self.current_model = self.backup_model
                    self.current_version = self.backup_version
                    logger.info(f"[HotSwap] Rolled back to version {self.current_version}")
                return False
    
    def _restore_model_state(self, model_state: Dict, model=None):
        """
        恢复模型状态
        
        Args:
            model_state: 模型状态字典
            model: 目标模型（如果为None，使用current_model）
        """
        if model is None:
            model = self.current_model
        
        # 恢复训练数据
        if 'X_samples' in model_state and 'y_samples' in model_state:
            model.X_samples = model_state['X_samples']
            model.y_samples = model_state['y_samples']
        
        # 恢复模型特定参数
        if 'model_params' in model_state:
            for key, value in model_state['model_params'].items():
                if hasattr(model, key):
                    setattr(model, key, value)
        
        # 标记为已训练
        model.is_trained = True
    
    def get_current_model(self):
        """获取当前模型"""
        return self.current_model
    
    def get_current_version(self) -> int:
        """获取当前模型版本"""
        return self.current_version
    
    def get_swap_history(self) -> List[Dict]:
        """获取热替换历史"""
        return self.swap_history


class DataWatcher:
    """
    数据监听器
    
    监听评估数据变化，触发回调。
    """
    
    def __init__(self, shared_memory: SharedMemoryManager, 
                 callback=None, poll_interval: float = 1.0):
        """
        初始化数据监听器
        
        Args:
            shared_memory: 共享内存管理器
            callback: 数据变化回调函数
            poll_interval: 轮询间隔（秒）
        """
        self.shared_memory = shared_memory
        self.callback = callback
        self.poll_interval = poll_interval
        
        self._running = False
        self._thread = None
        self._last_count = 0
    
    def start(self):
        """启动监听"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("[DataWatcher] Started watching for new evaluations")
    
    def stop(self):
        """停止监听"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("[DataWatcher] Stopped")
    
    def _watch_loop(self):
        """监听循环"""
        while self._running:
            try:
                # 检查新数据
                new_evals, current_count = self.shared_memory.get_new_evaluations(self._last_count)
                
                if new_evals:
                    logger.info(f"[DataWatcher] Detected {len(new_evals)} new evaluations (total: {current_count})")
                    
                    # 触发回调
                    if self.callback is not None:
                        self.callback(new_evals, current_count)
                    
                    # 更新计数
                    self._last_count = current_count
                
            except Exception as e:
                logger.info(f"[ERROR] DataWatcher error: {e}")
            
            time.sleep(self.poll_interval)
    
    def get_last_count(self) -> int:
        """获取上次处理的记录数"""
        return self._last_count
