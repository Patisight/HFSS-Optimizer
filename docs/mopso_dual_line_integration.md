# MOPSO双线架构集成方案

## 修改概述

将MOPSO算法修改为支持双线架构，主要修改点：

1. **初始化阶段**：根据配置选择单线或双线模式
2. **代理模型管理器**：双线模式使用`DualLineSurrogateManager`
3. **评估阶段**：检查模型热替换，移除训练逻辑
4. **数据写入**：真实仿真后写入共享内存

## 修改详情

### 1. 修改 `__init__` 方法

在 `algorithms/mopso.py` 的 `__init__` 方法中添加双线模式配置：

```python
def __init__(self, config: Dict):
    super().__init__(config)
    
    # ... 现有代码 ...
    
    # 双线架构配置
    self.dual_line_mode = config.get('dual_line_mode', False)
    self.shared_dir = config.get('shared_dir', './shared_data')
    
    # ... 其余代码 ...
```

### 2. 修改代理模型初始化

在 `run` 方法中，修改代理模型管理器的创建逻辑：

```python
# 初始化代理模型管理器
if self.use_surrogate:
    min_samples = self.surrogate_min_samples if self.surrogate_min_samples > 0 else self.population_size
    model_params = self.surrogate_model_params
    
    # 双线模式：使用支持热替换的管理器
    if self.dual_line_mode:
        from core.surrogate_hotswap import DualLineSurrogateManager
        
        self.surrogate_manager = DualLineSurrogateManager(
            n_objectives=self.n_objectives,
            model_type=self.surrogate_type,
            shared_dir=self.shared_dir,
            min_samples=min_samples,
            **model_params
        )
        
        print(f"[INFO] Dual-line mode enabled: {self.surrogate_type}")
        print(f"[INFO] Shared directory: {self.shared_dir}")
        
        # 尝试从共享内存加载已有模型
        if self.surrogate_manager.initialize_from_shared_memory():
            print(f"[INFO] Loaded existing model from shared memory")
    
    # 单线模式：使用原有管理器
    else:
        # ... 原有的单线模式代码 ...
```

### 3. 修改 `_evaluate` 方法

在 `_evaluate` 方法中，添加热替换检查，移除训练逻辑：

```python
def _evaluate(self, x: np.ndarray, evaluator, force_real: bool = False) -> Tuple[np.ndarray, bool]:
    """评估粒子"""
    self.evaluation_count += 1
    self._last_surrogate_pred = None
    
    # 双线模式：检查模型热替换
    if self.dual_line_mode and self.surrogate_manager:
        if self.surrogate_manager.check_for_model_update():
            print(f"  [HotSwap] Model updated to version {self.surrogate_manager.get_current_version()}")
    
    # 检查缓存
    cached = self._check_cache(x)
    if cached is not None:
        return cached, True
    
    # ... 代理模型预测逻辑（保持不变）...
    
    # 真实仿真
    y = self._real_evaluate(x, evaluator)
    self.real_evaluation_count += 1
    
    # 更新代理模型
    if self.use_surrogate and self.surrogate_manager:
        # 双线模式：只写入共享内存，不训练
        if self.dual_line_mode:
            self.surrogate_manager.add_sample(x, y, is_real=True)
        # 单线模式：训练模型
        else:
            self.surrogate_manager.add_sample(x, y)
            
            # 定期重训练
            if self.retrain_interval > 0 and self.surrogate_type in ['gp', 'rf', 'gpflow_svgp']:
                current_sample_count = len(self.surrogate_manager.X_samples)
                samples_since_retrain = current_sample_count - self._last_retrain_count
                
                if samples_since_retrain >= self.retrain_interval:
                    print(f"  [Retraining] Full retrain triggered")
                    self.surrogate_manager.retrain_all()
                    self._last_retrain_count = current_sample_count
    
    # 缓存结果
    self._add_to_cache(x, y)
    
    return y, True
```

### 4. 修改配置文件

在配置中添加双线模式选项：

```python
config = {
    # ... 其他配置 ...
    
    'algorithm': 'mopso',
    'dual_line_mode': True,  # 启用双线模式
    'shared_dir': './shared_data',  # 共享数据目录
    
    'surrogate_config': {
        'type': 'gp',  # 或 'rf', 'gpflow_svgp'
        'min_samples': 10,
        'uncertainty_threshold': 0.5,
        'model_params': {
            'n_estimators': 100,  # RF参数
            'n_inducing': 100,    # GPflow参数
            'kernel_type': 'matern52'
        }
    }
}
```

## 完整修改文件

由于修改较多，建议创建一个新的MOPSO版本：`mopso_dual_line.py`

主要改动：
1. 导入双线架构模块
2. 添加双线模式配置
3. 使用`DualLineSurrogateManager`
4. 在评估时检查热替换
5. 移除训练逻辑（双线模式下）

## 测试方案

1. **单元测试**：测试`DualLineSurrogateManager`的热替换功能
2. **集成测试**：测试优化线和训练线的通信
3. **端到端测试**：运行完整的优化流程

## 回滚方案

如果双线模式出现问题，可以通过配置开关快速切回单线模式：

```python
config['dual_line_mode'] = False
```

系统会自动使用原有的单线模式。
