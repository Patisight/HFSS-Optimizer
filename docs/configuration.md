# 配置说明

## 配置文件结构
配置文件支持两种格式：Python (.py) 和 JSON (.json)

### 核心配置项

#### HFSS配置
```python
"hfss": {
    "project_path": "C:/path/to/your/project.aedt",  # HFSS项目路径
    "design_name": "AntennaDesign",                  # 设计名称
    "setup_name": "Setup1",                          # Setup名称
    "sweep_name": "Sweep"                            # Sweep名称
}
```

#### 变量配置
```python
"variables": [
    {
        "name": "width",        # 变量名称（必须和HFSS中一致）
        "bounds": [1.0, 5.0],   # 变量范围 [最小值, 最大值]
        "unit": "mm"            # 单位（可选）
    },
    {
        "name": "length",
        "bounds": [2.0, 10.0],
        "unit": "mm"
    }
]
```

#### 目标配置
支持两种目标类型：
1. **direct**：直接获取HFSS结果（如增益、S参数等）
2. **formula**：使用公式计算目标值

```python
"objectives": [
    {
        "name": "max_gain",
        "type": "direct",
        "goal": 15.0,           # 目标值
        "target": "maximize",   # 优化方向：maximize/minimize
        "weight": 1.0           # 权重（多目标优化时使用）
    },
    {
        "name": "s11_band",
        "type": "formula",
        "formula": "abs(S(1,1)) < -10",  # 计算公式
        "freq_range": [2.4, 2.5],        # 频率范围（GHz）
        "goal": 1.0,
        "target": "maximize",
        "weight": 1.0
    }
]
```

#### 算法配置
```python
"algorithm": {
    "name": "nsga2",              # 算法名称：nsga2, mopso, surrogate, mobo
    "population_size": 50,        # 种群大小
    "max_iterations": 100,        # 最大迭代次数
    "initial_samples": 50         # 初始样本数（代理模型算法使用）
}
```

## 配置校验
程序启动时会自动校验配置，如果有错误会在日志中提示：
- 变量范围顺序错误（最小值 >= 最大值）
- 必填字段缺失
- 公式语法错误
