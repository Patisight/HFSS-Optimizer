# HFSS-AI-Optimizer - Project Document

> Version: 1.0  
> Updated: 2026-03-25  
> Maintainer: Jayden

---

## 1. Project Overview

### 1.1 Goal

HFSS-AI-Optimizer is a design optimization tool that integrates optimization algorithms and AI proxy models with HFSS (High Frequency Structure Simulator) automation control. The tool aims to achieve efficient parameter optimization of RF/microwave devices through surrogate model technologies such as Gaussian Process Regression (GPR) and Random Forest (RF), combined with active learning strategies, significantly reducing manual tuning costs and simulation time, and improving design efficiency.

> **Note**: Neural network proxy models are under development. Currently, GPR and RF surrogate models are available.

### 1.2 Core Features
- GUI configuration for optimization tasks
- Multiple optimization algorithm support
- Real-time visualization of progress
- Historical data reuse
- HFSS automation via PyAEDT

### 1.3 Tech Stack
- Python 3.x
- PyAEDT (HFSS Control)
- NumPy, SciPy (Scientific Computing)
- scikit-learn (Surrogate Models)
- Tkinter (GUI)

---

## 2. 项目结构

```
HFSS-Python-Optimizer/
├── run.py                     # 命令行入口
├── gui.py                     # 图形界面入口
├── launch_gui.py              # GUI 启动器
├── setup_env.py              # 环境配置工具
│
├── user_config.json           # 用户配置
│
├── core/                      # 核心模块
│   ├── __init__.py
│   ├── hfss_controller.py     # HFSS 连接和控制
│   ├── evaluator.py           # 目标评估器
│   └── surrogate.py          # 代理模型管理
│
├── algorithms/                # 优化算法
│   ├── __init__.py
│   ├── base.py               # 算法基类
│   ├── mobo.py               # 贝叶斯优化
│   ├── mopso.py              # 多目标粒子群 ⭐
│   ├── nsga2.py              # NSGA-II
│   ├── robust_optimizer.py   # 鲁棒优化
│   └── surrogate.py          # 代理模型辅助 NSGA-II
│
├── utils/                     # 工具模块
│   ├── __init__.py
│   └── visualization.py       # 可视化
│
├── config/                    # 配置模块
│   ├── __init__.py
│   └── default_config.py     # 默认配置
│
├── tests/                     # 测试工具
│   ├── checker.py            # 环境检测
│   ├── test_imports.py       # 导入测试
│   ├── test_validate.py      # 验证测试
│   └── test_message_api.py   # API 测试
│
├── optim_results/            # 优化结果输出
│   └── {algorithm}_{timestamp}/
│       ├── config.json       # 使用的配置
│       ├── results.json      # Pareto 前沿解
│       ├── report.json       # 统计报告
│       ├── evaluations.jsonl # 所有评估记录
│       └── progress_*.png   # 进度图
│
├── 环境配置.bat              # 环境配置菜单
├── 启动优化程序.bat          # GUI 启动器
├── requirements.txt
├── README.md                 # GitHub 首页
└── PROJECT.md               # 项目书
```

---

## 3. 功能模块

### 3.1 HFSS 控制器 (core/hfss_controller.py)

**职责**: 与 HFSS 软件通信

**功能**:
- 连接/断开 HFSS
- 设置变量值
- 执行仿真
- 获取 S 参数、增益等结果

**关键类**: `HFSSController`

### 3.2 目标评估器 (core/evaluator.py)

**职责**: 评估一组参数的目标值

**功能**:
- 根据目标配置计算目标值
- 支持多种目标类型
- 保存评估记录到 evaluations.jsonl

**目标类型**:
| 类型 | 说明 | 目标选项 |
|------|------|---------|
| `s_db` | S 参数 dB 值 | minimize, maximize |
| `s_mag` | S 参数幅值 | minimize, maximize, range |
| `peak_gain` | 峰值增益 | maximize |
| `z_real` | 阻抗实部 | target |
| `z_imag` | 阻抗虚部 | target |

**关键类**: `ObjectiveEvaluator`

### 3.3 代理模型 (core/surrogate.py)

**职责**: 用数学模型近似 HFSS 仿真

**支持模型**:
- `gp`: 高斯过程（平滑）
- `rf`: 随机森林（不连续）

**功能**:
- 训练模型
- 预测目标值
- 估计不确定性

**关键类**: `SurrogateManager`

---

## 4. 优化算法

### 4.1 MOPSO (多目标粒子群) ⭐

**文件**: `algorithms/mopso.py`

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 50 | 种群大小 |
| `n_generations` | 30 | 迭代代数 |
| `w_max` | 0.9 | 最大惯性权重 |
| `w_min` | 0.4 | 最小惯性权重 |
| `c1` | 1.5 | 认知学习因子 |
| `c2` | 1.5 | 社会学习因子 |
| `use_surrogate` | false | 是否使用代理模型 |
| `surrogate_threshold` | 1.0 | 不确定性阈值 |

**代理模型逻辑** (2026-03-25 更新):
```
不确定性 ≥ 阈值 → 真实仿真，用真实值迭代
不确定性 < 阈值 → 用预测值迭代，跳过仿真
```

**关键特性**:
- 自适应惯性权重
- 外部 Pareto 档案
- 拥挤距离多样性保持
- 历史数据加载复用
- Pareto 档案标记 `is_predicted`

### 4.2 MOBO (贝叶斯优化)

**文件**: `algorithms/mobo.py`

**特点**:
- 基于高斯过程的贝叶斯优化
- 自动选择最有潜力的参数点
-适合低维度问题

### 4.3 NSGA-II

**文件**: `algorithms/nsga2.py`

**特点**:
- 标准多目标进化算法
- 快速非支配排序
- 适合作为基准对比

### 4.4 Surrogate-NSGA-II

**文件**: `algorithms/surrogate.py`

**特点**:
- NSGA-II + 代理模型
- 减少仿真次数
- 适合高计算成本问题

---

## 5. 使用流程

### 5.1 图形界面 (推荐)

1. 启动：`python gui.py` 或双击 `启动优化程序.bat`
2. 配置项目路径、设计名称
3. 添加优化变量（名称、范围、单位、精度）
4. 添加优化目标（类型、频点、目标值）
5. 选择算法和参数
6. （可选）加载历史数据继续优化
7. 点击「开始优化」

### 5.2 命令行

```bash
python run.py --config user_config.json
```

---

## 6. 配置说明

### 6.1 变量配置

```json
{
  "name": "Rl",
  "bounds": [10.0, 30.0],
  "unit": "mm",
  "precision": 2
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 变量名（与 HFSS 一致） |
| `bounds` | ✅ | 参数范围 [最小, 最大] |
| `unit` | ❌ | 单位（默认 mm） |
| `precision` | ❌ | 小数位数（默认 4） |

### 6.2 目标配置

```json
{
  "name": "S11_max",
  "type": "s_db",
  "port": [1, 1],
  "freq_range": [5.1, 7.2],
  "target": "minimize",
  "goal": -10.0
}
```

### 6.3 算法配置 (MOPSO)

```json
{
  "algorithm": "mopso",
  "population_size": 20,
  "n_generations": 30,
  "use_surrogate": true,
  "surrogate_type": "rf",
  "surrogate_threshold": 1.0,
  "load_evaluations": null
}
```

---

## 7. 输出文件

### 7.1 evaluations.jsonl

每次评估的完整记录：

```json
{
  "eval_id": 1,
  "timestamp": "2026-03-25T10:00:00.000000",
  "parameters": [1.5, 0.63, ...],
  "objectives": {
    "S11_max": {
      "value": -12.5,
      "actual_value": -12.5,
      "goal_met": true
    },
    "PeakGain": {
      "value": -7.2,
      "actual_value": 7.2,
      "goal_met": true
    }
  }
}
```

**注意**:
- `value`: 优化用的目标函数值
  - minimize: 直接使用
  - maximize: 存储为 `-actual_value`
- `actual_value`: 真实的物理值
- `goal_met`: 是否达到目标

### 7.2 results.json

Pareto 前沿解：

```json
[
  {
    "parameters": [...],
    "objectives": [...],
    "is_predicted": false
  }
]
```

### 7.3 report.json

统计报告：

```json
{
  "n_evaluations": 170,
  "real_evaluations": 120,
  "surrogate_predictions": 50,
  "pareto_size": 15,
  "loaded_evaluations": 100,
  "best_objectives": [...]
}
```

---

## 8. 已知问题与解决方案

### 8.1 S11 返回正值

**问题**: `dB(S(1,1))` 返回正值（回波损耗格式）

**解决**: 程序自动取绝对值处理

### 8.2 maximize 目标显示

**问题**: maximize 目标存储为负值

**解决**: 输出时 `-actual_value` 转回正值

### 8.3 代理模型不确定性计算

**问题**: 不确定性阈值需要调优

**解决**: 默认 1.0，可根据实际效果调整

---

## 9. 开发记录

### 2026-03-25 更新

1. ✅ MOPSO 代理模型逻辑简化
   - 不确定性 ≥ 阈值 → 真实仿真
   - 不确定性 < 阈值 → 预测值迭代

2. ✅ Pareto 档案增加 `is_predicted` 标记

3. ✅ 变量小数点位数控制（precision 字段）

4. ✅ 加载历史数据继续优化

5. ✅ GUI 添加历史数据加载选项

6. ✅ 清理旧模型配置，只保留 minimax-portal

---

## 10. 下一步计划

- [ ] 完善代理模型不确定性计算
- [ ] 添加更多目标类型支持
- [ ] 优化可视化效果
- [ ] 添加批量测试功能
- [ ] 编写单元测试

---

*项目负责人: Jayden*  
*最后更新: 2026-03-25*
