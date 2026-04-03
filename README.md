# HFSS-Optimizer

HFSS 天线自动化优化工具，集成多种优化算法（MOPSO、MOBO、NSGA-II）和代理模型（GPR、RF、GPflow SVGP），通过代理模型技术大幅减少仿真次数。

## 功能特性

- 🖥️ **GUI 界面** - PyQt6 现代化图形配置界面
- 🚀 **多种算法** - MOPSO、MOBO、NSGA-II、Surrogate-NSGA-II
- ⚙️ **代理加速** - GP/RF/GPflow SVGP 代理模型减少仿真次数
- 📊 **实时可视化** - 优化进度图表自动生成
- 💾 **历史复用** - 加载历史数据继续优化
- 🔧 **公式目标** - 支持 `dB(S(1,1)) + max(dB(S(2,1)))` 等复杂表达式

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
pip install gpflow tensorflow  # 推荐安装
python setup_env.py             # 一键配置
```

### 2. 配置项目

编辑 `user_config.json` 或使用 GUI：

```json
{
  "hfss": {
    "project_path": "C:/path/to/project.aedt",
    "design_name": "HFSSDesign1",
    "setup_name": "Setup1",
    "sweep_name": "Sweep"
  },
  "variables": [
    {"name": "Wm", "bounds": [0.2, 1.5], "unit": "mm"},
    {"name": "Lp", "bounds": [8.0, 14.0], "unit": "mm"}
  ],
  "objectives": [
    {
      "name": "S11",
      "type": "formula",
      "freq_range": [5.7, 6.1],
      "formula": "dB(S(1,1))",
      "goal": -10.0,
      "target": "minimize",
      "weight": 1.0
    },
    {
      "name": "PeakGain",
      "type": "peak_gain",
      "freq": 5.9,
      "goal": 7.5,
      "target": "maximize",
      "weight": 1.0
    }
  ],
  "algorithm": {
    "algorithm": "mopso",
    "population_size": 20,
    "n_generations": 30,
    "use_surrogate": true,
    "surrogate_type": "gpflow_svgp"
  }
}
```

### 3. 启动优化

```bash
python gui_pyqt6.py    # 现代化 GUI（推荐）
python gui.py          # 传统 GUI
python run.py          # 命令行模式
```

## 算法

| 算法 | 描述 |
|------|------|
| MOPSO | 多目标粒子群优化 |
| MOBO | 多目标贝叶斯优化 |
| NSGA-II | 非支配排序遗传算法 |
| Surrogate-NSGA-II | 代理模型辅助 NSGA-II |

## 代理模型

| 模型 | 特点 |
|------|------|
| GPflow SVGP | 稀疏变分高斯过程，推荐使用 |
| GPR | 高斯过程回归 |
| RF | 随机森林 |

## 项目结构

```
HFSS-Optimizer/
├── run.py                 # CLI 入口
├── gui.py / gui_pyqt6.py  # GUI 入口
├── setup_env.py           # 环境配置
├── user_config.json       # 用户配置
├── core/                  # 核心模块
│   ├── hfss_controller.py # HFSS 控制
│   ├── evaluator.py       # 目标评估
│   └── surrogate.py       # 代理模型
├── algorithms/            # 优化算法
├── utils/                 # 工具函数
└── tests/                 # 测试文件
```

## 依赖

- Python 3.8+
- PyAEDT (HFSS 自动化)
- NumPy, SciPy, scikit-learn
- GPflow, TensorFlow (推荐)
- PyQt6

## 许可

MIT
