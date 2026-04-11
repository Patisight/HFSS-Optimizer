# HFSS-Python-Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

HFSS 天线自动化优化工具，集成多种优化算法（MOPSO、MOBO、NSGA-II）和代理模型（GPR、RF、GPflow SVGP），通过代理模型技术大幅减少仿真次数。

## 📥 下载与安装

### 从 GitHub 下载

```bash
# 方式一：Git 克隆
git clone https://github.com/Patisight/HFSS-Optimizer.git
cd HFSS-Optimizer

# 方式二：下载 ZIP 压缩包
# 点击 GitHub 页面的 "Code" -> "Download ZIP"，解压后进入目录
```

### 安装依赖

```bash
# 直接安装依赖
pip install -r requirements.txt

# 推荐安装代理模型增强依赖（可选）
pip install gpflow tensorflow
```

## 🚀 运行项目

### Windows 用户（推荐）

双击 `start.bat` 或直接运行：

```bash
python gui_pyqt6.py
```

### 命令行模式

```bash
python run.py --config user_config.json
```

### 配置项目

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
      "type": "S参数",
      "freq_range": [5.7, 6.1],
      "formula": "dB(S(1,1))",
      "goal": -10.0,
      "target": "minimize",
      "weight": 1.0
    },
    {
      "name": "Z11",
      "type": "Z参数",
      "freq_range": [5.7, 6.1],
      "formula": "mag(Z(1,1))",
      "goal": 50.0,
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

## 🔄 更新项目

### 从 GitHub 更新最新版本

```bash
# 如果是 Git 克隆的项目
cd HFSS-Optimizer
git pull origin main

# 重新安装依赖（如果有更新）
pip install -r requirements.txt --upgrade
```

## ✨ 功能特性

- 🖥️ **GUI 界面** - PyQt6 现代化图形配置界面
- 🚀 **多种算法** - MOPSO、MOBO、NSGA-II、Surrogate-NSGA-II
- ⚙️ **代理加速** - GP/RF/GPflow SVGP 代理模型减少仿真次数
- 📊 **实时可视化** - 优化进度图表自动生成
- 💾 **历史复用** - 加载历史数据继续优化
- 🔧 **公式目标** - 支持 `dB(S(1,1)) + max(dB(S(2,1)))` 等复杂表达式
- 📡 **Z 参数支持** - 新增 Z 参数公式计算，如 `mag(Z(1,1))`、`re(Z(2,1))`

## 📚 算法

| 算法 | 描述 |
|------|------|
| MOPSO | 多目标粒子群优化 |
| MOBO | 多目标贝叶斯优化 |
| NSGA-II | 非支配排序遗传算法 |
| Surrogate-NSGA-II | 代理模型辅助 NSGA-II |

## 🤖 代理模型

| 模型 | 特点 |
|------|------|
| GPflow SVGP | 稀疏变分高斯过程，推荐使用 |
| GPR | 高斯过程回归 |
| RF | 随机森林 |

## 📁 项目结构

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

## 📝 依赖与兼容性

### HFSS 版本要求

| HFSS 版本 | 兼容性 |
|----------|--------|
| HFSS 2022 R1+ | ✅ 推荐 |
| HFSS 2023 R1 | ✅ 已测试 |
| HFSS 2024 R1+ | ✅ 兼容 |

### Python 与 PyAEDT 版本

| 依赖 | 版本要求 |
|------|---------|
| **Python** | 3.8 - 3.12 |
| **PyAEDT** | >= 0.7.0 (推荐最新版 0.26.x) |

### 其他依赖

- NumPy, SciPy, scikit-learn
- GPflow, TensorFlow (推荐，用于代理模型)
- PyQt6

## 📄 许可

MIT
