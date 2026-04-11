# 快速开始指南

## 环境要求
- Windows 10/11
- HFSS 2021R1 或更高版本
- Python 3.8 ~ 3.11 (推荐 3.10)

## 安装步骤
1. 克隆或下载项目到本地
```bash
git clone https://github.com/your-repo/HFSS-Python-Optimizer.git
cd HFSS-Python-Optimizer
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 测试安装
```bash
python run.py --version
```
如果输出版本号则安装成功。

## 快速使用

### 1. 准备HFSS项目
- 创建或打开你的HFSS天线项目
- 确保已经设置好：
  - 辐射边界
  - 远场球体
  - 仿真Setup和Sweep

### 2. 配置优化参数
复制默认配置文件：
```bash
cp config/default_config.py user_config.py
```
编辑`user_config.py`，配置：
- HFSS项目路径
- 优化变量和范围
- 优化目标
- 算法参数

### 3. 启动优化
```bash
# 使用默认NSGA2算法
python run.py

# 使用代理模型优化（更快）
python run.py --algorithm surrogate

# 启动图形界面
python launch_gui.py
```

### 4. 查看结果
优化结果会保存在`optim_results/`目录下，包含：
- 所有评估数据
- Pareto前沿解
- 优化过程图表
- 日志文件
