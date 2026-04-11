"""
默认配置模块 - HFSS 天线优化程序

配置说明:
1. HFSS_CONFIG: HFSS 项目路径和设置
2. VARIABLES: 优化变量定义
3. OBJECTIVES: 优化目标定义
4. ALGORITHM_CONFIG: 优化算法参数
5. RUN_CONFIG: 运行配置

使用方法:
- 直接修改此文件，或
- 通过 GUI 界面修改（推荐）
"""
from typing import Dict, List

# ============================================================================
# HFSS 项目配置
# ============================================================================
HFSS_CONFIG = {
    # HFSS 项目文件路径 (.aedt)
    'project_path': r"C:\Users\16438\Desktop\pyHFSSProject\n.aedt",
    
    # 设计名称 (在 HFSS Project Manager 中显示的名称)
    'design_name': "broadB1",
    
    # Setup 名称 (求解设置)
    # 通常为 "Setup1"，可在 HFSS 中查看
    'setup_name': "Setup1",
    
    # Sweep 名称 (扫频设置)
    # 通常为 "Sweep"，可在 HFSS 中查看
    'sweep_name': "Sweep",
}

# ============================================================================
# 变量配置
# ============================================================================
# 变量格式: {'name': 变量名, 'bounds': (最小值, 最大值), 'unit': 单位}
#
# 变量名必须与 HFSS 项目中定义的变量名完全一致（区分大小写）
# bounds 定义变量的搜索范围，优化算法会在此范围内寻找最优值
# unit 单位: mm, GHz, nH, pF, deg 等
#
VARIABLES = [
    # Md: 介质基板宽度/接地层宽度
    # 影响: 影响天线的阻抗匹配和辐射特性
    {'name': 'Md', 'bounds': (22.0, 35.0), 'unit': 'mm'},
    
    # Wm2: 微带线宽度参数2
    # 影响: 影响阻抗匹配和带宽
    {'name': 'Wm2', 'bounds': (0.2, 2.0), 'unit': 'mm'},
    
    # Wm3: 微带线宽度参数3
    # 影响: 影响阻抗匹配和高频特性
    {'name': 'Wm3', 'bounds': (0.2, 2.0), 'unit': 'mm'},
    
    # Wp: 贴片宽度
    # 影响: 影响谐振频率和增益
    {'name': 'Wp', 'bounds': (7.0, 15.0), 'unit': 'mm'},
    
    # Lp: 贴片长度
    # 影响: 影响谐振频率，是关键参数
    {'name': 'Lp', 'bounds': (5.0, 15.0), 'unit': 'mm'},
]

# ============================================================================
# 目标配置
# ============================================================================
# 目标格式说明:
#
# 类型 's_db' - S 参数 (dB)
# {
#     'type': 's_db',
#     'port': (m, n),           # 端口号，如 (1,1) 表示 S11
#     'freq_range': (f1, f2),   # 频率范围 (GHz)，或 'freq': 单频点
#     'target': 'minimize',     # 优化方向: minimize 或 maximize
#     'constraint': 'max',      # 约束类型: max, min, mean
#     'goal': -10.0,            # 目标值
#     'name': 'S11_max',        # 目标名称
# }
#
# 类型 'peak_gain' - 峰值增益
# {
#     'type': 'peak_gain',
#     'freq': 5.9,              # 频率 (GHz)
#     'target': 'maximize',     # 优化方向: maximize
#     'goal': 6.0,              # 目标值 (dB)
#     'name': 'PeakGain',       # 目标名称
# }
#
# 注意:
# - 对于 Interpolating Sweep，增益只能在 Setup 频率点获取
# - 程序会自动检测目标频率是否在 Setup 中
# - 如果不在，会自动更新 Setup Frequency 为目标频率
# - 修改 Setup Frequency 后需要重新运行仿真
#
OBJECTIVES = [
    # 目标1: S11 最小化
    # 要求: 在 5.1-7.2 GHz 频带内，S11 的最大值 < -10 dB
    {
        'type': 's_db',
        'port': (1, 1),              # S11
        'freq_range': (5.1, 7.2),    # 频率范围 (GHz)
        'target': 'minimize',        # 最小化 S11
        'constraint': 'max',         # 取频带内的最大值
        'goal': -10.0,               # 目标: < -10 dB
        'name': 'S11_max',
    },
    
    # 目标2: PeakGain 最大化
    # 要求: 在指定频率点，PeakGain > 6 dB
    # 注意: freq 会自动设置为 Setup Frequency
    {
        'type': 'peak_gain',
        'freq': 5.9,                 # 频率 (GHz) - 会自动设置为 Setup Frequency
        'target': 'maximize',        # 最大化增益
        'goal': 6.0,                 # 目标: > 6 dB
        'name': 'PeakGain',
    },
]

# ============================================================================
# 算法配置
# ============================================================================
# NSGA-II 多目标遗传算法参数
#
ALGORITHM_CONFIG = {
    # ===== 算法选择 =====
    # algorithm: 优化算法类型
    # - 'mobo': 多目标贝叶斯优化（推荐！专为昂贵黑箱优化设计）
    # - 'robust': 鲁棒优化器（适合非凸、不连续问题）
    # - 'adaptive': 自适应优化器（自动检测问题特性）
    # - 'surrogate': 代理模型优化器（平滑函数效果好）
    # - 'nsga2': 纯遗传算法（无代理模型，最稳健）
    'algorithm': 'mobo',
    
    # surrogate_type: 代理模型类型（仅 surrogate/robust 模式）
    # - 'gp': 高斯过程（全量训练，适合平滑函数）
    # - 'rf': 随机森林（全量训练，对不连续函数更鲁棒）
    # - 'incremental': RFF+SGD增量学习（轻量级，单样本更新能力）
    # - 'gpflow_svgp': 稀疏变分GP（强大，非线性+增量学习+不确定性估计）
    #     推荐用于：多变量、非凸、有突变、复杂的天线优化场景
    #     需要安装: pip install gpflow tensorflow
    'surrogate_type': 'gpflow_svgp',
    
    # ===== 多起点优化 =====
    # n_restarts: 多起点次数
    # - 并行运行多个优化，避免局部最优
    # - 推荐: 3-5 次
    'n_restarts': 3,
    
    # ===== 种群参数 =====
    # population_size: 种群大小
    # - 较大种群可提高全局搜索能力，但计算时间增加
    # - 推荐: 变量数 × 4 到 变量数 × 10
    'population_size': 30,
    
    # n_generations: 迭代代数
    # - 更多迭代可得到更优解，但计算时间增加
    # - 推荐: 10-50 代
    'n_generations': 15,
    
    # ===== 初始样本 =====
    # initial_samples: 初始采样数
    # - 用于建立初始代理模型
    # - 推荐: 变量数 × 15 到 变量数 × 25
    'initial_samples': 80,
    
    # ===== 交叉变异参数 =====
    # crossover_prob: 交叉概率
    # - 推荐: 0.8-0.95
    'crossover_prob': 0.9,
    
    # mutation_prob: 变异概率
    # - 较高变异率可增加种群多样性，避免局部最优
    # - 推荐: 0.1-0.2（天线优化建议 0.15-0.2）
    'mutation_prob': 0.15,
    
    # eta_c: SBX 交叉分布指数
    # - 推荐: 15-30
    'eta_c': 20,
    
    # eta_m: 多项式变异分布指数
    # - 推荐: 15-30
    'eta_m': 20,
    
    # ===== 精英保留 =====
    # elite_ratio: 精英保留比例
    # - 保留最优个体的比例
    # - 推荐: 0.1-0.2
    'elite_ratio': 0.1,
    
    # ===== 代理模型参数 =====
    # surrogate_enabled: 是否启用代理模型
    'surrogate_enabled': True,
    
    # min_real_evals: 每代最小真实评估数
    'min_real_evals': 20,
    
    # update_interval: 代理模型更新间隔
    'update_interval': 10,

    # ===== 早停参数 =====
    # stop_when_goal_met: 当达到目标时是否提前停止
    'stop_when_goal_met': True,
    
    # n_solutions_to_stop: 达到目标的解数量阈值，触发早停
    # - 例如设置为5，表示当有5个以上解全部达标时停止
    'n_solutions_to_stop': 5,
}

# ============================================================================
# 运行配置
# ============================================================================
RUN_CONFIG = {
    # force_reanalyze: 是否强制重新仿真
    # - True: 每次都重新仿真（推荐，确保结果正确）
    # - False: 如果已有结果则跳过仿真
    'force_reanalyze': True,
    
    # clear_old_results: 是否清除旧结果
    # - True: 开始优化前清除之前的结果
    'clear_old_results': True,
    
    # output_dir: 结果输出目录
    'output_dir': r"C:\Users\16438\Desktop\HFSS-Python-Optimizer\optim_results",
}


def get_default_config() -> Dict:
    """获取默认配置"""
    return {
        'hfss': HFSS_CONFIG.copy(),
        'variables': VARIABLES.copy(),
        'objectives': OBJECTIVES.copy(),
        'algorithm': ALGORITHM_CONFIG.copy(),
        'run': RUN_CONFIG.copy(),
    }


def validate_config(config: Dict) -> bool:
    """
    验证配置
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_keys = ['hfss', 'variables', 'objectives', 'algorithm']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"[ERROR] Missing config key: {key}")
            return False
    
    # 检查变量
    if not config['variables']:
        logger.error(f"[ERROR] No variables defined")
        return False
    
    for var in config['variables']:
        if 'name' not in var or 'bounds' not in var:
            logger.error(f"[ERROR] Invalid variable: {var}")
            return False
    
    # 检查目标
    if not config['objectives']:
        logger.error(f"[ERROR] No objectives defined")
        return False
    
    for obj in config['objectives']:
        if 'type' not in obj or 'name' not in obj:
            logger.error(f"[ERROR] Invalid objective: {obj}")
            return False
    
    return True


# ============================================================================
# 目标类型参考
# ============================================================================
"""
支持的目标类型:

1. 's_db' - S 参数 (dB 值)
   示例: 在 5-6 GHz 频带内最小化 S11
   {
       'type': 's_db',
       'port': (1, 1),           # S11
       'freq_range': (5.0, 6.0), # 或 'freq': 5.9
       'target': 'minimize',
       'constraint': 'max',      # max/min/mean
       'goal': -10.0,
       'name': 'S11_max',
   }

2. 'peak_gain' - 峰值增益
   示例: 在 4 GHz 最大化增益
   {
       'type': 'peak_gain',
       'freq': 4.0,              # 必须是 Setup Frequency
       'target': 'maximize',
       'goal': 6.0,
       'name': 'PeakGain',
   }

3. 's_mag' - S 参数幅度 (线性值)
   示例: S11 幅度 < 0.3
   {
       'type': 's_mag',
       'port': (1, 1),
       'freq': 5.9,
       'target': 'minimize',
       'goal': 0.3,
       'name': 'S11_mag',
   }

4. 'z_real' - 阻抗实部
   示例: 阻抗实部接近 50Ω
   {
       'type': 'z_real',
       'freq': 5.9,
       'value': 50.0,            # 目标值
       'tolerance': 10.0,        # 容差
       'name': 'Z_real',
   }

5. 'z_imag' - 阻抗虚部
   示例: 阻抗虚部接近 0Ω
   {
       'type': 'z_imag',
       'freq': 5.9,
       'value': 0.0,
       'tolerance': 5.0,
       'name': 'Z_imag',
   }
"""