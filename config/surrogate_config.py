"""
代理模型配置定义

每种代理模型的：
1. 显示名称和描述
2. 专属配置参数
3. 默认值
4. 参数说明

使用方法：
    from config.surrogate_config import SURROGATE_MODELS, COMMON_PARAMS, get_model_default_config
"""

# 各模型配置定义
SURROGATE_MODELS = {
    "gp": {
        "display_name": "GP (高斯过程)",
        "description": "全量训练，适合平滑函数，不确定性估计准确",
        "is_incremental": False,
        "params": [
            {
                "key": "min_new_samples_to_train",
                "label": "最小新样本数触发训练",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 100,
                "unit": "个新样本",
                "tooltip": "训练线：至少积累N个新样本才触发训练\n避免训练过于频繁",
            },
        ],
    },
    "rf": {
        "display_name": "RF (随机森林)",
        "description": "全量训练，适合不连续函数，对异常值鲁棒",
        "is_incremental": False,
        "params": [
            {
                "key": "min_new_samples_to_train",
                "label": "最小新样本数触发训练",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 100,
                "unit": "个新样本",
                "tooltip": "训练线：至少积累N个新样本才触发训练\n避免训练过于频繁",
            },
            {
                "key": "n_estimators",
                "label": "决策树数量",
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 500,
                "tooltip": "更多的树可以提高精度，但会增加训练时间",
            },
        ],
    },
    "incremental": {
        "display_name": "Incremental (RFF+SGD)",
        "description": "增量学习，轻量级，每次仿真后自动更新",
        "is_incremental": True,
        "params": [
            {
                "key": "n_features",
                "label": "傅里叶特征维度",
                "type": "int",
                "default": 100,
                "min": 50,
                "max": 500,
                "tooltip": "越大模型表达能力越强，但计算量增加",
            },
            {
                "key": "gamma",
                "label": "核参数 (gamma)",
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "decimals": 2,
                "tooltip": "控制模型的平滑程度，值越大越平滑",
            },
        ],
    },
    "gpflow_svgp": {
        "display_name": "GPflow-SVGP (推荐)",
        "description": "全量训练，适合复杂场景，非线性拟合能力强",
        "is_incremental": False,  # 修复：GPflow SVGP不支持真正的增量学习
        "params": [
            {
                "key": "min_new_samples_to_train",
                "label": "最小新样本数触发训练",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 100,
                "unit": "个新样本",
                "tooltip": "训练线：至少积累N个新样本才触发训练\n避免训练过于频繁",
            },
            {
                "key": "n_inducing",
                "label": "诱导点数量",
                "type": "int",
                "default": 100,
                "min": 20,
                "max": 500,
                "tooltip": "影响模型精度和训练速度\n越大越精确但训练越慢",
            },
            {
                "key": "kernel_type",
                "label": "核函数类型",
                "type": "combo",
                "default": "matern52",
                "options": [
                    {"value": "matern52", "label": "Matern 5/2 (推荐)"},
                    {"value": "matern32", "label": "Matern 3/2"},
                    {"value": "rbf", "label": "RBF"},
                ],
                "tooltip": "Matern 5/2 适合大多数场景\nRBF 适合非常平滑的函数",
            },
        ],
    },
}

# 通用参数（所有模型共享）
COMMON_PARAMS = [
    {
        "key": "min_samples",
        "label": "最小训练样本",
        "type": "int",
        "default": 5,
        "min": 3,
        "max": 100,
        "tooltip": "开始训练代理模型所需的最小样本数",
    },
    {
        "key": "uncertainty_threshold",
        "label": "不确定性阈值",
        "type": "float",
        "default": 0.5,
        "min": 0.1,
        "max": 2.0,
        "decimals": 2,
        "tooltip": "低于此值使用代理预测，高于此值进行真实仿真\n建议: 0.3-0.8",
    },
]


def get_model_default_config(model_type: str) -> dict:
    """
    获取指定模型的默认配置

    Args:
        model_type: 模型类型 ('gp', 'rf', 'incremental', 'gpflow_svgp')

    Returns:
        包含通用参数和模型专属参数的字典
    """
    config = {
        "min_samples": COMMON_PARAMS[0]["default"],
        "uncertainty_threshold": COMMON_PARAMS[1]["default"],
        "model_params": {},
    }

    model_def = SURROGATE_MODELS.get(model_type, {})
    for param in model_def.get("params", []):
        config["model_params"][param["key"]] = param["default"]

    return config


def get_all_default_config() -> dict:
    """获取所有模型的默认配置"""
    return {
        "surrogate_type": "gpflow_svgp",
        "use_surrogate": False,
        "surrogate_config": get_model_default_config("gpflow_svgp"),
    }


def validate_config(model_type: str, config: dict) -> dict:
    """
    验证并修正配置参数

    Args:
        model_type: 模型类型
        config: 配置字典

    Returns:
        验证后的配置（超出范围的值会被修正）
    """
    validated = config.copy()

    # 验证通用参数
    for param in COMMON_PARAMS:
        key = param["key"]
        if key in validated:
            value = validated[key]
            if param["type"] == "int":
                validated[key] = max(param["min"], min(param["max"], int(value)))
            elif param["type"] == "float":
                validated[key] = max(param["min"], min(param["max"], float(value)))

    # 验证模型专属参数
    model_def = SURROGATE_MODELS.get(model_type, {})
    model_params = validated.get("model_params", {})

    for param in model_def.get("params", []):
        key = param["key"]
        if key in model_params:
            value = model_params[key]
            if param["type"] == "int":
                model_params[key] = max(param["min"], min(param["max"], int(value)))
            elif param["type"] == "float":
                model_params[key] = max(param["min"], min(param["max"], float(value)))
            elif param["type"] == "combo":
                valid_values = [opt["value"] for opt in param.get("options", [])]
                if value not in valid_values:
                    model_params[key] = param["default"]

    validated["model_params"] = model_params
    return validated
