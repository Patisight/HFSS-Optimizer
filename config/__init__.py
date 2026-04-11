"""
配置模块
"""

from .default_config import (
    ALGORITHM_CONFIG,
    HFSS_CONFIG,
    OBJECTIVES,
    RUN_CONFIG,
    VARIABLES,
    get_default_config,
    validate_config,
)
from .surrogate_config import (
    COMMON_PARAMS,
    SURROGATE_MODELS,
    get_all_default_config,
    get_model_default_config,
)
from .surrogate_config import validate_config as validate_surrogate_config

__all__ = [
    "HFSS_CONFIG",
    "VARIABLES",
    "OBJECTIVES",
    "ALGORITHM_CONFIG",
    "RUN_CONFIG",
    "get_default_config",
    "validate_config",
    # 代理模型配置
    "SURROGATE_MODELS",
    "COMMON_PARAMS",
    "get_model_default_config",
    "get_all_default_config",
    "validate_surrogate_config",
]
