from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any
import json

class VariableConfig(BaseModel):
    name: str
    bounds: List[Union[float, str]]
    unit: str = ""
    
class ObjectiveConfig(BaseModel):
    name: str
    type: str  # formula or direct
    formula: Optional[str] = None
    freq_range: Optional[List[float]] = None
    goal: float
    target: str  # minimize or maximize
    weight: float = 1.0
    
class HFSSConfig(BaseModel):
    project_path: str
    design_name: str
    setup_name: str
    sweep_name: str = ""

class AlgorithmConfig(BaseModel):
    algorithm: str = "nsga2"
    population_size: int = 50
    n_generations: int = 100
    surrogate_type: Optional[str] = None
    use_surrogate: bool = False
    surrogate_config: Optional[Dict[str, Any]] = None
    stop_when_goal_met: bool = True
    n_solutions_to_stop: int = 5
    load_evaluations: Optional[str] = None

class OptimizerConfig(BaseModel):
    hfss: HFSSConfig
    variables: List[VariableConfig]
    objectives: List[ObjectiveConfig]
    algorithm: Union[str, AlgorithmConfig] = "nsga2"
    run: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_json(cls, path: str) -> "OptimizerConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def validate(self) -> List[str]:
        """返回警告列表，空则校验通过"""
        warnings = []
        for var in self.variables:
            lower, upper = var.bounds
            # 只有两个都是数字的时候才比较顺序
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                if lower >= upper:
                    warnings.append(f"变量 {var.name} 的 bounds 顺序错误")
        return warnings
