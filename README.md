# HFSS-Python-Optimizer

An intelligent optimization framework for HFSS RF/microwave devices — integrating traditional optimization algorithms, AI surrogate models, and reinforcement learning into a unified optimization platform.

## Overview

This framework addresses the challenges of time-consuming parameter tuning and low manual optimization efficiency in HFSS simulations. It enables automated parameter optimization for antennas, filters, and other RF devices through multiple optimization strategies.

### Core Capabilities

- **Multi-Strategy Optimization**: Bayesian, PSO, CMA-ES, Differential Evolution
- **AI Surrogate Models**: Bayesian Neural Network (BNN), Gaussian Process Regression (GPR)
- **Reinforcement Learning**: Parameterized PPO agent with zero-shot generalization to different frequency constraints
- **HFSS Automation**: Stable simulation interface based on PyAEDT
- **Curriculum Learning**: Progressive difficulty scaling for faster model convergence

## Project Structure

```
HFSS-Python-Optimizer/
├── api.py                    # HFSS automation controller (core interface)
├── optim_framework.py        # Multi-strategy constraint optimization framework
│
├── src/                      # Reinforcement learning framework source
│   ├── environment/          # RL environments
│   │   ├── parameterized_env.py    # Parameterized pixel antenna environment
│   │   └── simple_env.py           # Simplified environment
│   │
│   ├── agent/                # RL agents
│   │   ├── generalized_agent.py    # Generalized PPO agent
│   │   ├── policy_networks.py      # Conditional policy networks
│   │   └── agent_config.py         # Agent configuration
│   │
│   ├── config/               # Constraint configuration
│   │   ├── constraint_config.py    # Constraint definitions
│   │   ├── constraint_manager.py   # Constraint management
│   │   └── constraint_sampler.py   # Constraint sampler
│   │
│   └── training/             # Training modules
│       ├── generalized_trainer.py  # Generalized trainer
│       ├── curriculum_scheduler.py # Curriculum learning scheduler
│       └── training_config.py      # Training configuration
│
├── examples/                 # Usage examples
│   ├── api_usage_example.py       # HFSS API usage demo
│   ├── simple_training.py         # Simplified training example
│   └── simple_inference.py        # Inference example
│
├── legacy/                   # Legacy code (retained for reference)
├── HFSS_Project/             # HFSS project files
├── models/                   # Saved models
├── checkpoints/              # Training checkpoints
└── logs/                     # Training logs
```

## Installation

### Requirements

- Python 3.8+
- Ansys HFSS 2023 R1 or later
- CUDA (optional, for GPU-accelerated training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20 | Numerical computing |
| pandas | >=1.3 | Data processing |
| scikit-learn | >=1.0 | Machine learning |
| scikit-optimize | >=0.9 | Bayesian optimization |
| torch | >=1.10 | Deep learning |
| gymnasium | >=0.26 | RL environment |
| pyDOE | >=0.3 | Experimental design |
| matplotlib | >=3.5 | Visualization |
| ansys-aedt | >=0.4 | HFSS integration |

## Quick Start

### 1. HFSS API Basics

```python
from api import HFSSController

# Use context manager for automatic connection handling
with HFSSController(
    project_path="path/to/project.aedt",
    design_name="HFSSDesign1"
) as hfss:
    # Get S-parameters
    s_params = hfss.get_s_params()
    
    # Set variable
    hfss.set_variable("Lp", 10, unit="mm")
    
    # Run simulation
    hfss.analyze()
    
    # Get far-field data
    farfield = hfss.get_farfield_data(
        sphere_name="3D",
        frequencies=[10e9],
        quantity="GainTotal"
    )
```

### 2. Traditional Constraint Optimization

```python
from optim_framework import HfssAdvancedConstraintOptimizer

# Define optimization variables
variables = [
    {'name': 'Lp', 'bounds': (3, 30), 'unit': 'mm'},
    {'name': 'Wp', 'bounds': (3, 25), 'unit': 'mm'}
]

# Define constraints
constraints = [
    {
        'expression': 'dB(S11)',
        'target': -15,
        'operator': '<',
        'weight': 1.0,
        'freq_range': (5e9, 7e9),
        'aggregate': 'max'
    }
]

# Port mapping
port_map = {'S11': ('1', '1')}

# Create optimizer
optimizer = HfssAdvancedConstraintOptimizer(
    project_path="project.aedt",
    variables=variables,
    constraints=constraints,
    global_port_map=port_map,
    max_iter=100
)

# Run optimization
result = optimizer.optimize(optimizer_type="bayesian")  # or "pso", "cmaes", "de"
```

### 3. Reinforcement Learning Training

```python
from src import (
    ParameterizedPixelAntennaEnv,
    GeneralizedPPOAgent,
    AgentConfig,
    GeneralizedTrainer,
    TrainingConfig,
    ConstraintManager
)

# Create environment
env = ParameterizedPixelAntennaEnv(
    project_path="project.aedt",
    grid_size=(10, 10),
    freq_samples=20,
    max_steps=50
)

# Configure agent
agent_config = AgentConfig(
    state_dim=126,      # params(99) + S11(20) + features(4) + constraint(3)
    action_dim=99,
    constraint_dim=3,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95
)

# Create agent
agent = GeneralizedPPOAgent(config=agent_config)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    
    for step in range(50):
        action, log_prob, value = agent.select_action(state, constraint)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store experience...
        
        if done or truncated:
            break
    
    # Update policy
    agent.update()
```

## Core Modules

### HFSS Controller (api.py)

Provides automated control of HFSS:

| Method | Description |
|--------|-------------|
| `connect()` | Connect to HFSS project |
| `set_variable()` | Set variable value (supports scalars and arrays) |
| `analyze()` | Run simulation |
| `get_s_params()` | Get S-parameter data |
| `get_farfield_data()` | Get far-field data |
| `save_s_params()` | Save S-parameters to CSV |

Features:
- Automatic lock file handling
- Connection retry on failure
- Timeout control support

### Optimization Framework (optim_framework.py)

Supports four optimization algorithms:

| Algorithm | Characteristics | Use Case |
|-----------|-----------------|----------|
| **Bayesian** (gp_minimize) | Efficient sampling, suitable for small samples | Expensive simulations, few variables |
| **PSO** | Strong global search capability | Multi-peak problems, continuous variables |
| **CMA-ES** | Adaptive covariance matrix | High-dimensional problems, non-convex optimization |
| **DE** | Differential evolution | Constrained optimization, robust performance |

Constraint expression support:
- S-parameters: `dB(S11)`, `S21`, `abs(S31)`
- Math functions: `max()`, `min()`, `mean()`, `log10()`
- Complex operations: `abs()`, `angle()`, `real()`, `imag()`

### Reinforcement Learning Environment (src/environment/)

Parameterized environment design:

```
Observation space: [params(99) + S11 samples(20) + physical features(4) + constraint vector(3)]
Action space: Discrete or continuous parameter adjustment
Reward function: Adaptive reward based on constraint satisfaction
```

Physical feature extraction:
- Resonant frequency
- Operating bandwidth
- Minimum S11 value
- Target deviation

### Constraint Management (src/config/)

Supports multi-band, multi-objective constraints:

```python
from src.config.constraint_config import ConstraintConfig, ConstraintGroup

# Single-band constraint
constraint = ConstraintConfig(
    freq_low=5e9,
    freq_high=6e9,
    target_s11=-15.0,
    tolerance=1.0
)

# Multi-band constraint group
group = ConstraintGroup(
    name="dual_band",
    constraints=[
        ConstraintConfig(freq_low=2.4e9, freq_high=2.5e9, target_s11=-10),
        ConstraintConfig(freq_low=5.1e9, freq_high=5.9e9, target_s11=-15)
    ]
)
```

## Advanced Features

### Curriculum Learning

Progressive difficulty scaling for faster model convergence:

```python
from src.training.curriculum_scheduler import CurriculumScheduler

scheduler = CurriculumScheduler(
    stages=["easy", "medium", "hard"],
    total_episodes=10000,
    performance_window=100
)
```

### Constraint Sampling Strategies

Supports multiple constraint sampling methods:
- **Uniform sampling**: Uniform sampling in constraint space
- **Difficulty-weighted**: Weighted by historical performance
- **Adaptive sampling**: Dynamically adjusted sampling distribution

### Model Save & Restore

```python
# Save model
agent.save("models/ppo_agent.pth")

# Load model
agent.load("models/ppo_agent.pth")

# Evaluate performance
results = agent.evaluate(env, constraints, n_episodes_per_constraint=10)
```

## Output Results

After optimization, results are saved in `optim_results/`:

```
optim_results/YYYYMMDD-HHMMSS/
├── optim_result.json      # Optimization summary
├── optim_history.csv      # Iteration history
├── best_s_params.csv      # Best S-parameters
├── optimization.png       # Convergence curve
└── s_params_plots/        # S-parameter plots per iteration
```

## Extending the Framework

### Adding New Optimization Algorithms

Implement in `optim_framework.py`:

```python
def my_optimizer(self, objective_func):
    """Custom optimizer"""
    # Implement your optimization logic
    # Return optimization result
    pass
```

### Adding New Constraint Types

Extend in `src/config/constraint_config.py`:

```python
@dataclass
class MyConstraint(ConstraintConfig):
    # Add new constraint attributes
    pass
```

## Notes

1. **HFSS Connection**: Ensure HFSS is properly installed and gRPC port is available
2. **Resource Management**: Use `with` statement to ensure proper connection closure
3. **Simulation Timeout**: Configure `iteration_timeout` to prevent simulation hang
4. **GPU Training**: Set `CUDA_VISIBLE_DEVICES` to specify GPU

## License

MIT License

## Contact

For questions or suggestions:
- Email: ayang1643816608@gmail.com
- GitHub Issues: [Project Issues Page]

---

**Performance**: Compared to manual tuning, this framework achieves **40%+ reduction in simulation time**, validated in antenna design, filter optimization, and EMI mitigation projects.