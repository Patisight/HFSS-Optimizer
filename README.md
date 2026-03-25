# HFSS-AI-Optimizer

A design optimization tool that integrates optimization algorithms and AI proxy models with HFSS (High Frequency Structure Simulator) automation control.

## Overview

HFSS-AI-Optimizer enables efficient parameter optimization of RF/microwave devices using surrogate-assisted optimization algorithms. By leveraging proxy models like Gaussian Process Regression (GPR) and Random Forest (RF) combined with active learning strategies, this tool significantly reduces manual tuning costs and simulation time.

> **Note**: Neural network proxy models are under development. Currently, GPR and RF surrogate models are available.

## Features

- 🖥️ **GUI Interface** - Full-featured visual configuration
- 🚀 **Multiple Algorithms** - MOPSO, MOBO, NSGA-II, Surrogate-NSGA-II
- ⚙️ **Surrogate Acceleration** - GP/RF proxy models reduce simulation count
- 📊 **Real-time Visualization** - Automatic optimization progress charts
- 💾 **History Reuse** - Load historical data to continue optimization
- 🔧 **HFSS Automation** - Direct control via PyAEDT

## Quick Start

### 1. Environment Setup

Run `环境配置.bat` or:
```bash
python setup_env.py
```

### 2. Configure Your Project

Edit `user_config.json`:
```json
{
  "hfss": {
    "project_path": "C:/path/to/project.aedt",
    "design_name": "HFSSDesign1",
    "setup_name": "Setup1",
    "sweep_name": "Sweep"
  },
  "variables": [
    {"name": "Rl", "bounds": [10.0, 30.0], "unit": "mm", "precision": 2}
  ],
  "objectives": [
    {"type": "s_db", "name": "S11_max", "goal": -10.0, "target": "minimize", "freq_range": [5.1, 7.2], "port": [1, 1]}
  ],
  "algorithm": {
    "algorithm": "mopso",
    "population_size": 20,
    "n_generations": 30,
    "use_surrogate": true,
    "surrogate_type": "rf"
  }
}
```

### 3. Launch Optimization

```bash
python gui.py
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **MOPSO** | Multi-Objective Particle Swarm Optimization |
| **MOBO** | Multi-Objective Bayesian Optimization |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II |
| **Surrogate-NSGA-II** | Proxy model assisted NSGA-II |

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   MOPSO     │────▶│  Evaluator   │────▶│    HFSS     │
│  Algorithm  │     │              │     │  Simulation │
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                    │
       │                    ▼
       │            ┌──────────────┐
       │            │   Surrogate  │ (GP/RF)
       │            │    Model     │
       │            └──────────────┘
       │                    │
       └────────────────────┘
              Train & Predict
```

### Surrogate Model Logic

The surrogate model predicts objective values and uncertainty:

- **Low uncertainty** (< threshold) → Use prediction, skip simulation
- **High uncertainty** (≥ threshold) → Run real simulation

This reduces total simulation count significantly.

## Project Structure

```
HFSS-AI-Optimizer/
├── run.py                 # CLI entry
├── gui.py                # GUI entry
├── launch_gui.py        # GUI launcher
├── setup_env.py         # Environment setup
├── user_config.json     # User configuration
├── core/                # Core modules
│   ├── hfss_controller.py   # HFSS control
│   ├── evaluator.py          # Objective evaluation
│   └── surrogate.py         # Proxy model
├── algorithms/           # Optimization algorithms
│   ├── mopso.py        # MOPSO
│   ├── mobo.py         # MOBO
│   ├── nsga2.py        # NSGA-II
│   └── surrogate.py     # Surrogate-NSGA-II
├── utils/               # Visualization
├── config/              # Configuration
└── tests/              # Test tools
```

## Configuration

### Variables

```json
{
  "name": "variable_name",
  "bounds": [min, max],
  "unit": "mm",
  "precision": 2
}
```

### Objectives

| Type | Description | Target |
|------|-------------|--------|
| `s_db` | S-parameter dB value | minimize/maximize |
| `s_mag` | S-parameter magnitude | minimize/maximize |
| `peak_gain` | Peak gain (dB) | maximize |

### MOPSO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 20 | Population size |
| `n_generations` | 30 | Number of generations |
| `use_surrogate` | true | Enable surrogate model |
| `surrogate_type` | rf | 'gp' or 'rf' |
| `surrogate_threshold` | 0.5 | Uncertainty threshold |

## Requirements

- Python 3.8+
- PyAEDT (HFSS automation)
- NumPy, SciPy
- scikit-learn
- matplotlib

## License

MIT
