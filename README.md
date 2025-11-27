# Adaptive Negative Correlation Learning Via Online Hypergradient Descent

**Master's Thesis Repository**  
**Date:** November, 2025  
**Author:** Frederico Portela  
**Institution:** NOVA Information Management School (IMS)

## 📌 Overview

This repository contains the complete source code, experimental framework, and results for the Master's Thesis titled **"Adaptive Negative Correlation Learning Via Online Hypergradient Descent."**

## 📂 Repository Structure

The project is organized as follows:

```text
.
├── src/                # Core implementation (models, training loops, HGD logic)
├── experiments/        # YAML configuration files defining experiment parameters
├── runs/               # Artifacts, logs, and results from experimental runs
├── notebooks/          # Jupyter notebooks for visualization and analysis
└── scripts/            # Shell scripts for batch execution
```


## 🛠️ Installation & Requirements

This project uses **[uv](https://github.com/astral-sh/uv)** for fast Python package management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fmportela/adaptive-ncl-hgd.git
    cd adaptive-ncl-hgd
    ```

2.  **Sync the environment:**
    ```bash
    uv sync
    ```
    or (preferred)
    ```bash
    make install
    ```


## 🚀 Usage

Experiments are driven by YAML configuration files located in the `experiments/` directory.

### 1. Run a Single Experiment
To run a specific configuration (e.g., a specific dataset or model variation), use the python entry point via `uv`:

```bash
uv run src/experiments/run_experiment.py -c experiments/path/to/config.yaml --overwrite
```

*   **`-c` / `--config`**: Path to the YAML configuration file.
*   **`--overwrite`**: If the experiment folder already exists in `runs/`, this flag overwrites it.

### 2. Run Batch Experiments
To reproduce the full suite of experiments presented in the thesis, use the provided shell script:

```bash
source scripts/run_exper.sh
```

## ⚙️ Configuration & Modes

Experiments are defined by YAML files in the `experiments/` directory. The configuration structure allows you to define the model architecture, the training loop, and the execution mode (e.g., single run, hyperparameter sweep, or Optuna tuning).

### Key Sections
*   **`dataset`**: Selects the target PMLB dataset and task type (regression/classification).
*   **`trainer`**: Defines the algorithm variant (i.e., `single`, `ncl`, `bagging`, `adaptive_ncl`, `multi_adaptive_ncl`) and specific hyperparameters like $\lambda$ initialization.
*   **`mode`**: Determines the execution strategy (`sweep` or `holdout`).
*   **`modes`**: specific settings for the selected mode (metrics, data splits, specific sweep lists).
*   **`param_grid`**: (Used in `holdout` mode) Defines the search space for Optuna hyperparameter optimization.

### Example 1: Optuna Optimization (Holdout Mode)
This configuration performs a Bayesian hyperparameter search using Optuna on the `churn` dataset. It utilizes a holdout set for validation and reserves a test set for final evaluation.

```yaml
description: |
  Adaptive NCL on churn dataset with Optuna tuning.

run_dir: "./runs/exper1/classification/churn/"
mode: holdout  # <--- execution mode

dataset:
  name: churn
  task_type: classification

trainer:
  name: adaptive_ncl
  params:
    lambda_init: 0.1
    use_dncc: true
    hvp_damping: 0.1

# ... [Model, Ensemble, Optim settings] ...

# Mode-specific settings
modes: 
  holdout:
    ratios: {train: 0.8, val: 0.0, test: 0.2}
    stratify: true
    tune:
      enabled: true
      n_trials: 50
      n_startup_trials: 20
      metric: binary_f1_score
      direction: maximize

# Optuna Search Space
param_grid:
  optim__lr:
    type: float
    low: 0.01
    high: 0.5
    log: true
  lambda_init:
    type: float
    low: 0.0
    high: 1.1
```

### Example 2: Hyperparameter Sweep
This configuration manually sweeps over specific lists of values (grid search). It is useful for analyzing the sensitivity of specific parameters like `lambda_lr`.

```yaml
description: |
  Sweep of multi-lambda adaptive NCL on 529_pollen.

run_dir: "./runs/exper2/regression/529_pollen/"
mode: sweep  # <--- execution mode

dataset:
  name: 529_pollen
  task_type: regression

trainer:
  name: multi_adaptive_ncl
  # ...

modes: 
  sweep:
    ratios: {train: 0.7, val: 0.1, test: 0.2}
    metrics: [mse, rmse, mae]
    # Define specific lists to iterate over:
    params:
      lambda_init: [0.0, 0.5, 0.9]
      lambda_lr: [0.0, 0.05, 0.1]
```

## 🚀 Usage

### 1. Run an Experiment
Pass the path to a configuration file. The script will automatically detect the mode (`sweep` or `holdout`) defined inside the YAML.

```bash
uv run src/experiments/run_experiment.py -c experiments/classification/churn_optuna.yaml --overwrite
```

### 2. Batch Execution
To reproduce the full thesis benchmark, use the provided shell script:

```bash
source scripts/run_exper.sh
```

## 📈 Results

Results are automatically saved to the `runs/` directory, structured by experiment name. You can analyze these results using the notebooks provided in `notebooks/`.

## 📄 Citation

If you use this code or findings in your research, please cite the thesis:

> **Portela, F.**. (2025). *Adaptive Negative Correlation Learning Via Online Hypergradient Descent*. Master's Thesis, NOVA Information Management School.

---
*For any questions regarding the implementation, please open an issue*