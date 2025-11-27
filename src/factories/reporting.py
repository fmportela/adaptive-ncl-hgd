from typing import Callable
from src.configs import ReportingMetricConfig
from src.metrics.reporting import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_binary_f1_score,
    compute_macro_f1_score,
    compute_roc_auc,
    compute_cross_entropy,
    compute_mse,
    compute_rmse,
    compute_mae,
)


def get_reporting_fn(cfg: ReportingMetricConfig) -> Callable:
    if cfg.name == "accuracy":
        return compute_accuracy
    elif cfg.name == "balanced_accuracy":
        return compute_balanced_accuracy
    elif cfg.name == "binary_f1_score":
        return compute_binary_f1_score
    elif cfg.name == "macro_f1_score":
        return compute_macro_f1_score
    elif cfg.name == "roc_auc":
        return compute_roc_auc
    elif cfg.name == "cross_entropy":
        return compute_cross_entropy
    elif cfg.name == "mse":
        return compute_mse
    elif cfg.name == "rmse":
        return compute_rmse
    elif cfg.name == "mae":
        return compute_mae
    else:
        raise ValueError(f"Unsupported metric: {cfg.name}")
