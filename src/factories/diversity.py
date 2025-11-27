from typing import Callable
from src.configs import DiversityMetricConfig
from src.metrics.diversity import (
    disagreement_rate,
    mean_pairwise_js,
    pairwise_pearson_correlation,
    mean_variance_around_ensemble,
)


def get_diversity_fn(cfg: DiversityMetricConfig) -> Callable:
    if cfg.name == "disagreement":
        return disagreement_rate
    elif cfg.name == "mean_pairwise_js_divergence":
        return mean_pairwise_js
    elif cfg.name == "pairwise_pearson_correlation":
        return pairwise_pearson_correlation
    elif cfg.name == "mean_variance_around_ensemble":
        return mean_variance_around_ensemble
    else:
        raise ValueError(f"Unsupported metric: {cfg.name}")
