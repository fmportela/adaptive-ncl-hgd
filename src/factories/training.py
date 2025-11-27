from typing import List
import torch
import torch.nn as nn
from src.training.trainers import (
    SingleModelTrainer,
    BaggingTrainer,
    NCLTrainer,
    AdaptiveNCLTrainer,
    MultiAdaptiveNCLTrainer,
)
from src.configs import (
    ModelConfig,
    EnsembleConfig,
    TrainerConfig,
)
from src.models import MLP


def get_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name == "mlp":
        return MLP(**cfg.params)
    else:
        raise ValueError(f"Unsupported model type: {cfg.name}")


def get_ensemble(cfg: EnsembleConfig, base_seed: int = 42) -> List[nn.Module]:
    """Create ensemble with different initialization seeds for each model."""
    models = []
    for i in range(cfg.n_models):
        torch.manual_seed(base_seed + i)
        model = get_model(cfg.model_cfg)
        models.append(model)
    return models


def get_trainer(trainer_cfg: TrainerConfig):
    models = get_ensemble(trainer_cfg.ens_cfg)
    if trainer_cfg.name == "single":
        return SingleModelTrainer(models[0], **trainer_cfg.params)
    elif trainer_cfg.name == "bagging":
        return BaggingTrainer(models, **trainer_cfg.params)
    elif trainer_cfg.name == "ncl":
        return NCLTrainer(models, **trainer_cfg.params)
    elif trainer_cfg.name == "adaptive_ncl":
        return AdaptiveNCLTrainer(models, **trainer_cfg.params)
    elif trainer_cfg.name == "multi_adaptive_ncl":
        return MultiAdaptiveNCLTrainer(models, **trainer_cfg.params)
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_cfg}")


# def get_trainer_cls(trainer_cfg: TrainerConfig):
#     if trainer_cfg.name == "single":
#         return SingleModelTrainer
#     elif trainer_cfg.name == "bagging":
#         return BaggingTrainer
#     elif trainer_cfg.name == "ncl":
#         return NCLTrainer
#     elif trainer_cfg.name == "adaptive_ncl":
#         return AdaptiveNCLTrainer
#     elif trainer_cfg.name == "multi_adaptive_ncl":
#         return MultiAdaptiveNCLTrainer
#     else:
#         raise ValueError(f"Unsupported trainer type: {trainer_cfg}")


def get_trainer_cls(name: str):
    if name == "single":
        return SingleModelTrainer
    elif name == "bagging":
        return BaggingTrainer
    elif name == "ncl":
        return NCLTrainer
    elif name == "adaptive_ncl":
        return AdaptiveNCLTrainer
    elif name == "multi_adaptive_ncl":
        return MultiAdaptiveNCLTrainer
    else:
        raise ValueError(f"Unsupported trainer type: {name}")
