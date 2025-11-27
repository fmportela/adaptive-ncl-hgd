import torch.nn as nn
from src.configs import LossConfig


def get_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "mse":
        return nn.MSELoss()
    elif cfg.name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.name}")
