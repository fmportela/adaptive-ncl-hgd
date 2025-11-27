import torch.nn as nn
import torch.optim as optim
from src.configs import OptimConfig


def get_optimizer(cfg: OptimConfig, model: nn.Module) -> optim.Optimizer:
    if cfg.name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr)
    elif cfg.name == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr)
    elif cfg.name == "custom":
        # Just want to have access to the LR
        return optim.Optimizer(model.parameters(), {"lr": cfg.lr})
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.name}")
