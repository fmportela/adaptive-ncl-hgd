from typing import Tuple
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden: Tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
