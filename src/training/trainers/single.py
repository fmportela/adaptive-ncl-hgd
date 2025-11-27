from typing import Tuple, Optional
import torch
import torch.nn as nn
from src.training.trainers.base import BaseEnsembleTrainer
from src.training.trainers.utils import _maybe_squeeze_for_regression


class SingleModelTrainer(BaseEnsembleTrainer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__([model])

    def _step_fn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> dict:
        xb, yb = batch
        (m,) = self.models
        out = m(xb)
        out = _maybe_squeeze_for_regression(out, yb, self.cfg.task_type)
        loss = self.criterion(out, yb)
        (opt,) = self.optimizers
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {"train_loss": float(loss.detach().item())}
