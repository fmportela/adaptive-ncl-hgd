import copy
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from src.configs.config import FitConfig, OptimConfig, LossConfig
from src.training.trainers.single import SingleModelTrainer
from src.training.callbacks import Callback
from src.data.utils import make_bootstrap_ds


# TODO: Currently not able to use OptunaPruningCallback due to independent model training
class BaggingTrainer:
    def __init__(self, models: List[nn.Module]) -> None:
        self.models = [SingleModelTrainer(m) for m in models]
        self._is_compiled = False

    def compile(
        self,
        optim_cfg: OptimConfig,
        loss_cfg: LossConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        for m in self.models:
            m.compile(optim_cfg, loss_cfg, device=device)
        self._is_compiled = True

    def fit(
        self,
        cfg: FitConfig,
        train_ds: TensorDataset,
        val_ds: Optional[TensorDataset] = None,
        *,
        callbacks: Optional[List[Callback]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> List[dict]:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        self.cfg = cfg

        per_model_ds = [
            make_bootstrap_ds(train_ds, seed=cfg.seed + i)
            for i in range(len(self.models))
        ]

        self.histories = {}
        for i, (m, ds) in enumerate(zip(self.models, per_model_ds)):
            cbs = None
            if callbacks is not None:
                cbs = [copy.deepcopy(cb) for cb in callbacks]
            self.histories[f"model_{i}"] = m.fit(
                cfg, ds, val_ds, callbacks=cbs, device=device
            )

        return self.histories

    # TODO: add mode e.g. mean or majority_vote
    def predict(
        self,
        xb: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> torch.Tensor:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        return torch.mean(
            torch.stack([m.predict(xb, device=device) for m in self.models]),
            dim=0,
        )

    def state_dict(self) -> dict:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        return {
            "models": [m.state_dict() for m in self.models],
            "cfg": self.cfg,
            "histories": self.histories,
        }

    def load_state_dict(self, state: dict) -> None:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        for m, sd in zip(self.models, state["models"]):
            m.load_state_dict(sd)
        self.cfg = state["cfg"]
        self.histories = state.get("histories", None)
