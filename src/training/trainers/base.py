from itertools import cycle
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from src.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    ProgressBarCallback,
    TimerCallback,
)
from src.configs import FitConfig, OptimConfig, LossConfig
from src.factories.loss import get_loss
from src.factories.optim import get_optimizer
from src.metrics.diversity import (
    disagreement_rate,
    mean_pairwise_js,
    pairwise_pearson_correlation,
    mean_variance_around_ensemble,
)
from src.training.utils import init_logging, compute_balanced_class_w
from src.data.utils import convert_dataset_to_loader
from src.training.trainers.utils import _maybe_squeeze_for_regression


class BaseEnsembleTrainer:
    def __init__(
        self,
        models: List[nn.Module],
    ) -> None:
        self.models = models
        self.n = len(models)
        self._is_compiled = False
        self.global_step = 0

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        return [m(xb) for m in self.models]

    def _step_fn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def _callback(self, name: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, name, lambda *_a, **_k: None)(self, *args, **kwargs)

    def state_dict(self) -> Dict[str, object]:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        return {
            "cfg": self.cfg,
            "models": [m.state_dict() for m in self.models],
            # "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        for m, sd in zip(self.models, state["models"]):
            m.load_state_dict(sd)

        # for opt, sd in zip(self.optimizers, state["optimizers"]):
        #     opt.load_state_dict(sd)

    def eval_mode(self) -> None:
        for m in self.models:
            m.eval()

    def train_mode(self) -> None:
        for m in self.models:
            m.train()

    def compile(
        self,
        optim_cfg: OptimConfig,
        loss_cfg: LossConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.models = [m.to(device) for m in self.models]
        self.optimizers = [get_optimizer(optim_cfg, m) for m in self.models]
        self.criterion = get_loss(loss_cfg).to(device)
        self._is_compiled = True

    def fit(
        self,
        cfg: FitConfig,
        train_ds: TensorDataset,
        val_ds: Optional[TensorDataset] = None,
        *,
        callbacks: Optional[List[Callback]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, List[float]]:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        # torch.autograd.set_detect_anomaly(True)

        self.cfg = cfg
        self.logger = init_logging(self.cfg.log_level)

        default_cbs = [
            ProgressBarCallback(),
            EarlyStoppingCallback(),
            TimerCallback(),
        ]
        self.callbacks = callbacks or default_cbs

        class_w = self.cfg.class_w
        if class_w is not None and self.cfg.task_type == "classification":
            if isinstance(class_w, str):
                if class_w == "balanced":
                    class_w = compute_balanced_class_w(train_ds.tensors[1])
            elif isinstance(class_w, list):
                class_w = torch.tensor(class_w, dtype=torch.float32)
            else:
                raise ValueError("Invalid class_w type")

            self.criterion.weight = class_w.to(device)

        train_loader = convert_dataset_to_loader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            seed=self.cfg.seed,
        )

        val_loader = None
        if val_ds is not None:
            val_loader = convert_dataset_to_loader(
                val_ds,
                batch_size=self.cfg.val_batch_size,
                shuffle=False,
                seed=self.cfg.seed,
            )
            self._val_iter = cycle(val_loader)

        self.history: Dict[str, List[float]] = {}
        self._stop_training = False

        def _log_epoch(d: Dict[str, float]):
            for k, v in d.items():
                self.history.setdefault(k, []).append(v)

        self._callback("on_train_start")

        for epoch in range(1, self.cfg.n_epochs + 1):
            self._callback("on_epoch_start", epoch)
            self.train_mode()

            accum: Dict[str, float] = {}
            num_samples = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                val_batch = None
                if self._val_iter is not None:
                    vxb, vyb = next(self._val_iter)
                    val_batch = (vxb.to(device), vyb.to(device))

                logs = self._step_fn((xb, yb), val_batch)
                self.global_step += 1

                # Report metrics every step instead of only per epoch
                step_metrics = getattr(self.cfg, "step_metrics", None)
                if step_metrics is not None:
                    for k in step_metrics:
                        if k in logs and isinstance(logs[k], (int, float)):
                            key = f"{k}_step"  # to avoid colliding with epoch means
                            self.history.setdefault(key, []).append(
                                float(logs[k])
                            )

                bs = xb.size(0)
                num_samples += bs
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        accum[k] = accum.get(k, 0.0) + float(v) * bs
                    elif isinstance(v, list):
                        for i, vi in enumerate(v):
                            accum[f"{k}_{i}"] = (
                                accum.get(f"{k}_{i}", 0.0) + float(vi) * bs
                            )
                    else:
                        raise ValueError(
                            "Values to log must be int, float or list of these."
                        )

                self._callback("on_batch_end", logs)

            epoch_logs = {k: v / max(1, num_samples) for k, v in accum.items()}

            # Validation
            if val_loader is not None:
                self.eval_mode()
                val_accum: Dict[str, float] = {}
                val_count = 0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv, yv = xv.to(device), yv.to(device)
                        bs = xv.size(0)
                        logs = self._compute_val_logs(xv, yv)
                        for k, v in logs.items():
                            val_accum[k] = (
                                val_accum.get(k, 0.0) + float(v) * bs
                            )
                        val_count += bs
                for k in val_accum:
                    epoch_logs[k] = val_accum[k] / max(1, val_count)

                self._callback("on_validation_end", epoch_logs)

            _log_epoch(epoch_logs)
            self._callback("on_epoch_end", epoch, epoch_logs, ds=val_ds)

            if self._stop_training:
                break

        self._callback("on_train_end")
        return self.history

    def predict(
        self,
        xb: torch.Tensor,
        *,
        aggregate: str = "mean",
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        assert aggregate in {"mean", "none"}, (
            "aggregate must be 'mean' or 'none'"
        )

        xb = xb.to(device)
        self.eval_mode()
        outs = self(xb)

        return torch.stack(outs).mean(0) if aggregate == "mean" else outs

    # TODO: to avoid unncessary computations additional metrics should be optional and passed in the config
    def _compute_val_logs(
        self, xv: torch.Tensor, yv: torch.Tensor
    ) -> Dict[str, float]:
        if not self._is_compiled:
            raise RuntimeError("Call .compile(...) first")

        with torch.no_grad():
            outs = self(xv)
            out_mean = torch.stack(outs).mean(0)
            out_mean = _maybe_squeeze_for_regression(
                out_mean, yv, self.cfg.task_type
            )
            val_loss = self.criterion(out_mean, yv).item()
            logs: Dict[str, float] = {"val_loss": float(val_loss)}

            if self.cfg.task_type == "classification" and len(outs) > 1:
                probs = [torch.softmax(o, dim=1) for o in outs]
                # preds = [o.argmax(dim=1) for o in outs]
                logs["js_mean"] = mean_pairwise_js(probs)
                logs["disag"] = disagreement_rate(
                    probs
                )  # will be converted to preds inside the function
            elif self.cfg.task_type == "regression" and len(outs) > 1:
                outs = [
                    _maybe_squeeze_for_regression(
                        o, yv, self.cfg.task_type
                    ).view(-1)
                    for o in outs
                ]
                logs["corr"] = pairwise_pearson_correlation(outs)
                logs["var"] = mean_variance_around_ensemble(outs)

        return logs
