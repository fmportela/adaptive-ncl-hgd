import time
import optuna


class Callback:
    def on_train_start(self, trainer, **kwargs): ...
    def on_epoch_start(self, trainer, epoch: int, **kwargs): ...
    def on_batch_end(self, trainer, logs: dict, **kwargs): ...
    def on_validation_end(self, trainer, logs: dict, **kwargs): ...
    def on_epoch_end(self, trainer, epoch: int, logs: dict, **kwargs): ...
    def on_train_end(self, trainer, **kwargs): ...


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 1e-3,
        restore_best: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = -float("inf") if mode == "max" else float("inf")
        self.wait = 0
        self.best_state = None

    def on_validation_end(self, trainer, logs, **kwargs):
        metric = logs.get(self.monitor)
        if metric is None:
            return
        better = (
            metric > self.best + 1e-12
            if self.mode == "max"
            else metric < self.best - 1e-12
        )
        if better:
            self.best = metric
            self.wait = 0
            if self.restore_best:
                self.best_state = trainer.state_dict()
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                trainer._stop_training = True

    def on_train_end(self, trainer, **kwargs):
        if self.restore_best and self.best_state is not None:
            trainer.load_state_dict(self.best_state)
            trainer.logger.info(
                f"Restored best model with {self.monitor}={self.best:.4f}"
            )


class ProgressBarCallback(Callback):
    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        self._t0 = time.time()
        trainer.logger.info(f"Epoch {epoch}/{trainer.cfg.n_epochs} started")

    def on_epoch_end(self, trainer, epoch: int, logs: dict, **kwargs):
        dt = time.time() - getattr(self, "_t0", time.time())
        msg = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        trainer.logger.info(f"Epoch {epoch} done in {dt:.1f}s | {msg}")


class TimerCallback(Callback):
    def on_train_start(self, trainer, **kwargs):
        self._t0 = time.time()

    def on_train_end(self, trainer, **kwargs):
        dt = time.time() - getattr(self, "_t0", time.time())
        trainer.logger.info(f"Training finished in {dt:.1f}s")


class OptunaPruningCallback:
    """
    Reports per-epoch validation metric to Optuna and prunes the trial if underperforming.
    """

    def __init__(self, trial: optuna.Trial, metric_fn=None):
        self.trial = trial
        self.metric_fn = metric_fn

    def on_epoch_end(self, trainer, epoch: int, logs: dict, **kwargs):
        ds = kwargs.get("ds", None)

        if (ds is None) or (self.trial is None) or (self.metric_fn is None):
            return

        value = self.metric_fn(
            trainer.predict(ds.tensors[0], aggregate="mean"), ds.tensors[1]
        )

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
