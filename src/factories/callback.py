from src.training.callbacks import (
    EarlyStoppingCallback,
    ProgressBarCallback,
    TimerCallback,
)
from src.configs import CallbackConfig


def get_callback(cfg: CallbackConfig):
    if cfg.name == "early_stopping":
        return EarlyStoppingCallback(**cfg.params)
    elif cfg.name == "progress_bar":
        return ProgressBarCallback(**cfg.params)
    elif cfg.name == "timer":
        return TimerCallback(**cfg.params)
    else:
        raise ValueError(f"Unsupported callback type: {cfg.name}")
