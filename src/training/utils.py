import logging
from sklearn.utils import compute_class_weight
import torch


def init_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger("logger")


def compute_balanced_class_w(labels: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        compute_class_weight(
            "balanced",
            classes=torch.unique(labels).cpu().numpy(),
            y=labels.cpu().numpy(),
        ),
        dtype=torch.float32,
    )
