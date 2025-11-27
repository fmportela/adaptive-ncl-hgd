import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


# *************************
# Classification
# *************************


def compute_cross_entropy(
    outputs: torch.Tensor, targets: torch.Tensor
) -> float:
    return float(F.cross_entropy(outputs, targets).item())


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    outputs = torch.argmax(outputs, dim=1)
    return float(
        accuracy_score(targets.detach().numpy(), outputs.detach().numpy())
    )


def compute_balanced_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor
) -> float:
    outputs = torch.argmax(outputs, dim=1)
    return float(
        balanced_accuracy_score(
            targets.detach().numpy(), outputs.detach().numpy()
        )
    )


def compute_roc_auc(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    outputs = outputs[:, 1]  # Probability of positive class
    return float(
        roc_auc_score(targets.detach().numpy(), outputs.detach().numpy())
    )


def compute_binary_f1_score(
    outputs: torch.Tensor, targets: torch.Tensor, average: str = "binary"
) -> float:
    outputs = torch.argmax(outputs, dim=1)
    return float(
        f1_score(
            targets.detach().numpy(),
            outputs.detach().numpy(),
            average="binary",
        )
    )


def compute_macro_f1_score(
    outputs: torch.Tensor, targets: torch.Tensor, average: str = "binary"
) -> float:
    outputs = torch.argmax(outputs, dim=1)
    return float(
        f1_score(
            targets.detach().numpy(), outputs.detach().numpy(), average="macro"
        )
    )


# *************************
# Regression
# *************************


def compute_mse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if outputs.dim() == 2:
        outputs = outputs.squeeze(-1)
    return float(F.mse_loss(outputs, targets).item())


def compute_rmse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if outputs.dim() == 2:
        outputs = outputs.squeeze(-1)
    return float(torch.sqrt(F.mse_loss(outputs, targets)).item())


def compute_mae(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if outputs.dim() == 2:
        outputs = outputs.squeeze(-1)
    return float(F.l1_loss(outputs, targets).item())
