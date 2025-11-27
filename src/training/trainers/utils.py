from typing import List
import torch
import torch.nn as nn


def _maybe_squeeze_for_regression(
    out: torch.Tensor, target: torch.Tensor, task_type: str
) -> torch.Tensor:
    if task_type == "regression":
        if out.dim() == target.dim() + 1 and out.size(-1) == 1:
            out = out.squeeze(-1)
    return out


def params_per_model(model_list: List[nn.Module]):
    return [[p for p in m.parameters()] for m in model_list]


def zeros_like_params_list(params_list):
    return [[torch.zeros_like(p) for p in plist] for plist in params_list]


def grads_or_zeros(grad_list, params_list):
    out = []
    for g, p in zip(grad_list, params_list):
        out.append(torch.zeros_like(p) if g is None else g)
    return out


def dot_param_lists(a_list, b_list):
    return sum((a * b).sum() for a, b in zip(a_list, b_list))
