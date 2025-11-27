from typing import Callable, List
import torch.nn as nn


def make_model_fns(
    model_fn: Callable[[], nn.Module], n: int
) -> List[Callable[[], nn.Module]]:
    return [model_fn for _ in range(n)]
