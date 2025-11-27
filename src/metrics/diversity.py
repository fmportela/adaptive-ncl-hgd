import math
from typing import List
import numpy as np
import torch


# ******************
# Classification
# ******************


def _js_pair(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=1)
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=1)
    js = 0.5 * (kl_pm + kl_qm)
    return js / math.log(2.0)


def mean_pairwise_js(probs: list[torch.Tensor], *args, **kwargs) -> float:
    vals = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            vals.append(_js_pair(probs[i], probs[j]).mean().item())
    return float(np.mean(vals)) if vals else float("nan")


def disagreement_rate(probs: List[torch.Tensor], *args, **kwargs) -> float:
    preds = [o.argmax(dim=1) for o in probs]
    dis = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            dis.append((preds[i] != preds[j]).float().mean().item())
    return float(np.mean(dis)) if dis else float("nan")


# ******************
# Regression
# ******************


def pairwise_pearson_correlation(
    outs: List[torch.Tensor], *args, **kwargs
) -> float:
    cors = []
    for i in range(len(outs)):
        xi = outs[i].double()
        xi = xi - xi.mean()
        xi_std = xi.std(unbiased=False)
        for j in range(i + 1, len(outs)):
            xj = outs[j].double()
            xj = xj - xj.mean()
            xj_std = xj.std(unbiased=False)
            denom = (xi_std * xj_std).item()
            if denom == 0:
                cors.append(float("nan"))
            else:
                num = (xi * xj).mean().item()
                cors.append(num / denom)
    cors = np.array(cors, dtype=float)
    return float(np.nanmean(cors)) if np.any(~np.isnan(cors)) else float(0.0)


def mean_variance_around_ensemble(
    outs: List[torch.Tensor], *args, **kwargs
) -> float:
    P = torch.stack([p.view(-1).double() for p in outs], dim=0)
    var_per_sample = P.var(dim=0, unbiased=False)
    return float(var_per_sample.mean().item())
