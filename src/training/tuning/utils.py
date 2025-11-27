import optuna
from typing import Any, Callable, Dict, List, Tuple, Type
import torch.nn as nn


def make_model_fns(
    model_fn: Callable[[], nn.Module], n: int
) -> List[Callable[[], nn.Module]]:
    return [model_fn for _ in range(n)]


def make_model_fns_builder(
    model_cls: Type[nn.Module],
    in_dim: int,
    out_dim: int,
    make_model_fns: Callable[
        [Callable[[], nn.Module], int], List[Callable[[], nn.Module]]
    ],
) -> Callable[[Dict[str, Any]], List[Callable[[], nn.Module]]]:
    def builder(model_params: dict) -> list:
        n_models = int(model_params.get("n_models", 1))
        hidden = tuple(model_params.get("hidden", (64,)))

        def model_fn() -> nn.Module:
            return model_cls(in_dim, out_dim=out_dim, hidden=hidden)

        return make_model_fns(model_fn, n_models)

    return builder


def _suggest_params(
    trial: optuna.Trial, param_space: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in param_space.items():
        t = spec.get("type", "float")
        if t == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step", None),
            )
        elif t == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown spec type for '{name}': {t}")
    return params


def _split_params(params: Dict[str, Any]) -> tuple:
    (
        cfg_updates,
        optim_updates,
        loss_updates,
        model_updates,
        ensemble_updates,
        trainer_updates,
    ) = {}, {}, {}, {}, {}, {}
    for k, v in params.items():
        if k.startswith("fit__"):
            cfg_updates[k[5:]] = v
        elif k.startswith("optim__"):
            optim_updates[k[7:]] = v
        elif k.startswith("loss__"):
            loss_updates[k[6:]] = v
        elif k.startswith("model__"):
            # because optuna does not accept tuples
            model_updates[k[7:]] = (
                _hidden_str_to_tuple(v) if k == "model__hidden" else v
            )
        elif k.startswith("ens__"):
            ensemble_updates[k[5:]] = v
        else:
            trainer_updates[k] = v
    return (
        cfg_updates,
        optim_updates,
        loss_updates,
        model_updates,
        ensemble_updates,
        trainer_updates,
    )


def _hidden_str_to_tuple(s: str) -> Tuple[int, ...]:
    """'128x64' -> (128, 64); '64' -> (64,)"""
    s = str(s).strip()
    if not s:
        return tuple()
    return tuple(int(p) for p in s.split("x"))
