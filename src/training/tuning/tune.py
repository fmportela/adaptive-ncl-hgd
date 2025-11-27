from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import optuna
import torch
from src.configs.config import (
    FitConfig,
    OptimConfig,
    LossConfig,
    ModelConfig,
    EnsembleConfig,
    ScalerConfig,
    ReportingMetricConfig,
    TrainerConfig,
)
from src.factories.reporting import get_reporting_fn
from src.factories.data import get_scaler
from src.factories.training import get_trainer
from src.training.callbacks import EarlyStoppingCallback, OptunaPruningCallback
from src.utils import set_seed
from src.training.tuning.utils import (
    _suggest_params,
    _split_params,
)
from src.data.utils import convert_arrays_to_dataset


# TODO: config updates and scaling can be made DRYer with what's in run_experiment.py
def optimize_blackbox_cv(
    *,
    X: np.array,
    y: np.array,
    param_space: Dict[str, Dict[str, Any]],
    task_type: str,
    direction: str = "minimize",
    n_trials: int = 50,
    timeout: Optional[int] = None,
    n_startup_trials: int = 10,
    seed: int = 42,
    prune: bool = False,
    n_warmup_steps: int = 0,
    base_trainer_cfg: TrainerConfig,
    base_fit_cfg: FitConfig,
    base_optim_cfg: OptimConfig,
    base_loss_cfg: LossConfig,
    base_model_cfg: ModelConfig,
    base_ensemble_cfg: EnsembleConfig,
    metric_cfg: ReportingMetricConfig,
    scaler_cfg: Optional[ScalerConfig] = None,
    cv: Optional[int] = 5,
    stratify: bool = False,
    shuffle_cv: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tuple[optuna.Study, Dict[str, Any], float]:
    """Optimize hyperparameters using Optuna with k-fold cross-validation.

    This function performs hyperparameter optimization using Optuna's
    optimization framework. It applies k-fold cross-validation to evaluate
    the performance of hyperparameter configurations. The first fold is used
    for pruning to avoid issues with Optuna's pruning mechanism and to ensure
    reliable pruning decisions.

    Args:
        X: Feature matrix.
        y: Target vector.
        param_space: Hyperparameter search space.
        task_type: "regression" or "classification".
        direction: "minimize" or "maximize" the metric.
        n_trials: Number of optimization trials.
        timeout: Time limit for optimization in seconds.
        n_startup_trials: Number of initial random trials for TPE sampler.
        seed: Random seed for reproducibility.
        prune: Whether to use pruning.
        n_warmup_steps: Number of warmup steps before pruning starts.
        base_trainer_cfg: Base trainer configuration.
        base_fit_cfg: Base fit configuration.
        base_optim_cfg: Base optimizer configuration.
        base_loss_cfg: Base loss configuration.
        base_model_cfg: Base model configuration.
        base_ensemble_cfg: Base ensemble configuration.
        metric_cfg: Metric configuration for evaluation.
        scaler_cfg: Scaler configuration for data preprocessing.
        cv: Number of cross-validation folds.
        stratify: Whether to use stratified splits for classification.
        shuffle_cv: Whether to shuffle data before CV splitting.
        device: Device to run training on.

    Returns:
        A tuple containing the Optuna study object, the best hyperparameters,
        and the best metric value.

    Rationale for pruning the first fold:
        Optuna's pruning mechanism expects strictly increasing step values,
        but during k-fold cross-validation, each fold restarts epoch counting,
        which would otherwise cause duplicate step reports and unreliable
        pruning. Moreover, validation loss typically “jumps” between folds
        because each split has different data distributions — this can mislead
        the pruner into stopping good trials early. Restricting pruning to the
        first fold avoids both issues while still catching poor hyperparameter
        configurations early, since a setup that fails badly on one fold is
        very likely to perform poorly across the others as well.
    """
    metric_fn = get_reporting_fn(metric_cfg)

    if prune:
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
    )
    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner
    )

    def objective(trial: optuna.Trial) -> float:
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        # Suggest parameters
        params = _suggest_params(trial, param_space)
        (
            fit_updates,
            optim_updates,
            loss_updates,
            model_updates,
            ens_updates,
            trainer_updates,
        ) = _split_params(params)

        # Split into CV folds
        if stratify:
            splitter = StratifiedKFold(
                n_splits=cv,
                shuffle=shuffle_cv,
                random_state=trial_seed,
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=cv,
                shuffle=shuffle_cv,
                random_state=trial_seed,
            )
            split_iter = splitter.split(X)

        prune_cb = (
            OptunaPruningCallback(trial=trial, metric_fn=metric_fn)
            if prune
            else None
        )
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
            set_seed(trial_seed + fold_idx)
            y_scaler = None

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scaler = get_scaler(scaler_cfg)
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                if task_type == "regression":
                    y_scaler = get_scaler(scaler_cfg)
                    y_train = y_scaler.fit_transform(
                        y_train.reshape(-1, 1)
                    ).ravel()
                    y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

            fold_train_ds = convert_arrays_to_dataset(X_train, y_train)
            fold_val_ds = convert_arrays_to_dataset(X_val, y_val)

            # Update configs per fold
            trainer_cfg = base_trainer_cfg.model_copy(
                update={"params": trainer_updates}
            )
            fit_cfg = base_fit_cfg.model_copy(update=fit_updates)
            optim_cfg = base_optim_cfg.model_copy(update=optim_updates)
            loss_cfg = base_loss_cfg.model_copy(update=loss_updates)
            model_cfg = base_model_cfg.model_copy(update=model_updates)
            ens_cfg = base_ensemble_cfg.model_copy(update=ens_updates)
            ens_cfg.model_cfg = model_cfg
            trainer_cfg.ens_cfg = ens_cfg

            callbacks = [EarlyStoppingCallback()]
            if prune_cb and fold_idx == 1:
                callbacks.append(prune_cb)

            trainer = get_trainer(trainer_cfg)
            trainer.compile(optim_cfg, loss_cfg)
            trainer.fit(
                fit_cfg,
                fold_train_ds,
                fold_val_ds,
                callbacks=callbacks,
                device=device,
            )

            preds = trainer.predict(fold_val_ds.tensors[0], aggregate="mean")

            if task_type == "regression" and y_scaler is not None:
                # Rescale predictions for regression
                preds = torch.tensor(
                    y_scaler.inverse_transform(
                        preds.detach().numpy().reshape(-1, 1)
                    ).ravel(),
                    dtype=torch.float32,
                )
                y_true = torch.tensor(
                    y_scaler.inverse_transform(
                        fold_val_ds.tensors[1].detach().numpy().reshape(-1, 1)
                    ).ravel(),
                    dtype=torch.float32,
                )
            else:
                y_true = fold_val_ds.tensors[1]

            score = float(metric_fn(preds, y_true))
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    return study, study.best_params, study.best_value
