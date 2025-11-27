import argparse
import os
import shutil
import time
import copy
import itertools
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from optuna.importance import get_param_importances

from src.utils import load_yaml
from src.factories.data import get_pmlb_dataset, get_scaler
from src.factories.training import get_trainer
from src.factories.reporting import get_reporting_fn
from src.factories.diversity import get_diversity_fn
from src.factories.callback import get_callback
from src.data.utils import convert_arrays_to_dataset
from src.configs import (
    FitConfig,
    LossConfig,
    OptimConfig,
    ModelConfig,
    EnsembleConfig,
    TrainerConfig,
    DatasetConfig,
    ScalerConfig,
    ReportingMetricConfig,
    DiversityMetricConfig,
    CallbackConfig,
)
from src.training.io import save_dict_to_disk, save_state_dict_to_disk
from src.training.tuning.tune import optimize_blackbox_cv
from src.utils import set_seed


def run_experiment(args: argparse.Namespace) -> None:
    # ********************************
    # Get experiment configs
    # ********************************

    exp_config = load_yaml(args.config)

    # ********************************
    # Prepare experiment directory
    # ********************************

    exp_name = args.config.split("/")[-1].replace(".yaml", "")
    exp_dir = Path(exp_config["run_dir"]) / exp_name

    if exp_dir.exists():
        if args.overwrite:
            shutil.rmtree(exp_dir)
            os.makedirs(exp_dir)
        else:
            raise FileExistsError(
                f"Experiment directory {exp_dir} already exists. Use --overwrite to overwrite."
            )
    else:
        os.makedirs(exp_dir)

    # ********************************
    # Get data
    # ********************************

    dataset_cfg = DatasetConfig(
        name=exp_config["dataset"]["name"],
        task_type=exp_config["dataset"]["task_type"],
        local_cache_dir=exp_config["dataset"]["local_cache_dir"],
    )

    X, y = get_pmlb_dataset(dataset_cfg)

    if exp_config["dataset"]["task_type"] == "classification":
        # # Remap arbitrary class labels (e.g., {1,2,7}) into a contiguous range 0..n_classes-1
        _, y = np.unique(y, return_inverse=True)

    if exp_config["dataset"]["name"] == "optdigits":
        y = np.clip(y - 1, 0, 5)

    # ********************************
    # Training
    # ********************************

    if exp_config["seed"] is not None:
        set_seed(exp_config["seed"])
    else:
        # set random seed
        exp_config["seed"] = random.randint(0, 2**32 - 1)
        set_seed(exp_config["seed"])
        print(f"Random seed set to {exp_config['seed']}")

    if exp_config["mode"] == "holdout":
        run_holdout(X, y, exp_config, exp_dir)
    elif exp_config["mode"] == "cv":
        run_cv(X, y, exp_config, exp_dir)
    elif exp_config["mode"] == "sweep":
        run_sweep(X, y, exp_config, exp_dir)
    else:
        raise ValueError(f"Unsupported training mode: {exp_config['mode']}")


def run_holdout(X, y, config, exp_dir):
    if (
        config["modes"]["holdout"]["tune"]["enabled"]
        and config["modes"]["holdout"]["ratios"]["val"] > 0.0
    ):
        raise ValueError(
            "Validation set should not be pre-defined when hyperparameter tuning is enabled."
        )

    if (
        not config["modes"]["holdout"]["tune"]["enabled"]
        and config["modes"]["holdout"]["ratios"]["val"] == 0.0
    ):
        raise ValueError(
            "Validation set should be pre-defined when hyperparameter tuning is disabled."
        )

    if (
        config["modes"]["holdout"]["tune"]["enabled"]
        and config["modes"]["holdout"]["tune"]["refit_val_ratio"] == 0.0
    ):
        raise ValueError(
            "refit_val_ratio must be > 0.0 when hyperparameter tuning is enabled."
        )

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    ratios = config["modes"]["holdout"]["ratios"]
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X,
        y,
        test_size=ratios["test"],
        random_state=config["seed"],
        stratify=(y if config["modes"]["holdout"]["stratify"] else None),
    )

    if ratios["val"] == 0.0:
        X_train, y_train = X_full_train, y_full_train
        X_val, y_val = None, None
    else:
        val_size = ratios["val"] / (1.0 - ratios["test"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_full_train,
            y_full_train,
            test_size=val_size,
            random_state=config["seed"],
            stratify=(
                y_full_train
                if config["modes"]["holdout"]["stratify"]
                else None
            ),
        )

    # Configs
    fit_cfg = FitConfig(
        task_type=config["dataset"]["task_type"],
        n_epochs=config["trainer"]["fit"]["n_epochs"],
        batch_size=config["trainer"]["fit"]["batch_size"],
        val_batch_size=config["trainer"]["fit"]["val_batch_size"],
        normalize=config["trainer"]["fit"]["normalize"],
        class_w=config["trainer"]["fit"]["class_w"],
        log_level=config["trainer"]["fit"]["log_level"],
    )
    optim_cfg = OptimConfig(
        name=config["trainer"]["optim"]["name"],
        lr=config["trainer"]["optim"]["lr"],
    )
    loss_cfg = LossConfig(name=config["trainer"]["loss"]["name"])
    model_cfg = ModelConfig(
        name=config["trainer"]["model"]["name"],
        params=config["trainer"]["model"]["params"],
    )
    ensemble_cfg = EnsembleConfig(
        model_cfg=model_cfg,
        n_models=config["trainer"]["ensemble"]["n_models"],
    )
    trainer_cfg = TrainerConfig(
        name=config["trainer"]["name"],
        ens_cfg=ensemble_cfg,
        params=config["trainer"].get("params", {}),
    )

    start_time = time.perf_counter()
    if config["modes"]["holdout"]["tune"]["enabled"]:
        # HPO
        study, best_params, _ = optimize_blackbox_cv(
            X=X_train,
            y=y_train,
            param_space=config["param_grid"],
            task_type=config["dataset"]["task_type"],
            direction=config["modes"]["holdout"]["tune"]["direction"],
            n_trials=config["modes"]["holdout"]["tune"]["n_trials"],
            timeout=config["modes"]["holdout"]["tune"]["timeout"],
            n_startup_trials=config["modes"]["holdout"]["tune"][
                "n_startup_trials"
            ],
            seed=config["seed"],
            prune=config["modes"]["holdout"]["tune"]["prune"],
            base_trainer_cfg=trainer_cfg,
            base_fit_cfg=fit_cfg,
            base_optim_cfg=optim_cfg,
            base_loss_cfg=loss_cfg,
            base_model_cfg=model_cfg,
            base_ensemble_cfg=ensemble_cfg,
            metric_cfg=ReportingMetricConfig(
                name=config["modes"]["holdout"]["tune"]["metric"]
            ),
            scaler_cfg=ScalerConfig(
                name=config["preprocessing"]["scaler"]["name"]
            ),
            cv=config["modes"]["holdout"]["tune"]["cv"],
            stratify=config["modes"]["holdout"]["stratify"],
            shuffle_cv=True,
            device=torch.device("cpu"),
        )

        # apply tuned params
        fit_cfg = fit_cfg.model_copy(
            update={
                k.split("fit__")[1]: v
                for k, v in best_params.items()
                if k.startswith("fit__")
            }
        )
        optim_cfg = optim_cfg.model_copy(
            update={
                k.split("optim__")[1]: v
                for k, v in best_params.items()
                if k.startswith("optim__")
            }
        )
        model_cfg = model_cfg.model_copy(
            update={
                k.split("model__")[1]: v
                for k, v in best_params.items()
                if k.startswith("model__")
            }
        )
        ensemble_cfg = ensemble_cfg.model_copy(
            update={
                k.split("ens__")[1]: v
                for k, v in best_params.items()
                if k.startswith("ens__")
            }
        ).model_copy(update={"model_cfg": model_cfg})
        trainer_cfg = trainer_cfg.model_copy(
            update={
                "ens_cfg": ensemble_cfg,
                "params": {
                    k: v for k, v in best_params.items() if "__" not in k
                },
            }
        )

        # internal validation split for early stopping
        refit_val_ratio = config["modes"]["holdout"]["tune"]["refit_val_ratio"]
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=refit_val_ratio,
                random_state=config["seed"],
                stratify=(
                    y_train if config["modes"]["holdout"]["stratify"] else None
                ),
            )

    # Scale
    scaler = get_scaler(
        ScalerConfig(name=config["preprocessing"]["scaler"]["name"])
    )
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        y_scaler = None
        if config["dataset"]["task_type"] == "regression":
            y_scaler = get_scaler(
                ScalerConfig(name=config["preprocessing"]["scaler"]["name"])
            )
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
            # y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    train_ds = convert_arrays_to_dataset(X_train, y_train)
    val_ds = convert_arrays_to_dataset(X_val, y_val)
    test_ds = convert_arrays_to_dataset(X_test, y_test)

    # Fit
    trainer = get_trainer(trainer_cfg)
    trainer.compile(optim_cfg, loss_cfg)
    history = trainer.fit(fit_cfg, train_ds, val_ds)

    # Evaluate
    preds = trainer.predict(test_ds.tensors[0])
    if config["dataset"]["task_type"] == "regression" and y_scaler is not None:
        preds = torch.tensor(
            y_scaler.inverse_transform(
                preds.detach().numpy().reshape(-1, 1)
            ).ravel(),
            dtype=torch.float32,
        )
        y_true = test_ds.tensors[1]
    else:
        y_true = test_ds.tensors[1]

    metrics = config["modes"]["holdout"]["metrics"]
    if isinstance(metrics, str):
        metrics = [metrics]

    results = {
        m: float(
            get_reporting_fn(ReportingMetricConfig(name=m))(preds, y_true)
        )
        for m in metrics
    }
    results["time_in_seconds"] = time.perf_counter() - start_time
    if config["modes"]["holdout"]["tune"]["enabled"]:
        results["best_hyperparameters"] = best_params
        results["hyperparameter_importances"] = get_param_importances(study)

    print("\nHoldout test results:")
    for m, v in results.items():
        if isinstance(v, dict):
            print(f"\n  {m}:")
            for sub_m, sub_v in v.items():
                print(
                    f"    {sub_m}: {sub_v:.6f}"
                    if isinstance(sub_v, (float, int))
                    else f"    {sub_m}: {sub_v}"
                )
        else:
            print(
                f"  {m}: {v:.6f}"
                if isinstance(v, (float, int))
                else f"  {m}: {v}"
            )

    # Save results
    save_state_dict_to_disk(trainer.state_dict(), exp_dir / "model.pt")
    save_dict_to_disk(history, exp_dir / "history.json")
    save_dict_to_disk(results, exp_dir / "results.json")

    print(f"\nHoldout experiment completed. Results saved to {exp_dir}")


def run_sweep(X, y, config, exp_dir):
    # Basic setup
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # sweep settings
    sweep_cfg = config["modes"]["sweep"]
    ratios = sweep_cfg["ratios"]

    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X,
        y,
        test_size=ratios["test"],
        random_state=config["seed"],
        stratify=(y if sweep_cfg["stratify"] else None),
    )

    if ratios["val"] == 0.0:
        X_train, y_train = X_full_train, y_full_train
        X_val, y_val = None, None
    else:
        val_size = ratios["val"] / (1.0 - ratios["test"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_full_train,
            y_full_train,
            test_size=val_size,
            random_state=config["seed"],
            stratify=(y_full_train if sweep_cfg["stratify"] else None),
        )

    # base configs
    fit_cfg = FitConfig(
        task_type=config["dataset"]["task_type"],
        n_epochs=config["trainer"]["fit"]["n_epochs"],
        batch_size=config["trainer"]["fit"]["batch_size"],
        val_batch_size=config["trainer"]["fit"]["val_batch_size"],
        normalize=config["trainer"]["fit"]["normalize"],
        class_w=config["trainer"]["fit"]["class_w"],
        log_level=config["trainer"]["fit"]["log_level"],
    )
    optim_cfg = OptimConfig(
        name=config["trainer"]["optim"]["name"],
        lr=config["trainer"]["optim"]["lr"],
    )
    loss_cfg = LossConfig(name=config["trainer"]["loss"]["name"])
    model_cfg = ModelConfig(
        name=config["trainer"]["model"]["name"],
        params=config["trainer"]["model"]["params"],
    )
    ensemble_cfg = EnsembleConfig(
        model_cfg=model_cfg,
        n_models=config["trainer"]["ensemble"]["n_models"],
    )
    trainer_cfg = TrainerConfig(
        name=config["trainer"]["name"],
        ens_cfg=ensemble_cfg,
        params=config["trainer"].get("params", {}),
    )

    sweep_params = sweep_cfg.get("params", {})
    sweep_items = sorted(sweep_params.items(), key=lambda kv: kv[0])
    keys = [k for k, _ in sweep_items]
    values_lists = [
        v if isinstance(v, (list, tuple)) else [v] for _, v in sweep_items
    ]
    combos = list(itertools.product(*values_lists)) if values_lists else [()]

    summary = {}
    start_all = time.perf_counter()

    for combo_idx, combo in enumerate(combos, start=1):
        # run name
        if combo:
            combo_pairs = [f"{k}={v}" for k, v in zip(keys, combo)]
            run_name = "sweep_" + "__".join(combo_pairs)
        else:
            run_name = "sweep_default"

        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        fit_cfg_run = copy.deepcopy(fit_cfg)
        optim_cfg_run = copy.deepcopy(optim_cfg)
        loss_cfg_run = copy.deepcopy(loss_cfg)
        # model_cfg_run = copy.deepcopy(model_cfg)
        # ensemble_cfg_run = copy.deepcopy(ensemble_cfg)
        trainer_cfg_run = copy.deepcopy(trainer_cfg)

        # Update trainer params with sweep combo (only for trainer params)
        # if combo:
        #     updated_params = {**trainer_cfg_run.params}
        #     for k, v in zip(keys, combo):
        #         updated_params[k] = v
        #     trainer_cfg_run = trainer_cfg_run.model_copy(
        #         update={"params": updated_params}
        #     )

        if combo:
            updated_trainer_params = dict(trainer_cfg_run.params or {})
            updated_ensemble_params = copy.deepcopy(trainer_cfg_run.ens_cfg)

            # TODO: currently only supports ensemble and trainer params
            for k, v in zip(keys, combo):
                if k.startswith("ens__"):
                    field = k.split("__", 1)[1]  # e.g. "n_models"
                    updated_ensemble_params.__dict__[field] = v
                else:
                    updated_trainer_params[k] = v

        # reassemble trainer cfg
        trainer_cfg_run = trainer_cfg_run.model_copy(
            update={
                "params": updated_trainer_params,
                "ens_cfg": updated_ensemble_params,
            }
        )

        # Scale
        scaler = get_scaler(
            ScalerConfig(name=config["preprocessing"]["scaler"]["name"])
        )
        X_train_run, X_val_run, X_test_run = X_train, X_val, X_test
        y_train_run, y_val_run, y_test_run = y_train, y_val, y_test
        y_scaler = None
        if scaler is not None:
            X_train_run = scaler.fit_transform(X_train_run)
            X_val_run = (
                scaler.transform(X_val_run) if X_val_run is not None else None
            )
            X_test_run = scaler.transform(X_test_run)

            if config["dataset"]["task_type"] == "regression":
                y_scaler = get_scaler(
                    ScalerConfig(
                        name=config["preprocessing"]["scaler"]["name"]
                    )
                )
                y_train_run = y_scaler.fit_transform(
                    y_train_run.reshape(-1, 1)
                ).ravel()
                if y_val_run is not None:
                    y_val_run = y_scaler.transform(
                        y_val_run.reshape(-1, 1)
                    ).ravel()

        # Datasets
        train_ds = convert_arrays_to_dataset(X_train_run, y_train_run)
        val_ds = convert_arrays_to_dataset(X_val_run, y_val_run)
        test_ds = convert_arrays_to_dataset(X_test_run, y_test_run)

        # Train
        trainer = get_trainer(trainer_cfg_run)
        trainer.compile(optim_cfg_run, loss_cfg_run)

        start_time = time.perf_counter()

        callbacks = []
        for cb_conf in config["trainer"].get("callbacks", []):
            callbacks.append(
                get_callback(
                    CallbackConfig(
                        name=cb_conf["name"], params=cb_conf.get("params", {})
                    )
                )
            )

        history = trainer.fit(
            fit_cfg_run, train_ds, val_ds, callbacks=callbacks
        )

        # Predict on test
        preds = trainer.predict(test_ds.tensors[0])
        if (
            config["dataset"]["task_type"] == "regression"
            and y_scaler is not None
        ):
            preds = torch.tensor(
                y_scaler.inverse_transform(
                    preds.detach().numpy().reshape(-1, 1)
                ).ravel(),
                dtype=torch.float32,
            )
            y_true = test_ds.tensors[1]
        else:
            y_true = test_ds.tensors[1]

        # Evaluate
        metrics = sweep_cfg["metrics"]
        if isinstance(metrics, str):
            metrics = [metrics]

        results = {
            m: float(
                get_reporting_fn(ReportingMetricConfig(name=m))(preds, y_true)
            )
            for m in metrics
        }

        # Diversity - Only for ensembles
        diversity_metrics = sweep_cfg.get("diversity_metrics", [])

        if diversity_metrics:
            if isinstance(diversity_metrics, str):
                diversity_metrics = [diversity_metrics]

            for dm in diversity_metrics:
                outs = [m(test_ds.tensors[0]) for m in trainer.models]
                if config["dataset"]["task_type"] == "classification":
                    outs = [
                        torch.softmax(o, dim=1)
                        if isinstance(o, torch.Tensor)
                        else torch.softmax(o[0], dim=1)
                        for o in outs
                    ]  # TODO this is a batch for bagging
                    results[dm] = float(
                        get_diversity_fn(DiversityMetricConfig(name=dm))(outs)
                    )
                elif config["dataset"]["task_type"] == "regression":
                    outs = [
                        o if isinstance(o, torch.Tensor) else o[0]
                        for o in outs
                    ]  # Use raw outputs for regression
                    results[dm] = float(
                        get_diversity_fn(DiversityMetricConfig(name=dm))(outs)
                    )

        results["time_in_seconds"] = time.perf_counter() - start_time
        results["params"] = (
            {k: v for k, v in zip(keys, combo)} if combo else {}
        )

        # Print run results
        print(f"\n[{combo_idx}/{len(combos)}] Sweep run: {run_name}")
        for m, v in results.items():
            if isinstance(v, dict):
                print(f"\n  {m}:")
                for sub_m, sub_v in v.items():
                    print(
                        f"    {sub_m}: {sub_v:.6f}"
                        if isinstance(sub_v, (float, int))
                        else f"    {sub_m}: {sub_v}"
                    )
                print()
            else:
                print(
                    f"  {m}: {v:.6f}"
                    if isinstance(v, (float, int))
                    else f"  {m}: {v}"
                )

        # Persist artifacts
        save_state_dict_to_disk(trainer.state_dict(), run_dir / "model.pt")
        save_dict_to_disk(history, run_dir / "history.json")
        save_dict_to_disk(results, run_dir / "results.json")

        # Aggregate summary
        summary[run_name] = results

    # Save overall summary
    summary["_sweep_time_seconds"] = time.perf_counter() - start_all
    save_dict_to_disk(summary, exp_dir / "sweep_summary.json")

    print(
        f"\nSweep completed over {len(combos)} setting(s). Summary saved to {exp_dir}"
    )


def run_cv(X, y, config, exp_dir):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if config["modes"]["cv"]["stratify"]:
        splitter = StratifiedKFold(
            n_splits=config["modes"]["cv"]["n_splits"],
            shuffle=True,
            random_state=config["seed"],
        )
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(
            n_splits=config["modes"]["cv"]["n_splits"],
            shuffle=True,
            random_state=config["seed"],
        )
        split_iter = splitter.split(X)

    metrics_cfg_val = config["modes"]["cv"]["metrics"]
    if isinstance(metrics_cfg_val, str):
        metrics_cfg_val = [metrics_cfg_val]

    per_metric_values = defaultdict(list)
    fold_results = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(split_iter, start=1):
        start_time = time.perf_counter()
        fold_dir = exp_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = config["seed"] + fold_idx

        # Outer split
        X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Inner split
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=config["modes"]["cv"]["inner_val_ratio"],
            random_state=fold_seed,
            stratify=(
                y_trainval if config["modes"]["cv"]["stratify"] else None
            ),
        )

        # Preprocess
        scaler = get_scaler(
            ScalerConfig(name=config["preprocessing"]["scaler"]["name"])
        )
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            if config["dataset"]["task_type"] == "regression":
                y_scaler = get_scaler(
                    ScalerConfig(
                        name=config["preprocessing"]["scaler"]["name"]
                    )
                )
                y_train = y_scaler.fit_transform(
                    y_train.reshape(-1, 1)
                ).ravel()
                y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
                # y_test  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        # Datasets
        train_ds = convert_arrays_to_dataset(X_train, y_train)
        val_ds = convert_arrays_to_dataset(X_val, y_val)
        test_ds = convert_arrays_to_dataset(X_test, y_test)

        # Normal Training
        trainer = get_trainer(
            TrainerConfig(
                name=config["trainer"]["name"],
                ens_cfg=EnsembleConfig(
                    model_cfg=ModelConfig(
                        name=config["trainer"]["model"]["name"],
                        params=config["trainer"]["model"]["params"],
                    ),
                    n_models=config["trainer"]["ensemble"]["n_models"],
                ),
                params=config["trainer"]["params"]
                if config["trainer"]["params"] is not None
                else {},
            )
        )

        trainer.compile(
            OptimConfig(
                name=config["trainer"]["optim"]["name"],
                lr=config["trainer"]["optim"]["lr"],
            ),
            LossConfig(name=config["trainer"]["loss"]["name"]),
        )
        history = trainer.fit(
            FitConfig(**config["trainer"]["fit"]), train_ds, val_ds
        )

        # Evaluate on outer test fold
        preds = trainer.predict(test_ds.tensors[0])

        if config["dataset"]["task_type"] == "regression":
            # Rescale predictions
            preds = torch.tensor(
                y_scaler.inverse_transform(
                    preds.detach().numpy().reshape(-1, 1)
                ).ravel(),
                dtype=torch.float32,
            )
            # y_true = torch.tensor(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), dtype=torch.float32)
            y_true = test_ds.tensors[1]
        else:
            y_true = test_ds.tensors[1]

        fold_out = {"fold": fold_idx}
        for metric in metrics_cfg_val:
            value = get_reporting_fn(ReportingMetricConfig(name=metric))(
                preds, y_true
            )
            fold_out[metric] = value
            per_metric_values[metric].append(float(value))

        fold_results.append(fold_out)
        fold_out["time_in_seconds"] = time.perf_counter() - start_time
        fold_out["fold_seed"] = fold_seed
        print(f"[Fold {fold_idx}/{splitter.get_n_splits()}] completed.")

        # Save fold artifacts
        save_state_dict_to_disk(trainer.state_dict(), fold_dir / "model.pt")
        save_dict_to_disk(history, fold_dir / "history.json")
        save_dict_to_disk(fold_out, fold_dir / "results.json")

    # Aggregate CV metrics
    summary = {"n_splits": splitter.get_n_splits(), "seed": config["seed"]}
    for metric, vals in per_metric_values.items():
        arr = np.array(vals, dtype=float)
        summary[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "per_fold": [float(v) for v in arr],
        }

    save_dict_to_disk(
        {"folds": fold_results, "summary": summary},
        exp_dir / "cv_results.json",
    )

    print("Cross-validation completed.")
    print("CV summary:")
    for metric in metrics_cfg_val:
        m, s = summary[metric]["mean"], summary[metric]["std"]
        print(f"  {metric}: {m:.6f} +/- {s:.6f}")
    print(f"Artifacts saved under {exp_dir}")


if __name__ == "__main__":
    from src.experiments.cli import build_parser

    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)
