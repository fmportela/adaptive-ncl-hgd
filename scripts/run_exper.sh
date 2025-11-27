#!/bin/bash

configs=(
  "experiments/exper1/classification/churn/single_holdout_tune.yaml"
  "experiments/exper1/classification/churn/bagging_holdout_tune.yaml"
  "experiments/exper1/classification/churn/ncl_holdout_tune.yaml"
  "experiments/exper1/classification/churn/adapt_ncl_holdout_tune.yaml"
  "experiments/exper1/classification/churn/multi_adapt_ncl_holdout_tune.yaml"
  "experiments/exper1/classification/waveform_21/single_holdout_tune.yaml"
  "experiments/exper1/classification/waveform_21/bagging_holdout_tune.yaml"
  "experiments/exper1/classification/waveform_21/ncl_holdout_tune.yaml"
  "experiments/exper1/classification/waveform_21/adapt_ncl_holdout_tune.yaml"
  "experiments/exper1/classification/waveform_21/multi_adapt_ncl_holdout_tune.yaml"
  "experiments/exper1/regression/529_pollen/single_holdout_tune.yaml"
  "experiments/exper1/regression/529_pollen/bagging_holdout_tune.yaml"
  "experiments/exper1/regression/529_pollen/ncl_holdout_tune.yaml"
  "experiments/exper1/regression/529_pollen/adapt_ncl_holdout_tune.yaml"
  "experiments/exper1/regression/529_pollen/multi_adapt_ncl_holdout_tune.yaml"
  "experiments/exper2/classification/churn/adapt_ncl_sweep.yaml"
  "experiments/exper2/classification/churn/multi_adapt_ncl_sweep.yaml"
  "experiments/exper2/classification/waveform_21/adapt_ncl_sweep.yaml"
  "experiments/exper2/classification/waveform_21/multi_adapt_ncl_sweep.yaml"
  "experiments/exper2/regression/529_pollen/adapt_ncl_sweep.yaml"
  "experiments/exper2/regression/529_pollen/multi_adapt_ncl_sweep.yaml"
)

for config in "${configs[@]}"; do
  echo "Running experiment: $config"
  uv run src/experiments/run_experiment.py -c "$config" --overwrite
  echo "------------------------------------"
done
