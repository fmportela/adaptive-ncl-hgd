from typing import Literal, Optional, Union, List
import logging
from pydantic import BaseModel, Field


class ReportingMetricConfig(BaseModel):
    name: Literal[
        # Classification
        "cross_entropy",
        "accuracy",
        "balanced_accuracy",
        "binary_f1_score",
        "macro_f1_score",
        "roc_auc",
        # Regression
        "rmse",
        "mse",
        "mae",
    ] = Field(default_factory=lambda: "balanced_accuracy")


class DiversityMetricConfig(BaseModel):
    name: Literal[
        # Classification
        "disagreement",
        "mean_pairwise_js_divergence",
        # Regression
        "pairwise_pearson_correlation",
        "mean_variance_around_ensemble",
    ] = Field(default_factory=lambda: "disagreement")


class DatasetConfig(BaseModel):
    name: Literal[
        # Classification
        "churn",
        "waveform_21",
        # Regression
        "529_pollen",
    ]
    task_type: Literal["classification", "regression"]
    local_cache_dir: Optional[str] = None


class ScalerConfig(BaseModel):
    name: Optional[Literal["standard", "minmax"]] = None


class ModelConfig(BaseModel):
    name: Literal["mlp"] = "mlp"
    params: Optional[dict] = None

    def model_post_init(self, __context):
        if self.params is None:
            self.params = {}


# TODO: add possibility of passing list of different model configs
class EnsembleConfig(BaseModel):
    model_cfg: ModelConfig
    n_models: int = 5


class TrainerConfig(BaseModel):
    name: Literal[
        "single", "bagging", "ncl", "adaptive_ncl", "multi_adaptive_ncl"
    ]
    ens_cfg: EnsembleConfig
    params: Optional[dict] = None

    def model_post_init(self, __context):
        if self.params is None:
            self.params = {}


class LossConfig(BaseModel):
    name: Literal["cross_entropy", "mse"] = "cross_entropy"


class OptimConfig(BaseModel):
    name: Literal["adamw", "sgd", "custom"] = "sgd"
    lr: float = 1e-3


class FitConfig(BaseModel):
    task_type: Literal["classification", "regression"] = "classification"
    n_epochs: int = 20
    batch_size: int = 32
    val_batch_size: Optional[int] = 32
    normalize: Optional[Literal["bn", "softmax"]] = None
    seed: int = 42
    log_level: int = logging.ERROR
    class_w: Optional[Union[Literal["balanced"], List[float]]] = None


class CallbackConfig(BaseModel):
    name: Literal["early_stopping", "progress_bar", "timer"] = Field(
        default_factory=lambda: "early_stopping"
    )
    params: Optional[dict] = None

    def model_post_init(self, __context):
        if self.params is None:
            self.params = {}
