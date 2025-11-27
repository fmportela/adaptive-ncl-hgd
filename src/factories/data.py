from typing import Optional, Union, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.configs import ScalerConfig, DatasetConfig
from src.data.pmlb import load_pmlb_dataset


def get_scaler(
    scaler_cfg: ScalerConfig,
) -> Optional[Union[StandardScaler, MinMaxScaler]]:
    if scaler_cfg.name is None:
        return None
    elif scaler_cfg.name == "standard":
        return StandardScaler()
    elif scaler_cfg.name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_cfg.name}")


def get_pmlb_dataset(
    dataset_cfg: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    return load_pmlb_dataset(
        name=dataset_cfg.name, local_cache_dir=dataset_cfg.local_cache_dir
    )
