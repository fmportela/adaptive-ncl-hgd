import numpy as np
from pmlb import fetch_data


def load_pmlb_dataset(
    name: str, local_cache_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_data(
        dataset_name=name,
        return_X_y=True,
        local_cache_dir=local_cache_dir,
    )

    return X, y
