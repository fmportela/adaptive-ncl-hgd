from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# TODO: maybe pass task type instead of inferring from y
def convert_arrays_to_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if np.issubdtype(y.dtype, np.integer):
        y_tensor = torch.tensor(y, dtype=torch.long)  # Classification targets
    else:
        y_tensor = torch.tensor(y, dtype=torch.float32)  # Regression targets
    return TensorDataset(X_tensor, y_tensor)


def convert_dataset_to_loader(
    dataset: TensorDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )


def make_bootstrap_ds(ds: TensorDataset, seed: int = 42) -> TensorDataset:
    n_samples = len(ds)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(high=n_samples, size=(n_samples,), generator=g)
    tensors = [t[indices] for t in ds.tensors]
    return TensorDataset(*tensors)


def split_sklearn_dataset(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: list[float],
    *,
    stratify: bool = True,
    random_state: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Split a dataset into multiple subsets based on provided ratios using
    sklearn's train_test_split.

    Args:
        X: The input features.
        y: The labels corresponding to the features.
        split_ratios: The ratios for each split. Must sum to 1.

    Returns:
        A list of NumPy arrays in the order [X1, y1, ..., XN, yN] based
            on the split_ratios.
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1."

    # Omit 0.0 ratios for splitting
    split_ratios = [r for r in split_ratios if r > 0.0]

    X_splits, y_splits = [], []
    remaining_X, remaining_y = X, y

    for i in range(len(split_ratios) - 1):
        split_ratio = split_ratios[i] / sum(
            split_ratios[i:]
        )  # Adjust to remaining portion
        X_split, remaining_X, y_split, remaining_y = train_test_split(
            remaining_X,
            remaining_y,
            test_size=(1 - split_ratio),
            stratify=remaining_y if stratify else None,
            random_state=random_state,
        )
        X_splits.append(X_split)
        y_splits.append(y_split)

    X_splits.append(remaining_X)
    y_splits.append(remaining_y)

    return X_splits + y_splits
