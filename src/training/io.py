import json
from pathlib import Path
from typing import Union
import torch


# TODO: add model load functions


def save_dict_to_disk(
    data: dict, filepath: Union[str, Path], indent: int = 4
) -> None:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def save_state_dict_to_disk(state_dict: dict, filepath: str) -> None:
    try:
        torch.save(state_dict, filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to save state dict to {filepath}: {e}")
