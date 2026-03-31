"""Data I/O utilities for FMT-SimGen."""

import numpy as np
import json
from pathlib import Path
from typing import Any, Dict


def save_npz(filepath: str, **kwargs) -> Path:
    """Save arrays to .npz file.

    Parameters
    ----------
    filepath : str
        Output file path.
    **kwargs
        Arrays to save.

    Returns
    -------
    Path
        Path to saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **kwargs)
    return filepath


def load_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load arrays from .npz file.

    Parameters
    ----------
    filepath : str
        Path to .npz file.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of loaded arrays.
    """
    return dict(np.load(filepath))


def save_json(filepath: str, data: Any) -> Path:
    """Save data to JSON file.

    Parameters
    ----------
    filepath : str
        Output file path.
    data : Any
        Data to serialize (must be JSON-compatible).

    Returns
    -------
    Path
        Path to saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return filepath


def load_json(filepath: str) -> Any:
    """Load data from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file.

    Returns
    -------
    Any
        Loaded data.
    """
    with open(filepath, "r") as f:
        return json.load(filepath)
