"""
Model serialization and persistence module.

Handles saving and reloading trained classical ML models and label mapping configurations.
"""

from pathlib import Path
from typing import Dict, Any, Union
import json
import joblib


def save_label_mapping(mapping: Dict[str, int], output_path: Union[str, Path]) -> None:
    """
    Saves class label mapping dictionary to a JSON file.
    
    Parameters
    ----------
    mapping : Dict[str, int]
        Class label mapping dictionary (e.g., {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}).
    output_path : Union[str, Path]
        Target JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)


def load_label_mapping(input_path: Union[str, Path]) -> Dict[str, int]:
    """
    Loads class label mapping dictionary from a JSON file.
    
    Parameters
    ----------
    input_path : Union[str, Path]
        Path to label mapping JSON file.
        
    Returns
    -------
    Dict[str, int]
        Loaded label mapping dictionary.
    """
    input_path = Path(input_path)
    with open(input_path, "r") as f:
        mapping = json.load(f)
    return {k: int(v) for k, v in mapping.items()}


def save_model(model: Any, output_path: Union[str, Path]) -> None:
    """
    Saves a trained model artifact using joblib or native XGBoost save.
    
    Parameters
    ----------
    model : Any
        Trained model instance.
    output_path : Union[str, Path]
        Target filepath for saved model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".json" and hasattr(model, "save_model"):
        model.save_model(str(output_path))
    else:
        joblib.dump(model, output_path)


def load_model(input_path: Union[str, Path], model_type: str = "auto") -> Any:
    """
    Loads a saved model artifact.
    
    Parameters
    ----------
    input_path : Union[str, Path]
        Filepath of saved model artifact.
    model_type : str, default="auto"
        Optional model type hint ('xgboost', 'joblib', or 'auto').
        
    Returns
    -------
    Any
        Reloaded model instance.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
        
    if input_path.suffix == ".json" or model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(input_path))
        return model
    else:
        return joblib.load(input_path)


def get_model_size_mb(filepath: Union[str, Path]) -> float:
    """
    Computes size of a serialized model file in Megabytes (MB).
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Filepath to inspect.
        
    Returns
    -------
    float
        Size in MB.
    """
    path = Path(filepath)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size) / (1024.0 * 1024.0)
