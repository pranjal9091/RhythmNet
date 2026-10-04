"""Configuration loading and path resolution utilities."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def get_project_root() -> Path:
    """Locate the project root directory by traversing upwards for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def resolve_path(path: Union[str, Path], root: Union[str, Path, None] = None) -> Path:
    """Resolve a path relative to project root if it is not absolute.

    Args:
        path: Path string or Path instance.
        root: Optional root path. If None, uses get_project_root().

    Returns:
        Resolved absolute Path instance.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else get_project_root()
    return (base / p).resolve()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Relative or absolute path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If YAML parsing fails.
    """
    resolved_path = resolve_path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
            return cfg or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config {resolved_path}: {exc}") from exc
