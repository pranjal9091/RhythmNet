"""Feature definition schema and deterministic ordering system for ECG features."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    """Dataclass holding metadata for a single engineered feature."""

    name: str
    group: str
    description: str
    dtype: str = "float32"


class FeatureSchema:
    """Manager for maintaining a deterministic feature schema and catalog."""

    def __init__(self, definitions: Optional[List[FeatureDefinition]] = None):
        self._definitions: List[FeatureDefinition] = definitions or []
        self._name_to_def: Dict[str, FeatureDefinition] = {
            d.name: d for d in self._definitions
        }

    def add_feature(self, name: str, group: str, description: str, dtype: str = "float32") -> None:
        """Register a feature in the schema."""
        if name in self._name_to_def:
            return
        feat_def = FeatureDefinition(name=name, group=group, description=description, dtype=dtype)
        self._definitions.append(feat_def)
        self._name_to_def[name] = feat_def

    @property
    def feature_names(self) -> List[str]:
        """Return deterministic ordered list of feature names."""
        return [d.name for d in self._definitions]

    @property
    def definitions(self) -> List[FeatureDefinition]:
        """Return copy of feature definitions list."""
        return list(self._definitions)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature schema catalog to a DataFrame."""
        rows = [
            {
                "feature_name": d.name,
                "feature_group": d.group,
                "description": d.description,
                "dtype": d.dtype,
            }
            for d in self._definitions
        ]
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._definitions)
