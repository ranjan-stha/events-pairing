from typing import Dict

from pydantic import BaseModel, RootModel


class WeightConfig(BaseModel):
    """Weight configuration"""

    spatial: float
    temporal: float


class ClusterConfig(BaseModel):
    """Cluster configuration"""

    eps: float
    min_samples: int


class HazardConfig(BaseModel):
    """Hazard level configurations"""

    weight_config: WeightConfig
    cluster_config: ClusterConfig


class GridSearchConfigs(RootModel[Dict[str, HazardConfig]]):
    """Base"""
