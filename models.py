from dataclasses import dataclass


@dataclass
class EventData:
    """Events originated from different sources"""

    id: str
    source: str
    hazard_type: str
    lat: float
    lon: float
    start_timestamp: float


@dataclass
class MergedEventData:
    """Merge events of type EventData"""

    cluster_id: int
    event_data: list[EventData]
    confidence: float  # within cluster level confidence


@dataclass
class WeightConfig:
    """Weight configuration"""

    spatial: float
    temporal: float

    def normalized(self) -> "WeightConfig":
        """Normalize the values"""
        total = self.spatial + self.temporal
        return WeightConfig(self.spatial / total, self.temporal / total)


@dataclass
class ClusterConfig:
    """Cluster configuration"""

    eps: float
    min_samples: int
