import math
import json

from enum import Enum
import pandas as pd
from models import EventData, MergedEventData
from validation import HazardConfig

class Source(str, Enum):
    """Event sources"""
    GDACS = "GDACS"
    USGS = "USGS"
    EMDAT = "EMDAT"
    PDC = "PDC"

class HazardType(str, Enum):
    """List of hazards"""
    FLOOD = "FLOOD"
    EARTHQUAKE = "EARTHQUAKE"
    STORM = "STORM"
    TROPICAL_CYCLONE = "TROPICAL_CYCLONE"
    VOLCANO = "VOLCANO"
    WILDFIRE = "WILDFIRE"

class Utils:
    """Utility functions"""
    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres."""
        R = 6371.0  # radius of earth in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def load_data(file_path: str) -> dict:
        """Load the data from the file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return data

    @staticmethod
    def preprocess_data(event_data: dict) -> pd.DataFrame:
        """Preprocess the data and return a dataframe with specific columns only"""
        event_features = event_data.get("features", {})
        df = pd.DataFrame(event_features)
        df_properties = pd.json_normalize(df['properties'])
        df = df.drop(['properties'], axis=1)
        df = pd.concat([df, df_properties], axis=1)
        # Only use the required columns
        useful_cols = ['id', 'bbox', 'geometry', 'collection', 'title', 'datetime', 'monty:corr_id', 'monty:hazard_codes', 'monty:country_codes']
        df = df[useful_cols]
        df['monty:hazard_codes'] = df['monty:hazard_codes'].apply(lambda x: x[1] if len(x)>1 else x[0])
        return df
    
    @staticmethod
    def convert_to_df(merged: list[MergedEventData]) -> pd.DataFrame:
        """Convert the raw data to a dataframe"""
        rows = []
        for m in merged:
            for r in m.event_data:
                rows.append({
                    "cluster_id": m.cluster_id,
                    "confidence": m.confidence,
                    "id": r.id,
                    "source": r.source,
                    "hazard_type": r.hazard_type,
                    "lat": r.lat,
                    "lon": r.lon,
                    "timestamp": r.start_timestamp,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def postprocess_event_df(events_df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the event dataframe"""
        df = events_df.copy()

        # 1. Apply mappings (still Python functions, but vectorized over Series)
        df["hazard_type"] = df["monty:hazard_codes"].apply(lambda x: Mappings.hazard_mapping(x))
        df["source"] = df["collection"].apply(lambda x: Mappings.source_mapping(x))

        # 2. Drop invalid rows
        df = df[df["hazard_type"].notna() & df["source"].notna()]

        # 3. Compute derived columns
        df["lat"] = ((df["bbox"].str[0] + df["bbox"].str[2]) / 2.0).round(3)
        df["lon"] = ((df["bbox"].str[1] + df["bbox"].str[3]) / 2.0).round(3)

        df["start_timestamp"] = (
            pd.to_datetime(df["datetime"], utc=True, format="ISO8601").astype("int64") // 10**9
        )
        # df.to_csv("./outputs/processed.csv")
        return df

class Mappings:
    """Mappings"""
    @staticmethod
    def hazard_mapping(hazard_code: str) -> str|None:
        """Maps hazard code to hazard type"""
        mappings = {
            "WF": HazardType.WILDFIRE.value,
            "EQ": HazardType.EARTHQUAKE.value,
            "FL": HazardType.FLOOD.value,
            "ST": HazardType.STORM.value,
            "TC": HazardType.TROPICAL_CYCLONE.value,
            "FF": HazardType.FLOOD.value,
            "VO": HazardType.VOLCANO.value
        }
        return mappings.get(hazard_code)

    @staticmethod
    def source_mapping(source: str) -> str|None:
        """Maps source to source type"""
        mappings = {
            "gdacs-events": Source.GDACS.value,
            "emdat-events": Source.EMDAT.value,
            "usgs-events": Source.USGS.value,
            "pdc-events": Source.PDC.value,
        }
        return mappings.get(source)

class NormalizedValues:
    """Lists normalized values"""
    @staticmethod
    def normalized_mappings(hazard: HazardType, km_value: float, hrs_value: float) -> tuple[float, float]:
        """Normalized mappings"""
        mappings_spatial = {
            HazardType.EARTHQUAKE: (
                1.0 if km_value <= 20 else
                0.75 if km_value <= 50 else
                0.3 if km_value <= 100 else
                0.0
            ),
            HazardType.FLOOD: (
                1.0 if km_value <= 30 else
                0.7 if km_value <= 50 else
                0.4 if km_value <= 90 else
                0.0
            )
        }
        mappings_temporal = {
            HazardType.EARTHQUAKE: (
                1.0 if hrs_value <= 1 else
                0.7 if hrs_value <= 6 else
                0.3 if hrs_value <= 24 else
                0.0
            ),
            HazardType.FLOOD: (
                1.0 if hrs_value <= 12 else
                0.7 if hrs_value <= 24 else
                0.4 if hrs_value <= 48 else
                0.0
            )
        }
        return (mappings_spatial.get(hazard), mappings_temporal.get(hazard))

class ComputeScore:
    """Computes Scores/Distances between Events"""
    def __init__(self, event1: EventData, event2: EventData):
        self.event1 = event1
        self.event2 = event2
        self.km = Utils.haversine_km(event1.lat, event1.lon, event2.lat, event2.lon) # in kms
        self.hrs = abs(event1.start_timestamp - event2.start_timestamp) / 3600 # in hrs

    def _score_earthquake(self, spatial_weight: float, temporal_weight: float) -> float:
        """Score Earthquake event"""

        spatial, temporal = NormalizedValues.normalized_mappings(
            hazard=HazardType.EARTHQUAKE,
            km_value=self.km,
            hrs_value=self.hrs
        )
    
        # # Spatial
        # spatial = (
        #     1.0 if self.km <= 50 else
        #     0.75 if self.km <= 100 else
        #     0.4 if self.km <= 200 else
        #     0.05
        # )

        # # Temporal
        # temporal = (
        #     1.0 if self.hrs <= 12 else
        #     0.7 if self.hrs <= 24 else
        #     0.4 if self.hrs <= 48 else
        #     0.05
        # )
        return spatial * spatial_weight + temporal * temporal_weight

    def _score_flood(self, spatial_weight: float, temporal_weight: float) -> float:
        """Score Flood event"""
        spatial, temporal = NormalizedValues.normalized_mappings(
            hazard=HazardType.FLOOD,
            km_value=self.km,
            hrs_value=self.hrs
        )
        # # Spatial
        # spatial = (
        #     1.0 if self.km <= 50 else
        #     0.75 if self.km <= 100 else
        #     0.4 if self.km <= 200 else
        #     0.05
        # )
        # # Temporal
        # temporal = (
        #     1.0 if self.hrs <= 24 else
        #     0.7 if self.hrs <= 48 else
        #     0.4 if self.hrs <= 72 else
        #     0.05
        # )
        return spatial * spatial_weight + temporal * temporal_weight

    def compute_distance(self, configs: HazardConfig) -> float:
        """Compute distance between two events"""
        if self.event1.hazard_type != self.event2.hazard_type:
            return 1.0 #  Max distance as those events are of different hazard types

        if self.event1.hazard_type == HazardType.EARTHQUAKE.value:
            score = self._score_earthquake(
                spatial_weight=configs.weight_config.spatial,
                temporal_weight=configs.weight_config.temporal
            )
        elif self.event1.hazard_type == HazardType.FLOOD.value:
            score = self._score_flood(
                spatial_weight=configs.weight_config.spatial,
                temporal_weight=configs.weight_config.temporal
            )
        else:
            score = self._score_flood(
                spatial_weight=configs.weight_config.spatial,
                temporal_weight=configs.weight_config.temporal
            )
            return 1.0
            #raise NotImplementedError(f"Hazard type {self.event1.hazard_type} not implemented")

        return 1.0 - score
