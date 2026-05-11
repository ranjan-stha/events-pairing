import json
import math
from enum import Enum
from pathlib import Path

import pandas as pd

from events_pairing.models import EventData, MergedEventData
from events_pairing.validation import HazardConfig


class Source(str, Enum):
    """Event sources"""

    GDACS = "GDACS"
    USGS = "USGS"
    EMDAT = "EMDAT"
    PDC = "PDC"
    GLIDE = "GLIDE"


class HazardType(str, Enum):
    """List of hazards"""

    FLOOD = "FLOOD"
    EARTHQUAKE = "EARTHQUAKE"
    STORM = "STORM"
    TROPICAL_CYCLONE = "TROPICAL_CYCLONE"
    VOLCANO = "VOLCANO"
    WILDFIRE = "WILDFIRE"
    DROUGHT = "DROUGHT"


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
    def load_data(file_path: Path) -> dict:
        """Load the data from the file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return data

    @staticmethod
    def preprocess_data(event_data: dict) -> pd.DataFrame:
        """Preprocess the data and return a dataframe with specific columns only"""
        event_features = event_data.get("features", {})
        df = pd.DataFrame(event_features)
        df_properties = pd.json_normalize(df["properties"])
        df = df.drop(["properties"], axis=1)
        df = pd.concat([df, df_properties], axis=1)
        # Only use the required columns
        useful_cols = [
            "id",
            "bbox",
            "geometry",
            "collection",
            "title",
            "datetime",
            "monty:corr_id",
            "monty:hazard_codes",
            "monty:country_codes",
        ]
        df = df[useful_cols]
        df["monty:hazard_codes"] = df["monty:hazard_codes"].apply(lambda x: x[1] if len(x) > 1 else x[0])
        return df

    @staticmethod
    def convert_to_df(merged: list[MergedEventData]) -> pd.DataFrame:
        """Convert the raw data to a dataframe"""
        rows = []
        for m in merged:
            for r in m.event_data:
                rows.append(
                    {
                        "cluster_id": m.cluster_id,
                        "confidence": m.confidence,
                        "id": r.id,
                        "source": r.source,
                        "hazard_type": r.hazard_type,
                        "lat": r.lat,
                        "lon": r.lon,
                        "timestamp": r.start_timestamp,
                    }
                )
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

        df["start_timestamp"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601").astype("int64") // 10**9
        # df.to_csv("./outputs/processed.csv")
        return df


class Mappings:
    """Mappings"""

    @staticmethod
    def hazard_mapping(hazard_code: str) -> str | None:
        """Maps hazard code to hazard type"""
        mappings = {
            "WF": HazardType.WILDFIRE.value,
            "EQ": HazardType.EARTHQUAKE.value,
            "FL": HazardType.FLOOD.value,
            "ST": HazardType.STORM.value,
            "TC": HazardType.TROPICAL_CYCLONE.value,
            "FF": HazardType.FLOOD.value,
            "VO": HazardType.VOLCANO.value,
        }
        return mappings.get(hazard_code)

    @staticmethod
    def source_mapping(source: str) -> str | None:
        """Maps source to source type"""
        mappings = {
            "gdacs-events": Source.GDACS.value,
            "emdat-events": Source.EMDAT.value,
            "usgs-events": Source.USGS.value,
            "pdc-events": Source.PDC.value,
            "glide-events": Source.GLIDE.value,
        }
        return mappings.get(source)


class NormalizedValues:
    """Lists normalized values"""

    @staticmethod
    def normalized_mappings(hazard: HazardType, km_value: float, hrs_value: float) -> tuple[float | None, float | None]:
        """Normalized mapping values for different hazards"""
        mappings_spatial = {
            # GDACS vs EMDAT earthquake: epicenter (GDACS) vs admin centroid (EMDAT)
            # Admin centroids can be 30-80km off in large provinces
            HazardType.EARTHQUAKE: (
                1.0
                if km_value <= 25
                # same point, minor geocoding noise
                else 0.8
                if km_value <= 60
                # epicenter vs. district centroid
                else 0.5
                if km_value <= 120
                # epicenter vs. province/state centroid
                else 0.1
                if km_value <= 200
                # country-level aggregation (EMDAT edge case)
                else 0.0
            ),
            # Floods reported at affected district centroid vs. river gauge location
            # EMDAT especially aggregates multi-district floods to province centroid
            HazardType.FLOOD: (
                1.0
                if km_value <= 15
                else 0.7
                if km_value <= 50
                # district centroid drift
                else 0.4
                if km_value <= 120
                # province-level aggregation
                else 0.1
                if km_value <= 250
                # EMDAT national-level flood event
                else 0.0
            ),
            # Cyclones: GDACS uses current eye position, PDC may use landfall point
            # Track displacement between reports can be significant
            HazardType.TROPICAL_CYCLONE: (
                1.0
                if km_value <= 50
                # same eye position, timing difference
                else 0.7
                if km_value <= 150
                # track segment displacement
                else 0.4
                if km_value <= 350
                # landfall vs. origin point
                else 0.0
            ),
            # Tsunamis: source location (GDACS) vs. impact location (EMDAT)
            # Can be hundreds of km apart — earthquake origin vs. coastal impact zone
            # HazardType.TSUNAMI: (
            #     1.0 if km_value <= 30 else
            #     0.6 if km_value <= 100 else
            #     0.4 if km_value <= 300 else  # source vs. impact zone mismatch
            #     0.1 if km_value <= 600 else  # trans-oceanic source vs. distant impact
            #     0.0
            # ),
            HazardType.WILDFIRE: (
                1.0
                if km_value <= 10
                else 0.6
                if km_value <= 40
                # perimeter centroid shift as fire grows
                else 0.2
                if km_value <= 80
                else 0.0
            ),
            HazardType.DROUGHT: (
                1.0
                if km_value <= 100
                # admin region centroids expected to differ widely
                else 0.7
                if km_value <= 300
                else 0.4
                if km_value <= 600
                else 0.0
            ),
            HazardType.VOLCANO: (
                1.0
                if km_value <= 10
                # volcano location is fixed and precise
                else 0.6
                if km_value <= 30
                # impact zone centroid vs. vent
                else 0.2
                if km_value <= 80
                else 0.0
            ),
        }

        # Temporal thresholds should reflect:
        # - Source reporting latency (how long after event onset does each source publish)
        # - Update cycles (GDACS updates hourly, EMDAT may finalize months later)
        # - Whether the "event date" in the source is onset, peak, or report date

        mappings_temporal = {
            # GDACS: ~1-2hr lag | PDC: ~2-6hr | GLIDE: 1-7 days | EMDAT: weeks-months
            # A 72hr window is needed to catch GLIDE registrations of fast events
            HazardType.EARTHQUAKE: (
                1.0
                if hrs_value <= 6
                # GDACS/PDC same-event window
                else 0.8
                if hrs_value <= 24
                # GLIDE registration lag
                else 0.5
                if hrs_value <= 72
                # slow source ingestion
                else 0.1
                if hrs_value <= 168
                # EMDAT preliminary entry lag
                else 0.0
            ),
            # Floods have onset ambiguity — sources disagree on "start date"
            # EMDAT uses onset of impact, GDACS uses trigger (rainfall peak), PDC uses forecast
            HazardType.FLOOD: (
                1.0
                if hrs_value <= 12
                else 0.8
                if hrs_value <= 48
                # onset date ambiguity across sources
                else 0.5
                if hrs_value <= 120
                # slow-onset flood, EMDAT vs GDACS divergence
                else 0.2
                if hrs_value <= 240
                # GLIDE/EMDAT finalization lag
                else 0.0
            ),
            # Cyclones: all sources track these well but use different reference points
            # (formation vs. named storm vs. landfall vs. dissipation)
            HazardType.TROPICAL_CYCLONE: (
                1.0
                if hrs_value <= 12
                else 0.7
                if hrs_value <= 48
                # formation vs. landfall date used differently
                else 0.4
                if hrs_value <= 96
                # EMDAT records landfall date only
                else 0.0
            ),
            # HazardType.TSUNAMI: (
            #     1.0 if hrs_value <= 6 else
            #     0.6 if hrs_value <= 24 else
            #     0.2 if hrs_value <= 72 else   # EMDAT lag for distant-impact tsunamis
            #     0.0
            # ),
            HazardType.WILDFIRE: (
                1.0
                if hrs_value <= 12
                else 0.6
                if hrs_value <= 48
                else 0.3
                if hrs_value <= 120
                # EMDAT records fire season aggregates
                else 0.0
            ),
            # Drought: EMDAT onset dates are estimated retrospectively
            # Cross-source temporal matching is inherently loose
            HazardType.DROUGHT: (
                1.0
                if hrs_value <= 720
                # 30 days — onset date estimation error
                else 0.6
                if hrs_value <= 2160
                # 90 days
                else 0.3
                if hrs_value <= 4320
                else 0.0
            ),
            HazardType.VOLCANO: (
                1.0
                if hrs_value <= 12
                else 0.6
                if hrs_value <= 48
                else 0.3
                if hrs_value <= 168
                # EMDAT records eruption episodes, not onset
                else 0.0
            ),
        }
        return (mappings_spatial.get(hazard), mappings_temporal.get(hazard))


class ComputeScore:
    """Computes Scores/Distances between Events"""

    def __init__(self, event1: EventData, event2: EventData):
        self.event1 = event1
        self.event2 = event2
        self.km = Utils.haversine_km(event1.lat, event1.lon, event2.lat, event2.lon)  # in kms
        self.hrs = abs(event1.start_timestamp - event2.start_timestamp) / 3600  # in hrs

    def _score_earthquake(self, spatial_weight: float, temporal_weight: float) -> float:
        """Score Earthquake event"""

        spatial, temporal = NormalizedValues.normalized_mappings(
            hazard=HazardType.EARTHQUAKE, km_value=self.km, hrs_value=self.hrs
        )
        return spatial * spatial_weight + temporal * temporal_weight

    def _score_flood(self, spatial_weight: float, temporal_weight: float) -> float:
        """Score Flood event"""
        spatial, temporal = NormalizedValues.normalized_mappings(
            hazard=HazardType.FLOOD, km_value=self.km, hrs_value=self.hrs
        )
        return spatial * spatial_weight + temporal * temporal_weight

    def compute_distance(self, configs: HazardConfig) -> float:
        """Compute distance between two events"""
        if self.event1.hazard_type != self.event2.hazard_type:
            return 1.0  #  Max distance as those events are of different hazard types

        if self.event1.hazard_type == HazardType.EARTHQUAKE.value:
            score = self._score_earthquake(
                spatial_weight=configs.weight_config.spatial, temporal_weight=configs.weight_config.temporal
            )
        elif self.event1.hazard_type == HazardType.FLOOD.value:
            score = self._score_flood(
                spatial_weight=configs.weight_config.spatial, temporal_weight=configs.weight_config.temporal
            )
        else:
            score = self._score_flood(
                spatial_weight=configs.weight_config.spatial, temporal_weight=configs.weight_config.temporal
            )
            return 1.0
            # raise NotImplementedError(f"Hazard type {self.event1.hazard_type} not implemented")

        return 1.0 - score
