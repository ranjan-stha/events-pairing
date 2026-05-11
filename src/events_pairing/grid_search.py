import typing
from venv import logger

import numpy as np
import pandas as pd
from hdbscan.validity import validity_index
from sklearn.cluster import DBSCAN

from events_pairing.models import ClusterConfig, EventData, WeightConfig
from events_pairing.utils import HazardType, NormalizedValues, Utils


class GridSearch:
    """Grid Search of the weights and cluster configs"""

    @staticmethod
    def compute_components(report1: EventData, report2: EventData, hazard: HazardType) -> dict:
        """Return component scores"""
        km = Utils.haversine_km(report1.lat, report1.lon, report1.lat, report2.lon)
        hrs = abs(report1.start_timestamp - report2.start_timestamp) / 3600

        spatial, temporal = NormalizedValues.normalized_mappings(hazard=hazard, km_value=km, hrs_value=hrs)

        return {"spatial": spatial, "temporal": temporal}

    @staticmethod
    def precompute_components(reports: typing.List[EventData], hazard: HazardType) -> tuple[list[EventData], np.ndarray]:
        """Get the component scores of all the events with same hazard type"""
        subset = [r for r in reports if r.hazard_type == hazard.value]
        logger.info(f"Total event records for {hazard.value} is {len(subset)}")
        n = len(subset)
        comp = np.zeros((n, n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                c = GridSearch.compute_components(subset[i], subset[j], hazard)
                v = [c["spatial"], c["temporal"]]
                comp[i, j] = v
                comp[j, i] = v
        return subset, comp

    @staticmethod
    def build_weight_grid() -> typing.List[WeightConfig]:
        """Build spatial and temporal weights grid"""
        spatial_steps = [0.2, 0.3, 0.4, 0.5, 0.6]
        configs = []

        for sp in spatial_steps:
            tm = round(1.0 - sp, 2)
            configs.append(WeightConfig(sp, tm))
        return configs

    @staticmethod
    def build_cluster_grid() -> typing.List[ClusterConfig]:
        """Generate DBSCAN hyperparameter candidates."""
        configs = []
        for eps in [0.10, 0.15, 0.20, 0.30, 0.40]:
            for min_pts in [2, 3]:
                configs.append(ClusterConfig(eps, min_pts))
        return configs

    @staticmethod
    def distance_matrix_from_weights(comp: np.ndarray, weight: WeightConfig) -> np.ndarray:
        """Build a distance matrix"""
        w = weight.normalized()
        w_arr = np.array([w.spatial, w.temporal])
        scores = comp @ w_arr
        D = 1.0 - scores
        np.clip(D, 0, 1, out=D)
        np.fill_diagonal(D, 0.0)
        return D

    @staticmethod
    def compute_dbcv(D: np.ndarray, labels: np.ndarray, dims: int = 2) -> float:
        """
        Compute DBCV (Density-Based Clustering Validation) scores
        """
        n_clusters = len(set(labels) - {-1})
        n_non_noise = (labels != -1).sum()

        # DBCV requires ≥ 2 clusters and ≥ 2 non-noise points per cluster
        if n_clusters < 2 or n_non_noise < 2 * n_clusters:
            return 0.0

        try:
            score = validity_index(D.astype(np.float64), labels, metric="precomputed", d=dims)
            if np.isnan(score) or np.isinf(score):
                return 0.0
            return round(float(np.clip(score, -1.0, 1.0)), 2)
        except Exception:
            return 0.0

    @staticmethod
    def run_grid_search(events_df: pd.DataFrame, hazard: HazardType, top_k: int = 10) -> typing.Optional[dict]:
        """Run grid search"""
        postprocessed_df = Utils.postprocess_event_df(events_df=events_df)
        all_events = [
            EventData(
                id=row.id,
                source=row.source,
                hazard_type=row.hazard_type,
                lat=row.lat,
                lon=row.lon,
                start_timestamp=row.start_timestamp,
            )
            for row in postprocessed_df.itertuples(index=False)
        ]
        subset, comp = GridSearch.precompute_components(reports=all_events, hazard=hazard)

        weight_grid = GridSearch.build_weight_grid()
        cluster_grid = GridSearch.build_cluster_grid()

        configurations = []
        for w_cfg in weight_grid:
            D = GridSearch.distance_matrix_from_weights(comp, w_cfg)
            for c_cfg in cluster_grid:
                db_clusters = DBSCAN(eps=c_cfg.eps, min_samples=c_cfg.min_samples, metric="precomputed")
                labels = db_clusters.fit_predict(D)
                try:
                    dbcv_score = GridSearch.compute_dbcv(D, labels)
                    if dbcv_score >= 0.5:
                        configuration = {
                            "labels": labels,
                            "w_cfg": w_cfg,
                            "c_cfg": c_cfg,
                            "dbcv_score": dbcv_score,
                        }
                        configurations.append(configuration)
                except Exception:
                    pass

        if not configurations:
            logger.info(f"Using default configuration for hazard {hazard.value}")
            return {"weight_config": {"spatial": 0.5, "temporal": 0.5}, "cluster_config": {"eps": 0.3, "min_samples": 3}}

        sorted_data = sorted(configurations, key=lambda x: x["dbcv_score"], reverse=True)
        max_dbcv_data = sorted_data[0]

        return {
            "weight_config": {
                "spatial": max_dbcv_data["w_cfg"].spatial,
                "temporal": max_dbcv_data["w_cfg"].temporal,
            },
            "cluster_config": {
                "eps": max_dbcv_data["c_cfg"].eps,
                "min_samples": max_dbcv_data["c_cfg"].min_samples,
            },
        }
