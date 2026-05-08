from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

from utils import Utils, HazardType, ComputeScore
from validation import GridSearchConfigs, HazardConfig
from models import EventData, MergedEventData
from grid_search import GridSearch
from plots import plot_clusters

# import warnings
# warnings.filterwarnings("error")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class Clusters:
    """Cluster builder"""
    METRIC: str = "precomputed"

    def __init__(self, events: list[EventData], configs: HazardConfig):
        self.events = events
        self.configs = configs

    def build_distance_matrix(self) -> np.ndarray:
        """Pairwise distance matrix"""
        n = len(self.events)
        D = np.ones((n, n), dtype=float)
        np.fill_diagonal(D, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                dist = ComputeScore(self.events[i], self.events[j]).compute_distance(self.configs)
                D[i, j] = D[j, i] = dist
        return D

    def build_clusters(self) -> list:
        """Run the clustering algorithm"""
        dist_matrices = self.build_distance_matrix()
        db = DBSCAN(
            eps=self.configs.cluster_config.eps,
            min_samples=self.configs.cluster_config.min_samples,
            metric=self.METRIC
        )
        labels = db.fit_predict(dist_matrices)
        return list(labels)


def run_pipeline(events_df: pd.DataFrame, search_configs):
    """Run the pipeline"""
    all_events: list[EventData] = []
    df = Utils.postprocess_event_df(events_df=events_df)

    for hazard, grp_df in df.groupby(by="hazard_type"):
        try:
            configs = search_configs.root[hazard]
        except KeyError:
            logger.warning(f"No configs for {hazard}")
            continue

        # Convert to EventData objects
        all_events = [
            EventData(
                id=row.id,
                source=row.source,
                hazard_type=row.hazard_type,
                lat=row.lat,
                lon=row.lon,
                start_timestamp=row.start_timestamp,
            )
            for row in grp_df.itertuples(index=False)
        ]

        clusters = Clusters(events=all_events, configs=configs)
        labels = clusters.build_clusters()

        unique_labels = set(labels)

        merged: list[MergedEventData] = []

        for label in sorted(unique_labels):
            is_noise = label == -1
            cluster_eventdata = [ed for ed, l in zip(all_events, labels) if l==label]
            if is_noise:
                for item in cluster_eventdata:
                    merged.append(
                        MergedEventData(
                            cluster_id=-1,
                            event_data=[item],
                            confidence=0.0
                        )
                    )
            else:
                confidence = 1 #cluster_confidence(cluster_eventdata)
                merged.append(
                    MergedEventData(
                        cluster_id=label,
                        event_data=cluster_eventdata,
                        confidence=confidence
                    )
                )

        df_final = Utils.convert_to_df(merged=merged)
        # Saving the file as csv
        output_path = Path(".") / "outputs" / f"{hazard}_{datetime.today().date().isoformat()}_clustered_events.csv"
        df_final.to_csv(output_path, index=False)
        # Generating the plot and saving the file as png
        plot_clusters(df=df_final, hazard=hazard)



if __name__ == "__main__":
    # Add new data source here
    gdacs_data = Utils.load_data(file_path="./datasets/gdacs_2020_2025_data.json")
    emdat_data = Utils.load_data(file_path="./datasets/emdat_2024_2025_data.json")
    pdc_data = Utils.load_data(file_path="./datasets/pdc_2020_2025_data.json")
    glide_data = Utils.load_data(file_path="./datasets/glide_2020_2025_data.json")

    # Add new source data processing here
    processed_gdacs_data = Utils.preprocess_data(event_data=gdacs_data)
    processed_emdat_data = Utils.preprocess_data(event_data=emdat_data)
    processed_pdc_data = Utils.preprocess_data(event_data=pdc_data)
    processed_glide_data = Utils.preprocess_data(event_data=glide_data)

    logger.info(f"Shape of GDACS data: {processed_gdacs_data.shape}")
    logger.info(f"Shape of EMDAT data: {processed_emdat_data.shape}")
    logger.info(f"Shape of PDC data: {processed_pdc_data.shape}")
    logger.info(f"Shape of Glide data: {processed_glide_data.shape}")

    # Add new source data for concatenation
    processed_events_df = pd.concat([
        processed_gdacs_data,
        processed_emdat_data,
        processed_pdc_data,
        processed_glide_data,
    ])
    print(processed_events_df.head(5))
    # Remove fully duplicated rows
    processed_events_df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    # Reset dataframe index
    processed_events_df.reset_index(drop=True, inplace=True)
    logger.info(f"Size of the processed events: {processed_events_df.shape}")

    grid_search_configs = {}

    for hazard_type in HazardType:
        logger.info(f"Processing {hazard_type}")
        if not hazard_type.value == "EARTHQUAKE":
            continue
        part_config = GridSearch.run_grid_search(events_df=processed_events_df, hazard=hazard_type)
        if not part_config:
            logger.warning(f"Grid configs is not available for {hazard_type.value}")
            continue
        grid_search_configs.update({
            hazard_type.value: part_config ## hazard.value
        })

    logger.info(f"Grid search configs: {grid_search_configs}")

    # Validate the configurations
    validated_configs = GridSearchConfigs.model_validate(grid_search_configs)
    # Run the clustering pipeline
    run_pipeline(events_df=processed_events_df, search_configs=validated_configs)
