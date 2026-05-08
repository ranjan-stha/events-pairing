import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import matplotlib.lines as mlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def plot_clusters(df: pd.DataFrame, hazard: str) -> None:
    """Plot the clusters"""

    # Create figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # -------------------------
    # Automatically assign markers
    # -------------------------
    available_markers = [
        "o", "s", "^", "D", "v", "P", "*", "X",
        "<", ">", "h", "H", "8", "p"
    ]

    sources = sorted(df["source"].unique())

    # Map each source to a marker automatically
    markers = {
        src: available_markers[i % len(available_markers)]
        for i, src in enumerate(sources)
    }

    # -------------------------
    # Plot points
    # -------------------------
    for src in sources:
        sub = df[df["source"] == src]
        ax.scatter(
            sub["lon"],
            sub["lat"],
            sub["timestamp"],
            c=sub["cluster_id"],
            cmap="tab20",
            marker=markers[src],
            s=60,
            alpha=0.8
        )

    # Labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Timestamp")
    ax.set_title("DBSCAN Clusters (Spatio-Temporal View)")

    # -------------------------
    # Legend: Sources
    # -------------------------
    source_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=markers[src],
            linestyle="None",
            markersize=8,
            label=src
        )
        for src in sources
    ]

    legend1 = ax.legend(
        handles=source_handles,
        title="Source",
        loc="upper left"
    )

    ax.add_artist(legend1)

    # Improve layout
    plt.tight_layout()

    # -------------------------
    # Save figure as PNG
    # -------------------------
    output_file = Path(".") / "outputs" / f"{hazard}_{datetime.today().date().isoformat()}_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    logger.info(f"Saved plot to: {output_file}")

    # plt.show()
