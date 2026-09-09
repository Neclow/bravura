import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src._config import (
    CLUSTERS,
    DEFAULT_CLUSTERING_FEATURES,
    DEFAULT_PROCESSED_DIR,
    FEATURE_LABELS,
    PALETTE,
)

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG3_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG3_DIR, exist_ok=True)

RADAR_LABELS = [FEATURE_LABELS[c] for c in DEFAULT_CLUSTERING_FEATURES]

GRID_RINGS = [0.25, 0.5, 0.75, 1.0]

CLUSTER_ORDER = [cl["label"] for cl in CLUSTERS]


def load_data():
    """Load behavioural features and compute MinMax-scaled cluster means.

    Returns
    -------
    cluster_means_a : ndarray, shape (3, n_features)
        Scaled cluster centroids for Cohort A (row i = label i).
    cluster_means_b : ndarray, shape (3, n_features)
        Scaled cluster centroids for Cohort B (row i = label i).
    """
    Xa = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")
    Xb = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xb.csv", index_col="Row")

    scaler = MinMaxScaler()
    scaler.fit(Xa[DEFAULT_CLUSTERING_FEATURES])

    cluster_means_a = scaler.transform(
        Xa.groupby("label")[DEFAULT_CLUSTERING_FEATURES].mean().sort_index()
    )
    cluster_means_b = scaler.transform(
        Xb.groupby("label")[DEFAULT_CLUSTERING_FEATURES].mean().sort_index()
    )

    return cluster_means_a, cluster_means_b


def plot_radar(cluster_means_a, cluster_means_b):
    """Plot per-cluster radar charts (Cohort A filled, Cohort B dashed)."""
    n_features = len(DEFAULT_CLUSTERING_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), subplot_kw=dict(polar=True))

        for ax, label in zip(axes, CLUSTER_ORDER):
            values_a = cluster_means_a[label].tolist()
            values_a += values_a[:1]

            ax.fill(angles, values_a, alpha=0.25, color=PALETTE[label], zorder=10)
            ax.plot(angles, values_a, color=PALETTE[label], linewidth=1.5)

            values_b = cluster_means_b[label].tolist()
            values_b += values_b[:1]
            ax.plot(
                angles,
                values_b,
                color=PALETTE[label],
                linewidth=1.5,
                linestyle="--",
            )

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(RADAR_LABELS, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.set_yticks(GRID_RINGS)
            ax.set_yticklabels(["", "0.5", "", "1.0"])
            ax.yaxis.grid(False)
            ax.spines["polar"].set_visible(False)
            for r in GRID_RINGS:
                style = (
                    {"color": "black", "linewidth": 0.8}
                    if r == 1.0
                    else {"color": "gray", "linewidth": 0.5, "alpha": 0.3}
                )
                ax.plot(angles, [r] * len(angles), **style)
            ax.tick_params(axis="x", pad=10)

        plt.tight_layout()
        stem = f"{FIG3_DIR}/fig3c_radar_charts"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


if __name__ == "__main__":
    means_a, means_b = load_data()
    plot_radar(means_a, means_b)
