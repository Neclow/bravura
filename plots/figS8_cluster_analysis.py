import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from src._config import (
    CLUSTER_NAMES,
    DEFAULT_CLUSTERING_FEATURES,
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_DIR,
    PALETTE,
)

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG_DIR, exist_ok=True)

SENSITIVITY_DIR = f"{DEFAULT_DATA_DIR}/processed/sensitivity_clustering"

K_BEST = 3


def load_data():
    """Load ablate-k table and Cohort A scaled features with consensus labels.

    Returns
    -------
    sens_k : DataFrame
        Silhouette, gap, and gap_diff per k (index = k).
    Xa_scaled : ndarray
        StandardScaler'd clustering features for Cohort A.
    labels : ndarray
        Consensus cluster labels for Cohort A.
    """
    sens_k = pd.read_csv(f"{SENSITIVITY_DIR}/best_solver_ablate_k.csv", index_col="k")

    Xa = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")
    Xa_scaled = StandardScaler().fit_transform(Xa[DEFAULT_CLUSTERING_FEATURES])
    labels = Xa["label"].values

    return sens_k, Xa_scaled, labels


def plot_silhouette_k(sens_k):
    """Plot silhouette score vs number of clusters."""
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(sens_k.index, sens_k["silhouette"], "o-", color="k")
        ax.axvline(K_BEST, linestyle="--", color="gray", alpha=0.5)
        ax.set_xlabel("# Clusters")
        ax.set_ylabel("Silhouette score")
        ax.set_xticks(sens_k.index)

        stem = f"{FIG_DIR}/figS4a_silhouette_k"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG_DIR}/tableS10_silhouette_k.md"
    sens_k.round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


def plot_gap_statistic(sens_k):
    """Plot gap statistic difference vs number of clusters."""
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(sens_k.index, sens_k["gap_diff"], marker="o", color="k")
        ax.axhline(0, ls="--", color="grey", lw=0.8)
        ax.axvline(K_BEST, ls="--", color="grey", lw=0.8)
        ax.set_xlabel("# Clusters")
        ax.set_ylabel("Gap difference")

        plt.tight_layout()
        stem = f"{FIG_DIR}/figS4b_gap_statistic"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_silhouette_subjects(Xa_scaled, labels):
    """Plot per-subject silhouette coefficients grouped by cluster."""
    sample_sils = silhouette_samples(Xa_scaled, labels)

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3, 4))

        y_lower = 0
        for c in range(K_BEST):
            cluster_sils = np.sort(sample_sils[labels == c])[::-1]
            ax.barh(
                range(y_lower, y_lower + len(cluster_sils)),
                cluster_sils,
                height=1.0,
                color=PALETTE[c],
                edgecolor="none",
            )
            ax.text(
                0.5,
                y_lower + len(cluster_sils) / 2,
                CLUSTER_NAMES[c],
                ha="right",
                va="center",
                fontsize=8,
            )
            y_lower += len(cluster_sils)

        ax.axvline(
            sample_sils.mean(),
            color="k",
            linestyle="--",
            linewidth=0.8,
            label=f"Mean: {sample_sils.mean():.3f}",
        )
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Subjects")
        ax.set_yticks([])
        ax.legend(fontsize=8)

        stem = f"{FIG_DIR}/figS4c_silhouette_plot"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"Mean silhouette: {sample_sils.mean():.3f}\n")
        for c in range(K_BEST):
            cluster_mean = sample_sils[labels == c].mean()
            f.write(f"  {CLUSTER_NAMES[c]}: {cluster_mean:.3f}\n")
    print(f"Saved {stem}_stats.txt")


if __name__ == "__main__":
    sens_k, Xa_scaled, labels = load_data()

    plot_silhouette_k(sens_k)
    plot_gap_statistic(sens_k)
    plot_silhouette_subjects(Xa_scaled, labels)
