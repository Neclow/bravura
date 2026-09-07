import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Ellipse, Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src._config import (
    CLUSTERS,
    DEFAULT_CLUSTERING_FEATURES,
    DEFAULT_PROCESSED_DIR,
    PALETTE,
    RANDOM_SEED,
)

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG3_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG3_DIR, exist_ok=True)


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=angle, **kwargs))


def load_data():
    """Load PCA-projected cluster data for both cohorts.

    Returns
    -------
    Xa : DataFrame
        Cohort A with PC1, PC2, label columns.
    Xb : DataFrame
        Cohort B with PC1, PC2, label columns.
    centroids_pca : ndarray
        Cluster centroids in PCA space (indexed by label).
    explained_variance_ratio : ndarray
        Explained variance ratio for PC1 and PC2.
    """
    Xa = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")
    Xb = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xb.csv", index_col="Row")

    Xa_scaled = StandardScaler().fit_transform(Xa[DEFAULT_CLUSTERING_FEATURES])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(Xa_scaled)

    n_clusters = len(CLUSTERS)
    centroids_pca = np.empty((n_clusters, 2))
    for cl in CLUSTERS:
        c = cl["label"]
        mask = Xa["label"] == c
        centroids_pca[c] = Xa.loc[mask, ["PC1", "PC2"]].mean().values

    return Xa, Xb, centroids_pca, pca.explained_variance_ratio_


def plot_pca(Xa, Xb, centroids_pca, explained_variance_ratio):
    """Plot PCA scatter with cluster ellipses and centroids (Fig. 3b)."""
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 3.5))

        plot_data = dict(x="PC1", y="PC2", hue="label", palette=PALETTE)

        sns.scatterplot(**plot_data, data=Xa, ax=ax, legend=False)
        sns.scatterplot(**plot_data, data=Xb, alpha=0.3, ax=ax, legend=False)

        for cl in CLUSTERS:
            c = cl["label"]
            mask = Xa["label"] == c

            confidence_ellipse(
                Xa.loc[mask, "PC1"].values,
                Xa.loc[mask, "PC2"].values,
                ax,
                n_std=2.0,
                facecolor=PALETTE[c],
                alpha=0.15,
                edgecolor=PALETTE[c],
                linewidth=1,
            )

            ax.annotate(
                f"n={mask.sum()}",
                xy=cl["annot_xy"],
                fontweight="bold",
                ha="center",
                color=PALETTE[c],
            )

            ax.scatter(
                centroids_pca[c, 0],
                centroids_pca[c, 1],
                marker="X",
                s=30,
                color=PALETTE[c],
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )

        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%})")

        handles = [
            (
                Patch(facecolor=PALETTE[cl["label"]]),
                Patch(facecolor=PALETTE[cl["label"]], alpha=0.3),
            )
            for cl in CLUSTERS
        ]
        ax.legend(
            handles,
            [cl["name"] for cl in CLUSTERS],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
            loc="upper center",
            bbox_to_anchor=(0.9, 1.0),
            frameon=True,
        )

        lim = 4
        ticks = np.linspace(-lim, lim, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        stem = f"{FIG3_DIR}/fig3b_pca_clusters"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


if __name__ == "__main__":
    Xa, Xb, centroids_pca, evr = load_data()
    plot_pca(Xa, Xb, centroids_pca, evr)
