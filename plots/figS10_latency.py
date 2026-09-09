import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src._config import CLUSTER_PALETTE, CLUSTERS, DEFAULT_BRMS_DIR

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG_DIR, exist_ok=True)

CLUSTER_HUE_ORDER = [cl["name"] for cl in CLUSTERS]

POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/latency"


def load_data():
    """Load shock-latency posterior predictions and Bayes factors.

    Returns
    -------
    posterior : DataFrame
        Posterior predicted latencies per cluster and opponent.
    bf : DataFrame
        Pairwise Bayes factors for cluster contrasts.
    """
    posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv").drop(
        columns=[".row", ".draw"], errors="ignore"
    )
    bf = pd.read_csv(f"{POSTERIOR_DIR}/bayes_factors.csv")
    return posterior, bf


def plot_latency(posterior, bf):
    """Plot shock latency by cluster and opponent."""
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        sns.barplot(
            data=posterior,
            x="opponent",
            y="latency",
            hue="Cluster",
            hue_order=CLUSTER_HUE_ORDER,
            palette=CLUSTER_PALETTE,
            errorbar=("pi", 95),
            capsize=0.1,
            ax=ax,
        )

        ax.set_xlabel("Opponent", fontweight="bold")
        ax.set_ylabel("Shock latency (s)", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = ax.legend(
            title="", frameon=False, bbox_to_anchor=(1, 1), loc="upper left"
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.tight_layout()
        stem = f"{FIG_DIR}/figS10_latency"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG_DIR}/figS10_latency_bf.md"
    bf[["opponent", "contrast", "BF10", "excl_zero"]].set_index("contrast").round(
        3
    ).to_markdown(table_path)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    posterior, bf = load_data()
    plot_latency(posterior, bf)
