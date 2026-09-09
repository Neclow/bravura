import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator

from src._config import CLUSTER_PALETTE, CLUSTERS, DEFAULT_BRMS_DIR

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG3_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG3_DIR, exist_ok=True)

CLUSTER_HUE_ORDER = [cl["name"] for cl in CLUSTERS]

POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/shocks"


def parse_contrast(contrast):
    """'(Non-aggressive) - Proactive' -> ('Non-aggressive', 'Proactive')."""
    parts = contrast.split(" - ")
    return parts[0].strip("() "), parts[1].strip("() ")


def load_data():
    """Load per-cluster shock posterior predictions and Bayes factors.

    Returns
    -------
    posterior : DataFrame
        Posterior predicted shocks per cluster and opponent.
    bf : DataFrame
        Pairwise Bayes factors for cluster contrasts.
    """
    posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv").drop(
        columns=[".row", ".draw"]
    )
    bf = pd.read_csv(f"{POSTERIOR_DIR}/bayes_factors.csv")
    return posterior, bf


def build_annotations(bf):
    """Build statannotations pairs and labels from significant BFs.

    Returns
    -------
    pairs : list of tuple
        Pairs for Annotator, e.g. (("Opponent 1", "Non-aggressive"), ...).
    labels : list of str
        BF labels for each pair.
    """
    sig = bf[bf["excl_zero"]]
    pairs = []
    labels = []
    for _, row in sig.iterrows():
        c1, c2 = parse_contrast(row["contrast"])
        bf_val = row["BF10"]
        label = "BF > 100" if bf_val > 100 else f"BF = {bf_val:.0f}"
        pairs.append(((row["opponent"], c1), (row["opponent"], c2)))
        labels.append(label)
    return pairs, labels


def plot_shocks_per_cluster(posterior, bf):
    """Plot shocks given by cluster and opponent (Fig. 3e)."""
    pairs, annot_labels = build_annotations(bf)

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        plot_kw = dict(
            data=posterior,
            x="opponent",
            y="shocks",
            hue="Cluster",
            hue_order=CLUSTER_HUE_ORDER,
        )

        sns.barplot(
            **plot_kw,
            palette=CLUSTER_PALETTE,
            errorbar=("pi", 95),
            capsize=0.1,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel("Shocks given [0-15]", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.legend().remove()
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if pairs:
            annot = Annotator(ax, pairs, **plot_kw)
            annot.set_custom_annotations(annot_labels)
            annot.annotate()

        fig.tight_layout()
        stem = f"{FIG3_DIR}/fig3e_shocks_by_opponent"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG3_DIR}/tableS7_shocks_cluster.md"
    bf.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    posterior, bf = load_data()
    plot_shocks_per_cluster(posterior, bf)
