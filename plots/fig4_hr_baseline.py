import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src._config import CLUSTER_PALETTE, DEFAULT_BRMS_DIR, DEFAULT_PHYSIO_DIR_A

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG4_DIR = f"{DEFAULT_IMG_DIR}/fig4"
os.makedirs(FIG4_DIR, exist_ok=True)

CLUSTER_HUE_ORDER = ["Non-aggressive", "Reactive", "Proactive"]

POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/baseline_hr"


def load_data():
    """Load baseline HR posterior predictions and raw observed values.

    Returns
    -------
    posterior : DataFrame
        Posterior predicted baseline HR per cluster.
    observed : DataFrame
        Raw baseline HR per subject with cluster labels.
    """
    posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv").drop(
        columns=[".row", ".draw"], errors="ignore"
    )
    observed = pd.read_csv(f"{DEFAULT_PHYSIO_DIR_A}/baseline_hr.csv")
    return posterior, observed


def plot_baseline_hr(posterior, observed):
    """Plot baseline HR by cluster: posterior bars + raw swarmplot (Fig. 4a)."""
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3.5))

        sns.barplot(
            data=posterior,
            x="Cluster",
            y="HR_Pre",
            order=CLUSTER_HUE_ORDER,
            palette=CLUSTER_PALETTE,
            errorbar=("pi", 95),
            capsize=0.15,
            err_kws={"linewidth": 1},
            alpha=0.9,
            ax=ax,
        )
        sns.swarmplot(
            data=observed,
            x="Cluster",
            y="HR_Pre",
            order=CLUSTER_HUE_ORDER,
            color="k",
            size=3,
            alpha=0.25,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel("Baseline HR (bpm)", fontweight="bold")
        ax.set_axisbelow(True)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        fig.tight_layout()
        stem = f"{FIG4_DIR}/fig4a_baseline_hr"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


if __name__ == "__main__":
    posterior, observed = load_data()
    plot_baseline_hr(posterior, observed)
