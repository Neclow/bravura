import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator

from src._config import DEFAULT_BRMS_DIR, N_OPPONENTS

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/beliefs_overview"
FIG2_DIR = f"{DEFAULT_IMG_DIR}/fig2"
os.makedirs(FIG2_DIR, exist_ok=True)


def plot_belief_vs_opponent(cohort):
    """Plot belief vs. opponent with posterior predictions.

    Parameters
    ----------
    cohort : str
        Cohort label, e.g. "A" or "B".
    """
    suffix = f"_{cohort.lower()}"

    belief_posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv")
    bf_beliefs = pd.read_csv(f"{POSTERIOR_DIR}/bayes_factors.csv")

    belief_post = belief_posterior[belief_posterior["cohort"] == cohort]

    stem = f"{FIG2_DIR}/fig2c_belief_vs_opponent{suffix}"

    with plt.style.context(DEFAULT_STYLE):
        plot_dict = {
            "data": belief_post,
            "x": "opponent",
            "y": "belief",
        }

        fig, ax = plt.subplots(figsize=(2.5, 3))
        sns.barplot(
            **plot_dict,
            hue="opponent",
            palette=["white", "grey"],
            edgecolor="black",
            capsize=0.15,
            err_kws={"linewidth": 1},
            errorbar=("pi", 95),
            legend=False,
            ax=ax,
        )
        ax.set_xticklabels(
            [f"{i + 1}" for i in range(N_OPPONENTS)],
            fontweight="bold",
        )
        ax.set_ylabel("Reported belief [0-10]", fontweight="bold")
        ax.set_xlabel("Opponent", fontweight="bold")
        ax.set_ylim(bottom=0, top=10)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        bf = bf_beliefs[(bf_beliefs["cohort"] == cohort) & bf_beliefs["excl_zero"]]
        if not bf.empty:
            annotator = Annotator(
                ax,
                pairs=[("Opponent 1", "Opponent 2")],
                **plot_dict,
                verbose=False,
            )
            annotator.configure(loc="outside", verbose=False)
            annotator.set_custom_annotations(["*"])
            annotator.annotate()

        ax.yaxis.set_major_locator(plt.MultipleLocator(2.5))
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG2_DIR}/tableS5_beliefs_opponent.md"
    bf_beliefs.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    plot_belief_vs_opponent("A")
