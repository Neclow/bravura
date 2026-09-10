import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator

from src._config import DEFAULT_BRMS_DIR, N_OPPONENTS

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE, SCRIPT_PATH

POSTERIOR_DIR = f"{DEFAULT_BRMS_DIR}/shocks_overview"
FIG2_DIR = f"{DEFAULT_IMG_DIR}/fig2"
os.makedirs(FIG2_DIR, exist_ok=True)


def plot_shocks_given(cohort):
    """Plot shocks given vs. opponent with posterior predictions.

    Parameters
    ----------
    cohort : str
        Cohort label, e.g. "A" or "B".
    """
    suffix = f"_{cohort.lower()}"

    script = pd.read_excel(SCRIPT_PATH, index_col="Session")
    shocked = script.loc["Shocked"].values.astype(int)

    shock_posterior = pd.read_csv(f"{POSTERIOR_DIR}/posterior_epred.csv")
    bf_shocks = pd.read_csv(f"{POSTERIOR_DIR}/bayes_factors.csv")

    shock_post = shock_posterior[shock_posterior["cohort"] == cohort]

    stem = f"{FIG2_DIR}/fig2b_shocks_given{suffix}"

    with plt.style.context(DEFAULT_STYLE):
        plot_dict = {
            "data": shock_post,
            "x": "opponent",
            "y": "shocks",
        }

        _, ax = plt.subplots(figsize=(2.5, 3))
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
        ax.set_ylabel("Shocks given [0-15]", fontweight="bold")
        ax.set_xlabel("Opponent", fontweight="bold")
        ax.set_ylim(bottom=0, top=7.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        bf = bf_shocks[(bf_shocks["cohort"] == cohort) & bf_shocks["excl_zero"]]
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

        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG2_DIR}/tableS4_shocks_given.md"
    bf_shocks.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    plot_shocks_given("A")
