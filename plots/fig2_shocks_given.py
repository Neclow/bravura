import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator

from src._config import DEFAULT_BRMS_DIR, DEFAULT_SHARED_DIR, N_OPPONENTS, N_TRIALS

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE, SCRIPT_FNAME

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

    script = pd.read_excel(f"{DEFAULT_SHARED_DIR}/{SCRIPT_FNAME}", index_col="Session")
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
            palette="Greys_r",
            capsize=0.05,
            errorbar=("pi", 95),
            ax=ax,
        )
        ax.set_xticklabels(
            [
                f"Opponent {i + 1}\n({shocked[i * N_TRIALS:(i + 1) * N_TRIALS].sum()} shocks)"
                for i in range(N_OPPONENTS)
            ]
        )
        ax.set_ylabel("Shocks given [0-15]")
        ax.set_xlabel("")
        ax.set_ylim(bottom=0, top=7.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))

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

    table_path = f"{FIG2_DIR}/tableS1_shocks_opponent{suffix}.md"
    bf_shocks.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        # Opponent effect: absolute difference (count scale)
        wide = shock_post.pivot_table(
            index=".draw", columns="opponent", values="shocks"
        )
        diff = wide["Opponent 2"] - wide["Opponent 1"]
        f.write(
            f"Difference: {diff.mean():.1f} [{diff.quantile(0.025):.1f}, {diff.quantile(0.975):.1f}]"
        )
        f.write("\n")

        # Opponent effect: odds ratio
        log_or = -bf_shocks.query("cohort == 'A'")["estimate"].values[
            0
        ]  # flip sign for Opp2/Opp1
        or_val = np.exp(log_or)
        or_lo = np.exp(-bf_shocks.query("cohort == 'A'")["Q97.5"].values[0])
        or_hi = np.exp(-bf_shocks.query("cohort == 'A'")["Q2.5"].values[0])
        f.write(f"OR (Opp2/Opp1): {or_val:.2f} [{or_lo:.2f}, {or_hi:.2f}]")
        f.write("\n")
    print(f"Saved {stem}_stats.txt")


if __name__ == "__main__":
    plot_shocks_given("A")
