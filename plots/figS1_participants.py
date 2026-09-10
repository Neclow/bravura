# pylint: disable=redefined-outer-name

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src._config import DEFAULT_DATA_DIR, MAX_SHOCKS, MIN_BELIEF, MIN_SHOCKS

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

PERFORMANCE_FNAME = "aggroPerformance.xlsx"
BELIEF_FNAME = "beliefs.xlsx"
FIG2_DIR = f"{DEFAULT_IMG_DIR}/fig2"
os.makedirs(FIG2_DIR, exist_ok=True)


def plot_shock_vs_belief(cohort, belief_scale=1):
    """Plot shock vs. belief joint distribution with outlier exclusion.

    Parameters
    ----------
    cohort : str
        Cohort subfolder name, e.g. "cohort_a" or "cohort_b".
    belief_scale : int
        Multiplier for raw belief scores (Cohort B uses 2).
    """
    cohort_key = cohort.split("_")[-1]
    suffix = f"_{cohort_key}"
    cohort_dir = f"{DEFAULT_DATA_DIR}/{cohort}"

    aggro_performance = pd.read_excel(
        f"{cohort_dir}/{PERFORMANCE_FNAME}", index_col="Subject"
    )
    shock_cols_only = [c for c in aggro_performance.columns if c.startswith("shock")]
    total_shocks = aggro_performance[shock_cols_only].sum(axis=1)

    opponent_beliefs = pd.read_excel(f"{cohort_dir}/{BELIEF_FNAME}", index_col="ID")
    if sorted(opponent_beliefs.columns) != ["opponent1", "opponent2"]:
        opponent_beliefs.rename(
            columns={k: k[4:] + k[0] for k in opponent_beliefs.columns}, inplace=True
        )
    opponent_beliefs.drop("opponent3", axis=1, errors="ignore", inplace=True)
    if belief_scale != 1:
        opponent_beliefs = opponent_beliefs.mul(belief_scale)

    mean_belief = opponent_beliefs.mean(axis=1)

    common_subj = total_shocks.index.intersection(mean_belief.dropna().index)
    total_shocks = total_shocks.loc[common_subj]
    mean_belief = mean_belief.loc[common_subj]

    with open(f"{cohort_dir}/outliers.txt", encoding="utf-8") as f:
        outlier_ids = [line.strip() for line in f if line.strip()]
    outliers = total_shocks.index.isin(outlier_ids)

    with plt.style.context(DEFAULT_STYLE):
        g = sns.JointGrid(x=total_shocks, y=mean_belief, height=4)

        sns.kdeplot(
            x=total_shocks,
            y=mean_belief,
            cut=1,
            cmap=sns.color_palette("Blues", as_cmap=True),
            fill=True,
            alpha=0.9,
            ax=g.ax_joint,
        )
        g.ax_joint.scatter(
            total_shocks[~outliers],
            mean_belief[~outliers],
            color="k",
            s=10,
            alpha=0.5,
            zorder=3,
        )

        np.random.seed(42)
        jitter_y = np.abs(np.random.normal(0, 0.2, size=outliers.sum()))
        g.ax_joint.scatter(
            total_shocks[outliers],
            mean_belief[outliers] + jitter_y,
            color="darkred",
            edgecolor="k",
            s=30,
            alpha=0.8,
            zorder=5,
            label=f"Excluded (N={outliers.sum()})",
        )

        for x_val in (MIN_SHOCKS, MAX_SHOCKS):
            g.ax_joint.plot(
                [x_val, x_val],
                [0, MIN_BELIEF],
                linestyle="--",
                color="k",
                alpha=0.5,
                linewidth=0.8,
            )
        for x_range in ([0, MIN_SHOCKS], [MAX_SHOCKS, 30]):
            g.ax_joint.plot(
                x_range,
                [MIN_BELIEF, MIN_BELIEF],
                linestyle="--",
                color="k",
                alpha=0.5,
                linewidth=0.8,
            )

        sns.histplot(
            x=total_shocks, ax=g.ax_marg_x, color="gray", bins=15, edgecolor="white"
        )
        sns.histplot(
            y=mean_belief, ax=g.ax_marg_y, color="gray", bins=10, edgecolor="white"
        )

        g.ax_joint.set_xlabel("Total shocks given [0-30]")
        g.ax_joint.set_ylabel("Mean belief [0-10]")
        g.ax_joint.legend(fontsize=7, loc="upper right")

        stem = f"{FIG2_DIR}/figS1_shock_vs_belief{suffix}"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {FIG2_DIR}/shock_vs_belief{suffix}.pdf/.png")
        plt.show()



if __name__ == "__main__":
    plot_shock_vs_belief("cohort_a", belief_scale=1)
    plot_shock_vs_belief("cohort_b", belief_scale=2)
