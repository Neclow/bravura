import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.io import loadmat
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample

from src._config import (
    CLUSTER_PALETTE,
    CLUSTERS,
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_DIR,
)

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE, SCRIPT_PATH

FIG3_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG3_DIR, exist_ok=True)


def load_data(cohort):
    cohort_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"
    pred = pd.read_csv(f"{cohort_dir}/predictions.csv", header=None)
    actual_vba = pd.read_csv(f"{cohort_dir}/decisions.csv", header=None)
    ids = pd.read_csv(f"{cohort_dir}/subject_ids.csv")
    outliers = loadmat(f"{cohort_dir}/outliers.mat", squeeze_me=True)["outliers"]

    # Align indices (excluding outliers)
    pred.index = ids["subject"]
    actual_vba.index = ids["subject"]
    included = pred.index[~pred.index.isin(outliers)]
    pred_clean = pred.loc[included].values
    actual_clean = actual_vba.loc[included].values

    # Sort by total shocks
    sort_order = actual_clean.sum(axis=1).argsort()
    actual_sorted = actual_clean[sort_order]
    pred_sorted = pred_clean[sort_order]

    script = pd.read_excel(SCRIPT_PATH, index_col="Session")
    shocked = script.loc["Shocked"].values.astype(int)
    wins = script.loc["Win"].values.astype(int)

    corr_preds = loadmat(f"{cohort_dir}/corr_preds.mat")[
        "corr_preds"
    ]  # (n_simulations, 4, 4)

    sim_rec = loadmat(f"{cohort_dir}/cov_stats.mat", squeeze_me=True)
    cov_stats = sim_rec["cov_stats"]  # (126, 2): [determinant, condition_number]

    return actual_sorted, pred_sorted, shocked, wins, corr_preds, cov_stats


def plot_decisions(actual_sorted, pred_sorted, shocked, wins):
    # Plot
    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(5, 4.2),
            gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.08},
            layout="constrained",
        )

        cmap = "YlOrRd"

        sns.heatmap(
            actual_sorted,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            ax=axes[0],
        )
        axes[0].set_title("Observed", y=1.02, fontweight="bold")
        axes[0].set_ylabel("Subject (sorted by total shocks given)", fontweight="bold")
        sns.heatmap(
            pred_sorted,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            ax=axes[1],
        )
        axes[1].set_title("Predicted", y=1.02, fontweight="bold")
        ticks = [1, 5, 10, 15, 20, 25, 30]
        for ax in axes[:-1]:
            ax.set_xticks([t - 0.5 for t in ticks])
            ax.set_xticklabels(ticks, fontweight="bold")
            ax.set_xlabel("Trial", fontweight="bold")
            ax.axvline(x=15, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_yticks([])
            for i in range(len(actual_sorted)):
                ax.axhline(y=i, color="k", linewidth=0.1)
        for t in range(30):
            if shocked[t]:
                ax.plot(
                    t + 0.5,
                    -2.0,
                    marker="v",
                    color="orange",
                    markersize=4,
                    clip_on=False,
                )
            if not wins[t]:
                ax.plot(
                    t + 0.5,
                    -0.5,
                    marker="x",
                    color="red",
                    markersize=3,
                    clip_on=False,
                    markeredgewidth=1,
                )
        axes[1].text(
            31.5, -0.25, "Trial lost", color="red", va="center", fontweight="bold"
        )
        axes[1].text(
            31.5,
            -2.25,
            "Shock received",
            color="orange",
            va="center",
            fontweight="bold",
        )

        cbar = fig.colorbar(axes[1].collections[0], cax=axes[2], label="P(shock)")
        cbar.set_label("P(shock)", fontweight="bold")
        axes[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        for label in axes[2].get_yticklabels():
            label.set_fontweight("bold")
        axes[2].set_position(
            [axes[2].get_position().x0, 0.3, axes[2].get_position().width, 0.3]
        )
        stem = f"{FIG3_DIR}/fig3a_heatmaps"
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_roc(actual_sorted, pred_sorted):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # Individual ROCs
        # pylint: disable=consider-using-enumerate
        for i in range(len(pred_sorted)):
            y_true = actual_sorted[i]
            y_pred = pred_sorted[i]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            aucs.append(roc_auc_score(y_true, y_pred))
            ax.plot(fpr, tpr, color="lightcoral", alpha=0.1, linewidth=0.5)
            # Interpolate to common FPR grid for averaging
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        # pylint: enable=consider-using-enumerate

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        # mean_auc = roc_auc_score(actual_sorted.flatten(), pred_sorted.flatten())
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="darkred",
            linewidth=2,
            label=f"Mean (AUC = {np.mean(aucs):.3f})",
        )

        # 0.5 line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.8)

        # Bootstrap confidence intervals for mean AUC
        n_bootstrap = 1000
        tprs_array = np.array(tprs)  # (n_subjects, 100)

        boot_means = []
        for _ in range(n_bootstrap):
            idx = resample(np.arange(len(tprs_array)), replace=True)
            boot_means.append(tprs_array[idx].mean(axis=0))

        boot_means = np.array(boot_means)
        tpr_lower = np.percentile(boot_means, 2.5, axis=0)
        tpr_upper = np.percentile(boot_means, 97.5, axis=0)

        ax.fill_between(
            mean_fpr, tpr_lower, tpr_upper, color="red", alpha=0.15, label="95% CI"
        )

        ax.set_xlabel("False positive rate", fontweight="bold")
        ax.set_ylabel("True positive rate", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontweight("bold")
        stem = f"{FIG3_DIR}/figS6a_roc_curve"
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_calibration(actual_sorted, pred_sorted):
    y_true = actual_sorted.flatten()
    y_pred = pred_sorted.flatten()

    # Remove NaN if any
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)

    fraction_pos, mean_predicted = calibration_curve(
        y_true[mask], y_pred[mask], n_bins=10
    )

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax.plot(
            mean_predicted,
            fraction_pos,
            "o-",
            color="darkred",
            markersize=4,
            linewidth=1.5,
            label="Model",
        )
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration"
        )
        ax.fill_between(
            [0, 1],
            [0, 0],
            [1, 1],
            where=[True, True],
            alpha=0.03,
            color="gray",
        )
        ax.set_xlabel("Predicted P(shock)", fontweight="bold")
        ax.set_ylabel("Observed fraction of shocks", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        legend = ax.legend(loc="lower right")
        for text in legend.get_texts():
            text.set_fontweight("bold")
        ax.grid(alpha=0.3)
        stem = f"{FIG3_DIR}/figS6b_calibration"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_vba_corr_matrix(corr_preds):
    # Average across simulations
    corr_mean = corr_preds.mean(axis=0)
    mask = np.triu(np.ones(corr_mean.shape), k=0).astype(bool)
    labels = [r"$K_{r_1}$", r"$K_{r_c}$", r"$K_p$", r"$K_{w_c}$"]

    # Annotation: mean values as text
    annot = np.array([[f"{corr_mean[i, j]:.2f}" for j in range(4)] for i in range(4)])

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(
            corr_mean,
            mask=mask,
            annot=annot,
            fmt="",
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Pearson's $r$", "shrink": 0.6},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar.ax.get_ylabel(), fontweight="bold")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_yticks(ax.get_yticks()[1:])
        ax.set_xticks(ax.get_xticks()[:-1])
        # ax.set_title("Model parameter correlations\n(simulation-recovery)", fontweight="bold")

        stem = f"{FIG3_DIR}/figS2a_vba_corr_matrix"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

        with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
            f.write(f"Correlation matrix (mean across {len(corr_preds)} subjects):\n")
            f.write(np.array2string(np.round(corr_mean, 3), separator=", "))
            f.write("\n")
        print(f"Saved {stem}_stats.txt")


def plot_vba_cov_stats(cov_stats):
    cond_numbers = cov_stats[:, 1]

    determinants = cov_stats[:, 0]
    ranks = (determinants > 1e-10).astype(int) * 4  # full rank if det > 0

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

        sns.histplot(
            cond_numbers,
            bins=20,
            color="k",
            edgecolor="white",
            linewidth=0.5,
            kde=True,
            ax=ax,
        )
        ax.set_xlabel(r"Cond($\Sigma$)", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        # axes[0].hist(cond_numbers, bins=20, color="k", edgecolor="white", linewidth=0.5)
        ax.axvline(
            np.median(cond_numbers),
            color="red",
            linestyle="--",
            label=f"median = {np.median(cond_numbers):.1f}",
        )
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_fontweight("bold")

        fig.tight_layout()
        stem = f"{FIG3_DIR}/figS2b_condition_numbers"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

        with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Condition number: median={np.median(cond_numbers):.1f}, "
                f"min={cond_numbers.min():.1f}, max={cond_numbers.max():.1f}\n"
            )
            f.write(f"All full rank: {(ranks == 4).all()}\n")
        print(f"Saved {stem}_stats.txt")


COEF_LABELS = {
    "Kp": r"$K_p$ (baseline)",
    "Kr1": r"$K_{r_1}$ (immediate reaction)",
    "Krc": r"$K_{r_c}$ (cumulative reaction)",
    "Kwc": r"$K_{w_c}$ (win-loss response)",
}


def plot_coef_distributions(cohort="a"):
    """Plot VBA coefficient distributions for included subjects (Fig. S7)."""
    cohort_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"
    coefs = pd.read_csv(f"{cohort_dir}/coefficients.csv", index_col="Row")

    ids = pd.read_csv(f"{cohort_dir}/subject_ids.csv")
    outliers = loadmat(f"{cohort_dir}/outliers.mat", squeeze_me=True)["outliers"]
    included = ids["subject"][~ids["subject"].isin(outliers)]
    coefs = coefs.loc[included]

    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=True)

        for ax, (col, label) in zip(axes, COEF_LABELS.items()):
            sns.histplot(
                coefs[col], kde=True, color="k", ax=ax,
                bins=15, edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel(label, fontweight="bold")
            ax.set_ylabel("")
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        axes[0].set_ylabel("Count", fontweight="bold")
        plt.tight_layout()
        stem = f"{FIG3_DIR}/figS7_coef_distributions"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"N = {len(coefs)}\n\n")
        f.write(coefs.describe().round(3).to_string())
        f.write("\n")
    print(f"Saved {stem}_stats.txt")


def load_trial_pshock_data(cohort="a"):
    """Load per-cluster trial-by-trial P(shock) summaries.

    Returns
    -------
    cluster_means : DataFrame
        Mean predicted P(shock) per cluster per trial.
    cluster_sems : DataFrame
        SEM of predicted P(shock) per cluster per trial.
    shocked : ndarray
        Binary opponent-shock schedule (length 30).
    """
    cohort_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"
    pred = pd.read_csv(f"{cohort_dir}/predictions.csv", header=None)
    ids = pd.read_csv(f"{cohort_dir}/subject_ids.csv")
    pred.index = ids["subject"]

    Xa = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")
    pred_cluster = pred.loc[Xa.index].copy()
    pred_cluster["Cluster"] = Xa["Cluster"].values

    cluster_means = pred_cluster.groupby("Cluster").mean()
    cluster_sems = pred_cluster.groupby("Cluster").sem()

    script = pd.read_excel(SCRIPT_PATH, index_col="Session")
    shocked = script.loc["Shocked"].values.astype(int)

    return cluster_means, cluster_sems, shocked


def plot_trial_pshock(cluster_means, cluster_sems, shocked):
    """Plot trial-by-trial P(shock) by cluster (Fig. 3d)."""
    markers = ["o", "s", "D"]
    linestyles = ["-", "--", ":"]

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(8.12, 3.5))

        x = np.arange(1, 31)
        for i, cl in enumerate(CLUSTERS):
            name = cl["name"]
            mean = cluster_means.loc[name].values
            sem = cluster_sems.loc[name].values
            color = CLUSTER_PALETTE[name]
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
            ax.plot(
                x,
                mean,
                color=color,
                marker=markers[i],
                linestyle=linestyles[i],
                label=name,
                markersize=4,
                linewidth=1.5,
            )

        for t in range(30):
            if shocked[t]:
                ax.plot(
                    t + 1, -0.03, marker="v", color="red", markersize=4, clip_on=False
                )

        ax.axvline(x=15.5, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.text(8, 1.05, "Opponent 1", ha="center", fontsize=10, fontweight="bold")
        ax.text(23, 1.05, "Opponent 2", ha="center", fontsize=10, fontweight="bold")

        ax.set_xlabel("Trial", fontweight="bold")
        ax.set_ylabel("P(shock)", fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0.5, 30.5)
        legend = ax.legend(title="", frameon=False, bbox_to_anchor=(1, 0.95))
        for text in legend.get_texts():
            text.set_fontweight("bold")
        ax.set_axisbelow(True)

        stem = f"{FIG3_DIR}/fig3d_trial_pshock"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


if __name__ == "__main__":
    actual_sorted, pred_sorted, shocked, wins, vba_corr_preds, vba_cov_stats = (
        load_data(cohort="a")
    )

    # Fig 3a
    plot_decisions(actual_sorted, pred_sorted, shocked, wins)

    # Fig 3d
    cl_means, cl_sems, shocked_schedule = load_trial_pshock_data()
    plot_trial_pshock(cl_means, cl_sems, shocked_schedule)

    # Fig S2a
    plot_roc(actual_sorted, pred_sorted)

    # Fig S2b
    plot_calibration(actual_sorted, pred_sorted)

    # Fig S3a
    plot_vba_corr_matrix(vba_corr_preds)

    # Fig S3b
    plot_vba_cov_stats(vba_cov_stats)

    # Fig S7
    plot_coef_distributions()
