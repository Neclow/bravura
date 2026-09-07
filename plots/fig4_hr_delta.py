import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from statannotations.Annotator import Annotator

from src._config import (
    CLUSTER_NAMES,
    CLUSTER_PALETTE,
    CLUSTERS,
    DEFAULT_BRMS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_DIR,
    PALETTE,
)

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG4_DIR = f"{DEFAULT_IMG_DIR}/fig4"
os.makedirs(FIG4_DIR, exist_ok=True)

CLUSTER_HUE_ORDER = [cl["name"] for cl in CLUSTERS]
CLUSTER_ORDER = [cl["label"] for cl in CLUSTERS]

HR_COLS = ["HR_Pre", "HR_Op1T1", "HR_Op1T2", "HR_Op2T1", "HR_Op2T2"]
BLOCK_LABELS = ["Pre", "1.1", "1.2", "2.1", "2.2"]

BF_THRESHOLD = 10**0.5


def parse_contrast(contrast):
    """'(Non-aggressive) - Proactive' -> ('Non-aggressive', 'Proactive')."""
    parts = contrast.split(" - ")
    return parts[0].strip("() "), parts[1].strip("() ")


# -- Data loaders -------------------------------------------------------------


def load_delta_hr_data():
    """Load delta-HR brms posterior and Bayes factors (Fig. 4b).

    Returns
    -------
    posterior : DataFrame
        Posterior predicted delta-HR per cluster and block.
    bf : DataFrame
        Pairwise Bayes factors for cluster contrasts.
    """
    brms_dir = f"{DEFAULT_BRMS_DIR}/delta_hr"
    posterior = pd.read_csv(f"{brms_dir}/posterior_epred.csv").drop(
        columns=[".row", ".draw"]
    )
    bf = pd.read_csv(f"{brms_dir}/bayes_factors.csv")
    return posterior, bf


def load_delta1_data():
    """Load delta1-HR (block 1.1 - Pre) for both cohorts (Fig. 4c).

    Returns
    -------
    delta1 : DataFrame
        Delta1-HR per subject with Cluster and cohort columns.
    """
    delta_a = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/delta_hr_long.csv")
    delta_b = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/delta_hr_long_b.csv")

    d1_a = delta_a[delta_a["block"] == "1.1"][["subject", "Cluster", "delta_hr"]].copy()
    d1_a["cohort"] = "A"
    d1_b = delta_b[delta_b["block"] == "1.1"][["subject", "Cluster", "delta_hr"]].copy()
    d1_b["cohort"] = "B"

    return pd.concat([d1_a, d1_b], ignore_index=True)


def load_hr_timecourse_data():
    """Load raw HR data and compute per-cluster time course stats for Fig. S7.

    Returns
    -------
    hr_stats : DataFrame
        Mean and SEM of HR per cluster and block.
    grand_stats : DataFrame
        Grand mean and SEM of HR per block (all subjects).
    physio_hr : DataFrame
        Raw HR per subject and block (for heatmap).
    labels : Series
        Cluster labels aligned to physio_hr index.
    """
    behav = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")
    physio = pd.read_excel(
        f"{DEFAULT_DATA_DIR}/cohort_a/physPerformance.xlsx", index_col="Subject"
    )

    shared = behav.index.intersection(physio.index)
    physio_hr = physio.loc[shared, HR_COLS].dropna()
    labels = behav.loc[physio_hr.index, "label"]

    hr_long = physio_hr.copy()
    hr_long["label"] = labels
    hr_long = hr_long.melt(
        id_vars="label", value_vars=HR_COLS, var_name="block", value_name="HR"
    )
    hr_long["block"] = pd.Categorical(
        hr_long["block"].map(dict(zip(HR_COLS, BLOCK_LABELS))),
        categories=BLOCK_LABELS,
        ordered=True,
    )

    hr_stats = (
        hr_long.groupby(["label", "block"], observed=True)["HR"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    hr_stats["Cluster"] = hr_stats["label"].map(CLUSTER_NAMES)

    grand_stats = hr_long.groupby("block", observed=True)["HR"].agg(["mean", "sem"])

    return hr_stats, grand_stats, physio_hr, labels


def load_replication_bf():
    """Load replication Bayes factors (Supp. Table 8).

    Returns
    -------
    rep_bf : DataFrame
        Replication BFs for delta-HR contrasts.
    """
    return pd.read_csv(f"{DEFAULT_BRMS_DIR}/delta_hr_replication/replication_bf.csv")


# -- Annotation helpers --------------------------------------------------------


def build_delta_hr_annotations(bf):
    """Build annotation pairs from significant delta-HR BFs.

    Returns
    -------
    pairs : list of tuple
    labels : list of str
    """
    sig = bf[(bf["excl_zero"]) & (bf["BF10"] >= BF_THRESHOLD)]
    pairs = []
    labels = []
    for _, row in sig.iterrows():
        c1, c2 = parse_contrast(row["contrast"])
        bf_val = row["BF10"]
        label = "BF > 100" if bf_val > 100 else f"BF = {bf_val:.1f}"
        pairs.append(((row["block"], c1), (row["block"], c2)))
        labels.append(label)
    return pairs, labels


# -- Plot functions ------------------------------------------------------------


def plot_delta_hr(posterior, bf):
    """Plot delta-HR by cluster x block with BF annotations (Fig. 4b)."""
    pairs, annot_labels = build_delta_hr_annotations(bf)

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        plot_kw = dict(
            data=posterior,
            x="block",
            y="delta_hr",
            hue="Cluster",
            hue_order=CLUSTER_HUE_ORDER,
        )

        sns.barplot(
            **plot_kw,
            palette=CLUSTER_PALETTE,
            errorbar=("pi", 95),
            capsize=0.05,
            ax=ax,
        )

        ax.set_xlabel("Block")
        ax.set_ylabel(r"$\Delta$HR (bpm)")
        ax.legend(title="", loc="upper right", frameon=False)
        ax.set_axisbelow(True)

        if pairs:
            annot = Annotator(ax, pairs, **plot_kw)
            annot.configure(text_offset=2.0)
            annot.set_custom_annotations(annot_labels)
            annot.annotate()

        fig.tight_layout()
        stem = f"{FIG4_DIR}/fig4b_delta_hr"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG4_DIR}/tableS7_delta_hr.md"
    bf.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


def plot_delta1_cohorts(delta1):
    """Plot delta1-HR by cluster, Cohort A vs B boxplot (Fig. 4c)."""
    n_cohorts = 2
    colors = [CLUSTER_PALETTE[c] for c in CLUSTER_HUE_ORDER]

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 3.5))

        plot_kw = dict(
            data=delta1,
            hue="Cluster",
            y="delta_hr",
            x="cohort",
            order=["A", "B"],
            hue_order=CLUSTER_HUE_ORDER,
        )

        sns.boxplot(
            **plot_kw,
            palette=CLUSTER_PALETTE,
            showfliers=False,
            ax=ax,
        )

        box_patches = [p for p in ax.patches if isinstance(p, PathPatch)]
        for i, patch in enumerate(box_patches):
            cluster_idx = i // n_cohorts
            cohort_idx = i % n_cohorts
            c = colors[cluster_idx]
            alpha = 1.0 if cohort_idx == 0 else 0.4
            patch.set_facecolor(to_rgba(c, alpha))
            patch.set_edgecolor(c)

        sns.swarmplot(
            **plot_kw,
            dodge=True,
            palette="dark:k",
            size=3,
            alpha=0.8,
            ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel(r"$\Delta_1$HR (bpm)")
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="", frameon=True)

        fig.tight_layout()
        stem = f"{FIG4_DIR}/fig4c_delta1_hr_cohorts"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_hr_timecourse(hr_stats, grand_stats):
    """Plot HR time course per cluster with grand mean (Fig. S7)."""
    markers = ["o", "s", "D"]
    linestyles = ["-", "--", ":"]

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.grid(which="major", axis="y", color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)

        x = np.arange(len(BLOCK_LABELS))

        gm = grand_stats["mean"].values
        gsem = grand_stats["sem"].values
        ax.fill_between(x, gm - gsem, gm + gsem, color="0.8", alpha=0.5, zorder=0)
        ax.plot(
            x,
            gm,
            color="0.6",
            linestyle="-",
            linewidth=1,
            zorder=1,
            label="Grand mean",
        )

        for i, c in enumerate(CLUSTER_ORDER):
            mask = hr_stats["label"] == c
            mean = hr_stats.loc[mask, "mean"].values
            name = CLUSTER_NAMES[c]
            color = CLUSTER_PALETTE[name]

            ax.plot(
                x,
                mean,
                color=color,
                marker=markers[i],
                linestyle=linestyles[i],
                label=name,
                markersize=6,
                linewidth=1.5,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(BLOCK_LABELS)
        ax.set_xlabel("Block")
        ax.set_ylabel("Heart rate (bpm)")
        ax.legend(title="", frameon=False, ncol=4, columnspacing=0.5)

        fig.tight_layout()
        stem = f"{FIG4_DIR}/figS7_hr_timecourse"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def plot_delta_hr_heatmap(physio_hr, labels):
    """Plot per-subject delta-HR heatmap grouped by cluster (Fig. S7b)."""
    clusters = labels.map(CLUSTER_NAMES)

    delta_wide = physio_hr[HR_COLS[1:]].sub(physio_hr["HR_Pre"], axis=0)
    delta_wide.columns = BLOCK_LABELS[1:]
    delta_wide["Cluster"] = pd.Categorical(
        clusters, categories=CLUSTER_HUE_ORDER, ordered=True
    )
    delta_wide = delta_wide.sort_values(["Cluster", "1.1"])

    mat = delta_wide[BLOCK_LABELS[1:]].values
    counts = delta_wide["Cluster"].value_counts().reindex(CLUSTER_HUE_ORDER).values
    boundaries = np.cumsum(counts)
    starts = np.concatenate([[0], boundaries[:-1]])

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 6))
        vmax = 20
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(BLOCK_LABELS[1:])))
        ax.set_xticklabels(BLOCK_LABELS[1:])
        ax.set_xlabel("Block")
        ax.set_yticks([])

        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="k", lw=1)
        for name, s, c in zip(CLUSTER_HUE_ORDER, starts, counts):
            ax.text(
                -0.75,
                s + c / 2 - 0.5,
                name,
                rotation=90,
                va="center",
                ha="center",
                color=CLUSTER_PALETTE[name],
                fontweight="bold",
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta$HR (bpm)")

        fig.tight_layout()
        stem = f"{FIG4_DIR}/figS7b_delta_hr_heatmap"
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()


def save_replication_table(rep_bf):
    """Save replication BF table (Supp. Table 8)."""
    cols = [
        "block",
        "contrast",
        "d_original",
        "d_replication",
        "n_original",
        "n_replication",
        "BFr",
        "BFs",
    ]
    table = rep_bf[cols].rename(
        columns={
            "block": "Block",
            "contrast": "Contrast",
            "d_original": "d (original)",
            "d_replication": "d (replication)",
            "n_original": "N (original)",
            "n_replication": "N (replication)",
        }
    )
    table_path = f"{FIG4_DIR}/tableS8_replication_bf.md"
    table.round(3).to_markdown(table_path, index=False)
    print(f"Saved {table_path}")


if __name__ == "__main__":
    # Fig 4b
    posterior, bf = load_delta_hr_data()
    plot_delta_hr(posterior, bf)

    # Fig 4c
    delta1 = load_delta1_data()
    plot_delta1_cohorts(delta1)

    # Fig S7
    hr_stats, grand_stats, physio_hr, labels = load_hr_timecourse_data()
    plot_hr_timecourse(hr_stats, grand_stats)
    plot_delta_hr_heatmap(physio_hr, labels)

    # Supp Table 8
    rep_bf = load_replication_bf()
    save_replication_table(rep_bf)
