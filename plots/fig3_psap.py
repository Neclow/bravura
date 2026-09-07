import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import adjusted_rand_score
from statannotations.Annotator import Annotator

from src._config import (
    CLUSTER_PALETTE,
    CLUSTERS,
    DEFAULT_BRMS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_DIR,
    RANDOM_SEED,
)
from src.cluster2 import fit_predict

from ._config import DEFAULT_IMG_DIR, DEFAULT_STYLE

FIG3_DIR = f"{DEFAULT_IMG_DIR}/fig3"
os.makedirs(FIG3_DIR, exist_ok=True)

CLUSTER_HUE_ORDER = [cl["name"] for cl in CLUSTERS]

PSAP_FEATURES = ["pA", "pB", "pC", "rA", "rB", "rC"]

BUTTON_ORDER = ["Earn", "Steal", "Protect"]

GRID_RINGS = [0.25, 0.5, 0.75, 1.0]


# -- Fig 3f data + plot -------------------------------------------------------


def load_concurrent_data():
    """Load PSAP Dirichlet posterior predictions and Bayes factors.

    Returns
    -------
    posterior : DataFrame
        Posterior predicted button proportions per cluster, phase, and button.
    bf : DataFrame
        Pairwise Bayes factors for cluster contrasts.
    """
    brms_dir = f"{DEFAULT_BRMS_DIR}/psap"
    posterior = pd.read_csv(f"{brms_dir}/posterior_epred.csv")
    bf = pd.read_csv(f"{brms_dir}/bayes_factors.csv")
    return posterior, bf


def build_concurrent_annotations(bf):
    """Build per-phase annotation pairs from significant BFs.

    Returns
    -------
    sig_contrasts : dict
        {phase: [(pairs, labels), ...]} for Annotator.
    """
    sig = bf[bf["excl_zero"]]
    sig_contrasts = {}
    for phase in ["Proactive", "Reactive"]:
        phase_sig = sig[sig["phase"] == phase]
        pairs = []
        labels = []
        for _, row in phase_sig.iterrows():
            c1, c2 = row["contrast"].split(" - ")
            pairs.append(((row["button"], c1), (row["button"], c2)))
            labels.append("*")
        sig_contrasts[phase] = (pairs, labels)
    return sig_contrasts


def plot_psap_concurrent(posterior, bf):
    """Plot PSAP button proportions by cluster and phase (Fig. 3f)."""
    sig_contrasts = build_concurrent_annotations(bf)

    with plt.style.context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)

        for ax, phase in zip(axes, ["Proactive", "Reactive"]):
            phase_data = posterior[posterior["phase"] == phase]

            plot_kw = dict(
                data=phase_data,
                x="button",
                y="proportion",
                hue="Cluster",
                hue_order=CLUSTER_HUE_ORDER,
                order=BUTTON_ORDER,
            )

            sns.barplot(
                **plot_kw,
                palette=CLUSTER_PALETTE,
                errorbar=("pi", 95),
                capsize=0.1,
                ax=ax,
            )
            ax.set_title(f"{phase} phase")
            ax.set_xlabel("")
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1)
            ax.get_legend().remove()

            pairs, labels = sig_contrasts.get(phase, ([], []))
            if pairs:
                annot = Annotator(ax, pairs, **plot_kw)
                annot.set_custom_annotations(labels)
                annot.annotate()

        axes[0].set_ylabel("Proportion")

        plt.tight_layout()
        stem = f"{FIG3_DIR}/fig3f_psap_concurrent"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    table_path = f"{FIG3_DIR}/tableS6_psap.md"
    bf.set_index("contrast").round(3).to_markdown(table_path)
    print(f"Saved {table_path}")


# -- Fig S6 data + plots ------------------------------------------------------


def load_psap_clustering_data():
    """Load raw PSAP data, merge with cluster labels, run PSAP clustering.

    Returns
    -------
    profiles : DataFrame
        Mean PSAP proportions per PSAP cluster.
    ari : float
        Adjusted Rand Index (Bravura clusters vs PSAP clusters).
    ct : DataFrame
        Contingency table (Bravura × PSAP clusters).
    psap_scatter : DataFrame
        Scatter data with shock_prop and B_prop columns.
    """
    psap = pd.read_excel(f"{DEFAULT_DATA_DIR}/raw/additional.xlsx").set_index("Subject")

    Xa = pd.read_csv(f"{DEFAULT_PROCESSED_DIR}/behav_Xa.csv", index_col="Row")

    psap_cluster = (
        psap[PSAP_FEATURES].join(Xa[["label", "Cluster"]], how="inner").dropna()
    )
    psap_dir = psap_cluster.reset_index()

    # PSAP k-means clustering
    psap_clust = psap_dir[["Subject", "Cluster"] + PSAP_FEATURES].dropna()
    X_psap = (
        psap_clust[PSAP_FEATURES]
        .sub(psap_clust[PSAP_FEATURES].mean())
        .div(psap_clust[PSAP_FEATURES].std())
    )
    res_psap = fit_predict(
        X_psap.values, solver="k-means", k=3, random_state=RANDOM_SEED
    )
    psap_clust["psap_cluster"] = res_psap.labels

    ari = adjusted_rand_score(psap_clust["Cluster"], psap_clust["psap_cluster"])
    ct = pd.crosstab(
        psap_clust["Cluster"],
        psap_clust["psap_cluster"],
        rownames=["Bravura"],
        colnames=["PSAP"],
    )
    profiles = psap_clust.groupby("psap_cluster")[PSAP_FEATURES].mean()

    # Scatter data
    psap_scatter = psap_dir[["Subject", "Cluster"]].drop_duplicates()
    psap_scatter = psap_scatter.merge(
        psap[["NumShocks", "pB", "rB"]],
        left_on="Subject",
        right_index=True,
    )
    psap_scatter["B_prop"] = (psap_scatter["pB"] + psap_scatter["rB"]) / 2
    psap_scatter = psap_scatter.dropna()
    psap_scatter["shock_prop"] = psap_scatter["NumShocks"] / 30

    return profiles, ari, ct, psap_scatter


def plot_psap_clustering(profiles, ari, ct):
    """Plot PSAP cluster radar charts and contingency heatmap (Fig. S6a)."""
    n_feat = len(PSAP_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    psap_palette = sns.color_palette("magma", 4)[:-1]

    with plt.style.context(DEFAULT_STYLE):
        fig = plt.figure(figsize=(13, 3.5))

        for i, cluster_id in enumerate(sorted(profiles.index)):
            ax = fig.add_subplot(1, 4, i + 1, projection="polar")
            ax.set_theta_offset(np.pi / 2)

            values = profiles.loc[cluster_id].tolist()
            values += values[:1]

            ax.fill(
                angles, values, alpha=0.25, color=psap_palette[cluster_id], zorder=10
            )
            ax.plot(angles, values, color=psap_palette[cluster_id], linewidth=1.5)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(PSAP_FEATURES)
            ax.set_ylim(0, 1)
            ax.set_yticks(GRID_RINGS)
            ax.set_yticklabels(["", "0.5", "", "1.0"])
            ax.yaxis.grid(False)
            ax.spines["polar"].set_visible(False)
            for r in GRID_RINGS:
                style = (
                    {"color": "black", "linewidth": 0.8}
                    if r == 1.0
                    else {"color": "gray", "linewidth": 0.5, "alpha": 0.3}
                )
                ax.plot(angles, [r] * len(angles), **style)
            ax.tick_params(axis="x", pad=10)
            ax.set_title(f"Cluster {cluster_id}", pad=15, fontweight="bold")

        ax_ct = fig.add_subplot(1, 4, 4)
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax_ct)
        ax_ct.set_title(f"ARI = {ari:.3f}")

        plt.tight_layout()
        stem = f"{FIG3_DIR}/figS6a_psap_clustering"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"ARI = {ari:.3f}\n")
        f.write(f"\n{ct}\n")
    print(f"Saved {stem}_stats.txt")


def plot_psap_scatter(psap_scatter):
    """Plot Bravura shocks vs PSAP B-presses scatter (Fig. S6b)."""
    r = psap_scatter["shock_prop"].corr(psap_scatter["B_prop"], method="spearman")

    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.regplot(
            data=psap_scatter,
            x="shock_prop",
            y="B_prop",
            scatter_kws={"s": 10, "alpha": 0.6},
            color="k",
            line_kws={"lw": 1},
            ax=ax,
        )
        ax.set_xlabel("Average % shocks (Bravura)")
        ax.set_ylabel("Average % B presses (PSAP)")
        ax.set_ylim(-0.05, 1)
        ax.set_title(f"Spearman's r = {r:.2f}")
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        stem = f"{FIG3_DIR}/figS6b_psap_scatter"
        plt.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        print(f"Saved {stem}.pdf/.png")
        plt.show()

    with open(f"{stem}_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"Spearman's r = {r:.3f}\n")
        f.write(f"N = {len(psap_scatter)}\n")
    print(f"Saved {stem}_stats.txt")


if __name__ == "__main__":
    # Fig 3f
    posterior, bf = load_concurrent_data()
    plot_psap_concurrent(posterior, bf)

    # Fig S6
    profiles, ari, ct, psap_scatter = load_psap_clustering_data()
    plot_psap_clustering(profiles, ari, ct)
    plot_psap_scatter(psap_scatter)
