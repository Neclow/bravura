# pylint: disable=redefined-outer-name, invalid-name

"""Cluster participants based on behavioural data.

Phase 1:  Grid search (deterministic, fast):
    k × solver silhouette + gap statistic table.
Phase 2: Fit_predict with chosen k and solver (deterministic):
    Save clusters.csv (with cluster labels).
Phase 3: MC-based clustering:
    fuzzy_fit_predict with chosen k and solver, save consensus robustness metrics.
Phase 4: Metric ablation (deterministic):
    fit_predict with various VBA metrics with chosen k and solver, save ablation table.
"""

import os

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src._config import (
    CLUSTER_NAMES,
    DEFAULT_CLUSTER_DIR_A,
    DEFAULT_CLUSTER_DIR_B,
    DEFAULT_CLUSTERING_FEATURES,
    DEFAULT_DATA_DIR,
    MAX_BELIEF_COHORT_A,
    MAX_BELIEF_COHORT_B,
    RANDOM_SEED,
)
from src.cluster2 import CLUSTERERS, ablate_k, ablate_X, fit_predict, fuzzy_fit_predict
from src.preprocessing import (
    collect_metrics,
    load_behavioral_features,
    sample_behavioral_features,
)

DEFAULT_COLS_TO_DROP = ["Brier", "AUC", "accuracy", "balanced_accuracy", "log_evidence"]
K_RANGE = range(2, 11)
METRIC_COLS = ["R2", "Brier", "AUC", "balanced_accuracy", "accuracy", "log_evidence"]
N_SAMPLES = 1000


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-kbest",
        default="auto",
        help=(
            "Override the number of clusters for consensus clustering and metric "
            "ablation. 'auto' uses the optimal k from the gap statistic (default: auto)."
        ),
    )
    parser.add_argument(
        "-sbest",
        default="auto",
        help=(
            "Override the clustering algorithm. 'auto' picks the solver with the "
            "highest mean silhouette from the grid search (default: auto)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run consensus clustering even if output files already exist.",
    )
    args = parser.parse_args()
    if args.kbest != "auto":
        args.kbest = int(args.kbest)
    return args


def load_data(cohort):
    """Load VBA outputs, questionnaires, and build behavioural features.

    Outliers are excluded using the pre-computed list from
    pipeline/prepare_behavior_pre.py.

    Returns
    -------
    iterator : generator
        MC-sampled feature matrices (for fuzzy clustering).
    bma : pd.DataFrame
        Full feature set including all metrics.
    """
    cohort_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"
    vba_dir = f"{cohort_dir}/vba"

    with open(f"{cohort_dir}/outliers.txt", encoding="utf-8") as f:
        outlier_ids = [line.strip() for line in f if line.strip()]

    vba_metrics = pd.read_csv(f"{vba_dir}/fit_metrics.csv", index_col=0)
    vba_preds = pd.read_csv(f"{vba_dir}/predictions.csv", header=None)
    vba_actual = pd.read_csv(f"{vba_dir}/decisions.csv", header=None)

    ids = pd.read_csv(f"{cohort_dir}/subject_ids.csv")
    vba_preds.index = ids["subject"]
    vba_actual.index = ids["subject"]

    all_metrics = collect_metrics(vba_metrics, vba_preds, vba_actual)

    vba_posteriors = loadmat(f"{vba_dir}/vba_posteriors.mat")
    coefs_mu = vba_posteriors["mu_all"]
    coefs_sigma = vba_posteriors["sigma_all"]

    aggro = pd.read_excel(f"{cohort_dir}/aggroPerformance.xlsx", index_col="Subject")
    beliefs = pd.read_excel(f"{cohort_dir}/beliefs.xlsx", index_col="ID")
    if sorted(beliefs.columns) != ["opponent1", "opponent2"]:
        beliefs.rename(columns={k: k[4:] + k[0] for k in beliefs.columns}, inplace=True)
    beliefs.drop("opponent3", axis=1, errors="ignore", inplace=True)
    if cohort == "b":
        beliefs = beliefs * (MAX_BELIEF_COHORT_A / MAX_BELIEF_COHORT_B)

    # Drop outliers from posteriors (aligned by row with subject_ids)
    keep_mask = ~ids["subject"].isin(outlier_ids).values
    coefs_mu = coefs_mu[keep_mask]
    coefs_sigma = coefs_sigma[keep_mask]

    # Drop outliers from tabular sources
    all_metrics = all_metrics.drop(index=outlier_ids, errors="ignore")
    aggro = aggro.drop(index=outlier_ids, errors="ignore")
    beliefs = beliefs.drop(index=outlier_ids, errors="ignore")

    iterator = sample_behavioral_features(
        coefs_mu=coefs_mu,
        coefs_sigma=coefs_sigma,
        metrics=all_metrics.drop(columns=DEFAULT_COLS_TO_DROP),
        aggro=aggro,
        beliefs=beliefs,
        n_samples=N_SAMPLES,
        cols_to_use=DEFAULT_CLUSTERING_FEATURES,
    )

    coefs = pd.read_csv(f"{vba_dir}/coefficients.csv", index_col=0)
    coefs = coefs.drop(index=outlier_ids, errors="ignore")
    bma = load_behavioral_features(coefs, all_metrics, aggro, beliefs)

    return iterator, bma


def grid_search(Xa_scaled, out_dir):
    """Phase 1: k × solver silhouette grid, then gap statistic for best solver."""
    solvers = list(CLUSTERERS.keys())

    # 1a: silhouette pivot across all solvers
    rows = []
    for solver in solvers:
        sens_k = ablate_k(
            Xa_scaled, solver=solver, k_range=K_RANGE, random_state=RANDOM_SEED
        )
        for k_val in K_RANGE:
            rows.append(
                {
                    "solver": solver,
                    "k": k_val,
                    "silhouette": sens_k.loc[k_val, "silhouette"],
                    "gap": sens_k.loc[k_val, "gap"],
                    "gap_diff": sens_k.loc[k_val, "gap_diff"],
                }
            )
    grid = pd.DataFrame(rows)

    pivot = grid.pivot(index="k", columns="solver", values="silhouette").round(3)
    print(pivot.to_markdown())
    out_path = f"{out_dir}/grid_k_solver.csv"
    grid.round(3).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print()

    # 1b: pick best solver (highest mean silhouette), print full ablate_k
    mean_sil = grid.groupby("solver")["silhouette"].mean()
    best_solver = mean_sil.idxmax()
    print(
        f"Best solver by mean silhouette: {best_solver} ({mean_sil[best_solver]:.3f})"
    )
    print()

    best = grid[grid["solver"] == best_solver].set_index("k")
    print(f"Best solver: {best_solver}")
    print(best[["silhouette", "gap", "gap_diff"]].round(3).to_markdown())
    best_path = f"{out_dir}/ablate_k.csv"
    best[["silhouette", "gap", "gap_diff"]].round(3).to_csv(best_path)
    print(f"Saved: {best_path}")

    positive = best["gap_diff"].dropna()
    positive = positive[positive > 0]
    optimal_k = positive.index.min() if len(positive) > 0 else None
    print(f"\nOptimal k (first gap_diff > 0): {optimal_k}\n")

    return best_solver, optimal_k


def cluster_and_save(Xa, Xb, Xa_scaled, Xb_scaled, k, solver):
    """Phase 2: fit_predict with chosen k and solver, save cluster assignments."""
    resa = fit_predict(X=Xa_scaled, solver=solver, k=k, random_state=RANDOM_SEED)
    resb_labels = resa.clusterer.predict(Xb_scaled)

    Xa["label"] = resa.labels
    Xa["Cluster"] = Xa["label"].map(CLUSTER_NAMES)
    Xb["label"] = resb_labels
    Xb["Cluster"] = Xb["label"].map(CLUSTER_NAMES)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Xa[["PC1", "PC2"]] = pca.fit_transform(Xa_scaled)
    Xb[["PC1", "PC2"]] = pca.transform(Xb_scaled)

    out_a = f"{DEFAULT_CLUSTER_DIR_A}/clusters.csv"
    out_b = f"{DEFAULT_CLUSTER_DIR_B}/clusters.csv"
    Xa.to_csv(out_a)
    print(f"Saved: {out_a}")
    Xb.to_csv(out_b)
    print(f"Saved: {out_b}\n")


def mc_clustering(
    iterator_a, iterator_b, Xa, Xb, labels_a, labels_b, k, solver, npz_path, txt_path
):
    """Phase 2: MC consensus clustering with chosen k and solver."""
    res_fuzzy = fuzzy_fit_predict(
        iterator_a=iterator_a,
        iterator_b=iterator_b,
        solver=solver,
        k=k,
        random_state=RANDOM_SEED,
        n_a=len(Xa),
        n_b=len(Xb),
        ref_labels=labels_a,
        n_samples=N_SAMPLES,
    )

    # Save consensus robustness metrics
    agree_a = int((labels_a == res_fuzzy.consensus_a).sum())
    agree_b = int((labels_b == res_fuzzy.consensus_b).sum())
    ari_a = adjusted_rand_score(labels_a, res_fuzzy.consensus_a)
    ari_b = adjusted_rand_score(labels_b, res_fuzzy.consensus_b)

    robustness_lines = [
        f"Consensus robustness ({N_SAMPLES} MC draws, {solver}, k={k})",
        f"Stability (cohort A): {res_fuzzy.stability_a:.3f}",
        f"Stability (cohort B): {res_fuzzy.stability_b:.3f}",
        f"ARI deterministic vs consensus (A): {ari_a:.3f}",
        f"ARI deterministic vs consensus (B): {ari_b:.3f}",
        f"Agreement (A): {agree_a}/{len(Xa)} ({agree_a/len(Xa):.1%})",
        f"Agreement (B): {agree_b}/{len(Xb)} ({agree_b/len(Xb):.1%})",
    ]
    for line in robustness_lines:
        print(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(robustness_lines) + "\n")
    print(f"Saved: {txt_path}")

    np.savez_compressed(npz_path, **vars(res_fuzzy))
    print(f"Saved: {npz_path}\n")


def metric_ablation(bma_a, out_dir, k, solver):
    """Phase 3: which VBA fit metric to include (given chosen k and solver).

    Uses the non-metric clustering features as the base, then adds one metric
    column at a time from the full feature set (bma_a).
    """
    base_cols = [c for c in DEFAULT_CLUSTERING_FEATURES if c not in METRIC_COLS]

    Xs = {}
    for metric in METRIC_COLS + ["none"]:
        if metric == "none":
            cols = base_cols
        else:
            cols = base_cols + [metric]
        Xs[metric] = StandardScaler().fit_transform(bma_a[cols])

    sens_X = ablate_X(Xs, solver=solver, k=k, random_state=RANDOM_SEED)
    sens_X_out = (
        sens_X.drop(["labels", "sizes", "clusterer"], axis=1)
        .sort_values(by="silhouette", ascending=False)
        .round(3)
    )
    print(sens_X_out.to_markdown())
    out_path = f"{out_dir}/ablate_metric.csv"
    sens_X_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    args = parse_args()

    # Phase 0a: Load data for both cohorts
    print("Loading data for both cohorts (excluding outliers)")
    iterator_a, bma_a = load_data("a")
    iterator_b, bma_b = load_data("b")

    # Phase 0b: Prepare feature matrices for clustering
    Xa = bma_a.drop(columns=DEFAULT_COLS_TO_DROP)
    Xb = bma_b.drop(columns=DEFAULT_COLS_TO_DROP)

    features = DEFAULT_CLUSTERING_FEATURES
    print(f"{len(features)} features: {features}\n")

    scaler = StandardScaler()
    # pylint: disable=unsubscriptable-object
    Xa_scaled = scaler.fit_transform(Xa[features])
    Xb_scaled = scaler.transform(Xb[features])

    # Phase 1: Grid search (k × solver) → pick best solver + optimal k
    print("Phase 1: k × solver grid search (silhouette + gap statistic)")
    best_solver, optimal_k = grid_search(Xa_scaled, out_dir=DEFAULT_CLUSTER_DIR_A)

    k = optimal_k if args.kbest == "auto" else args.kbest
    solver = best_solver if args.sbest == "auto" else args.sbest
    print(f"\nUsing k={k}, solver={solver}\n")

    # Phase 2: Cluster with chosen k and solver, save cluster assignments.
    print("Phase 2: Cluster and save with chosen k and solver")
    cluster_and_save(Xa, Xb, Xa_scaled, Xb_scaled, k=k, solver=solver)
    labels_a = Xa["label"].values
    labels_b = Xb["label"].values
    # pylint: enable=unsubscriptable-object

    # Phase 3: MC-based clustering
    print("Phase 3: MC-based clustering")
    npz_path = f"{DEFAULT_CLUSTER_DIR_A}/mc_{solver}_{k}.npz"
    txt_path = f"{DEFAULT_CLUSTER_DIR_A}/mc_{solver}_{k}_stats.txt"
    if not args.overwrite and os.path.exists(npz_path):
        print(f"\tMC output exists ({npz_path}), skipping (use --overwrite to re-run)")
    else:
        print(f"\tRunning MC clustering (k={k}, {solver})")
        mc_clustering(
            iterator_a,
            iterator_b,
            Xa,
            Xb,
            labels_a,
            labels_b,
            k=k,
            solver=solver,
            npz_path=npz_path,
            txt_path=txt_path,
        )

    # Phase 4: Metric ablation
    print("Phase 4: Metric ablation")
    metric_ablation(bma_a, out_dir=DEFAULT_CLUSTER_DIR_A, k=k, solver=solver)

    print("Done.")
