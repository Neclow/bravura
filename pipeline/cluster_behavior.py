# pylint: disable=redefined-outer-name, invalid-name

"""Cluster participants based on behavioural data.

Phase 1:  Grid search (deterministic, fast):
    k × solver silhouette table.
Phase 2: Consensus clustering (MC, ~2 min):
    fuzzy_fit_predict with chosen k/solver → behav_Xa.csv, behav_Xb.csv.
Phase 3: Metric ablation (deterministic):
    Which VBA fit metric to include, given k=3 / k-means.
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


def parse_args():
    parser = ArgumentParser(description=__doc__)
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


N_SAMPLES = 1000
DEFAULT_COLS_TO_DROP = ["Brier", "AUC", "accuracy", "balanced_accuracy", "log_evidence"]
METRIC_COLS = ["R2", "Brier", "AUC", "balanced_accuracy", "accuracy", "log_evidence"]
K_RANGE = range(2, 11)


def load_data(cohort):
    """Load VBA outputs, questionnaires, and build behavioural features.

    Returns
    -------
    iterator : generator
        MC-sampled feature matrices (for fuzzy clustering).
    X : pd.DataFrame
        Deterministic features, metrics in COLS_TO_DROP removed.
    bma : pd.DataFrame
        Full feature set including all metrics.
    """
    vba_metrics = pd.read_csv(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/fit_metrics.csv", index_col=0
    )
    vba_preds = pd.read_csv(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/predictions.csv", header=None
    )
    vba_actual = pd.read_csv(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/decisions.csv", header=None
    )

    ids = pd.read_csv(f"{DEFAULT_DATA_DIR}/cohort_{cohort}/subject_ids.csv")

    vba_preds.index = ids["subject"]
    vba_actual.index = ids["subject"]

    all_metrics = collect_metrics(vba_metrics, vba_preds, vba_actual)

    vba_posteriors = loadmat(f"{DEFAULT_DATA_DIR}/cohort_{cohort}/vba_posteriors.mat")
    coefs_mu = vba_posteriors["mu_all"]
    coefs_sigma = vba_posteriors["sigma_all"]

    aggro = pd.read_excel(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/aggroPerformance.xlsx", index_col="Subject"
    )
    beliefs = pd.read_excel(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/beliefs.xlsx", index_col="ID"
    )
    if sorted(beliefs.columns) != ["opponent1", "opponent2"]:
        beliefs.rename(columns={k: k[4:] + k[0] for k in beliefs.columns}, inplace=True)
    beliefs.drop("opponent3", axis=1, errors="ignore", inplace=True)
    if cohort == "b":
        beliefs = beliefs * (MAX_BELIEF_COHORT_A / MAX_BELIEF_COHORT_B)

    iterator = sample_behavioral_features(
        coefs_mu=coefs_mu,
        coefs_sigma=coefs_sigma,
        metrics=all_metrics.drop(columns=DEFAULT_COLS_TO_DROP),
        aggro=aggro,
        beliefs=beliefs,
        n_samples=N_SAMPLES,
        cols_to_use=DEFAULT_CLUSTERING_FEATURES,
    )

    coefs = pd.read_csv(
        f"{DEFAULT_DATA_DIR}/cohort_{cohort}/coefficients.csv", index_col=0
    )
    bma = load_behavioral_features(
        coefs,
        all_metrics,
        aggro,
        beliefs,
    )

    return iterator, bma


def grid_search(Xa_scaled, out_dir):
    """Phase 1: k × solver silhouette grid, then gap statistic for best solver."""
    solvers = list(CLUSTERERS.keys())

    # 1a: silhouette pivot across all solvers
    print("=== k × solver silhouette ===")
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
    grid.round(3).to_csv(f"{out_dir}/grid_k_solver.csv", index=False)
    print(f"Saved: {out_dir}/grid_k_solver.csv")
    print()

    # 1b: pick best solver (highest mean silhouette), print full ablate_k
    mean_sil = grid.groupby("solver")["silhouette"].mean()
    best_solver = mean_sil.idxmax()
    print(
        f"Best solver by mean silhouette: {best_solver} ({mean_sil[best_solver]:.3f})"
    )
    print()

    best = grid[grid["solver"] == best_solver].set_index("k")
    print(f"=== {best_solver}: silhouette + gap ===")
    print(best[["silhouette", "gap", "gap_diff"]].round(3).to_markdown())
    best_path = f"{out_dir}/best_solver_ablate_k.csv"
    best[["silhouette", "gap", "gap_diff"]].round(3).to_csv(best_path)
    print(f"Saved: {best_path}")

    positive = best["gap_diff"].dropna()
    positive = positive[positive > 0]
    optimal_k = positive.index.min() if len(positive) > 0 else None
    print(f"\nOptimal k (first gap_diff > 0): {optimal_k}")

    return best_solver, optimal_k


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

    print("=== Metric ablation ===")
    sens_X = ablate_X(Xs, solver=solver, k=k, random_state=RANDOM_SEED)
    sens_X_out = (
        sens_X.drop(["labels", "sizes", "clusterer"], axis=1)
        .sort_values(by="silhouette", ascending=False)
        .round(3)
    )
    print(sens_X_out.to_markdown())
    sens_X_out.to_csv(f"{out_dir}/ablate_metric.csv")
    print(f"Saved: {out_dir}/ablate_metric.csv")
    print()


def consensus_clustering(
    iterator_a, iterator_b, Xa, Xb, Xa_scaled, Xb_scaled, k, solver
):
    """Phase 2: MC consensus clustering with chosen k and solver."""
    resa = fit_predict(X=Xa_scaled, solver=solver, k=k, random_state=RANDOM_SEED)
    resb_labels = resa.clusterer.predict(Xb_scaled)
    res = fuzzy_fit_predict(
        iterator_a=iterator_a,
        iterator_b=iterator_b,
        solver=solver,
        k=k,
        random_state=RANDOM_SEED,
        n_a=len(Xa),
        n_b=len(Xb),
        ref_labels=resa.labels,
        n_samples=N_SAMPLES,
    )

    print(f"Stability (cohort A): {res.stability_a:.3f}")
    print(f"Stability (cohort B): {res.stability_b:.3f}")
    print(
        "ARI (consensus vs deterministic):",
        adjusted_rand_score(resa.labels, res.consensus_a),
    )
    print(
        "ARI (cohort A vs. cohort B):",
        adjusted_rand_score(resb_labels, res.consensus_b),
    )

    Xa["label"] = res.consensus_a
    Xa["Cluster"] = Xa["label"].map(CLUSTER_NAMES)
    Xb["label"] = res.consensus_b
    Xb["Cluster"] = Xb["label"].map(CLUSTER_NAMES)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Xa[["PC1", "PC2"]] = pca.fit_transform(Xa_scaled)
    Xb[["PC1", "PC2"]] = pca.transform(Xb_scaled)

    out = f"{DEFAULT_DATA_DIR}/processed"
    Xa.to_csv(f"{out}/behav_Xa.csv")
    print(f"Saved: {out}/behav_Xa.csv")
    Xb.to_csv(f"{out}/behav_Xb.csv")
    print(f"Saved: {out}/behav_Xb.csv")
    npz_path = f"{out}/mc_consensus_{solver}_{k}.npz"
    np.savez_compressed(
        npz_path,
        label_counts_a=res.label_counts_a,
        label_counts_b=res.label_counts_b,
        consensus_a=res.consensus_a,
        consensus_b=res.consensus_b,
        stability_a=res.stability_a,
        stability_b=res.stability_b,
        scaler_means=res.scaler_means,
        scaler_scales=res.scaler_scales,
        n_samples=res.n_samples,
    )
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    args = parse_args()

    out_dir = f"{DEFAULT_DATA_DIR}/processed/sensitivity_clustering"
    os.makedirs(out_dir, exist_ok=True)

    iterator_a, bma_a = load_data("a")
    iterator_b, bma_b = load_data("b")

    Xa = bma_a.drop(columns=DEFAULT_COLS_TO_DROP)
    Xb = bma_b.drop(columns=DEFAULT_COLS_TO_DROP)

    features = DEFAULT_CLUSTERING_FEATURES
    print(f"{len(features)} features: {features}\n")

    scaler = StandardScaler()
    # pylint: disable=unsubscriptable-object
    Xa_scaled = scaler.fit_transform(Xa[features])
    Xb_scaled = scaler.transform(Xb[features])
    # pylint: enable=unsubscriptable-object

    # Phase 1: Grid search (k × solver) → pick best solver + optimal k
    best_solver, optimal_k = grid_search(Xa_scaled, out_dir)

    k = optimal_k if args.kbest == "auto" else args.kbest
    solver = best_solver if args.sbest == "auto" else args.sbest
    print(f"\nUsing k={k}, solver={solver}\n")

    # Phase 2: Consensus clustering
    processed = f"{DEFAULT_DATA_DIR}/processed"
    npz_path = f"{processed}/mc_consensus_{solver}_{k}.npz"
    if not args.overwrite and os.path.exists(npz_path):
        print(
            f"Consensus output exists ({npz_path}), skipping (use --overwrite to re-run)"
        )
    else:
        print(f"=== Consensus clustering (k={k}, {solver}) ===")
        consensus_clustering(
            iterator_a, iterator_b, Xa, Xb, Xa_scaled, Xb_scaled, k=k, solver=solver
        )

    # Phase 3: Metric ablation
    metric_ablation(bma_a, out_dir, k=k, solver=solver)
