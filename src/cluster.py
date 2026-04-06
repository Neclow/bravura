"""Clustering utilities for behavioral feature analysis."""

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    brier_score_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

# Features used for clustering (10 features including R² and beliefs)
CLUSTER_FEATURES = [
    "Kr1",
    "Krc",
    "Kp",
    "Kwc",
    "R2",
    "shock_opp1",
    "shock_opp2",
    "first_shock",
    "belief_opp1",
    "belief_opp2",
]

CLUSTER_NAMES = {0: "Nonaggressive", 1: "Reactive", 2: "Proactive"}
CLUSTER_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

RANDOM_SEED = 42
K_MIN, K_MAX = 2, 13

CLUSTERERS = {
    "k-means": lambda k, rs: KMeans(n_clusters=k, random_state=rs, n_init=10),
    "k-medoids": lambda k, rs: KMedoids(
        n_clusters=k, method="fasterpam", metric="euclidean", random_state=rs
    ),
    "spectral": lambda k, rs: SpectralClustering(
        n_clusters=k, random_state=rs, affinity="rbf"
    ),
    "hac": lambda k, _: AgglomerativeClustering(n_clusters=k, linkage="ward"),
    "gmm": lambda k, rs: GaussianMixture(n_components=k, random_state=rs, n_init=5),
}


def standardize(df, features):
    """Standardize features using pandas (ddof=1), matching the original analysis."""
    x = df[features]
    return x.sub(x.mean()).div(x.std())


def _assign_labels(df, labels):
    """Map raw cluster IDs to semantic labels (0=nonaggressive, 1=reactive, 2=proactive)."""
    total_shocks = df["shock_opp1"] + df["shock_opp2"]
    mean_shocks = pd.Series(total_shocks.values, index=labels).groupby(level=0).mean()
    mean_kp = pd.Series(df["Kp"].values, index=labels).groupby(level=0).mean()

    nonagg = mean_shocks.idxmin()
    remaining = [c for c in mean_shocks.index if c != nonagg]
    proactive = mean_kp[remaining].idxmax()
    reactive = [c for c in remaining if c != proactive][0]

    label_map = {nonagg: 0, reactive: 1, proactive: 2}
    return np.array([label_map[lab] for lab in labels])


def _cluster(X, clusterer, k, random_state):
    """Run a single clustering algorithm and return raw labels."""
    algo = CLUSTERERS[clusterer](k, random_state)
    if hasattr(algo, "fit_predict"):
        return algo.fit_predict(X)
    algo.fit(X)
    return algo.predict(X)


def fit_predict(
    X,
    clusterer="k-means",
    k=3,
    random_state=RANDOM_SEED,
):
    """Standardize, cluster, assign semantic labels, and compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with behavioral features
    features : list[str]
        Feature columns to cluster on
    clusterer : str
        One of: "k-means", "k-medoids", "spectral", "hac", "gmm"
    k : int
        Number of clusters
    random_state : int
        Random seed
    reference_labels : np.ndarray, optional
        If provided, compute ARI against these labels

    Returns
    -------
    dict with keys:
        labels : np.ndarray -- semantic cluster labels (0=nonagg, 1=reactive, 2=proactive)
        silhouette : float
        ari : float or None
        sizes : list[int]
        X_scaled : np.ndarray -- standardized feature matrix
    """
    X = standardize(df, features).values
    raw_labels = _cluster(X, clusterer, k, random_state)
    labels = _assign_labels(df, raw_labels)
    sil = silhouette_score(X, labels)
    ari = (
        adjusted_rand_score(reference_labels, labels)
        if reference_labels is not None
        else None
    )

    return {
        "labels": labels,
        "silhouette": sil,
        "ari": ari,
        "sizes": np.bincount(labels, minlength=k).tolist(),
        "X_scaled": X,
    }


def silhouette_scan(
    df,
    features=CLUSTER_FEATURES,
    clusterer="k-means",
    k_range=range(K_MIN, K_MAX),
    random_state=RANDOM_SEED,
):
    """Silhouette scores for a range of k values."""
    return pd.DataFrame(
        [
            {
                "k": k,
                "silhouette": fit_predict(df, features, clusterer, k, random_state)[
                    "silhouette"
                ],
            }
            for k in k_range
        ]
    )


def ablate_methods(
    df, features=CLUSTER_FEATURES, k=3, reference_labels=None, random_state=RANDOM_SEED
):
    """Compare all clustering methods on the same data."""
    results = []
    for name in CLUSTERERS:
        res = fit_predict(df, features, name, k, random_state, reference_labels)
        scan = silhouette_scan(df, features, name, random_state=random_state)
        best = scan.loc[scan["silhouette"].idxmax()]

        results.append(
            {
                "method": name,
                "k_opt": int(best["k"]),
                f"silhouette (k={k})": round(res["silhouette"], 3),
                "best silhouette": round(best["silhouette"], 3),
                "ARI vs reference": (
                    round(res["ari"], 3) if res["ari"] is not None else np.nan
                ),
                f"sizes (k={k})": res["sizes"],
            }
        )
    return pd.DataFrame(results).set_index("method")


def ablate_features(
    df,
    pred_probs,
    actual_decisions,
    reference_features=CLUSTER_FEATURES,
    random_state=RANDOM_SEED,
):
    """Compare clustering with different goodness-of-fit metrics replacing R²."""
    brier = np.array(
        [brier_score_loss(actual_decisions[i], pred_probs[i]) for i in range(len(df))]
    )
    aucs = np.array(
        [
            (
                roc_auc_score(actual_decisions[i], pred_probs[i])
                if len(np.unique(actual_decisions[i])) > 1
                else 0.5
            )
            for i in range(len(df))
        ]
    )

    df_ext = df.copy()
    df_ext["Brier"] = brier
    df_ext["AUC"] = aucs

    base = [f for f in reference_features if f != "R2"]
    configs = {
        "R² (original)": base + ["R2"],
        "No fit metric": base,
        "Brier score": base + ["Brier"],
        "AUC": base + ["AUC"],
    }

    ref = fit_predict(df_ext, configs["R² (original)"], random_state=random_state)
    total_shocks = (df_ext["shock_opp1"] + df_ext["shock_opp2"]).values

    results = []
    for name, features in configs.items():
        res = fit_predict(
            df_ext, features, random_state=random_state, reference_labels=ref["labels"]
        )
        fit_col = [f for f in features if f not in base]
        corr = (
            np.corrcoef(df_ext[fit_col[0]].values, total_shocks)[0, 1]
            if fit_col
            else np.nan
        )

        results.append(
            {
                "feature set": name,
                "silhouette": round(res["silhouette"], 3),
                "ARI vs original": (
                    round(res["ari"], 3) if res["ari"] is not None else np.nan
                ),
                "corr(metric, shocks)": (
                    round(corr, 3) if not np.isnan(corr) else np.nan
                ),
                "sizes": res["sizes"],
            }
        )
    return pd.DataFrame(results).set_index("feature set")
