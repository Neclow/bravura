from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from kmedoids import KMedoids
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


class GaussianMixtureWrapper(GaussianMixture):
    """
    Wrapper to make GaussianMixture
    use `n_clusters` for consistency with other clusterers.
    """

    def __init__(self, n_clusters, **kwargs):
        super().__init__(n_components=n_clusters, **kwargs)
        self.n_clusters = n_clusters


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    """
    Wrapper to make AgglomerativeClustering
    use `random_state` for consistency with other clusterers.
    """

    def __init__(self, random_state, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state


@dataclass
class ClusterResult:
    clusterer: object
    labels: np.ndarray
    silhouette: float
    sizes: list


CLUSTERERS = {
    "k-means": partial(KMeans, n_init=10),
    "k-medoids": partial(KMedoids, method="fasterpam", metric="euclidean"),
    "hac": partial(AgglomerativeClusteringWrapper, linkage="ward"),
    "gmm": partial(GaussianMixtureWrapper, n_init=5),
    # "spectral": partial(SpectralClustering, affinity="rbf"),
}


def fit_predict(X: np.ndarray, solver: str, k: int, random_state: int) -> np.ndarray:
    try:
        clusterer = CLUSTERERS[solver](n_clusters=k, random_state=random_state)
    except KeyError as exc:
        raise ValueError(
            (
                f"Unsupported solver: {solver}.\n"
                f"Supported solvers are: {list(CLUSTERERS.keys())}"
            )
        ) from exc

    labels = clusterer.fit_predict(X)

    sil = silhouette_score(X, labels)

    return ClusterResult(
        clusterer=clusterer,
        labels=labels,
        silhouette=sil,
        sizes=np.bincount(labels, minlength=k).tolist(),
    )


def ablate_k(X, solver, k_range, random_state):
    results = {}
    for k in k_range:
        res = fit_predict(X, solver, k, random_state)
        results[k] = res
    return pd.DataFrame.from_dict(results, orient="index")


def ablate_solver(X, solvers, k, random_state):
    results = {}
    for solver in solvers:
        res = fit_predict(X, solver, k, random_state)
        results[solver] = res
    return pd.DataFrame.from_dict(results, orient="index")


def ablate_X(Xs, solver, k, random_state):
    results = {}
    for name, X in Xs.items():
        res = fit_predict(X, solver, k, random_state)
        results[name] = res
    return pd.DataFrame.from_dict(results, orient="index")
