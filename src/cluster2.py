from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from kmedoids import KMedoids
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler


class GaussianMixtureWrapper(GaussianMixture):
    """
    Wrapper to make GaussianMixture
    use `n_clusters` for consistency with other clusterers.
    """

    def __init__(self, n_clusters, **kwargs):
        super().__init__(n_components=n_clusters, **kwargs)
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        super().fit(X, y)
        self.cluster_centers_ = self.means_
        return self

    def fit_predict(self, X, y=None):
        labels = super().fit_predict(X, y)
        self.cluster_centers_ = self.means_
        return labels


class KMedoidsWrapper(KMedoids):
    """
    Wrapper to make KMedoids
    output a correct `predict` when metric is not "precomputed"

    https://github.com/kno10/python-kmedoids/blob/ef2b3b68ec0aaa9ee63d6e4bf9724cdfff874d17/kmedoids/__init__.py#L966
    """

    def predict(self, X):
        from sklearn.metrics.pairwise import pairwise_distances_argmin

        Y = self.cluster_centers_
        X = pairwise_distances_argmin(X, Y=Y, metric=self.metric)
        return X


@dataclass
class ClusterResult:
    clusterer: object
    labels: np.ndarray
    silhouette: float
    sizes: list


@dataclass
class FuzzyClusterResult:
    consensus_a: np.ndarray
    consensus_b: np.ndarray
    stability_a: float
    stability_b: float
    label_counts_a: np.ndarray
    label_counts_b: np.ndarray
    scaler_means: np.ndarray
    scaler_scales: np.ndarray
    n_samples: int


CLUSTERERS = {
    "k-means": partial(KMeans, n_init=10),
    "k-medoids": partial(KMedoidsWrapper, method="fasterpam", metric="euclidean"),
    "gmm": partial(GaussianMixtureWrapper, n_init=5),
    # "hac": partial(AgglomerativeClusteringWrapper, linkage="ward"),
    # "spectral": partial(SpectralClustering, affinity="rbf"),
}


def fit_predict(
    X: np.ndarray, solver: str, k: int, random_state: int, **kwargs
) -> np.ndarray:
    try:
        clusterer = CLUSTERERS[solver](
            n_clusters=k, random_state=random_state, **kwargs
        )
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


def fuzzy_fit_predict(
    iterator_a,
    iterator_b,
    solver,
    k,
    random_state,
    n_a,
    n_b,
    ref_labels,
    n_samples,
):
    scalers = []
    label_counts_a = np.zeros((n_a, k), dtype=int)
    label_counts_b = np.zeros((n_b, k), dtype=int)
    for X_mc_a, X_mc_b in zip(iterator_a, iterator_b):
        scaler = StandardScaler()
        X_a_scaled = scaler.fit_transform(X_mc_a)
        X_b_scaled = scaler.transform(X_mc_b)
        scalers.append(scaler)

        res_mc = fit_predict(X_a_scaled, solver=solver, k=k, random_state=random_state)
        labels_a_matched, label_map = _match_labels(res_mc.labels, ref_labels, k)

        labels_b_raw = res_mc.clusterer.predict(X_b_scaled)
        labels_b_matched = np.array([label_map[lb] for lb in labels_b_raw])

        for s in range(n_a):
            label_counts_a[s, labels_a_matched[s]] += 1
        for s in range(n_b):
            label_counts_b[s, labels_b_matched[s]] += 1

    consensus_a = label_counts_a.argmax(axis=1)
    consensus_b = label_counts_b.argmax(axis=1)
    stability_a = (label_counts_a.max(axis=1) / n_samples).mean()
    stability_b = (label_counts_b.max(axis=1) / n_samples).mean()

    np.savez_compressed(
        f"data/processed/mc_consensus_{solver}_{k}.npz",
        label_counts_a=label_counts_a,
        label_counts_b=label_counts_b,
        consensus_a=consensus_a,
        consensus_b=consensus_b,
        stability_a=stability_a,
        stability_b=stability_b,
        scaler_means=np.array([s.mean_ for s in scalers]),
        scaler_scales=np.array([s.scale_ for s in scalers]),
        n_samples=n_samples,
    )

    return FuzzyClusterResult(
        consensus_a=consensus_a,
        consensus_b=consensus_b,
        stability_a=stability_a,
        stability_b=stability_b,
        label_counts_a=label_counts_a,
        label_counts_b=label_counts_b,
        scaler_means=np.array([s.mean_ for s in scalers]),
        scaler_scales=np.array([s.scale_ for s in scalers]),
        n_samples=n_samples,
    )


def _shift(arr, num, fill_value=np.nan):
    """Shift an array.
    Source: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))


def ablate_k(X, solver, k_range, random_state):
    results = {}
    gaps, sks = np.zeros((len(k_range),)), np.zeros((len(k_range),))
    for i, k in enumerate(k_range):
        res = fit_predict(X, solver, k, random_state)
        y = res.labels
        gap, sk, method = gap_score(res.clusterer, X, y, random_state=random_state)
        gaps[i] = gap
        sks[i] = sk
        results[k] = res
    df = pd.DataFrame.from_dict(results, orient="index")

    sks_shifted = _shift(sks, -1)
    gaps_shifted = _shift(gaps, -1)
    diff = gaps - gaps_shifted + sks_shifted

    df["gap"] = gaps
    df["gap_diff"] = diff

    return df


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


def _match_labels(labels_mc, labels_ref, k):
    cost = np.zeros((k, k), dtype=int)
    for ref_c in range(k):
        for mc_c in range(k):
            cost[ref_c, mc_c] = -np.sum((labels_ref == ref_c) & (labels_mc == mc_c))
    row_ind, col_ind = linear_sum_assignment(cost)
    label_map = {col_ind[i]: row_ind[i] for i in range(k)}
    labels_matched = np.array([label_map[l] for l in labels_mc])
    return labels_matched, label_map


def pooled_distortion_score(X, labels, metric="sqeuclidean"):
    """
    Compute the pooled distortion score for each cluster.

    The distortion is computed as the the sum of the squared distances between
    each observation and its closest centroid. Logically, this is the metric
    that K-Means attempts to minimize as it is fitting the model.

    Source:
    https://github.com/DistrictDataLabs/yellowbrick/blob/develop/yellowbrick/cluster/elbow.py

    Parameters
    ----------
    X : array, shape = [n_samples, n_features] or [n_samples_a, n_samples_a]
        Array of pairwise distances between samples if metric == "precomputed"
        or a feature array for computing distances against the labels.
    labels : array, shape = [n_samples]
        Predicted labels for each sample
    metric : string
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by `sklearn.metrics.pairwise.pairwise_distances
        <http://bit.ly/2Z7Dxnn>`_
    .. todo:: add sample_size and random_state kwds similar to silhouette_score
    """
    # Encode labels to get unique centers and groups
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    unique_labels = label_encoder.classes_

    # Sum of the distortions
    distortion = 0

    # Loop through each label (center) to compute the centroid
    for current_label in unique_labels:
        # Mask the instances that belong to the current label
        mask = labels == current_label
        instances = X[mask]

        # # Compute the center of these instances
        # center = instances.mean(axis=0) # Isn't that wrong if diff from kmeans?

        # if not issparse(instances):
        #     center = np.array([center])

        # Compute the square distances from the instances to the center
        distances = pdist(instances, metric=metric)

        # Compute the square distances from the instances to the center
        # distances = pairwise_distances(instances, center, metric=metric)

        # Add half of the mean of square distance to the distortion
        # https://web.stanford.edu/~hastie/Papers/gap.pdf
        distortion += distances.sum() / (2 * instances.shape[0])

    return distortion


def gap_score(
    clusterer,
    X,
    labels,
    n_refs=10,
    method="star",
    distribution="normal",
    random_state=None,
):
    """Compute the gap statistic of a given clusterer.

    Sources:
    https://github.com/milesgranger/gap_statistic/blob/2b149e98af755390ef679a3610a61eb733a135e4/gap_statistic/optimalK.py
    https://statweb.stanford.edu/~gwalther/gap
    https://arxiv.org/pdf/1103.4767.pdf
    https://doi.org/10.1111/j.1541-0420.2007.00784.x

    Parameters
    ----------
    clusterer : sklearn.base.ClusterMixin
        A scikit-learn clusterer.
    X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        Training instances to cluster.
    labels : array, shape = [n_samples]
        Predicted labels for each sample
    n_refs : int, optional
        Number of random reference data sets used as inertia reference to actual data, by default 10
        https://stackoverflow.com/questions/51032086/recommended-number-of-simulated-reference-datasets-for-gap-statistic
    method : str, optional
        gap score computation method, by default 'log'.
        Available methods: 'log' (original method) and 'star' (without logs)

    Returns
    -------
    gap : float
        Gap statistic.
    sdk : float
        Standard deviation
    sk  : float
        s_k value defined in https://statweb.stanford.edu/~gwalther/gap

    Raises
    ------
    ValueError
        If the chosen gap score method is not available.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute real dispersion
    real_dispersion = pooled_distortion_score(X, labels)

    # Compute reference dispersions
    ref_dispersions = np.zeros(n_refs)

    # For n_references, generate random samples and get clsutering distortion scores
    for i in range(n_refs):

        # Create new random reference set uniformly over the range of each feature
        if distribution == "normal":
            random_data = np.random.normal(loc=X.mean(0), scale=X.std(0), size=X.shape)
        elif distribution == "uniform":
            x_min, x_max = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
            random_data = (
                np.random.random_sample(size=X.shape) * (x_max - x_min) + x_min
            )
        else:
            raise ValueError("Unknown distribution.")

        # Fit clusterer and compute distortion score
        labels = clusterer.fit_predict(random_data)

        dispersion = pooled_distortion_score(random_data, labels)

        ref_dispersions[i] = dispersion

    if method == "log":
        # https://statweb.stanford.edu/~gwalther/gap
        final_ref_dispersion = np.log(ref_dispersions)
        final_real_dispersion = np.log(real_dispersion)

    elif method == "star":
        # https://arxiv.org/pdf/1103.4767.pdf
        final_ref_dispersion = ref_dispersions
        final_real_dispersion = real_dispersion

    else:
        raise ValueError(f'Method {method} not available. Use "log" or "star".')

    gap = np.mean(final_ref_dispersion) - final_real_dispersion
    sdk = np.std(final_ref_dispersion)
    sk = np.sqrt(1.0 + 1.0 / n_refs) * sdk
    return gap, sk, method
