"""Prepare physiological data for brms modelling.

Phase 1: Baseline HR:
    Resting HR per subject with cluster labels → baseline_hr.csv.
Phase 2: Delta HR:
    Change-from-baseline HR in long format → delta_hr_long.csv,
    delta_hr_long_b.csv.
Phase 3: Cardiac multivariate:
    Delta HR + varimax-rotated HRV RC1-3 → physio_cardiac_long.csv.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src._config import (
    CLUSTER_NAMES,
    DEFAULT_CLUSTER_DIR_A,
    DEFAULT_CLUSTER_DIR_B,
    DEFAULT_PHYSIO_DIR_A,
    DEFAULT_PHYSIO_DIR_B,
    RANDOM_SEED,
)

CLUSTER_DIRS = {
    "a": DEFAULT_CLUSTER_DIR_A,
    "b": DEFAULT_CLUSTER_DIR_B,
}
PHYSIO_DIRS = {
    "a": DEFAULT_PHYSIO_DIR_A,
    "b": DEFAULT_PHYSIO_DIR_B,
}

# Feature sets
HRV_FEATS = [
    "SDNN",
    "RMSSD",
    "pnn50",
    "SD1",
    "SD2",
    "SD1SD2",
    "ac",
    "dc",
    "rrHRV",
    "lf",
    "hf",
    "lfhf",
    "ttlpwr",
    "SampEn",
    "ApEn",
    "TINN",
    "TRI",
]

# Block structure
ALL_BLOCKS = ["Pre", "Op1T1", "Op1T2", "Op2T1", "Op2T2"]
TASK_BLOCKS = ["Op1T1", "Op1T2", "Op2T1", "Op2T2"]
TASK_BLOCK_LABELS = ["1.1", "1.2", "2.1", "2.2"]


# Data loading
def load_physio(cohort):
    """Load physiology + behavioural cluster labels for one cohort.

    Cohort A: subjects with complete HR + HRV across all blocks (N=112).
    Cohort B: subjects with complete HR across all blocks (N=37).

    Parameters
    ----------
    cohort : str
        "a" or "b".

    Returns
    -------
    physio : pd.DataFrame
        Full physiology data (all columns) for qualifying subjects.
    labels : pd.Series
        Integer cluster labels indexed by subject.
    """
    behav = pd.read_csv(f"{CLUSTER_DIRS[cohort]}/clusters.csv", index_col=0)
    physio = pd.read_excel(
        f"{PHYSIO_DIRS[cohort]}/physPerformance.xlsx",
        index_col="Subject",
    )

    shared = behav.index.intersection(physio.index)

    if cohort == "a":
        cardiac_cols = [f"HR_{b}" for b in ALL_BLOCKS] + [
            f"{f}_{b}"
            for f in HRV_FEATS
            for b in ALL_BLOCKS
            if f"{f}_{b}" in physio.columns
        ]
        keep = physio.loc[shared, cardiac_cols].dropna().index
    else:
        hr_cols = [f"HR_{b}" for b in ALL_BLOCKS]
        keep = physio.loc[shared, hr_cols].dropna().index

    labels = behav.loc[keep, "label"]

    n_dropped = len(shared) - len(keep)
    print(f"Cohort {cohort.upper()}: {len(keep)} subjects with complete physio")
    if n_dropped:
        missing = sorted(shared.difference(keep))
        print(f"  Dropped {n_dropped}: {missing}")

    return physio.loc[keep], labels


# Helpers
def varimax(Phi, max_iter=100, tol=1e-6):
    """Varimax rotation on a loadings matrix (n_features x n_components)."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        d_old = d
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T
            @ (Lambda**3 - (Lambda * np.sum(Lambda**2, axis=0, keepdims=True)) / p)
        )
        R = u @ vt
        d = np.sum(s)
        if d - d_old < tol:
            break
    return Phi @ R


def _available_feats(physio, feat_list):
    """Return features from feat_list present across all blocks in physio."""
    return [
        f for f in feat_list if all(f"{f}_{b}" in physio.columns for b in ALL_BLOCKS)
    ]


def _pool_and_fit_pca(physio, feats, n_components):
    """Fit StandardScaler + PCA on raw features pooled across all blocks.

    Returns
    -------
    scaler : StandardScaler
    pca : PCA
    pre_scores : ndarray
        Scores for the Pre block.
    """
    pooled = pd.concat(
        [
            pd.DataFrame({f: physio[f"{f}_{b}"].values for f in feats})
            for b in ALL_BLOCKS
        ],
        ignore_index=True,
    )
    scaler = StandardScaler().fit(pooled)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(
        scaler.transform(pooled)
    )

    pre_raw = pd.DataFrame({f: physio[f"{f}_Pre"].values for f in feats})
    pre_scores = pca.transform(scaler.transform(pre_raw))

    return scaler, pca, pre_scores


def _block_delta_scores(physio, feats, blk, scaler, pca, pre_scores):
    """Compute delta PCA scores (block minus Pre) for one block."""
    raw = pd.DataFrame({f: physio[f"{f}_{blk}"].values for f in feats})
    blk_scores = pca.transform(scaler.transform(raw))
    return blk_scores - pre_scores


# Phase 1: Baseline HR
def export_baseline_hr(physio, labels, out_dir):
    """Export resting HR per subject with cluster labels."""
    df = pd.DataFrame(
        {
            "subject": physio.index,
            "HR_Pre": physio["HR_Pre"].values,
            "Cluster": labels.map(CLUSTER_NAMES).values,
        }
    )
    path = f"{out_dir}/baseline_hr.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print()


# Phase 2: Delta HR
def export_delta_hr(physio, labels, cohort, out_dir):
    """Export delta-HR (change from Pre) in long format for one cohort."""
    task_hr = [f"HR_{b}" for b in TASK_BLOCKS]
    delta = physio[task_hr].sub(physio["HR_Pre"], axis=0)
    delta["subject"] = delta.index
    delta["Cluster"] = labels.map(CLUSTER_NAMES).values

    long = delta.melt(
        id_vars=["subject", "Cluster"],
        value_vars=task_hr,
        var_name="block",
        value_name="delta_hr",
    )
    long["block"] = long["block"].map(dict(zip(task_hr, TASK_BLOCK_LABELS)))

    suffix = "" if cohort == "a" else f"_{cohort}"
    path = f"{out_dir}/delta_hr_long{suffix}.csv"
    long.to_csv(path, index=False)
    print(f"Saved {len(long)} rows to {path}")


# Phase 3: Cardiac multivariate
def export_cardiac(physio, labels, out_dir):
    """Export delta HR + varimax-rotated HRV RC1-3 in long format."""
    available = _available_feats(physio, HRV_FEATS)

    scaler, pca, pre_scores_raw = _pool_and_fit_pca(physio, available, n_components=3)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotated = varimax(loadings)

    rot_var = (rotated**2).sum(axis=0)
    order = np.argsort(rot_var)[::-1]
    rotated = rotated[:, order]

    R_varimax = np.linalg.lstsq(loadings, rotated, rcond=None)[0]

    pre_raw = pd.DataFrame({f: physio[f"{f}_Pre"].values for f in available})
    pre_scores = scaler.transform(pre_raw) @ pca.components_.T @ R_varimax

    cluster_labels = labels.map(CLUSTER_NAMES)

    rows = []
    for subj in physio.index:
        for blk, blk_label in zip(TASK_BLOCKS, TASK_BLOCK_LABELS):
            rows.append(
                {
                    "subject": subj,
                    "Cluster": cluster_labels[subj],
                    "block": blk_label,
                    "HR": physio.loc[subj, f"HR_{blk}"] - physio.loc[subj, "HR_Pre"],
                }
            )
    df = pd.DataFrame(rows)

    for blk, blk_label in zip(TASK_BLOCKS, TASK_BLOCK_LABELS):
        raw = pd.DataFrame({f: physio[f"{f}_{blk}"].values for f in available})
        blk_scores = scaler.transform(raw) @ pca.components_.T @ R_varimax
        delta_scores = blk_scores - pre_scores

        mask = df["block"] == blk_label
        df.loc[mask, "HRV_RC1"] = delta_scores[:, 0]
        df.loc[mask, "HRV_RC2"] = delta_scores[:, 1]
        df.loc[mask, "HRV_RC3"] = delta_scores[:, 2]

    path = f"{out_dir}/physio_cardiac_long.csv"
    df.to_csv(path, index=False)
    print(f"Saved {df.shape} to {path}")

    rc_names = ["Overall HRV power", "Vagal/parasympathetic", "Complexity/entropy"]
    for i in range(3):
        top_idx = np.argsort(np.abs(rotated[:, i]))[::-1][:4]
        top = ", ".join(f"{available[j]}={rotated[j, i]:+.2f}" for j in top_idx)
        print(f"  RC{i+1} ({rc_names[i]}): {top}")
    print()


# Main
if __name__ == "__main__":
    physio_a, labels_a = load_physio("a")
    physio_b, labels_b = load_physio("b")
    print()

    print("=== Phase 1: Baseline HR (Cohort A)===")
    export_baseline_hr(physio_a, labels_a, out_dir=DEFAULT_PHYSIO_DIR_A)

    print("=== Phase 2: Delta HR (both cohorts) ===")
    export_delta_hr(physio_a, labels_a, "a", out_dir=DEFAULT_PHYSIO_DIR_A)
    export_delta_hr(physio_b, labels_b, "b", out_dir=DEFAULT_PHYSIO_DIR_B)
    print()

    print("=== Phase 3: Cardiac (HR + varimax HRV RC1-3; Cohort A) ===")
    export_cardiac(physio_a, labels_a, out_dir=DEFAULT_PHYSIO_DIR_A)
