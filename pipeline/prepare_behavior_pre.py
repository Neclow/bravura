"""Detect and save outlier participants for each cohort.

Saves to data_v2/cohort_{cohort}/:
    outliers.mat    MATLAB-compatible cell array (legacy)
    outliers.txt    One subject ID per line
"""

import os

import numpy as np
import pandas as pd

from scipy.io import savemat

from src._config import (
    DEFAULT_DATA_DIR,
    MAX_BELIEF_COHORT_A,
    MAX_BELIEF_COHORT_B,
    MAX_SHOCKS,
    MIN_BELIEF,
    MIN_SHOCKS,
)

COHORTS = {
    "a": {"belief_scale": 1},
    "b": {"belief_scale": MAX_BELIEF_COHORT_A / MAX_BELIEF_COHORT_B},
}


def detect_outliers(cohort):
    """Identify outlier participants based on shock and belief thresholds.

    Parameters
    ----------
    cohort : str
        'a' or 'b'.

    Returns
    -------
    pd.Index
        Subject IDs flagged as outliers.
    """
    cohort_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"

    aggro = pd.read_excel(f"{cohort_dir}/aggroPerformance.xlsx", index_col="Subject")
    beliefs = pd.read_excel(f"{cohort_dir}/beliefs.xlsx", index_col="ID")

    if sorted(beliefs.columns) != ["opponent1", "opponent2"]:
        beliefs.rename(columns={k: k[4:] + k[0] for k in beliefs.columns}, inplace=True)
    beliefs.drop("opponent3", axis=1, errors="ignore", inplace=True)

    scale = COHORTS[cohort]["belief_scale"]
    if scale != 1:
        beliefs = beliefs * scale

    shock_cols = [c for c in aggro.columns if c.startswith("shock")]
    total_shocks = aggro[shock_cols].sum(axis=1)
    mean_belief = beliefs.mean(axis=1)

    common = total_shocks.index.intersection(mean_belief.dropna().index)
    shocks = total_shocks.loc[common]
    belief = mean_belief.loc[common]

    mask = ((shocks < MIN_SHOCKS) | (shocks > MAX_SHOCKS)) & (belief < MIN_BELIEF)
    return mask[mask].index


def load_outliers(cohort):
    """Read saved outlier IDs for a cohort.

    Parameters
    ----------
    cohort : str
        'a' or 'b'.

    Returns
    -------
    list[str]
        Subject IDs to exclude.
    """
    path = f"{DEFAULT_DATA_DIR}/cohort_{cohort}/outliers.txt"
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    for cohort in COHORTS:
        outlier_ids = detect_outliers(cohort)
        out_dir = f"{DEFAULT_DATA_DIR}/cohort_{cohort}"
        os.makedirs(out_dir, exist_ok=True)

        savemat(
            f"{out_dir}/outliers.mat",
            {"outliers": np.array(sorted(outlier_ids), dtype=object)},
        )

        with open(f"{out_dir}/outliers.txt", "w", encoding="utf-8") as f:
            for sid in sorted(outlier_ids):
                f.write(f"{sid}\n")

        print(f"Cohort {cohort.upper()}: {len(outlier_ids)} outliers -> {out_dir}")
