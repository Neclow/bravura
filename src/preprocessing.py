"""Preprocessing utilities for behavioral data."""

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import brier_score_loss, roc_auc_score

from src._config import (
    MAX_BELIEF_COHORT_A,
    MIN_BELIEF_COHORT_A,
    N_OPPONENTS,
    N_TRIALS,
    RANDOM_SEED,
)


def collect_metrics(vba_metrics, pred, actual):
    """Merge MATLAB fit metrics with per-subject AUC and Brier score.

    Parameters
    ----------
    matlab_metrics : pd.DataFrame
        Fit metrics from VBA export (index=subject IDs).
    pred : pd.DataFrame
        Predicted P(shock) per trial (n_subjects x n_trials).
    actual : pd.DataFrame
        Actual binary decisions per trial (n_subjects x n_trials).

    Returns
    -------
    pd.DataFrame
        ``matlab_metrics`` with ``AUC`` and ``Brier`` columns appended.
    """
    pred = np.asarray(pred)
    actual = np.asarray(actual)

    briers = {}
    aucs = {}
    for i, subject in enumerate(vba_metrics.index):
        mask = ~np.isnan(actual[i])
        y_true = actual[i][mask]
        y_pred = pred[i][mask]
        briers[subject] = brier_score_loss(y_true, y_pred)
        if len(np.unique(y_true)) > 1:
            aucs[subject] = roc_auc_score(y_true, y_pred)
        else:
            aucs[subject] = 0.5

    metrics = vba_metrics.copy()
    metrics["AUC"] = aucs
    metrics["Brier"] = briers
    return metrics


def load_behavioral_features(coefs, metrics, aggro, beliefs=None):
    """Build the behavioral feature dataframe from raw data sources.

    Parameters
    ----------
    coefs : pd.DataFrame
        BMA coefficients (index=subject IDs, columns=Kr1, Krc, Kp, Kwc)
    metrics : pd.DataFrame
        Fit metrics per subject (index=subject IDs).
    aggro : pd.DataFrame
        aggroPerformance data (index=subject IDs)
    beliefs : pd.DataFrame
        Opponent belief data (columns=opponent1, opponent2; index=subject IDs)

    Returns
    -------
    pd.DataFrame
        Combined behavioral features, with imputed beliefs and first_shock.
        Outlier removal is handled upstream (pipeline/prepare_behavior_pre.py).
    """
    shock_cols = sorted(
        [c for c in aggro.columns if c.startswith("shock")],
        key=lambda c: int(c.strip().split("_")[-1]),
    )

    shock_opp1 = aggro[shock_cols[:N_TRIALS]].sum(axis=1)
    shock_opp2 = aggro[shock_cols[N_TRIALS : N_TRIALS * N_OPPONENTS]].sum(axis=1)

    shock_data = aggro[shock_cols]
    first_shock = shock_data.apply(
        lambda row: row.values.nonzero()[0][0] if row.any() else np.nan, axis=1
    )

    df = coefs.copy()
    df[metrics.columns.tolist()] = metrics.reindex(df.index)
    df["shock_opp1"] = shock_opp1.reindex(df.index)
    df["shock_opp2"] = shock_opp2.reindex(df.index)
    df["first_shock"] = first_shock.reindex(df.index)
    if beliefs is not None:
        df["belief_opp1"] = beliefs["opponent1"].reindex(df.index)
        df["belief_opp2"] = beliefs["opponent2"].reindex(df.index)

    df = impute_missing(df)

    return df


def sample_behavioral_features(
    coefs_mu,
    coefs_sigma,
    metrics,
    aggro,
    beliefs,
    n_samples=1000,
    random_state=RANDOM_SEED,
    cols_to_use=None,
):
    """Generate MC samples of behavioral features from VBA posteriors.

    Resamples the 4 VBA coefficients from their posterior distributions;
    all other features (metrics, shocks, beliefs) stay fixed.
    Outlier removal is handled upstream (pipeline/prepare_behavior_pre.py).

    Parameters
    ----------
    coefs_mu : np.ndarray
        Posterior means of coefficients (shape: n_subjects x 4).
    coefs_sigma : np.ndarray
        Posterior covariances of coefficients (shape: n_subjects x 4 x 4).
    metrics : pd.DataFrame
        Fit metrics per subject (index=subject IDs), from ``collect_metrics``.
    aggro : pd.DataFrame
        aggroPerformance data (index=subject IDs).
    beliefs : pd.DataFrame
        Opponent belief data (columns=opponent1, opponent2; index=subject IDs).
    n_samples : int
        Number of MC samples to draw.
    random_state : int
        Random seed for reproducibility.

    Yields
    ------
    pd.DataFrame
        Sampled behavioral features (beliefs imputed).
    """
    rng = np.random.default_rng(random_state)
    coef_cols = ["Kr1", "Krc", "Kp", "Kwc"]

    for _ in range(n_samples):
        coefs_sampled = np.array(
            [
                rng.multivariate_normal(mu, sigma)
                for mu, sigma in zip(coefs_mu, coefs_sigma)
            ]
        )
        df_sampled = load_behavioral_features(
            coefs=pd.DataFrame(coefs_sampled, index=metrics.index, columns=coef_cols),
            metrics=metrics,
            aggro=aggro,
            beliefs=beliefs,
        )
        if cols_to_use is not None:
            df_sampled = df_sampled[cols_to_use]
        yield df_sampled


def impute_missing(df, random_state=RANDOM_SEED):
    """Impute missing values in behavioral features.

    - first_shock NaN → N_TRIALS * N_OPPONENTS (never shocked)
    - belief columns → IterativeImputer with BayesianRidge, bounded [0, 10]

    Matches the original analysis in dev/figure24.py.

    Parameters
    ----------
    df : pd.DataFrame
        Behavioral features with possible NaN in first_shock and belief columns
    random_state : int
        Random seed for imputer

    Returns
    -------
    pd.DataFrame
        Imputed dataframe (no NaN)
    """
    df = df.copy()

    # Fill first_shock NaN with "never shocked"
    df["first_shock"] = df["first_shock"].fillna(N_TRIALS * N_OPPONENTS)

    # Impute belief columns using iterative imputation
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        random_state=random_state,
        min_value=MIN_BELIEF_COHORT_A,
        max_value=MAX_BELIEF_COHORT_A,
    )

    df_imputed = pd.DataFrame(
        imp.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )

    return df_imputed
