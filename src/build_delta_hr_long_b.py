"""Generate delta_hr_long_b_v{2,3}.csv for the brms replication analysis.

For each cohort B version, the output mirrors the v1
``data/processed/delta_hr_long_b.csv`` layout (long-format with columns
``subject, Cluster, block, delta_hr``). Cluster labels come from
projecting the cohort B 10-feature matrix onto cohort A's k-means
centroids (the same procedure ``notebooks/fig2.ipynb`` uses).

Run with:

    pixi run python -m src.build_delta_hr_long_b
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src._config import RANDOM_SEED, MAX_BELIEF_COHORT_A, MAX_BELIEF_COHORT_B
from src.preprocessing import load_behavioral_features

FEATURES = [
    "Kr1", "Krc", "Kp", "Kwc", "R2",
    "shock_opp1", "shock_opp2", "first_shock",
    "belief_opp1", "belief_opp2",
]
N_CLUSTERS = 3
BLOCK_TO_COL = {
    "1.1": "HR_Op1T1",
    "1.2": "HR_Op1T2",
    "2.1": "HR_Op2T1",
    "2.2": "HR_Op2T2",
}


def _load_cohort_a():
    coefs = pd.read_csv("data/cohort_a/coefficients.csv", index_col=0)
    metrics = pd.read_csv("data/cohort_a/fit_metrics.csv", index_col=0)
    aggro = pd.read_excel("data/cohort_a/aggroPerformance.xlsx", index_col=0)
    beliefs = pd.read_excel("data/cohort_a/beliefs.xlsx", index_col="ID")
    beliefs.rename(columns={k: k[4:] + k[0] for k in beliefs.columns}, inplace=True)
    beliefs.drop("opponent3", axis=1, errors="ignore", inplace=True)
    return load_behavioral_features(coefs, metrics, aggro, beliefs)


def _load_cohort_b(version):
    data_dir = Path(f"data/cohort_b_{version}")
    coefs = pd.read_csv(data_dir / "coefficients.csv", index_col=0)
    metrics = pd.read_csv(data_dir / "fit_metrics.csv", index_col=0)
    aggro = pd.read_excel(data_dir / "aggroPerformance.xlsx", index_col="Subject")
    beliefs = pd.read_excel(data_dir / "beliefs.xlsx", index_col="ID")
    beliefs = beliefs * (MAX_BELIEF_COHORT_A / MAX_BELIEF_COHORT_B)
    return load_behavioral_features(coefs, metrics, aggro, beliefs)


def _cluster_b(df_a, df_b):
    scaler = StandardScaler().fit(df_a[FEATURES])
    km = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10
    ).fit(scaler.transform(df_a[FEATURES]))
    labels_b = km.predict(scaler.transform(df_b[FEATURES]))

    # Re-map cluster ids to {Non-aggressive, Reactive, Proactive} by mean
    # shocks in cohort A (matches notebooks/fig2.ipynb convention).
    df_a = df_a.copy()
    df_a["_cl"] = km.labels_
    df_a["_total"] = df_a["shock_opp1"] + df_a["shock_opp2"]
    means = df_a.groupby("_cl")["_total"].mean().sort_values()
    name = {means.index[0]: "Non-aggressive",
            means.index[1]: "Reactive",
            means.index[2]: "Proactive"}
    return pd.Series([name[c] for c in labels_b], index=df_b.index, name="Cluster")


def _build_long(df_b, labels_b, phys):
    rows = []
    for sid in df_b.index:
        if sid not in phys.index:
            continue
        cluster = labels_b.loc[sid]
        hr_pre = phys.loc[sid, "HR_Pre"]
        for block, col in BLOCK_TO_COL.items():
            hr = phys.loc[sid, col]
            rows.append({
                "subject": sid,
                "Cluster": cluster,
                "block": float(block),
                "delta_hr": hr - hr_pre,
            })
    return pd.DataFrame(rows)


def main():
    df_a = _load_cohort_a()
    for version in ["v2", "v3"]:
        df_b = _load_cohort_b(version)
        labels_b = _cluster_b(df_a, df_b)
        phys = pd.read_excel(
            f"data/cohort_b_{version}/physPerformance.xlsx", index_col=0
        )
        long = _build_long(df_b, labels_b, phys)
        out = Path(f"data/processed/delta_hr_long_b_{version}.csv")
        long.to_csv(out, index=False)
        counts = long.drop_duplicates("subject")["Cluster"].value_counts().to_dict()
        print(f"{out}: {len(long)} rows, "
              f"{long['subject'].nunique()} subjects, clusters={counts}")


if __name__ == "__main__":
    main()
