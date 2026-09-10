"""Prepare intermediate behavioural CSVs for brms models.

Reads behav_Xa.csv / behav_Xb.csv (from cluster_behavior) and raw data,
produces:
    - shock_long.csv          (shocks.R)
    - psap_ilr.csv            (psap.R)
    - shock_latency_long.csv  (latency.R)
"""

import os

import numpy as np
import pandas as pd

from src._config import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_DIR, RANDOM_SEED

PROCESSED = DEFAULT_PROCESSED_DIR
SHARED = f"{DEFAULT_DATA_DIR}/shared"
RAW = f"{DEFAULT_DATA_DIR}/raw"


def make_shock_long():
    """Melt shock_opp1/shock_opp2 into long format (both cohorts)."""
    dfs = []
    for cohort, path in [("A", "behav_Xa.csv"), ("B", "behav_Xb.csv")]:
        df = pd.read_csv(f"{PROCESSED}/{path}", index_col=0)
        df.index.name = "subject"
        long = (
            df[["shock_opp1", "shock_opp2", "label", "Cluster"]]
            .reset_index()
            .melt(
                id_vars=["subject", "label", "Cluster"],
                value_vars=["shock_opp1", "shock_opp2"],
                var_name="opponent",
                value_name="shocks",
            )
        )
        long["opponent"] = long["opponent"].map(
            {"shock_opp1": "Opponent 1", "shock_opp2": "Opponent 2"}
        )
        long["cohort"] = cohort
        dfs.append(long)

    out = pd.concat(dfs, ignore_index=True)
    out_path = f"{PROCESSED}/shock_long.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")


def make_psap_ilr():
    """Noise-replace compositional zeros in PSAP data, merge cluster labels."""
    Xa = pd.read_csv(f"{PROCESSED}/behav_Xa.csv", index_col=0)
    psap = pd.read_excel(f"{RAW}/additional.xlsx", index_col="Subject")

    psap_cluster = psap[["pA", "pB", "pC", "rA", "rB", "rC"]].join(
        Xa[["label", "Cluster"]], how="inner"
    ).dropna()

    psap_dir = psap_cluster.reset_index()

    psap_pro = psap_dir[["Subject", "label", "Cluster", "pA", "pB", "pC"]].copy()
    psap_pro.columns = ["Subject", "label", "Cluster", "Earn", "Steal", "Protect"]
    psap_pro["phase"] = "Proactive"

    psap_rea = psap_dir[["Subject", "label", "Cluster", "rA", "rB", "rC"]].copy()
    psap_rea.columns = ["Subject", "label", "Cluster", "Earn", "Steal", "Protect"]
    psap_rea["phase"] = "Reactive"

    psap_comp = pd.concat([psap_pro, psap_rea], ignore_index=True)

    rng = np.random.default_rng(RANDOM_SEED)
    buttons = ["Earn", "Steal", "Protect"]
    for i, row in psap_comp.iterrows():
        vals = row[buttons].values.astype(float)
        zero_mask = vals == 0
        if zero_mask.any():
            noise = rng.normal(0.01, 0.0025, size=zero_mask.sum()).clip(1e-6, 0.025)
            vals[zero_mask] = noise
        vals = vals / vals.sum()
        psap_comp.loc[i, buttons] = vals

    out_path = f"{PROCESSED}/psap_ilr.csv"
    psap_comp.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(psap_comp)} rows)")


def make_shock_latency_long():
    """Filter trial events to shock trials, join cluster labels."""
    trials = pd.read_csv(f"{SHARED}/trial_events.csv")
    Xa = pd.read_csv(f"{PROCESSED}/behav_Xa.csv", index_col=0)

    shocks = trials[trials["choice"] == "shock"].copy()
    shocks = shocks[shocks["subject"].isin(Xa.index)]
    Xa_cluster = Xa[["Cluster"]].copy()
    Xa_cluster.index.name = "subject"
    shocks = shocks.merge(Xa_cluster, on="subject", how="left")
    out = shocks[["subject", "trial", "opponent", "duration", "Cluster"]].rename(
        columns={"duration": "latency"}
    )

    out_path = f"{PROCESSED}/shock_latency_long.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    make_shock_long()
    make_psap_ilr()
    make_shock_latency_long()
