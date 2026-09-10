# Behavioural preparation

Reshapes clustered behavioural data into long-format CSVs consumed by brms
models ([5_brms.md](5_brms.md)).

- **Command:** `pixi run prepare_behavior_post`
- **Script:** `pipeline/prepare_behavior_post.py`
- **Requires:** `behav_Xa.csv`, `behav_Xb.csv` from clustering
  ([2_clustering.md](2_clustering.md)).

## Shock counts (long format)

Melts `shock_opp1`/`shock_opp2` from both cohorts into one long dataframe.

- **Inputs:** `data_v2/processed/behav_Xa.csv`, `behav_Xb.csv`
- **Outputs:** `data_v2/processed/shock_long.csv`
- **Columns:** `subject`, `label`, `Cluster`, `opponent`, `shocks`, `cohort`
- **Consumed by:** `brms_shocks`, `brms_shocks_overview`

## PSAP compositions

Prepares compositional PSAP button-press data (Earn, Steal, Protect) for
Dirichlet regression. Zeros are noise-replaced with draws from
N(0.01, 0.0025) clipped to [1e-6, 0.025], then rows renormalised to sum
to 1. Proactive and reactive phases stacked into long format.

- **Inputs:** `data_v2/processed/behav_Xa.csv`, `data_v2/raw/additional.xlsx`
- **Outputs:** `data_v2/processed/psap_ilr.csv`
- **Columns:** `Subject`, `label`, `Cluster`, `Earn`, `Steal`, `Protect`, `phase`
- **Consumed by:** `brms_psap`
- **Note:** Cohort A only.

## Shock latency

Filters trial events to shock trials only, joins cluster labels.

- **Inputs:** `data_v2/shared/trial_events.csv`, `data_v2/processed/behav_Xa.csv`
- **Outputs:** `data_v2/processed/shock_latency_long.csv`
- **Columns:** `subject`, `trial`, `opponent`, `latency`, `Cluster`
- **Consumed by:** `brms_latency`
- **Note:** Cohort A only.
