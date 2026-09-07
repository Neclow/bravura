# Physiology preparation

Computes physiological features and exports long-format CSVs for brms models
([5_brms.md](5_brms.md)).

- **Command:** `pixi run prepare_physio`
- **Script:** `pipeline/prepare_physio.py`
- **Requires:** `behav_Xa.csv`, `behav_Xb.csv` from clustering
  ([2_clustering.md](2_clustering.md)); raw physiology and hormone data.

## Exclusions

Cohort A: subjects with incomplete HR or HRV data across all 5 blocks are
dropped (N=112 retained; P039 and P113 excluded). Cohort B: subjects with
incomplete HR (N=37 retained).

## Experimental blocks

| Code | Label | Description |
| ---- | ----- | ----------- |
| `Pre` | — | Resting baseline |
| `Op1T1` | 1.1 | Opponent 1, first half |
| `Op1T2` | 1.2 | Opponent 1, second half |
| `Op2T1` | 2.1 | Opponent 2, first half |
| `Op2T2` | 2.2 | Opponent 2, second half |

## Phase 1: Baseline HR

Resting HR per subject with cluster labels.

- **Outputs:** `data_v2/processed/baseline_hr.csv`
- **Columns:** `subject`, `HR_Pre`, `Cluster`
- **Consumed by:** `brms_baseline_hr`
- **Note:** Cohort A only.

## Phase 2: Delta HR

Change-from-baseline HR for each task block.

- **Outputs:** `data_v2/processed/delta_hr_long.csv` (Cohort A),
  `delta_hr_long_b.csv` (Cohort B)
- **Columns:** `subject`, `Cluster`, `block`, `delta_hr`
- **Consumed by:** `brms_delta_hr`, `brms_delta_hr_rep`

## Phase 3: Cardiac multivariate

Delta HR combined with varimax-rotated HRV components. PCA (3 components) is
fit on 17 HRV features pooled across blocks, then varimax-rotated:

| Component | Interpretation | Variance |
| --------- | -------------- | -------- |
| RC1 | Overall HRV power | 59% |
| RC2 | Vagal/parasympathetic | 24% |
| RC3 | Complexity/entropy | 17% |

Delta scores are computed as task-block minus Pre for each component.

- **Outputs:** `data_v2/processed/physio_cardiac_long.csv`
- **Columns:** `subject`, `Cluster`, `block`, `HR`, `HRV_RC1`, `HRV_RC2`, `HRV_RC3`
- **Consumed by:** `brms_physio_cardiac`
- **Note:** Cohort A only.

## Phase 4: Physio multivariate

Six physiological DVs: one representative feature plus one PC1 composite per
domain (cardiac, respiratory, electrodermal). Three separate 1-component PCAs
are fit on HRV, respiration, and EDA features respectively. Delta scores
computed as task minus Pre.

- **Outputs:** `data_v2/processed/physio_multivariate_long.csv`
- **Columns:** `subject`, `Cluster`, `block`, `HR`, `HRV_PC1`, `RespRate`,
  `Resp_PC1`, `nSCRcda`, `EDA_PC1`
- **Consumed by:** `brms_physio_mv`
- **Note:** Cohort A only. Subjects with incomplete respiratory/EDA data are
  additionally dropped.

## Phase 5: Hormones

Salivary cortisol and testosterone by cluster. Computes mean cortisol, mean
testosterone, T:C ratio (testosterone converted pg/ml to ug/dl), and
circadian-corrected stress reactivity.

- **Inputs:** `data_v2/raw/CortisolData.xlsx`,
  `data_v2/raw/VR main-testosterone-september2019-longxlsx.xlsx`,
  `data_v2/processed/behav_Xa.csv`
- **Outputs:** `data_v2/processed/hormones.csv`
- **Columns:** `subject`, `Cluster`, `Condition`, `TotalCort`, `Cmean`,
  `CortBase`, `StressChange`, `StressChange_corrected`, `hour`, `Testo_mean`,
  `TC_ratio`
- **Consumed by:** `brms_hormones`
- **Note:** Cohort A only. Subjects with missing testosterone are dropped.
