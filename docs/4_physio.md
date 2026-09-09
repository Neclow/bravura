# Physiology preparation

Computes physiological features and exports long-format CSVs for brms models
([5_brms.md](5_brms.md)).

- **Command:** `pixi run prepare_physio`
- **Script:** `pipeline/prepare_physio.py`
- **Requires:** `behav_Xa.csv`, `behav_Xb.csv` from clustering
  ([2_clustering.md](2_clustering.md)); raw physiology data.

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
