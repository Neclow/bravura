# plots/

Plotting scripts for the Bravura paper. Each script exposes one or more functions that produce a figure panel.

## Figure-to-script mapping

### Main figures

| Panel | Script | Description |
|-------|--------|-------------|
| 1 | ✗ (not code-generated) | Paradigm explainer |
| 2a | ✗ (not code-generated) | Paradigm schematic |
| 2b | [fig2_shocks_given.py](fig2_shocks_given.py) | Shocks given by opponent |
| 2c | [fig2_belief.py](fig2_belief.py) | Belief ratings by opponent |
| 2d | [fig2_trial_duration.py](fig2_trial_duration.py) | Trial duration by decision type |
| 3a | [fig3_decision_modelling.py](fig3_decision_modelling.py) | Observed vs predicted shock heatmaps |
| 3b | [fig3_pca.py](fig3_pca.py) | PCA projection of clusters |
| 3c | [fig3_radar.py](fig3_radar.py) | Radar charts of clustering features |
| 3d | [fig3_decision_modelling.py](fig3_decision_modelling.py) | Trial-by-trial P(shock) by cluster |
| 3e | [fig3_shock_per_cluster.py](fig3_shock_per_cluster.py) | Shocks by cluster and opponent |
| 3f | [fig3_psap.py](fig3_psap.py) | PSAP concurrent validity |
| 4a | [fig4_hr_baseline.py](fig4_hr_baseline.py) | Baseline heart rate by cluster |
| 4b | [fig4_hr_delta.py](fig4_hr_delta.py) | Delta-HR by cluster and block |
| 4c | [fig4_hr_delta.py](fig4_hr_delta.py) | Delta-1 HR by cluster and cohort |
| 4e | TODO | Cardiac pairwise contrasts (forest plot) |
| 4f | TODO | HRV PCA loadings |

### Supplementary figures

| Panel | Script | Description |
|-------|--------|-------------|
| S1 | [figS1_participants.py](figS1_participants.py) | Shock-vs-belief joint distribution with exclusion boundaries |
| S2a | [fig3_decision_modelling.py](fig3_decision_modelling.py) | Per-subject ROC curves |
| S2b | [fig3_decision_modelling.py](fig3_decision_modelling.py) | Model calibration curve |
| S3a | [fig3_decision_modelling.py](fig3_decision_modelling.py) | VBA parameter correlation matrix (simulation recovery) |
| S3b | [fig3_decision_modelling.py](fig3_decision_modelling.py) | Posterior covariance condition numbers |
| S4a | [figS4_cluster_analysis.py](figS4_cluster_analysis.py) | Silhouette score vs k |
| S4b | [figS4_cluster_analysis.py](figS4_cluster_analysis.py) | Gap statistic vs k |
| S4c | [figS4_cluster_analysis.py](figS4_cluster_analysis.py) | Per-subject silhouette plot (k=3) |
| S5 | [figS5_latency.py](figS5_latency.py) | Shock latency by cluster and opponent |
| S6a | [fig3_psap.py](fig3_psap.py) | PSAP cluster radar charts and contingency heatmap |
| S6b | [fig3_psap.py](fig3_psap.py) | Bravura shocks vs PSAP B-presses scatter |
| S7a | [fig4_hr_delta.py](fig4_hr_delta.py) | HR time course per cluster with grand mean |
| S7b | [fig4_hr_delta.py](fig4_hr_delta.py) | Per-subject delta-HR heatmap by cluster |
| S8 | [figS8_cardiac.py](figS8_cardiac.py) | Block 1.1 cardiac pairwise contrasts (forest plot) |

### Supplementary tables

Each table is saved as a `.md` file alongside its figure via `to_markdown()`.

| Table | Script | Description |
|-------|--------|-------------|
| T1 | [fig2_shocks_given.py](fig2_shocks_given.py) | BFs for shocks-by-opponent model |
| T2 | [fig2_belief.py](fig2_belief.py) | BFs for beliefs-by-opponent model |
| T3 | [fig2_trial_duration.py](fig2_trial_duration.py) | BFs for trial-duration model (pairwise decision contrasts) |
| T4 | [cluster_behavior.py](../pipeline/cluster_behavior.py) | Clustering solver sensitivity (k x solver silhouette grid) |
| T5 | [fig3_shock_per_cluster.py](fig3_shock_per_cluster.py) | BFs for shocks-by-cluster model (per opponent) |
| T6 | [fig3_psap.py](fig3_psap.py) | BFs for PSAP Dirichlet regression (cluster contrasts) |
| T7 | [fig4_hr_delta.py](fig4_hr_delta.py) | BFs for delta-HR model (cluster contrasts per block) |
| T8 | [fig4_hr_delta.py](fig4_hr_delta.py) | Replication BFs for delta-HR (sceptical BFr/BFs) |
| T9 | [cluster_behavior.py](../pipeline/cluster_behavior.py) | Clustering feature sensitivity (BMA metric ablation) |
| T10 | [figS4_cluster_analysis.py](figS4_cluster_analysis.py) | Silhouette/gap scores per k |
| T11 | [figS8_cardiac.py](figS8_cardiac.py) | Block 1.1 cardiac pairwise contrasts |
