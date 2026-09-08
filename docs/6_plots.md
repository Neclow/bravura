# Plots

All plot scripts live in `plots/` and use the custom matplotlib style
(`.matplotlib/paper.mplstyle`). Shared constants (image directory, style path,
script filename) are in `plots/_config.py`. Figures are saved as both PDF and
PNG (300 dpi) to `img_v2/`.

## Figure dependency table

| Figure | Panel | Script | Data |
| ------ | ----- | ------ | ---- |
| Fig 2 | b: Shocks given vs opponent | `fig2_shocks_given.py` | `brms/shocks_overview/posterior_epred.csv`, `bayes_factors.csv`, `AggressionScript.xlsx` |
| Fig 2 | c: Belief vs opponent | `fig2_belief.py` | `brms/beliefs_overview/posterior_epred.csv`, `bayes_factors.csv` |
| Fig 2 | Trial duration | `fig2_trial_duration.py` | `shared/trial_events.csv`, `cohort_{a,b}/outliers.mat`, `brms/trial_duration/` |
| Fig 3 | Decision modelling | `fig3_decision_modelling.py` | `cohort_{a,b}/predictions.csv`, `decisions.csv`, `subject_ids.csv`, `outliers.mat`, `corr_preds.mat`, `cov_stats.mat` |
| Fig 3 | PCA scatter | `fig3_pca.py` | `behav_Xa.csv`, `behav_Xb.csv` (passed as arguments) |
| Fig S1 | Participant exclusion | `figS1_participants.py` | `cohort_{a,b}/aggroPerformance.xlsx`, `beliefs.xlsx` |

Scripts with `pixi` tasks:

```bash
pixi run fig2_belief
pixi run fig2_shocks_given
pixi run fig2_trial_duration
```

The remaining scripts (`fig3_*`, `figS1_*`) do not have pixi tasks and are
called from notebooks or run directly.

## Not yet implemented

The following plot scripts exist as empty files:

- `fig3_psap.py`
- `fig3_radar.py`
- `fig3_shock_per_cluster.py`
- `fig3_shockprob.py`
