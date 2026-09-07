# Documentation

## Environment setup

```bash
pixi install                # Python + R dependencies
pixi run post_install       # brms (installed from CRAN)
```

## Pipeline

All commands use [pixi](https://pixi.sh) task definitions from `pixi.toml`. The
pipeline runs in the `default` environment unless noted otherwise.

| Stage                        | Docs                                   | Environment |
| ---------------------------- | -------------------------------------- | ----------- |
| 1. Computational modelling   | [1_vba.md](1_vba.md)                  | MATLAB      |
| 2. Clustering                | [2_clustering.md](2_clustering.md)    | `default`   |
| 3. Behavioural preparation   | [3_behavior.md](3_behavior.md)        | `default`   |
| 4. Physiology preparation    | [4_physio.md](4_physio.md)            | `default`   |
| 5. Bayesian models           | [5_brms.md](5_brms.md)               | `default`   |
| 6. Plots                     | [6_plots.md](6_plots.md)             | `default`   |
| 7. Core library              | [7_core.md](7_core.md)               | —           |

See also the [data README](../data_v2/README.md) for dataset and output
descriptions.
