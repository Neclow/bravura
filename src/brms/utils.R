library(brms)
library(tidybayes)
library(dplyr)

# ── Constants ────────────────────────────────────────────────────────────────

SEED <- 42
CHAINS <- 4
ITER <- 4000
WARMUP <- 2000

# ── Helpers ──────────────────────────────────────────────────────────────────

fit_or_load <- function(name, out_dir, ...) {
  path <- file.path(out_dir, paste0(name, ".rds"))
  if (file.exists(path)) {
    cat("Loading cached model:", name, "\n")
    readRDS(path)
  } else {
    cat("Fitting model:", name, "\n")
    fit <- brm(...)
    saveRDS(fit, path)
    fit
  }
}

kfold_or_load <- function(fit, name, out_dir, K = 10) {
  path <- file.path(out_dir, paste0(name, "_kfold.rds"))
  if (file.exists(path)) {
    cat("Loading cached kfold:", name, "\n")
    readRDS(path)
  } else {
    cat("Computing kfold:", name, "\n")
    kf <- kfold(fit, K = K)
    saveRDS(kf, path)
    kf
  }
}

save_diagnostics <- function(best, best_label, out_dir, prior_fit = NULL) {
  sink(file.path(out_dir, "summary.txt"))
  cat(paste0("Model: ", best_label, "\n\n"))
  summary(best)
  cat("\n\nPrior summary:\n")
  prior_summary(best)
  sink()

  if (!is.null(prior_fit)) {
    png(file.path(out_dir, "prior_predictive_check.png"),
        width = 6, height = 4, units = "in", res = 300)
    print(pp_check(prior_fit, ndraws = 100))
    dev.off()
  }

  png(file.path(out_dir, "posterior_predictive_check.png"),
      width = 6, height = 4, units = "in", res = 300)
  print(pp_check(best, ndraws = 100))
  dev.off()

  png(file.path(out_dir, "trace_plots.png"),
      width = 10, height = 8, units = "in", res = 300)
  print(plot(best))
  dev.off()
}
