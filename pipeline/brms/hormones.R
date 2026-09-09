# TEMPORARY: exploratory hormone models (Cohort A) by cluster.
#
# Fits one Student-t model per hormone variable: dv ~ Cluster + Condition.
# Condition (stress/control) is included because the cortisol samples are not
# aligned across groups; cluster effects are marginalised over Condition.
#
# Variables: TotalCort (tonic level), StressChange (raw reactivity proxy;
# circadian-corrected version unavailable), Testo_mean, TC_ratio.
#
# Outputs per variable to data/brms/physio_hormones/<var>/ (bayes_factors.csv,
# posterior_epred.csv, predicted_means.csv, diagnostics), matching baseline_hr.R.
#
# Run with: pixi run Rscript src/brms/_hormones.R [--overwrite]

source("src/brms/utils.R")

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

CLUSTER_LEVELS <- c("Non-aggressive", "Proactive", "Reactive")

# Per-variable priors (scales differ by orders of magnitude across hormones).
# Each entry: intercept mean/sd, slope sd, sigma scale.
VAR_CONFIG <- list(
  TotalCort = list(
    int_mean = 0.8,
    int_sd = 0.5,
    b_sd = 0.5,
    sigma_scale = 0.5,
    ppc_x = expression("Total cortisol (" * mu * "g/dL)"),
    ppc_xlim = c(0, 2.5)
  ),
  StressChange_corrected = list(
    int_mean = 0.0,
    int_sd = 0.1,
    b_sd = 0.1,
    sigma_scale = 0.1,
    ppc_x = expression(Delta * " cortisol (" * mu * "g/dL)"),
    ppc_xlim = NULL
  ),
  Testo_mean = list(
    int_mean = 145,
    int_sd = 60,
    b_sd = 60,
    sigma_scale = 60,
    ppc_x = "Mean testosterone (pg/mL)",
    ppc_xlim = c(0, 400)
  ),
  TC_ratio = list(
    int_mean = 0.1,
    int_sd = 0.1,
    b_sd = 0.1,
    sigma_scale = 0.1,
    ppc_x = "T/C ratio",
    ppc_xlim = c(0, 0.5)
  )
)

df <- read.csv(file.path(DEFAULT_PROCESSED_DIR, "hormones.csv"))
df$Cluster <- factor(df$Cluster, levels = CLUSTER_LEVELS)
df$Condition <- factor(df$Condition)

cat("N:", nrow(df), "| Per cluster:", table(df$Cluster), "\n\n")

fit_one <- function(var, cfg) {
  cat("\n========== ", var, " ==========\n", sep = "")
  out_dir <- file.path(DEFAULT_BRMS_DIR, "physio_hormones", var)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  formula <- as.formula(paste(var, "~ Cluster + Condition"))
  priors <- c(
    prior_string(
      paste0("normal(", cfg$int_mean, ", ", cfg$int_sd, ")"),
      class = "Intercept"
    ),
    prior_string(paste0("normal(0, ", cfg$b_sd, ")"), class = "b"),
    prior_string(
      paste0("student_t(3, 0, ", cfg$sigma_scale, ")"),
      class = "sigma"
    )
  )

  fit <- fit_or_load(
    paste0("fit_", var),
    out_dir,
    formula = formula,
    data = df,
    family = student(),
    prior = priors,
    chains = CHAINS,
    cores = CORES,
    iter = ITER,
    warmup = WARMUP,
    seed = SEED,
    overwrite = OVERWRITE
  )
  cat("Fixed effects:\n")
  print(round(fixef(fit), 3))

  fit_prior <- fit_or_load(
    paste0("fit_", var, "_prior"),
    out_dir,
    formula = formula,
    data = df,
    family = student(),
    prior = priors,
    sample_prior = "only",
    chains = CHAINS,
    cores = CORES,
    iter = 12000,
    warmup = 2000,
    seed = SEED,
    overwrite = OVERWRITE
  )

  # Savage-Dickey BFs on pairwise cluster contrasts (marginal over Condition).
  em_post <- emmeans(fit, pairwise ~ Cluster)
  em_prior <- emmeans(fit_prior, pairwise ~ Cluster)
  bf_results <- bf_table(em_post, em_prior)
  cat("\nPairwise contrasts (Savage-Dickey BF):\n")
  print(
    bf_results[, c(
      "contrast",
      "estimate",
      "Q2.5",
      "Q97.5",
      "BF10",
      "excl_zero"
    )],
    digits = 3
  )
  write.csv(
    bf_results,
    file.path(out_dir, "bayes_factors.csv"),
    row.names = FALSE
  )

  # Posterior predicted cluster means, marginalised over Condition.
  newdata <- expand.grid(
    Cluster = levels(df$Cluster),
    Condition = levels(df$Condition)
  )
  epred_long <- newdata %>%
    add_epred_draws(fit, re_formula = NA) %>%
    group_by(Cluster, .draw) %>%
    summarise(value = mean(.epred), .groups = "drop")
  names(epred_long)[names(epred_long) == "value"] <- var
  write.csv(
    epred_long[, c("Cluster", var, ".draw")],
    file.path(out_dir, "posterior_epred.csv"),
    row.names = FALSE
  )

  pred_summary <- epred_long %>%
    group_by(Cluster) %>%
    summarise(
      mean = mean(.data[[var]]),
      Q2.5 = quantile(.data[[var]], 0.025),
      Q97.5 = quantile(.data[[var]], 0.975),
      .groups = "drop"
    )
  write.csv(
    pred_summary,
    file.path(out_dir, "predicted_means.csv"),
    row.names = FALSE
  )
  cat("\nPredicted means:\n")
  print(pred_summary)

  save_diagnostics(
    fit,
    paste("student-t", var, "~ Cluster + Condition"),
    out_dir,
    prior_fit = fit_prior,
    ppc_labs = labs(x = cfg$ppc_x, y = "Density"),
    ppc_xlim = cfg$ppc_xlim
  )
  cat("\nDone:", var, "->", out_dir, "\n")
}

for (var in names(VAR_CONFIG)) {
  fit_one(var, VAR_CONFIG[[var]])
}

cat("\nAll hormone models complete.\n")
