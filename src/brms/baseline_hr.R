source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────
# Test whether baseline HR differs between clusters.
# Expected result: no difference (supports state vs trait argument).

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/baseline_hr"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ────────────────────────────────────────────────────────────────

df <- read.csv("data/processed/baseline_hr.csv")
df$Cluster <- factor(df$Cluster, levels = c("Non-aggressive", "Proactive", "Reactive"))

cat("N:", nrow(df), "\n")
cat("Per cluster:", table(df$Cluster), "\n")
cat("Mean HR_Pre per cluster:\n")
print(tapply(df$HR_Pre, df$Cluster, function(x) round(c(mean = mean(x), sd = sd(x)), 2)))
cat("\n")

# ── Fit model ────────────────────────────────────────────────────────────────

priors <- c(
  prior(normal(85, 15), class = "Intercept"),
  prior(normal(0, 10), class = "b"),
  prior(student_t(3, 0, 15), class = "sigma")
)

fit <- fit_or_load(
  "fit_baseline_hr",
  out_dir,
  formula = HR_Pre ~ Cluster,
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
print(round(fixef(fit), 2))

# ── Savage-Dickey BFs ────────────────────────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_baseline_hr_prior",
  out_dir,
  formula = HR_Pre ~ Cluster,
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

em_post <- emmeans(fit, pairwise ~ Cluster)
em_prior <- emmeans(fit_prior, pairwise ~ Cluster)

bf_obj <- bayesfactor_parameters(em_post$contrasts, prior = em_prior$contrasts)

contrasts_summary <- as.data.frame(em_post$contrasts)
bf_df <- as.data.frame(bf_obj)

bf_table <- contrasts_summary
bf_table$BF10 <- exp(bf_df$log_BF)
bf_table$excl_zero <- contrasts_summary$lower.HPD > 0 | contrasts_summary$upper.HPD < 0

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_table[, c("contrast", "estimate", "lower.HPD", "upper.HPD", "BF10", "excl_zero")], digits = 3)

write.csv(bf_table, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

# ── Posterior predicted means per cluster ─────────────────────────────────────

newdata <- data.frame(Cluster = levels(df$Cluster))

epred_long <- newdata %>%
  add_epred_draws(fit, re_formula = NA) %>%
  rename(HR_Pre = .epred) %>%
  select(Cluster, HR_Pre, .draw)

write.csv(
  epred_long,
  file.path(out_dir, "posterior_epred.csv"),
  row.names = FALSE
)

pred_summary <- epred_long %>%
  group_by(Cluster) %>%
  summarise(
    mean = mean(HR_Pre),
    Q2.5 = quantile(HR_Pre, 0.025),
    Q97.5 = quantile(HR_Pre, 0.975),
    .groups = "drop"
  )
write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
cat("\nPredicted means:\n")
print(pred_summary)

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(fit, "student-t baseline HR", out_dir, prior_fit = fit_prior)

cat("\nDone. Outputs saved to", out_dir, "\n")
