source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────
# Shock latency model: do reactive subjects shock faster?
# Lognormal family (reaction times are right-skewed).
# Only trials where a shock was given.

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/latency"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ────────────────────────────────────────────────────────────────

df <- read.csv("data/processed/shock_latency_long.csv")

df$Cluster <- factor(df$Cluster, levels = c("Non-aggressive", "Proactive", "Reactive"))
df$opponent <- factor(df$opponent)
df$subject <- factor(df$subject)

cat("Design:\n")
cat("  Shock events:", nrow(df), "\n")
cat("  Subjects:", nlevels(df$subject), "\n")
cat("  Per cluster:", table(df$Cluster[!duplicated(df$subject)]), "\n")
cat("  Mean latency:", round(mean(df$latency), 3), "s\n\n")

# ── Formula ──────────────────────────────────────────────────────────────────

formula_ri <- latency ~ Cluster * opponent + (1 | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Log-scale: log(0.75) ≈ -0.29, typical latencies ~0.7-0.9s

priors <- c(
  prior(normal(-0.3, 0.5), class = "Intercept"),
  prior(normal(0, 0.3), class = "b"),
  prior(student_t(3, 0, 0.5), class = "sd"),
  prior(student_t(3, 0, 0.5), class = "sigma")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_lognormal",
  out_dir,
  formula = formula_ri,
  data = df,
  family = lognormal(),
  prior = priors,
  chains = CHAINS,
  cores = CORES,
  iter = ITER,
  warmup = WARMUP,
  seed = SEED,
  overwrite = OVERWRITE
)

cat("Model fitted\n\n")

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(fit, "lognormal RI (shock latency)", out_dir)

# ── Posterior predicted means ────────────────────────────────────────────────

write.csv(as.data.frame(fixef(fit)), file.path(out_dir, "fixed_effects.csv"))

newdata <- expand.grid(
  Cluster = levels(df$Cluster),
  opponent = levels(df$opponent)
)

epred_long <- newdata %>%
  add_epred_draws(fit, re_formula = NA) %>%
  rename(latency = .epred) %>%
  select(Cluster, opponent, latency, .draw)

write.csv(
  epred_long,
  file.path(out_dir, "posterior_epred.csv"),
  row.names = FALSE
)

pred_summary <- epred_long %>%
  group_by(Cluster, opponent) %>%
  summarise(
    mean = mean(latency),
    Q2.5 = quantile(latency, 0.025),
    Q97.5 = quantile(latency, 0.975),
    .groups = "drop"
  )
write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
cat("Predicted means (response scale):\n")
print(pred_summary)

# ── Pairwise contrasts (Savage-Dickey) ───────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior_lognormal",
  out_dir,
  formula = formula_ri,
  data = df,
  family = lognormal(),
  prior = priors,
  sample_prior = "only",
  chains = CHAINS,
  cores = CORES,
  iter = 12000,
  warmup = 2000,
  seed = SEED,
  overwrite = OVERWRITE
)

em_post <- emmeans(fit, pairwise ~ Cluster | opponent)
em_prior <- emmeans(fit_prior, pairwise ~ Cluster | opponent)

bf_results <- bf_table(em_post, em_prior)

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("\nDone. Outputs saved to", out_dir, "\n")
