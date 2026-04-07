source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────
# Multivariate Bayesian model for 5 physiological DVs:
#   HR, HRV_PC1, RespRate, nSCRcda, EDA_PC1
# One representative feature + one PC1 composite per domain.
# Student-t family, Cluster × block + (1|p|subject), residual correlations.

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/physio_multivariate"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ────────────────────────────────────────────────────────────────

df <- read.csv("data/processed/physio_multivariate_long.csv")

df$Cluster <- factor(df$Cluster, levels = c("Non-aggressive", "Proactive", "Reactive"))
df$block <- factor(df$block, levels = c("1.1", "1.2", "2.1", "2.2"))
df$subject <- factor(df$subject)

cat("Design:\n")
cat("  Subjects:", nlevels(df$subject), "\n")
cat("  Clusters:", levels(df$Cluster), "\n")
cat("  Blocks:", levels(df$block), "\n")
cat("  Observations:", nrow(df), "\n")
cat("  DVs: HR, HRV_PC1, RespRate, Resp_PC1, nSCRcda, EDA_PC1\n\n")

# ── Formula ──────────────────────────────────────────────────────────────────

formula_mv <- bf(
  mvbind(HR, HRV_PC1, RespRate, Resp_PC1, nSCRcda, EDA_PC1) ~
    Cluster * block + (1 |p| subject)
) + set_rescor(TRUE)

# ── Priors (per-DV, weakly informative) ──────────────────────────────────────

priors <- c(
  # Intercepts
  prior(normal(5, 5), class = "Intercept", resp = "HR"),
  prior(normal(0, 3), class = "Intercept", resp = "HRVPC1"),
  prior(normal(0, 3), class = "Intercept", resp = "RespRate"),
  prior(normal(0, 3), class = "Intercept", resp = "RespPC1"),
  prior(normal(0, 5), class = "Intercept", resp = "nSCRcda"),
  prior(normal(0, 3), class = "Intercept", resp = "EDAPC1"),
  # Fixed effects
  prior(normal(0, 5), class = "b", resp = "HR"),
  prior(normal(0, 3), class = "b", resp = "HRVPC1"),
  prior(normal(0, 3), class = "b", resp = "RespRate"),
  prior(normal(0, 3), class = "b", resp = "RespPC1"),
  prior(normal(0, 5), class = "b", resp = "nSCRcda"),
  prior(normal(0, 3), class = "b", resp = "EDAPC1"),
  # Random effects SD
  prior(student_t(3, 0, 5), class = "sd", resp = "HR"),
  prior(student_t(3, 0, 3), class = "sd", resp = "HRVPC1"),
  prior(student_t(3, 0, 3), class = "sd", resp = "RespRate"),
  prior(student_t(3, 0, 3), class = "sd", resp = "RespPC1"),
  prior(student_t(3, 0, 5), class = "sd", resp = "nSCRcda"),
  prior(student_t(3, 0, 3), class = "sd", resp = "EDAPC1"),
  # Residual SD
  prior(student_t(3, 0, 10), class = "sigma", resp = "HR"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "HRVPC1"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "RespRate"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "RespPC1"),
  prior(student_t(3, 0, 10), class = "sigma", resp = "nSCRcda"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "EDAPC1"),
  # Residual correlations
  prior(lkj(2), class = "rescor")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_mv_student",
  out_dir,
  formula = formula_mv,
  data = df,
  family = student(),
  prior = priors,
  chains = CHAINS,
  cores = CORES,
  iter = 8000,
  warmup = 4000,
  seed = SEED,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  overwrite = OVERWRITE
)

cat("Model fitted\n\n")

# ── Diagnostics ──────────────────────────────────────────────────────────────

sink(file.path(out_dir, "summary.txt"))
cat("Multivariate student-t RI model (5 DVs)\n\n")
summary(fit)
cat("\n\nPrior summary:\n")
prior_summary(fit)
sink()

png(
  file.path(out_dir, "trace_plots.png"),
  width = 14, height = 10, units = "in", res = 300
)
plot(fit, ask = FALSE)
dev.off()

# ── Per-DV emmeans contrasts ─────────────────────────────────────────────────

dvs <- c("HR", "HRVPC1", "RespRate", "RespPC1", "nSCRcda", "EDAPC1")
dv_labels <- c("HR", "HRV_PC1", "RespRate", "Resp_PC1", "nSCRcda", "EDA_PC1")

all_contrasts <- list()
for (i in seq_along(dvs)) {
  em <- emmeans(fit, pairwise ~ Cluster | block, resp = dvs[i])
  contr <- as.data.frame(em$contrasts)
  contr$DV <- dv_labels[i]
  all_contrasts[[i]] <- contr
}

contrasts_df <- do.call(rbind, all_contrasts)

cat("Pairwise contrasts (all DVs):\n")
print(contrasts_df[, c("DV", "block", "contrast", "estimate", "lower.HPD", "upper.HPD")],
      digits = 3)

write.csv(contrasts_df, file.path(out_dir, "pairwise_contrasts.csv"), row.names = FALSE)

# ── Fixed effects per DV ─────────────────────────────────────────────────────

for (i in seq_along(dvs)) {
  fe <- fixef(fit, resp = dvs[i])
  write.csv(
    as.data.frame(fe),
    file.path(out_dir, paste0("fixed_effects_", dv_labels[i], ".csv"))
  )
}

cat("\nDone. Outputs saved to", out_dir, "\n")
