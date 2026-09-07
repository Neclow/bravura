source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────
# Multivariate Bayesian model for cardiovascular DVs:
#   HR + 3 varimax-rotated HRV components:
#     RC1 = overall HRV power (SDNN, SD2, TRI, ttlpwr)
#     RC2 = vagal/parasympathetic (SD1SD2, hf, lfhf, RMSSD)
#     RC3 = complexity/entropy (SampEn, ApEn, rrHRV)
# Student-t family, Cluster × block + (1|p|subject), residual correlations.

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/physio_cardiac"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ────────────────────────────────────────────────────────────────

df <- read.csv("data/processed/physio_cardiac_long.csv")

df$Cluster <- factor(df$Cluster, levels = c("Non-aggressive", "Proactive", "Reactive"))
df$block <- factor(df$block, levels = c("1.1", "1.2", "2.1", "2.2"))
df$subject <- factor(df$subject)

cat("Design:\n")
cat("  Subjects:", nlevels(df$subject), "\n")
cat("  Clusters:", levels(df$Cluster), "\n")
cat("  Blocks:", levels(df$block), "\n")
cat("  Observations:", nrow(df), "\n")
cat("  DVs: HR, HRV_RC1 (power), HRV_RC2 (vagal), HRV_RC3 (entropy)\n\n")

# ── Formula ──────────────────────────────────────────────────────────────────

formula_mv <- bf(
  mvbind(HR, HRV_RC1, HRV_RC2, HRV_RC3) ~
    Cluster * block + (1 |p| subject)
) + set_rescor(TRUE)

# ── Priors ───────────────────────────────────────────────────────────────────

priors <- c(
  prior(normal(5, 5), class = "Intercept", resp = "HR"),
  prior(normal(0, 3), class = "Intercept", resp = "HRVRC1"),
  prior(normal(0, 3), class = "Intercept", resp = "HRVRC2"),
  prior(normal(0, 3), class = "Intercept", resp = "HRVRC3"),
  prior(normal(0, 5), class = "b", resp = "HR"),
  prior(normal(0, 3), class = "b", resp = "HRVRC1"),
  prior(normal(0, 3), class = "b", resp = "HRVRC2"),
  prior(normal(0, 3), class = "b", resp = "HRVRC3"),
  prior(student_t(3, 0, 5), class = "sd", resp = "HR"),
  prior(student_t(3, 0, 3), class = "sd", resp = "HRVRC1"),
  prior(student_t(3, 0, 3), class = "sd", resp = "HRVRC2"),
  prior(student_t(3, 0, 3), class = "sd", resp = "HRVRC3"),
  prior(student_t(3, 0, 10), class = "sigma", resp = "HR"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "HRVRC1"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "HRVRC2"),
  prior(student_t(3, 0, 5), class = "sigma", resp = "HRVRC3"),
  prior(lkj(2), class = "rescor")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_cardiac_varimax",
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
cat("Multivariate student-t RI model (HR + 3 varimax HRV RCs)\n\n")
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

dvs <- c("HR", "HRVRC1", "HRVRC2", "HRVRC3")
dv_labels <- c("HR", "HRV_RC1", "HRV_RC2", "HRV_RC3")

all_contrasts <- list()
for (i in seq_along(dvs)) {
  em <- emmeans(fit, pairwise ~ Cluster | block, resp = dvs[i])
  contr <- contrasts_eti(em$contrasts)
  contr$DV <- dv_labels[i]
  all_contrasts[[i]] <- contr
}

contrasts_df <- do.call(rbind, all_contrasts)

cat("Pairwise contrasts (all DVs):\n")
print(contrasts_df[, c("DV", "block", "contrast", "estimate", "Q2.5", "Q97.5")],
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
