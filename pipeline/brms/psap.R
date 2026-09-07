source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/psap"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# psap_ilr.csv has noise-replaced zeros and renormalized proportions
df <- read.csv("data/processed/psap_ilr.csv")

# ── Model ────────────────────────────────────────────────────────────────────

formula <- bf(cbind(Earn, Steal, Protect) ~ Cluster * phase + (1 |p| Subject))

fit <- fit_or_load(
  name = "fit_dirichlet",
  out_dir = out_dir,
  formula = formula,
  family = dirichlet(),
  data = df,
  chains = CHAINS,
  iter = ITER,
  warmup = WARMUP,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Diagnostics ──────────────────────────────────────────────────────────────

dirichlet_dir <- file.path(out_dir, "fit_dirichlet")
dir.create(dirichlet_dir, showWarnings = FALSE)

sink(file.path(dirichlet_dir, "summary.txt"))
cat("Model: Dirichlet\n\n")
print(summary(fit))
cat("\n\nPrior summary:\n")
print(prior_summary(fit))
sink()

png(file.path(dirichlet_dir, "trace_plots.png"), width = 10, height = 8, units = "in", res = 300)
print(plot(fit, ask = FALSE))
dev.off()

# ── Posterior predictions ────────────────────────────────────────────────────

newdata <- expand.grid(
  Cluster = unique(df$Cluster),
  phase = unique(df$phase)
)

epred <- newdata %>%
  add_epred_draws(fit, re_formula = NA)

# tidybayes returns .category = "Earn", "Steal", "Protect"
all_epred <- epred %>%
  select(Cluster, phase, .draw, .category, .epred) %>%
  rename(button = .category, proportion = .epred)

all_summary <- all_epred %>%
  group_by(Cluster, phase, button) %>%
  summarise(
    mean = mean(proportion),
    Q2.5 = quantile(proportion, 0.025),
    Q97.5 = quantile(proportion, 0.975),
    .groups = "drop"
  )

# ── Pairwise contrasts (CrI-based) ──────────────────────────────────────────

bf_table <- pairwise_bf(
  epred = as.data.frame(all_epred),
  group_col = "Cluster",
  value_col = "proportion",
  by_cols = c("phase", "button"),
  pairs = list(
    c("Proactive", "Non-aggressive"),
    c("Reactive", "Non-aggressive"),
    c("Proactive", "Reactive")
  )
)

cat("\nPairwise contrasts:\n")
print(bf_table, digits = 3)

# ── Save outputs ─────────────────────────────────────────────────────────────

write.csv(
  as.data.frame(fixef(fit)),
  file.path(out_dir, "fit_dirichlet_fixed_effects.csv")
)
write.csv(all_epred, file.path(out_dir, "posterior_epred.csv"), row.names = FALSE)
write.csv(all_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
write.csv(bf_table, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("\nDone. Outputs saved to", out_dir, "\n")
