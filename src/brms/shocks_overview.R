source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/shocks_overview"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Data preparation ─────────────────────────────────────────────────────────
# Derive shock counts from behav_Xa and behav_Xb (both cohorts, included only)

load_shocks <- function(path, cohort_label) {
  raw <- read.csv(path, row.names = 1)
  data.frame(
    subject = rep(rownames(raw), 2),
    cohort = cohort_label,
    opponent = factor(rep(c("Opponent 1", "Opponent 2"), each = nrow(raw))),
    shocks = c(raw$shock_opp1, raw$shock_opp2)
  )
}

df <- rbind(
  load_shocks("data/processed/behav_Xa.csv", "A"),
  load_shocks("data/processed/behav_Xb.csv", "B")
)
df$cohort <- factor(df$cohort)

cat("N subjects per cohort:\n")
print(df %>% distinct(subject, cohort) %>% count(cohort))
cat("\nMean shocks per opponent x cohort:\n")
print(df %>% group_by(cohort, opponent) %>% summarise(
  mean = mean(shocks), sd = sd(shocks), .groups = "drop"
))

# ── Formula ──────────────────────────────────────────────────────────────────
# Population-level opponent effect (no Cluster), random intercept by subject

formula <- shocks | trials(15) ~ opponent * cohort + (1 | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Mean ~5 shocks/15 => logit(5/15) ~ -0.4; opponent 2 provokes more => positive slope

priors <- c(
  prior(normal(0, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(normal(0, 1.5), class = "sd")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_binomial",
  out_dir,
  formula = formula,
  family = binomial(),
  prior = priors,
  data = df,
  chains = CHAINS,
  iter = ITER,
  warmup = WARMUP,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Prior-only model for Savage-Dickey BF ────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior",
  out_dir,
  formula = formula,
  family = binomial(),
  prior = priors,
  data = df,
  sample_prior = "only",
  chains = CHAINS,
  iter = 12000,
  warmup = 2000,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(fit, "binomial RI (opponent * cohort)", out_dir, prior_fit = fit_prior)

# ── Predicted means per opponent ─────────────────────────────────────────────

newdata <- expand.grid(
  opponent = levels(df$opponent),
  cohort = levels(df$cohort)
)

ppe_long <- newdata %>%
  add_epred_draws(fit, re_formula = NA) %>%
  rename(shocks = .epred) %>%
  select(opponent, cohort, shocks, .draw)

pred_summary <- ppe_long %>%
  group_by(cohort, opponent) %>%
  summarise(
    mean = mean(shocks),
    Q2.5 = quantile(shocks, 0.025),
    Q97.5 = quantile(shocks, 0.975),
    .groups = "drop"
  )

write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
write.csv(ppe_long, file.path(out_dir, "posterior_epred.csv"), row.names = FALSE)

# ── Opponent effect BF (Savage-Dickey) ───────────────────────────────────────

em_posterior <- emmeans(fit, pairwise ~ opponent | cohort)
em_prior <- emmeans(fit_prior, pairwise ~ opponent | cohort)

bf_obj <- bayesfactor_parameters(em_posterior$contrasts, prior = em_prior$contrasts)

bf_results <- bf_table(em_posterior, em_prior)

cat("\nOpponent contrast (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
