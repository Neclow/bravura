source("src/brms/utils.R")

# ── Setup ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
OVERWRITE <- "--overwrite" %in% args

out_dir <- "data/brms/trial_duration"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Data preparation ─────────────────────────────────────────────────────────

# Load trial-level data (both cohorts)
trials <- read.csv("data/shared/trial_events.csv")

# Get included subjects from behav_Xa (N=114) and behav_Xb (N=37)
included_a <- rownames(read.csv("data/processed/behav_Xa.csv", row.names = 1))
included_b <- rownames(read.csv("data/processed/behav_Xb.csv", row.names = 1))
included <- c(included_a, included_b)
trials <- trials %>% filter(subject %in% included)
trials$cohort <- factor(trials$cohort)

# Code decision: shock, ring (enlarge), or nothing (NA choice)
trials <- trials %>%
  mutate(
    decision = case_when(
      choice == "shock" ~ "Shock",
      choice == "ring" ~ "Enlarge",
      is.na(choice) | choice == "" ~ "Nothing"
    ),
    decision = factor(decision, levels = c("Enlarge", "Shock", "Nothing"))
  )

# Remove extreme durations (>30s likely reflects technical issues)
trials <- trials %>% filter(duration > 0, duration <= 30)

cat("Decision counts:\n")
print(table(trials$decision))
cat("\nDuration summary by decision:\n")
print(trials %>% group_by(decision) %>% summarise(
  n = n(), mean = mean(duration), sd = sd(duration),
  median = median(duration), .groups = "drop"
))

# ── Formula ──────────────────────────────────────────────────────────────────
# Trial duration ~ decision type, random intercept by subject
# Student-t for robustness to outliers

formula <- duration ~ decision * cohort + (1 | subject)

# ── Priors ───────────────────────────────────────────────────────────────────
# Mean duration ~5s; differences expected to be small (~0.5s)

priors <- c(
  prior(normal(5, 2), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(normal(0, 1.5), class = "sd"),
  prior(exponential(1), class = "sigma"),
  prior(gamma(2, 0.1), class = "nu")
)

# ── Fit model ────────────────────────────────────────────────────────────────

fit <- fit_or_load(
  "fit_student",
  out_dir,
  formula = formula,
  family = student(),
  prior = priors,
  data = trials,
  chains = CHAINS,
  iter = 16000,
  warmup = 8000,
  seed = SEED,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  overwrite = OVERWRITE
)

# ── Prior-only model for Savage-Dickey BF ────────────────────────────────────

fit_prior <- fit_or_load(
  "fit_prior",
  out_dir,
  formula = formula,
  family = student(),
  prior = priors,
  data = trials,
  sample_prior = "only",
  chains = CHAINS,
  iter = 12000,
  warmup = 2000,
  seed = SEED,
  overwrite = OVERWRITE
)

# ── Diagnostics ──────────────────────────────────────────────────────────────

save_diagnostics(fit, "Student-t RI (decision * cohort)", out_dir, prior_fit = fit_prior)

# ── Predicted means per decision ─────────────────────────────────────────────

newdata <- expand.grid(
  decision = levels(trials$decision),
  cohort = levels(trials$cohort)
)

ppe_long <- newdata %>%
  add_epred_draws(fit, re_formula = NA) %>%
  rename(duration = .epred) %>%
  select(decision, cohort, duration, .draw)

pred_summary <- ppe_long %>%
  group_by(cohort, decision) %>%
  summarise(
    mean = mean(duration),
    Q2.5 = quantile(duration, 0.025),
    Q97.5 = quantile(duration, 0.975),
    .groups = "drop"
  )

write.csv(pred_summary, file.path(out_dir, "predicted_means.csv"), row.names = FALSE)
write.csv(ppe_long, file.path(out_dir, "posterior_epred.csv"), row.names = FALSE)

# ── Pairwise contrasts with Bayes Factors (Savage-Dickey) ────────────────────

em_posterior <- emmeans(fit, pairwise ~ decision | cohort)
em_prior <- emmeans(fit_prior, pairwise ~ decision | cohort)

bf_obj <- bayesfactor_parameters(em_posterior$contrasts, prior = em_prior$contrasts)

bf_results <- bf_table(em_posterior, em_prior)

cat("\nPairwise contrasts (Savage-Dickey BF):\n")
print(bf_results, digits = 3)

write.csv(bf_results, file.path(out_dir, "bayes_factors.csv"), row.names = FALSE)

cat("Done. Outputs saved to", out_dir, "\n")
