library(BayesRep)

# ── Setup ────────────────────────────────────────────────────────────────────
# Replication analysis for Δ₁HR using BayesRep (Pawel & Held, 2022).
# Effect estimates: pairwise mean differences + Welch SEs from raw data.
# Computed on blocks 1.1 and 2.1 (where Cohort A brms showed strong effects).

out_dir <- "data/brms/delta_hr_replication"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ────────────────────────────────────────────────────────────────

df_a <- read.csv("data/processed/delta_hr_long.csv")
df_b <- read.csv("data/processed/delta_hr_long_b.csv")

cat("Cohort A:", length(unique(df_a$subject)), "subjects\n")
cat("Cohort B:", length(unique(df_b$subject)), "subjects\n\n")

# ── Pairwise replication analysis per block ──────────────────────────────────

clusters <- c("Non-aggressive", "Proactive", "Reactive")
pairs <- combn(clusters, 2)
blocks <- c(1.1, 2.1)

results <- list()
idx <- 1

for (blk in blocks) {
  df_a_blk <- df_a[df_a$block == blk, ]
  df_b_blk <- df_b[df_b$block == blk, ]

  for (i in seq_len(ncol(pairs))) {
    c1 <- pairs[1, i]
    c2 <- pairs[2, i]

    # Cohort A: effect estimate + Welch SE
    x1_a <- df_a_blk$delta_hr[df_a_blk$Cluster == c1]
    x2_a <- df_a_blk$delta_hr[df_a_blk$Cluster == c2]
    to <- mean(x2_a) - mean(x1_a)
    so <- sqrt(var(x2_a) / length(x2_a) + var(x1_a) / length(x1_a))

    # Cohort B: effect estimate + Welch SE
    x1_b <- df_b_blk$delta_hr[df_b_blk$Cluster == c1]
    x2_b <- df_b_blk$delta_hr[df_b_blk$Cluster == c2]
    tr <- mean(x2_b) - mean(x1_b)
    sr <- sqrt(var(x2_b) / length(x2_b) + var(x1_b) / length(x1_b))

    # Cohen's d (pooled SD)
    n1_a <- length(x1_a); n2_a <- length(x2_a)
    pooled_sd_a <- sqrt(((n1_a - 1) * var(x1_a) + (n2_a - 1) * var(x2_a)) / (n1_a + n2_a - 2))
    d_a <- (mean(x2_a) - mean(x1_a)) / pooled_sd_a

    n1_b <- length(x1_b); n2_b <- length(x2_b)
    pooled_sd_b <- sqrt(((n1_b - 1) * var(x1_b) + (n2_b - 1) * var(x2_b)) / (n1_b + n2_b - 2))
    d_b <- (mean(x2_b) - mean(x1_b)) / pooled_sd_b

    # BayesRep
    bf_r <- BFr(to = to, so = so, tr = tr, sr = sr)
    bf_s <- tryCatch(BFs(to = to, so = so, tr = tr, sr = sr), error = function(e) NA)
    bf_e <- BFe(to = to, so = so, tr = tr, sr = sr, tau = 0.3)

    results[[idx]] <- data.frame(
      block = blk,
      contrast = paste0(c2, " - ", c1),
      to = round(to, 2), so = round(so, 2),
      tr = round(tr, 2), sr = round(sr, 2),
      d_original = round(d_a, 2), d_replication = round(d_b, 2),
      n_original = paste0(n2_a, " vs ", n1_a),
      n_replication = paste0(n2_b, " vs ", n1_b),
      BFr = round(bf_r, 3), BFs = round(bf_s, 3), BFe = round(bf_e, 3),
      stringsAsFactors = FALSE
    )
    idx <- idx + 1
  }
}

bf_table <- do.call(rbind, results)

cat("Pairwise replication analysis (blocks 1.1 and 2.1):\n\n")
cat("BFr (Verhagen & Wagenmakers): < 1 = replication success\n")
cat("BFs (Pawel & Held, sceptical): < 1 = replication success\n")
cat("BFe (equality of effect sizes): > 1 = equal effects\n\n")
print(bf_table, row.names = FALSE)

write.csv(bf_table, file.path(out_dir, "replication_bf.csv"), row.names = FALSE)

# ── Posterior plots ──────────────────────────────────────────────────────────

for (blk in blocks) {
  df_a_blk <- df_a[df_a$block == blk, ]
  df_b_blk <- df_b[df_b$block == blk, ]

  for (i in seq_len(ncol(pairs))) {
    c1 <- pairs[1, i]; c2 <- pairs[2, i]

    x1_a <- df_a_blk$delta_hr[df_a_blk$Cluster == c1]
    x2_a <- df_a_blk$delta_hr[df_a_blk$Cluster == c2]
    to <- mean(x2_a) - mean(x1_a)
    so <- sqrt(var(x2_a) / length(x2_a) + var(x1_a) / length(x1_a))

    x1_b <- df_b_blk$delta_hr[df_b_blk$Cluster == c1]
    x2_b <- df_b_blk$delta_hr[df_b_blk$Cluster == c2]
    tr <- mean(x2_b) - mean(x1_b)
    sr <- sqrt(var(x2_b) / length(x2_b) + var(x1_b) / length(x1_b))

    fname <- gsub(" ", "_", tolower(paste0("block", blk, "_", c2, "_vs_", c1)))
    png(
      file.path(out_dir, paste0("posterior_", fname, ".png")),
      width = 6, height = 4, units = "in", res = 300
    )
    repPosterior(to = to, so = so, tr = tr, sr = sr)
    title(main = paste0(c2, " - ", c1, " (block ", blk, ")"))
    dev.off()
  }
}

cat("\nDone. Outputs saved to", out_dir, "\n")
