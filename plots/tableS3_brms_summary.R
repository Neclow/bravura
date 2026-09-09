library(brms)

DEFAULT_BRMS_DIR <- "data_v2/brms"
DEFAULT_IMG_DIR <- "img_v2"
FIG3_DIR <- file.path(DEFAULT_IMG_DIR, "fig3")
dir.create(FIG3_DIR, showWarnings = FALSE, recursive = TRUE)

MODELS <- list(
  "Opponent provocation" = file.path(
    DEFAULT_BRMS_DIR,
    "shocks_overview/fit_binomial.rds"
  ),
  "Opponent belief" = file.path(
    DEFAULT_BRMS_DIR,
    "beliefs_overview/fit_student.rds"
  ),
  "Trial duration" = file.path(
    DEFAULT_BRMS_DIR,
    "trial_duration/fit_student.rds"
  ),
  "Cluster shocks" = file.path(DEFAULT_BRMS_DIR, "shocks/fit_binomial.rds"),
  "PSAP" = file.path(DEFAULT_BRMS_DIR, "psap/fit_dirichlet.rds"),
  "Shock latency" = file.path(DEFAULT_BRMS_DIR, "latency/fit_lognormal.rds"),
  "Baseline HR" = file.path(
    DEFAULT_BRMS_DIR,
    "baseline_hr/fit_baseline_hr.rds"
  ),
  "Delta-HR" = file.path(DEFAULT_BRMS_DIR, "delta_hr/fit_student_ri.rds"),
  "Cardiac multivariate" = file.path(
    DEFAULT_BRMS_DIR,
    "physio_cardiac/fit_cardiac_varimax.rds"
  ),
  "Delta-HR replication (1.1)" = file.path(
    DEFAULT_BRMS_DIR,
    "delta_hr_replication/block_1_1/fit_joint_1_1.rds"
  ),
  "Delta-HR replication (2.1)" = file.path(
    DEFAULT_BRMS_DIR,
    "delta_hr_replication/block_2_1/fit_joint_2_1.rds"
  )
)

format_priors <- function(fit) {
  p <- prior_summary(fit)
  # Prefer user-set priors; fall back to defaults if none were set
  user <- p[p$source == "user", , drop = FALSE]
  if (nrow(user) == 0) {
    user <- p[p$source == "default", , drop = FALSE]
  }

  # Per class, drop (flat) entries when a proper prior exists
  has_proper <- user$class[user$prior != ""]
  user <- user[user$prior != "" | !(user$class %in% has_proper), , drop = FALSE]

  # Deduplicate: one entry per (class, prior) pair
  user$key <- paste(user$class, user$prior)
  user <- user[!duplicated(user$key), , drop = FALSE]

  # Order: Intercept first, then alphabetical
  user <- user[order(user$class != "Intercept", user$class), , drop = FALSE]

  # Mark truncated priors: lb=0 → "+", ub=0 → "-"
  suffix <- ifelse(!is.na(user$lb) & user$lb == 0, "+", "")

  paste(
    paste0(user$class, suffix, ": ", user$prior),
    collapse = "; "
  )
}

extract_formula <- function(fit) {
  f <- formula(fit)
  if (!is.null(f$formula)) {
    return(deparse(f$formula, width.cutoff = 200))
  }
  # Multivariate: build mvbind(...) ~ RHS from the per-response forms
  responses <- paste(f$responses, collapse = ", ")
  rhs <- deparse(f$forms[[1]][[3]], width.cutoff = 200)
  paste0("mvbind(", responses, ") ~ ", rhs)
}

extract_family <- function(fit) {
  fit_family <- family(fit)
  if (is.list(fit_family) && is.null(fit_family$family)) {
    paste("MV", fit_family[[1]]$family)
  } else {
    fit_family$family
  }
}

extract_diagnostics <- function(s) {
  all_pars <- rbind(s$fixed, s$spec_pars)
  if (!is.null(s$random)) {
    for (i in seq_along(s$random)) {
      all_pars <- rbind(all_pars, s$random[[i]])
    }
  }
  list(
    max_rhat = round(max(all_pars$Rhat, na.rm = TRUE), 3),
    min_bulk = round(min(all_pars$Bulk_ESS, na.rm = TRUE)),
    min_tail = round(min(all_pars$Tail_ESS, na.rm = TRUE))
  )
}

extract_row <- function(model_name, rds_path) {
  if (!file.exists(rds_path)) {
    cat("WARNING:", rds_path, "not found, skipping", model_name, "\n")
    return(NULL)
  }
  fit <- readRDS(rds_path)
  diag <- extract_diagnostics(summary(fit))

  data.frame(
    Model = model_name,
    Family = extract_family(fit),
    Formula = gsub("\\|", "\\\\|", extract_formula(fit)),
    Priors = format_priors(fit),
    Max_Rhat = diag$max_rhat,
    Min_Bulk_ESS = diag$min_bulk,
    Min_Tail_ESS = diag$min_tail,
    stringsAsFactors = FALSE
  )
}

# Build table

rows <- list()
for (nm in names(MODELS)) {
  row <- extract_row(nm, MODELS[[nm]])
  if (!is.null(row)) rows <- c(rows, list(row))
}
df <- do.call(rbind, rows)
names(df) <- c(
  "Model",
  "Family",
  "Formula",
  "Priors",
  "Max R-hat",
  "Min Bulk ESS",
  "Min Tail ESS"
)

# Write pipe table

pipe_table <- function(df) {
  pad <- function(x, w) formatC(x, width = w, flag = "-")
  widths <- mapply(
    function(col, nm) max(nchar(nm), max(nchar(as.character(col)))),
    df,
    names(df)
  )
  header <- paste0(
    "| ",
    paste(mapply(pad, names(df), widths), collapse = " | "),
    " |"
  )
  sep <- paste0(
    "|",
    paste(
      sapply(widths, function(w) paste(rep("-", w + 2), collapse = "")),
      collapse = "|"
    ),
    "|"
  )
  body <- apply(df, 1, function(row) {
    paste0("| ", paste(mapply(pad, row, widths), collapse = " | "), " |")
  })
  c(header, sep, body)
}

table_path <- file.path(FIG3_DIR, "tableS3_brms_summary.md")
lines <- pipe_table(df)
writeLines(lines, table_path)
cat("Saved", table_path, "\n\n")
cat(paste(lines, collapse = "\n"), "\n")
