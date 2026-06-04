#!/usr/bin/env Rscript
# =====================================================================
# SI_Gam_Plots.R
# =====================================================================
# Fits GAM smooths for critical word predictability (prob + surp)
# across all estimators, for supplementary analyses:
#
#   SI-A: log(SUM_3RT_trimmed) as DV  (both datasets)
#   SI-C: Raw SUM_3RT_trimmed, HC-only and LC-only subsets (both datasets)
#
# Usage (on cluster):
#   Rscript SI_Gam_Plots.R
#
# Outputs per (dataset x analysis):
#   {tag}_gam_smooth_data.csv   — for downstream Python plotting
#   {tag}_gam_raw_data.csv      — raw x/y for optional scatter underlay
#
# Tags produced:
#   bkr_SIA_log, bko_SIA_log,
#   bkr_SIC_HC, bko_SIC_HC,
#   bkr_SIC_LC, bko_SIC_LC
# =====================================================================

library(tidyverse)
library(mgcv)
library(readr)

# ── Config ───────────────────────────────────────────────────────────
WORK_DIR   <- "/afs/cs.stanford.edu/u/sgcxliv"
setwd(WORK_DIR)

DATASETS <- list(
  list(file = "bkr21_spr.csv", prefix = "bkr"),
  list(file = "bko21_spr.csv", prefix = "bko")
)

curve_labels <- tribble(
  ~prob_col,       ~surp_col,     ~model_name,
  "clozeprob",     "cloze",       "Cloze",
  "gpt2prob",      "gpt2",        "GPT-2",
  "gptneoprob",    "gptneo",      "GPT-Neo",
  "gptneoxprob",   "gptneox",     "GPT-NeoX",
  "gptjprob",      "gptj",        "GPT-J",
  "gpt2xlprob",    "gpt2xl",      "GPT-2XL",
  "olmoprob",      "olmo",        "OLMO-2",
  "llama2prob",    "llama2",      "LLaMA-2"
)

# ── Helpers ──────────────────────────────────────────────────────────

preprocess_data <- function(df) {
  df %>%
    mutate(
      SUB  = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed)
    ) %>%
    drop_na(SUM_3RT_trimmed, clozeprob, SUB, ITEM) %>%
    filter(is.finite(SUM_3RT_trimmed))
}


extract_gam_plot_curve <- function(model, select = 1, n = 100, se_with_mean = TRUE) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit({ grDevices::dev.off(); unlink(pdf_path) }, add = TRUE)

  plot_data <- plot(model, select = select, seWithMean = se_with_mean, n = n)

  if (is.null(plot_data) || length(plot_data) < 1L || is.null(plot_data[[1]])) {
    stop("plot.gam() did not return smooth plotting data")
  }
  pd  <- plot_data[[1]]
  fit <- as.numeric(pd$fit)
  se  <- as.numeric(pd$se)
  data.frame(
    x     = as.numeric(pd$x),
    y     = fit,
    se    = se,
    lower = fit - 1.96 * se,
    upper = fit + 1.96 * se
  )
}


extract_raw_data <- function(model, term_index = 1) {
  pred_var     <- model$smooth[[term_index]]$term[1]
  response_var <- names(model$model)[1]
  data.frame(
    x = model$model[[pred_var]],
    y = model$model[[response_var]]
  )
}


fit_and_extract <- function(df, predictor_col, dv_col = "SUM_3RT_trimmed") {
  # Drop rows missing this specific predictor
  df <- df %>%
    filter(is.finite(.data[[predictor_col]])) %>%
    drop_na(all_of(c(dv_col, predictor_col, "SUB", "ITEM")))

  formula <- as.formula(paste0(
    dv_col, " ~ 1 + s(", predictor_col, ", bs='cs') +",
    " s(SUB, bs='re') + s(ITEM, bs='re')"
  ))

  cat("    Fitting:", deparse(formula), " (n =", nrow(df), ")\n")
  fit <- gam(formula, data = df)

  smooth <- extract_gam_plot_curve(fit, n = 100, se_with_mean = TRUE)
  raw    <- extract_raw_data(fit)

  list(smooth = smooth, raw = raw)
}


run_analysis <- function(df, tag, dv_col = "SUM_3RT_trimmed") {
  # Fits all prob + surp GAMs and saves CSVs
  smooth_list <- list()
  raw_list    <- list()

  for (i in seq_len(nrow(curve_labels))) {
    m_name   <- curve_labels$model_name[[i]]
    prob_col <- curve_labels$prob_col[[i]]
    surp_col <- curve_labels$surp_col[[i]]

    # ---- Probability ----
    cat("  [", tag, "]", m_name, "Probability ...\n")
    if (prob_col %in% names(df)) {
      res <- fit_and_extract(df, prob_col, dv_col)
      res$smooth$model   <- m_name; res$smooth$measure <- "Probability"
      res$raw$model      <- m_name; res$raw$measure    <- "Probability"
      smooth_list[[paste0(m_name, "_prob")]] <- res$smooth
      raw_list[[paste0(m_name, "_prob")]]    <- res$raw
    } else {
      cat("    SKIP: column", prob_col, "not found\n")
    }

    # ---- Surprisal ----
    cat("  [", tag, "]", m_name, "Surprisal ...\n")
    if (surp_col %in% names(df)) {
      res <- fit_and_extract(df, surp_col, dv_col)
      res$smooth$model   <- m_name; res$smooth$measure <- "Surprisal"
      res$raw$model      <- m_name; res$raw$measure    <- "Surprisal"
      smooth_list[[paste0(m_name, "_surp")]] <- res$smooth
      raw_list[[paste0(m_name, "_surp")]]    <- res$raw
    } else {
      cat("    SKIP: column", surp_col, "not found\n")
    }
  }

  smooth_combined <- bind_rows(smooth_list)
  raw_combined    <- bind_rows(raw_list)

  smooth_file <- paste0(tag, "_gam_smooth_data.csv")
  raw_file    <- paste0(tag, "_gam_raw_data.csv")

  write_csv(smooth_combined, smooth_file)
  write_csv(raw_combined, raw_file)

  cat("  Saved:", smooth_file, "(", nrow(smooth_combined), "rows )\n")
  cat("  Saved:", raw_file,    "(", nrow(raw_combined),    "rows )\n")
}


# =====================================================================
# MAIN
# =====================================================================

for (ds in DATASETS) {
  cat("\n", strrep("=", 60), "\n")
  cat("Loading:", ds$file, "\n")
  cat(strrep("=", 60), "\n")

  if (!file.exists(ds$file)) {
    cat("  ERROR: file not found, skipping.\n")
    next
  }

  df_raw <- read_csv(ds$file, show_col_types = FALSE)
  df_raw <- preprocess_data(df_raw)
  cat("  Rows after preprocessing:", nrow(df_raw), "\n")

  # ────────────────────────────────────────────────────────────────
  # SI-A: log transform
  # ────────────────────────────────────────────────────────────────
  cat("\n--- SI-A: log(SUM_3RT_trimmed) ---\n")
  df_log <- df_raw %>%
    filter(SUM_3RT_trimmed > 0) %>%                 # log requires positive
    mutate(log_SUM_3RT = log(SUM_3RT_trimmed))

  cat("  Rows with positive RT:", nrow(df_log), "\n")

  tag_log <- paste0(ds$prefix, "_SIA_log")
  run_analysis(df_log, tag_log, dv_col = "log_SUM_3RT")

  # ────────────────────────────────────────────────────────────────
  # SI-C: HC only
  # ────────────────────────────────────────────────────────────────
  cat("\n--- SI-C: HC items only ---\n")
  df_hc <- df_raw %>% filter(condition == "HC")
  cat("  HC rows:", nrow(df_hc), "\n")

  if (nrow(df_hc) > 0) {
    tag_hc <- paste0(ds$prefix, "_SIC_HC")
    run_analysis(df_hc, tag_hc, dv_col = "SUM_3RT_trimmed")
  } else {
    cat("  WARNING: No HC rows found. Check 'condition' column values.\n")
  }

  # ────────────────────────────────────────────────────────────────
  # SI-C: LC only
  # ────────────────────────────────────────────────────────────────
  cat("\n--- SI-C: LC items only ---\n")
  df_lc <- df_raw %>% filter(condition == "LC")
  cat("  LC rows:", nrow(df_lc), "\n")

  if (nrow(df_lc) > 0) {
    tag_lc <- paste0(ds$prefix, "_SIC_LC")
    run_analysis(df_lc, tag_lc, dv_col = "SUM_3RT_trimmed")
  } else {
    cat("  WARNING: No LC rows found. Check 'condition' column values.\n")
  }
}

cat("\n", strrep("=", 60), "\n")
cat("ALL DONE.\n")
cat("Output CSVs in:", WORK_DIR, "\n")
cat(strrep("=", 60), "\n")
