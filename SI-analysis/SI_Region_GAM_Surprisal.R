#!/usr/bin/env Rscript
# =====================================================================
# SI_Region_GAM_Surprisal.R
# =====================================================================
# Fits GAM smooths for REGION surprisal 
#
#   SI-A: log(SUM_3RT_trimmed) 
#   SI-C: Raw SUM_3RT_trimmed, HC-only and LC-only subsets 
#
# Outputs per (dataset x analysis):
#   {tag}_region_gam_smooth_data.csv
#   {tag}_region_gam_raw_data.csv
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

# Region surprisal predictors only
region_predictors <- tribble(
  ~predictor_col,   ~model_name,
  "clozeregion",    "Cloze",
  "gpt2region",     "GPT-2",
  "gptneoregion",   "GPT-Neo",
  "gptneoxregion",  "GPT-NeoX",
  "gptjregion",     "GPT-J",
  "gpt2xlregion",   "GPT-2XL",
  "olmoregion",     "OLMO-2",
  "llama2region",   "LLaMA-2"
)

preprocess_data <- function(df, predictor_col) {
  required <- c("SUM_3RT_trimmed", "SUB", "ITEM", predictor_col)
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  df %>%
    mutate(
      SUB  = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
      !!predictor_col := as.numeric(.data[[predictor_col]])
    ) %>%
    drop_na(all_of(required)) %>%
    filter(is.finite(SUM_3RT_trimmed), is.finite(.data[[predictor_col]]))
}


extract_gam_plot_curve <- function(model, select = 1, n = 100, se_with_mean = TRUE) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit({ grDevices::dev.off(); unlink(pdf_path) }, add = TRUE)

  plot_data <- plot(model, select = select, seWithMean = se_with_mean, n = n)

  if (is.null(plot_data) || length(plot_data) < 1L || is.null(plot_data[[1]])) {
    stop("plot.gam() did not return smooth data")
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


run_region_analysis <- function(df_full, tag, dv_col = "SUM_3RT_trimmed") {
  smooth_list <- list()
  raw_list    <- list()

  for (i in seq_len(nrow(region_predictors))) {
    pred_col <- region_predictors$predictor_col[[i]]
    m_name   <- region_predictors$model_name[[i]]

    cat("  [", tag, "]", m_name, "(", pred_col, ") ...\n")

    if (!pred_col %in% names(df_full)) {
      cat("    SKIP: column not found\n")
      next
    }

    df_model <- preprocess_data(df_full, pred_col)

    # For SI-A log transform: add column and filter
    if (dv_col == "log_SUM_3RT") {
      df_model <- df_model %>%
        filter(SUM_3RT_trimmed > 0) %>%
        mutate(log_SUM_3RT = log(SUM_3RT_trimmed))
    }

    cat("    n =", nrow(df_model), "\n")

    formula <- as.formula(paste0(
      dv_col, " ~ 1 + s(", pred_col, ", bs='cs') +",
      " s(SUB, bs='re') + s(ITEM, bs='re')"
    ))

    fit <- gam(formula, data = df_model)

    smooth <- extract_gam_plot_curve(fit, n = 100, se_with_mean = TRUE)
    smooth$model   <- m_name
    smooth$measure <- "Surprisal"
    smooth_list[[m_name]] <- smooth

    raw <- extract_raw_data(fit)
    raw$model   <- m_name
    raw$measure <- "Surprisal"
    raw_list[[m_name]] <- raw
  }

  smooth_combined <- bind_rows(smooth_list)
  raw_combined    <- bind_rows(raw_list)

  smooth_file <- paste0(tag, "_region_gam_smooth_data.csv")
  raw_file    <- paste0(tag, "_region_gam_raw_data.csv")

  write_csv(smooth_combined, smooth_file)
  write_csv(raw_combined, raw_file)

  cat("  Saved:", smooth_file, "(", nrow(smooth_combined), "rows )\n")
  cat("  Saved:", raw_file,    "(", nrow(raw_combined),    "rows )\n")
}


# MAIN

for (ds in DATASETS) {
  cat("\n", strrep("=", 60), "\n")
  cat("Loading:", ds$file, "\n")
  cat(strrep("=", 60), "\n")

  if (!file.exists(ds$file)) {
    cat("  ERROR: file not found, skipping.\n")
    next
  }

  df_raw <- read_csv(ds$file, show_col_types = FALSE)
  cat("  Total rows:", nrow(df_raw), "\n")

  # SI-A: log transform
  cat("\n--- SI-A: log(SUM_3RT_trimmed) region surprisal ---\n")
  tag_log <- paste0(ds$prefix, "_SIA_log")
  run_region_analysis(df_raw, tag_log, dv_col = "log_SUM_3RT")

  # SI-C: HC only
  cat("\n--- SI-C: HC items only, region surprisal ---\n")
  df_hc <- df_raw %>% filter(condition == "HC")
  cat("  HC rows:", nrow(df_hc), "\n")

  if (nrow(df_hc) > 0) {
    tag_hc <- paste0(ds$prefix, "_SIC_HC")
    run_region_analysis(df_hc, tag_hc, dv_col = "SUM_3RT_trimmed")
  } else {
    cat("  WARNING: No HC rows found.\n")
  }

  # SI-C: LC only
  cat("\n--- SI-C: LC items only, region surprisal ---\n")
  df_lc <- df_raw %>% filter(condition == "LC")
  cat("  LC rows:", nrow(df_lc), "\n")

  if (nrow(df_lc) > 0) {
    tag_lc <- paste0(ds$prefix, "_SIC_LC")
    run_region_analysis(df_lc, tag_lc, dv_col = "SUM_3RT_trimmed")
  } else {
    cat("  WARNING: No LC rows found.\n")
  }
}

cat("\n", strrep("=", 60), "\n")
cat("ALL DONE — Region surprisal GAMs.\n")
cat(strrep("=", 60), "\n")
