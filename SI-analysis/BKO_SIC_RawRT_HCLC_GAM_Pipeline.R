library(tidyverse)
library(mgcv)
library(readr)

# =====================================================================
# BKO SI-C pipeline (raw RT):
# - Dataset: bko21_spr.csv
# - DV: SUM_3RT_trimmed (NOT log-transformed)
# - Separate analyses: HC and LC
# - Outputs:
#   1) Normal GAM curves (Probability + Surprisal)
#   2) Region-surprisal GAM curves
# - Smooth extraction uses plot.gam(..., seWithMean=TRUE) so Python plots
#   match R smooth/CI behavior.
# =====================================================================

WORK_DIR <- "~/Desktop"
setwd(WORK_DIR)

INPUT_FILE <- "bko21_spr.csv"

normal_predictors <- tribble(
  ~model,     ~prob_col,      ~surp_col,
  "Cloze",    "clozeprob",    "cloze",
  "GPT-2",    "gpt2prob",     "gpt2",
  "GPT-Neo",  "gptneoprob",   "gptneo",
  "GPT-NeoX", "gptneoxprob",  "gptneox",
  "GPT-J",    "gptjprob",     "gptj",
  "GPT-2XL",  "gpt2xlprob",   "gpt2xl",
  "OLMO-2",   "olmoprob",     "olmo",
  "LLaMA-2",  "llama2prob",   "llama2"
)

region_surp_predictors <- tribble(
  ~model,     ~pred_col,
  "Cloze",    "clozeregion",
  "GPT-2",    "gpt2region",
  "GPT-Neo",  "gptneoregion",
  "GPT-NeoX", "gptneoxregion",
  "GPT-J",    "gptjregion",
  "GPT-2XL",  "gpt2xlregion",
  "OLMO-2",   "olmoregion",
  "LLaMA-2",  "llama2region"
)

preprocess_data <- function(df) {
  df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed)
    ) %>%
    filter(is.finite(SUM_3RT_trimmed)) %>%
    drop_na(SUM_3RT_trimmed, SUB, ITEM, condition)
}

extract_smooth <- function(fit, select = 1, n = 200) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit(
    {
      grDevices::dev.off()
      unlink(pdf_path)
    },
    add = TRUE
  )

  plot_data <- plot(fit, select = select, seWithMean = TRUE, n = n)
  pd <- plot_data[[1]]

  data.frame(
    x = as.numeric(pd$x),
    fit = as.numeric(pd$fit),
    se = as.numeric(pd$se),
    se_upper = as.numeric(pd$fit) + as.numeric(pd$se),
    se_lower = as.numeric(pd$fit) - as.numeric(pd$se)
  )
}

fit_single_gam <- function(df, predictor_col) {
  required <- c("SUM_3RT_trimmed", "SUB", "ITEM", predictor_col)
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  df_model <- df %>%
    mutate(!!predictor_col := as.numeric(.data[[predictor_col]])) %>%
    drop_na(all_of(required)) %>%
    filter(is.finite(.data[[predictor_col]]))

  formula <- as.formula(
    paste0(
      "SUM_3RT_trimmed ~ 1 + s(", predictor_col, ", bs='cs') + ",
      "s(SUB, bs='re') + s(ITEM, bs='re')"
    )
  )

  # bam for speed on this dataset size.
  bam(formula, data = df_model)
}

run_normal_pipeline <- function(df_subset, subset_label) {
  rows <- list()

  for (i in seq_len(nrow(normal_predictors))) {
    m <- normal_predictors[i, ]

    # Probability
    if (m$prob_col %in% names(df_subset)) {
      cat("[", subset_label, "] Fitting ", m$model, " Probability (", m$prob_col, ")\n", sep = "")
      fit_prob <- fit_single_gam(df_subset, m$prob_col)
      sm_prob <- extract_smooth(fit_prob, select = 1, n = 200)
      sm_prob$model <- m$model
      sm_prob$measure <- "Probability"
      sm_prob$predictor <- m$prob_col
      sm_prob$subset <- subset_label
      rows[[length(rows) + 1]] <- sm_prob
    } else {
      cat("[", subset_label, "] SKIP missing column: ", m$prob_col, "\n", sep = "")
    }

    # Surprisal
    if (m$surp_col %in% names(df_subset)) {
      cat("[", subset_label, "] Fitting ", m$model, " Surprisal (", m$surp_col, ")\n", sep = "")
      fit_surp <- fit_single_gam(df_subset, m$surp_col)
      sm_surp <- extract_smooth(fit_surp, select = 1, n = 200)
      sm_surp$model <- m$model
      sm_surp$measure <- "Surprisal"
      sm_surp$predictor <- m$surp_col
      sm_surp$subset <- subset_label
      rows[[length(rows) + 1]] <- sm_surp
    } else {
      cat("[", subset_label, "] SKIP missing column: ", m$surp_col, "\n", sep = "")
    }
  }

  out <- bind_rows(rows)
  out_file <- paste0("bko21_SIC_", subset_label, "_raw_gam_curves.csv")
  write_csv(out, out_file)
  cat("Saved ", nrow(out), " rows to ", out_file, "\n", sep = "")
}

run_region_surp_pipeline <- function(df_subset, subset_label) {
  rows <- list()

  for (i in seq_len(nrow(region_surp_predictors))) {
    m <- region_surp_predictors[i, ]

    if (!m$pred_col %in% names(df_subset)) {
      cat("[", subset_label, "] SKIP missing column: ", m$pred_col, "\n", sep = "")
      next
    }

    cat("[", subset_label, "] Fitting ", m$model, " Region Surprisal (", m$pred_col, ")\n", sep = "")
    fit <- fit_single_gam(df_subset, m$pred_col)
    sm <- extract_smooth(fit, select = 1, n = 200)
    sm$model <- m$model
    sm$measure <- "Surprisal"
    sm$predictor <- m$pred_col
    sm$subset <- subset_label
    rows[[length(rows) + 1]] <- sm
  }

  out <- bind_rows(rows)
  out_file <- paste0("bko21_SIC_", subset_label, "_raw_region_surprisal_gam_curves.csv")
  write_csv(out, out_file)
  cat("Saved ", nrow(out), " rows to ", out_file, "\n", sep = "")
}

main <- function() {
  if (!file.exists(INPUT_FILE)) {
    stop("Input file not found: ", INPUT_FILE)
  }

  df_raw <- read_csv(INPUT_FILE, show_col_types = FALSE)
  df <- preprocess_data(df_raw)
  cat("Rows after preprocessing:", nrow(df), "\n")

  df_hc <- df %>% filter(condition == "HC")
  df_lc <- df %>% filter(condition == "LC")

  cat("HC rows:", nrow(df_hc), "\n")
  cat("LC rows:", nrow(df_lc), "\n")

  if (nrow(df_hc) > 0) {
    run_normal_pipeline(df_hc, "HC")
    run_region_surp_pipeline(df_hc, "HC")
  } else {
    cat("WARNING: no HC rows found.\n")
  }

  if (nrow(df_lc) > 0) {
    run_normal_pipeline(df_lc, "LC")
    run_region_surp_pipeline(df_lc, "LC")
  } else {
    cat("WARNING: no LC rows found.\n")
  }

  cat("\nDone.\n")
}

main()
