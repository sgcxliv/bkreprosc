#!/usr/bin/env Rscript
# =====================================================================
# SI_C_Refit.R   (bam version)
# =====================================================================
# Refits SI-C HC + LC GAMs using bam() for speed at this data size.
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(mgcv)
  library(readr)
})

WORK_DIR <- "/afs/cs.stanford.edu/u/sgcxliv"
if (dir.exists(WORK_DIR)) setwd(WORK_DIR)

N_THREADS <- 4

DATASETS <- list(
  list(file = "bkr21_spr.csv", prefix = "bkr"),
  list(file = "bko21_spr.csv", prefix = "bko")
)

CONDITIONS <- c("HC", "LC")

curve_labels <- tribble(
  ~prob_col,     ~surp_col,  ~model_name,
  "clozeprob",   "cloze",    "Cloze",
  "gpt2prob",    "gpt2",     "GPT-2-small",
  "gptneoprob",  "gptneo",   "GPT-Neo",
  "gptneoxprob", "gptneox",  "GPT-NeoX",
  "gptjprob",    "gptj",     "GPT-J",
  "gpt2xlprob",  "gpt2xl",   "GPT-2XL",
  "olmoprob",    "olmo",     "OLMO-2",
  "llama2prob",  "llama2",   "LLaMA-2"
)
model_order <- curve_labels$model_name

region_predictors <- tribble(
  ~predictor_col,  ~model_name,
  "clozeregion",   "Cloze",
  "gpt2region",    "GPT-2-small",
  "gptneoregion",  "GPT-Neo",
  "gptneoxregion", "GPT-NeoX",
  "gptjregion",    "GPT-J",
  "gpt2xlregion",  "GPT-2XL",
  "olmoregion",    "OLMO-2",
  "llama2region",  "LLaMA-2"
)
# Core fit + smooth extraction 
fit_smooth <- function(df, predictor_col, dv_col, label) {
  required <- c(dv_col, "SUB", "ITEM", predictor_col)
  if (!all(required %in% names(df))) return(NULL)

  df <- df %>%
    mutate(SUB = as.factor(SUB), ITEM = as.factor(ITEM)) %>%
    drop_na(all_of(required)) %>%
    filter(is.finite(.data[[dv_col]]),
           is.finite(.data[[predictor_col]]))

  if (nrow(df) < 20) return(NULL)

  nu <- length(unique(df[[predictor_col]]))
  if (nu < 4) return(NULL)
  k_use <- max(3L, min(10L, nu - 1L))

  form <- as.formula(sprintf(
    "%s ~ 1 + s(%s, bs='tp', k=%d) + s(SUB, bs='re') + s(ITEM, bs='re')",
    dv_col, predictor_col, k_use
  ))

  t0 <- Sys.time()
  cat(sprintf("    %s (n=%d, k=%d) ... ", label, nrow(df), k_use))

  fit <- tryCatch(
    bam(form,
        data     = df,
        method   = "fREML",
        discrete = TRUE,
        nthreads = N_THREADS),
    error = function(e) { cat("ERROR:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(fit)) return(NULL)
  cat(sprintf("%.1fs\n", as.numeric(Sys.time() - t0, units = "secs")))

  pdf(file = tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  pd <- plot(fit, select = 1, seWithMean = TRUE, n = 100)[[1]]

  lo <- quantile(df[[predictor_col]], 0.01, na.rm = TRUE)
  hi <- quantile(df[[predictor_col]], 0.99, na.rm = TRUE)

  data.frame(
    x     = as.numeric(pd$x),
    y     = as.numeric(pd$fit),
    lower = as.numeric(pd$fit) - as.numeric(pd$se),
    upper = as.numeric(pd$fit) + as.numeric(pd$se)
  ) %>% filter(x >= lo, x <= hi)
}

# Main
cat("bam() with discrete=TRUE, method='fREML', nthreads=", N_THREADS, "\n", sep = "")

for (ds in DATASETS) {
  cat("\n", strrep("=", 64), "\n", sep = "")
  cat("Loading:", ds$file, "\n")
  cat(strrep("=", 64), "\n", sep = "")

  if (!file.exists(ds$file)) {
    cat("  ERROR: file not found, skipping.\n"); next
  }

  df_raw <- read_csv(ds$file, show_col_types = FALSE)
  cat("  Rows:", nrow(df_raw),
      " SUBs:", length(unique(df_raw$SUB)),
      " ITEMs:", length(unique(df_raw$ITEM)), "\n")

  if (!"condition" %in% names(df_raw)) {
    cat("  ERROR: no 'condition' column, skipping.\n"); next
  }

  for (cond in CONDITIONS) {
    df_cond <- df_raw %>% filter(condition == cond)
    cat(sprintf("\n--- %s [%s] n = %d ---\n", ds$prefix, cond, nrow(df_cond)))
    if (nrow(df_cond) == 0) next

    cat(" Critical-word predictability:\n")
    main_rows <- list()
    for (i in seq_len(nrow(curve_labels))) {
      m_name <- curve_labels$model_name[i]
      for (kind in c("Probability", "Surprisal")) {
        pcol <- if (kind == "Probability") curve_labels$prob_col[i]
                else                       curve_labels$surp_col[i]
        sm <- fit_smooth(df_cond, pcol, "SUM_3RT_trimmed",
                         sprintf("%s/%s", m_name, kind))
        if (is.null(sm)) next
        sm$model   <- m_name
        sm$measure <- kind
        main_rows[[paste(m_name, kind)]] <- sm
      }
    }
    main_df <- bind_rows(main_rows)
    write_csv(main_df, sprintf("%s_SIC_%s_refit_smooth.csv", ds$prefix, cond))
    plot_main(
      main_df,
      sprintf("%s SIC %s items: Critical-Word GAM (refit)",
              toupper(ds$prefix), cond),
      sprintf("%s_SIC_%s_main_refit.png", ds$prefix, cond)
    )

    cat(" Region surprisal:\n")
    reg_rows <- list()
    for (i in seq_len(nrow(region_predictors))) {
      pcol   <- region_predictors$predictor_col[i]
      m_name <- region_predictors$model_name[i]
      sm <- fit_smooth(df_cond, pcol, "SUM_3RT_trimmed",
                       sprintf("%s region", m_name))
      if (is.null(sm)) next
      sm$model   <- m_name
      sm$measure <- "Region Surprisal"
      reg_rows[[m_name]] <- sm
    }
    reg_df <- bind_rows(reg_rows)
    write_csv(reg_df, sprintf("%s_SIC_%s_refit_region_smooth.csv",
                              ds$prefix, cond))
    plot_region(
      reg_df,
      sprintf("%s SIC %s items: Region Surprisal GAM (refit)",
              toupper(ds$prefix), cond),
      sprintf("%s_SIC_%s_region_refit.png", ds$prefix, cond)
    )
  }
}

cat("\nDONE.\n")
