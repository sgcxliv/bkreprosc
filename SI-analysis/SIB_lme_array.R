#!/usr/bin/env Rscript
# ============================================================================
# SIB_lme_array.R
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

ESTIMATORS <- tribble(
  ~name,       ~surp_w1,             ~surp_w2,     ~surp_w3,     ~prob_w1,              ~prob_w2,        ~prob_w3,
  "Cloze",     "cloze_w1",           "cloze_w2",   "cloze_w3",   "clozeprob_w1",        "clozeprob_w2",  "clozeprob_w3",
  "GPT-2",     "surp_gpt2new",       "gpt2w2",     "gpt2w3",     "surp_gpt2newprob",    "gpt2probw2",    "gpt2probw3",
  "GPT-2 XL",  "surp_gpt2xlnew",     "gpt2xlw2",   "gpt2xlw3",   "surp_gpt2xlnewprob",  "gpt2xlprobw2",  "gpt2xlprobw3",
  "GPT-J",     "surp_gptjnew",       "gptjw2",     "gptjw3",     "surp_gptjnewprob",    "gptjprobw2",    "gptjprobw3",
  "GPT-Neo",   "surp_gptneonew",     "gptneow2",   "gptneow3",   "surp_gptneonewprob",  "gptneoprobw2",  "gptneoprobw3",
  "GPT-NeoX",  "surp_gptneoxnew",    "gptneoxw2",  "gptneoxw3",  "surp_gptneoxnewprob", "gptneoxprobw2", "gptneoxprobw3",
  "LLaMA-2",   "surp_llama2new",     "llama2w2",   "llama2w3",   "surp_llama2newprob",  "llama2probw2",  "llama2probw3",
  "OLMo-2",    "surp_olmonew",       "olmow2",     "olmow3",     "surp_olmonewprob",    "olmoprobw2",    "olmoprobw3"
)

RT_COLS  <- c("W1_RT", "W2_RT", "W3_RT")
RT_LOG   <- TRUE
RT_MIN   <- 100
RT_MAX   <- 2000
SCALE_PREDS <- TRUE

INPUT_CSV <- "/nlp/scr/$USER/SIB/tritems_with_surprisal.csv"
OUT_DIR <- "/nlp/scr/$USER/SIB/lme_results"
dir.create(OUT_DIR, showWarnings = FALSE)

# ---- Get job's estimator -----------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
idx  <- as.integer(args[1])
if (is.na(idx) || idx < 1 || idx > nrow(ESTIMATORS)) {
  stop("Usage: Rscript SIB_lme_array.R <1-", nrow(ESTIMATORS), ">")
}
est       <- ESTIMATORS$name[idx]
surp_cols <- c(ESTIMATORS$surp_w1[idx], ESTIMATORS$surp_w2[idx], ESTIMATORS$surp_w3[idx])
prob_cols <- c(ESTIMATORS$prob_w1[idx], ESTIMATORS$prob_w2[idx], ESTIMATORS$prob_w3[idx])
cat(sprintf("Job %d: estimator = %s\n", idx, est))

# ---- Load data --------------------------------------------------------------
csv_path <- INPUT_CSV
cat("Loading", csv_path, "...\n")
df <- read_csv(csv_path, show_col_types = FALSE, guess_max = 50000)
cat(sprintf("  %d rows x %d cols\n", nrow(df), ncol(df)))

# Check W1 columns exist
for (col in c(surp_cols[1], prob_cols[1])) {
  if (!(col %in% names(df))) stop("Missing required W1 column: ", col)
}

# Trim and log RT
df_lme <- df
for (rtc in intersect(RT_COLS, names(df_lme))) {
  df_lme[[rtc]] <- ifelse(
    df_lme[[rtc]] < RT_MIN | df_lme[[rtc]] > RT_MAX,
    NA_real_, df_lme[[rtc]]
  )
  if (RT_LOG) df_lme[[rtc]] <- log(df_lme[[rtc]])
}

# ---- Fit 6 models (lower triangle: j=1,2,3 x predictors W1..Wj) ------------
lme_rows <- list()

for (j in 1:3) {
  rt_col <- RT_COLS[j]
  if (!(rt_col %in% names(df_lme))) {
    cat(sprintf("  [skip] %s not in data\n", rt_col))
    next
  }

  sj_surp  <- surp_cols[1:j]
  sj_prob  <- prob_cols[1:j]
  have_s   <- !is.na(sj_surp) & (sj_surp %in% names(df_lme))
  have_p   <- !is.na(sj_prob) & (sj_prob %in% names(df_lme))
  use_surp <- sj_surp[have_s]
  use_prob <- sj_prob[have_p]

  if (length(use_surp) == 0 || length(use_prob) == 0) {
    cat(sprintf("  [skip] %s: no usable surp or prob predictors\n", rt_col))
    next
  }

  fdf <- df_lme %>%
    select(all_of(c("SUB", "ITEM", rt_col, use_surp, use_prob))) %>%
    drop_na()
  cat(sprintf("  %s ~ %d preds   n=%d\n", rt_col,
              length(use_surp) + length(use_prob), nrow(fdf)))

  if (SCALE_PREDS) {
    for (cc in c(use_surp, use_prob))
      fdf[[cc]] <- as.numeric(scale(fdf[[cc]]))
  }

  form <- as.formula(sprintf(
    "%s ~ %s + (1|SUB) + (1|ITEM)",
    rt_col, paste(c(use_surp, use_prob), collapse = " + ")
  ))

  t0  <- Sys.time()
  fit <- tryCatch(
    lmer(form, data = fdf, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa",
                               optCtrl  = list(maxfun = 1e5))),
    error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL }
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("    done in %.1fs\n", elapsed))
  if (is.null(fit)) next

  co     <- summary(fit)$coefficients
  co_df  <- as_tibble(co, rownames = "term")
  map_df <- bind_rows(
    tibble(term = use_surp, var = "surp",
           pred_word = paste0("W", which(have_s))),
    tibble(term = use_prob, var = "prob",
           pred_word = paste0("W", which(have_p)))
  )
  lme_rows[[j]] <- co_df %>%
    inner_join(map_df, by = "term") %>%
    transmute(
      estimator     = est,
      var           = var,
      pred_word     = pred_word,
      response_word = paste0("W", j),
      coef          = Estimate,
      se            = `Std. Error`,
      t_val         = `t value`,
      n_obs         = nrow(fdf)
    )
}

# ---- Write output -----------------------------------------------------------
out <- bind_rows(lme_rows)
out_path <- file.path(OUT_DIR, sprintf("lme_coefs_%02d_%s.csv", idx,
                                        gsub("[^A-Za-z0-9]", "_", est)))
write_csv(out, out_path)
cat(sprintf("Wrote %s (%d rows)\n", out_path, nrow(out)))
