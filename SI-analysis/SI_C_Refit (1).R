#!/usr/bin/env Rscript
# =====================================================================
# SI_C_refit_compare.R
# =====================================================================
# FAST SI-C refit (BKR + BKO, HC + LC, non-region prob/surp + region
# surprisal) that fits EACH panel TWO ways and plots them overlaid:
#
#   * "With s(ITEM)"  -> conservative / suppressed curves (your current ones)
#   * "No s(ITEM)"    -> un-suppressed curves (the item random effect no
#                        longer competes with the item-constant predictor)

# Usage:  Rscript SI_C_refit_compare.R [WORK_DIR]
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse); library(mgcv); library(readr)
})

# ── Config ───────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = TRUE)
WORK_DIR <- if (length(args) >= 1) args[[1]] else "/afs/cs.stanford.edu/u/sgcxliv"
setwd(WORK_DIR)
OUT_DIR <- file.path(WORK_DIR, "SIC_refit"); dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SMOOTH_BS  <- "tp"; SMOOTH_K <- 6; GAM_METHOD <- "fREML"
SE_WITH_MEAN <- TRUE; N_GRID <- 100; MIN_ROWS <- 30
NTHREADS   <- max(1L, parallel::detectCores(logical = TRUE) - 1L)
MAKE_R_PLOTS <- TRUE

SPEC_WITH <- "With s(ITEM)"   # suppressed / conservative
SPEC_NO   <- "No s(ITEM)"     # un-suppressed

cat("bam() discrete=TRUE method=", GAM_METHOD, " nthreads=", NTHREADS,
    "  | fitting BOTH specs per panel\n", sep = "")

DATASETS <- list(
  list(file = "bkr21_spr.csv", prefix = "bkr"),
  list(file = "bko21_spr.csv", prefix = "bko")
)

curve_labels <- tribble(
  ~prob_col,     ~surp_col,  ~model_name,
  "clozeprob",   "cloze",    "Cloze",
  "gpt2prob",    "gpt2",     "GPT-2",
  "gptneoprob",  "gptneo",   "GPT-Neo",
  "gptneoxprob", "gptneox",  "GPT-NeoX",
  "gptjprob",    "gptj",     "GPT-J",
  "gpt2xlprob",  "gpt2xl",   "GPT-2XL",
  "olmoprob",    "olmo",     "OLMO-2",
  "llama2prob",  "llama2",   "LLaMA-2"
)
region_predictors <- tribble(
  ~predictor_col,  ~model_name,
  "clozeregion",   "Cloze",   "gpt2region",    "GPT-2",
  "gptneoregion",  "GPT-Neo", "gptneoxregion", "GPT-NeoX",
  "gptjregion",    "GPT-J",   "gpt2xlregion",  "GPT-2XL",
  "olmoregion",    "OLMO-2",  "llama2region",  "LLaMA-2"
) %>% as_tibble()

region_predictors <- tribble(
  ~predictor_col,  ~model_name,
  "clozeregion",   "Cloze",
  "gpt2region",    "GPT-2",
  "gptneoregion",  "GPT-Neo",
  "gptneoxregion", "GPT-NeoX",
  "gptjregion",    "GPT-J",
  "gpt2xlregion",  "GPT-2XL",
  "olmoregion",    "OLMO-2",
  "llama2region",  "LLaMA-2"
)

MODEL_ORDER <- c("Cloze","GPT-2","GPT-Neo","GPT-NeoX",
                 "GPT-J","GPT-2XL","OLMO-2","LLaMA-2")
DIAG <- list()

# ── Helpers ──────────────────────────────────────────────────────────
coerce_numeric_report <- function(v) {
  if (is.numeric(v)) return(list(num = v, n_na_introduced = 0L, was_type = "numeric"))
  was_type <- class(v)[1]
  v_chr <- trimws(as.character(v))
  v_chr[v_chr %in% c("", "NA", "N/A", "na", "NaN")] <- NA
  num <- suppressWarnings(as.numeric(v_chr))
  list(num = num, was_type = was_type,
       n_na_introduced = sum(is.na(num) & !is.na(v_chr)))
}
prop_var_within_item <- function(x, item) {
  ok <- is.finite(x); x <- x[ok]; item <- item[ok]
  if (length(x) < 2) return(NA_real_)
  total <- sum((x - mean(x))^2); if (total == 0) return(0)
  within <- tapply(x, item, function(v) sum((v - mean(v))^2))
  sum(within, na.rm = TRUE) / total
}
extract_gam_plot_curve <- function(model, select = 1, n = N_GRID, se_with_mean = SE_WITH_MEAN) {
  pdf_path <- tempfile(fileext = ".pdf"); grDevices::pdf(file = pdf_path)
  on.exit({ grDevices::dev.off(); unlink(pdf_path) }, add = TRUE)
  pdat <- plot(model, select = select, seWithMean = se_with_mean, n = n)
  if (is.null(pdat) || length(pdat) < 1L || is.null(pdat[[1]])) stop("no smooth data")
  pd <- pdat[[1]]; fit <- as.numeric(pd$fit); se <- as.numeric(pd$se)
  data.frame(x = as.numeric(pd$x), y = fit, se = se,
             lower = fit - 1.96 * se, upper = fit + 1.96 * se)
}
extract_raw_data <- function(model, term_index = 1) {
  pred_var <- model$smooth[[term_index]]$term[1]
  response_var <- names(model$model)[1]
  data.frame(x = model$model[[pred_var]], y = model$model[[response_var]])
}
smooth_edf <- function(fit, pred_col) {
  st <- summary(fit)$s.table
  idx <- grep(paste0("s(", pred_col, ")"), rownames(st), fixed = TRUE)
  if (length(idx)) st[idx[1], "edf"] else NA_real_
}

# Fit a panel BOTH ways; return list with $with, $no curves + $raw, log diag.
fit_panel_both <- function(df_full, predictor_col, model_name, measure, tag,
                           dv_col = "SUM_3RT_trimmed") {
  cat(sprintf("    %-9s %-11s ", model_name, measure))
  if (!predictor_col %in% names(df_full)) {
    cat("SKIP: column '", predictor_col, "' not in data\n", sep = "")
    DIAG[[length(DIAG)+1]] <<- tibble(tag=tag, model=model_name, measure=measure,
      predictor=predictor_col, n=NA_integer_, status="SKIP: column missing"); return(NULL)
  }
  cr <- coerce_numeric_report(df_full[[predictor_col]])
  dvr <- coerce_numeric_report(df_full[[dv_col]])
  d <- df_full
  d[[predictor_col]] <- cr$num; d[[dv_col]] <- dvr$num
  d$SUB <- as.factor(d$SUB); d$ITEM <- as.factor(d$ITEM)
  d <- d[is.finite(d[[predictor_col]]) & is.finite(d[[dv_col]]) &
         !is.na(d$SUB) & !is.na(d$ITEM), , drop = FALSE]
  x <- d[[predictor_col]]; n <- nrow(d)
  n_unique <- length(unique(x)); x_sd <- if (n_unique > 1) sd(x) else 0
  pvw <- prop_var_within_item(x, d$ITEM)

  base_diag <- tibble(tag=tag, model=model_name, measure=measure,
    predictor=predictor_col, dv=dv_col, n=n, n_unique_x=n_unique,
    x_min=if(n) min(x) else NA, x_max=if(n) max(x) else NA, x_sd=x_sd,
    prop_var_within_item=pvw, edf_with_ITEM=NA_real_, edf_no_ITEM=NA_real_,
    status="ok")

  if (n < MIN_ROWS || n_unique < 4 || x_sd == 0) {
    reason <- sprintf("SKIP: n=%d unique=%d sd=%.3g", n, n_unique, x_sd)
    cat(reason, "\n"); base_diag$status <- reason
    DIAG[[length(DIAG)+1]] <<- base_diag; return(NULL)
  }
  k_use <- max(3, min(SMOOTH_K, n_unique - 1))

  f_with <- as.formula(sprintf(
    "%s ~ 1 + s(%s, bs='%s', k=%d) + s(SUB, bs='re') + s(ITEM, bs='re')",
    dv_col, predictor_col, SMOOTH_BS, k_use))
  f_no <- as.formula(sprintf(
    "%s ~ 1 + s(%s, bs='%s', k=%d) + s(SUB, bs='re')",
    dv_col, predictor_col, SMOOTH_BS, k_use))

  fit_w <- tryCatch(bam(f_with, data=d, method=GAM_METHOD, discrete=TRUE, nthreads=NTHREADS),
                    error=function(e){cat("ERR(with):",conditionMessage(e),"\n");NULL})
  fit_n <- tryCatch(bam(f_no,   data=d, method=GAM_METHOD, discrete=TRUE, nthreads=NTHREADS),
                    error=function(e){cat("ERR(no):", conditionMessage(e),"\n");NULL})
  if (is.null(fit_w) || is.null(fit_n)) {
    base_diag$status <- "ERR (see console)"; DIAG[[length(DIAG)+1]] <<- base_diag; return(NULL)
  }

  base_diag$edf_with_ITEM <- smooth_edf(fit_w, predictor_col)
  base_diag$edf_no_ITEM   <- smooth_edf(fit_n, predictor_col)
  DIAG[[length(DIAG)+1]] <<- base_diag

  cw <- extract_gam_plot_curve(fit_w); cw$model <- model_name; cw$measure <- measure; cw$spec <- SPEC_WITH
  cn <- extract_gam_plot_curve(fit_n); cn$model <- model_name; cn$measure <- measure; cn$spec <- SPEC_NO
  rw <- extract_raw_data(fit_w); rw$model <- model_name; rw$measure <- measure

  cat(sprintf("OK  n=%d  edf: with=%.2f  no=%.2f\n",
              n, base_diag$edf_with_ITEM, base_diag$edf_no_ITEM))
  list(with = cw, no = cn, raw = rw)
}

save_compare <- function(curves, raws, tag, suffix = "") {
  with_df <- bind_rows(lapply(curves, `[[`, "with"))
  no_df   <- bind_rows(lapply(curves, `[[`, "no"))
  raw_df  <- bind_rows(raws)
  write_csv(with_df %>% select(-spec),
            file.path(OUT_DIR, paste0(tag, suffix, "_gam_smooth_data.csv")))
  write_csv(no_df %>% select(-spec),
            file.path(OUT_DIR, paste0(tag, suffix, "_noITEM_gam_smooth_data.csv")))
  write_csv(bind_rows(with_df, no_df),
            file.path(OUT_DIR, paste0(tag, suffix, "_compare_smooth_data.csv")))
  write_csv(raw_df, file.path(OUT_DIR, paste0(tag, suffix, "_gam_raw_data.csv")))
  cat("    -> saved CSVs (with / noITEM / compare / raw) for", paste0(tag, suffix), "\n")
  bind_rows(with_df, no_df)
}

plot_compare <- function(cmp_df, title, file, grid = FALSE) {
  if (!MAKE_R_PLOTS || is.null(cmp_df) || !nrow(cmp_df)) return(invisible())
  cmp_df$model <- factor(cmp_df$model, levels = MODEL_ORDER)
  cmp_df$spec  <- factor(cmp_df$spec, levels = c(SPEC_WITH, SPEC_NO))
  p <- ggplot(cmp_df, aes(x, y, colour = spec, fill = spec)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.18, colour = NA) +
    geom_line(linewidth = 0.8) +
    scale_colour_manual(values = c("#444444", "#2c7fb8")) +
    scale_fill_manual(values   = c("#444444", "#2c7fb8")) +
    labs(title = title, x = NULL, y = "Partial effect on SUMMED_3RT (ms)",
         colour = NULL, fill = NULL) +
    theme_minimal(base_size = 11) +
    theme(strip.text = element_text(face = "bold"), legend.position = "top")
  if (grid) {
    cmp_df$measure <- factor(cmp_df$measure, levels = c("Probability","Surprisal"))
    p <- p + facet_grid(measure ~ model, scales = "free"); h <- 4.8
  } else { p <- p + facet_wrap(~ model, scales = "free", nrow = 1); h <- 3.2 }
  ggsave(file.path(OUT_DIR, file), p, width = 20, height = h, dpi = 150)
  cat("    -> plotted", file, "\n")
}

# ── Drivers ──────────────────────────────────────────────────────────
run_nonregion <- function(df, prefix, subset_label) {
  tag <- paste0(prefix, "_SIC_", subset_label); cat("  [non-region]", tag, "\n")
  curves <- list(); raws <- list()
  for (i in seq_len(nrow(curve_labels))) {
    mn <- curve_labels$model_name[[i]]
    for (kind in c("prob","surp")) {
      col <- curve_labels[[paste0(kind,"_col")]][[i]]
      measure <- if (kind == "prob") "Probability" else "Surprisal"
      res <- fit_panel_both(df, col, mn, measure, tag)
      if (!is.null(res)) { curves[[paste0(mn,"_",kind)]] <- res; raws[[paste0(mn,"_",kind)]] <- res$raw }
    }
  }
  cmp <- save_compare(curves, raws, tag)
  plot_compare(cmp, paste(toupper(prefix), "SIC", subset_label,
               "Items: SUMMED_3RT  (with vs no s(ITEM))"),
               paste0(tag, "_main_compare.png"), grid = TRUE)
}
run_region <- function(df, prefix, subset_label) {
  tag <- paste0(prefix, "_SIC_", subset_label); cat("  [region]    ", tag, "\n")
  curves <- list(); raws <- list()
  for (i in seq_len(nrow(region_predictors))) {
    col <- region_predictors$predictor_col[[i]]; mn <- region_predictors$model_name[[i]]
    res <- fit_panel_both(df, col, mn, "Surprisal", tag)
    if (!is.null(res)) { curves[[mn]] <- res; raws[[mn]] <- res$raw }
  }
  cmp <- save_compare(curves, raws, tag, suffix = "_region")
  plot_compare(cmp, paste(toupper(prefix), "SIC", subset_label,
               "Items: Region Surprisal  (with vs no s(ITEM))"),
               paste0(tag, "_region_compare.png"), grid = FALSE)
}

# ── Main ─────────────────────────────────────────────────────────────
for (ds in DATASETS) {
  cat("\n", strrep("=", 60), "\n", "Loading: ", ds$file, "\n", strrep("=", 60), "\n", sep = "")
  if (!file.exists(ds$file)) { cat("  ERROR: file not found, skipping.\n"); next }
  df_raw <- read_csv(ds$file, show_col_types = FALSE)
  cat("  total rows:", nrow(df_raw), "\n")
  if (!"condition" %in% names(df_raw)) { cat("  ERROR: no 'condition' column.\n"); next }
  cond <- toupper(trimws(as.character(df_raw$condition)))
  cat("  condition counts:\n"); print(table(cond, useNA = "ifany"))
  for (subset_label in c("HC","LC")) {
    df_sub <- df_raw[cond == subset_label, , drop = FALSE]
    cat("  --", subset_label, "rows:", nrow(df_sub), "\n")
    if (nrow(df_sub) < MIN_ROWS) { cat("     WARNING: too few rows; skipping\n"); next }
    run_nonregion(df_sub, ds$prefix, subset_label)
    run_region(df_sub,    ds$prefix, subset_label)
  }
}

diag_df <- bind_rows(DIAG)
write_csv(diag_df, file.path(OUT_DIR, "SIC_refit_diagnostics.csv"))
cat("\n", strrep("=", 60), "\n", sep = "")
cat("Compare edf_with_ITEM vs edf_no_ITEM in the diagnostics:\n",
    " big gap (e.g. 1.0 vs 4-5) = s(ITEM) was suppressing a real curve.\n",
    " similar & both ~1          = genuinely flat (e.g. GPT-J); not an artifact.\n", sep="")
print(diag_df, n = nrow(diag_df), width = Inf)
cat("\nALL DONE. Outputs in:", OUT_DIR, "\n")