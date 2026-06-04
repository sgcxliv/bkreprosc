library(tidyverse)
library(mgcv)
library(readr)

# =====================================================================
# SI-A Region Surprisal GAM Extraction
# - DV: log(SUM_3RT_trimmed)
# - Predictors: region-level surprisal columns only
# - Exports smooth points using plot.gam(..., seWithMean=TRUE)
# =====================================================================

WORK_DIR <- "~/Desktop"
setwd(WORK_DIR)

DATASETS <- list(
  # list(file = "bkr21_spr.csv", label = "bkr21"),
  list(file = "bko21_spr.csv", label = "bko21")
)

region_surp <- tribble(
  ~pred_col,       ~model,
  "clozeregion",   "Cloze",
  "gpt2region",    "GPT-2",
  "gptneoregion",  "GPT-Neo",
  "gptneoxregion", "GPT-NeoX",
  "gptjregion",    "GPT-J",
  "gpt2xlregion",  "GPT-2XL",
  "olmoregion",    "OLMO-2",
  "llama2region",  "LLaMA-2"
)

preprocess_data <- function(df) {
  df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
      log_SUM_3RT = if_else(SUM_3RT_trimmed > 0, log(SUM_3RT_trimmed), NA_real_)
    ) %>%
    drop_na(log_SUM_3RT, SUB, ITEM)
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

fit_region_surprisal <- function(df, pred_col) {
  required <- c("log_SUM_3RT", "SUB", "ITEM", pred_col)
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  df_model <- df %>%
    mutate(!!pred_col := as.numeric(.data[[pred_col]])) %>%
    drop_na(all_of(required)) %>%
    filter(is.finite(.data[[pred_col]]), is.finite(log_SUM_3RT))

  formula <- as.formula(
    paste0(
      "log_SUM_3RT ~ 1 + s(", pred_col, ", bs='cs') + ",
      "s(SUB, bs='re') + s(ITEM, bs='re')"
    )
  )

  bam(formula, data = df_model)
}

run_dataset <- function(ds) {
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Loading:", ds$file, "\n")
  cat(strrep("=", 60), "\n", sep = "")

  if (!file.exists(ds$file)) {
    cat("ERROR: file not found, skipping:", ds$file, "\n")
    return(invisible(NULL))
  }

  df_raw <- read_csv(ds$file, show_col_types = FALSE)
  df <- preprocess_data(df_raw)
  cat("Rows after preprocessing:", nrow(df), "\n")

  all_rows <- list()

  for (i in seq_len(nrow(region_surp))) {
    pred_col <- region_surp$pred_col[[i]]
    model_name <- region_surp$model[[i]]
    cat("Fitting:", model_name, "(", pred_col, ")\n")

    if (!pred_col %in% names(df)) {
      cat("  SKIP: missing column", pred_col, "\n")
      next
    }

    fit <- fit_region_surprisal(df, pred_col)
    sm <- extract_smooth(fit, select = 1, n = 200)
    sm$model <- model_name
    sm$measure <- "Surprisal"
    sm$pred_col <- pred_col
    sm$dataset <- ds$label

    all_rows[[length(all_rows) + 1]] <- sm
  }

  if (length(all_rows) == 0) {
    cat("No models were fit for", ds$label, "\n")
    return(invisible(NULL))
  }

  out <- bind_rows(all_rows)
  out_file <- paste0(ds$label, "_SIA_log_region_surprisal_gam_curves.csv")
  write_csv(out, out_file)
  cat("Saved", nrow(out), "rows to", out_file, "\n")
}

for (ds in DATASETS) {
  run_dataset(ds)
}

cat("\nDone.\n")
