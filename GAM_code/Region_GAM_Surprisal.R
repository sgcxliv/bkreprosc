library(tidyverse)
library(mgcv)
library(readr)
setwd("~/Desktop")

DATA_FILE <- "/afs/cs.stanford.edu/u/sgcxliv/bkr21_spr.csv"

OUTPUT_CSV <- "/afs/cs.stanford.edu/u/sgcxliv/gam_plot_data_region_surprisal.csv"

OUTPUT_CSV_LOCAL <- "gam_plot_data_region_surprisal.csv"
OUTPUT_RAW_CSV <- "region_gam_raw_data_surprisal.csv"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
preprocess_data <- function(df, predictor_col) {
  required <- c("SUM_3RT_trimmed", "SUB", "ITEM", predictor_col)
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
      !!predictor_col := as.numeric(.data[[predictor_col]])
    ) %>%
    drop_na(all_of(required)) %>%
    filter(
      is.finite(SUM_3RT_trimmed),
      is.finite(.data[[predictor_col]])
    )
}


extract_gam_plot_curve <- function(model, select = 1, n = 100, se_with_mean = TRUE) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit(
    {
      grDevices::dev.off()
      unlink(pdf_path)
    },
    add = TRUE
  )

  plot_data <- plot(
    model,
    select = select,
    seWithMean = se_with_mean,
    n = n
  )

  if (is.null(plot_data) || length(plot_data) < 1L || is.null(plot_data[[1]])) {
    stop("plot.gam() did not return smooth data for model.")
  }

  pd <- plot_data[[1]]
  fit <- as.numeric(pd$fit)
  se <- as.numeric(pd$se)

  data.frame(
    x = as.numeric(pd$x),
    y = fit,
    se = se,
    lower = fit - 1.96 * se,
    upper = fit + 1.96 * se
  )
}


extract_raw_data <- function(model, term_index = 1) {
  pred_var <- model$smooth[[term_index]]$term[1]
  response_var <- names(model$model)[1]
  data.frame(
    x = model$model[[pred_var]],
    y = model$model[[response_var]]
  )
}


fit_region_surprisal_model <- function(df, predictor_col) {
  formula <- as.formula(
    paste0(
      "SUM_3RT_trimmed ~ 1 + s(",
      predictor_col,
      ", bs = 'cs') + s(SUB, bs = 're') + s(ITEM, bs = 're')"
    )
  )
  gam(formula, data = df)
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if (!file.exists(DATA_FILE)) {
  stop("Could not find DATA_FILE: ", DATA_FILE)
}

df_items <- read_csv(DATA_FILE, show_col_types = FALSE)

region_surprisal_predictors <- tribble(
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

smooth_data_list <- list()
raw_data_list <- list()

for (i in seq_len(nrow(region_surprisal_predictors))) {
  predictor_col <- region_surprisal_predictors$predictor_col[[i]]
  model_name <- region_surprisal_predictors$model_name[[i]]

  cat("Fitting region surprisal GAM for:", model_name, "(", predictor_col, ")\n")

  df_model <- preprocess_data(df_items, predictor_col)
  fit <- fit_region_surprisal_model(df_model, predictor_col)

  smooth_data <- extract_gam_plot_curve(fit, n = 100, se_with_mean = TRUE)
  smooth_data$model <- model_name
  smooth_data$measure <- "Surprisal"
  smooth_data_list[[model_name]] <- smooth_data

  raw_data <- extract_raw_data(fit)
  raw_data$model <- model_name
  raw_data$measure <- "Surprisal"
  raw_data_list[[model_name]] <- raw_data
}

smooth_data_combined <- bind_rows(smooth_data_list)
raw_data_combined <- bind_rows(raw_data_list)

output_dir <- dirname(OUTPUT_CSV)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

write_csv(smooth_data_combined, OUTPUT_CSV)
write_csv(smooth_data_combined, OUTPUT_CSV_LOCAL)
write_csv(raw_data_combined, OUTPUT_RAW_CSV)

cat("\nDone.\n")
cat("Saved smooth curves to:\n")
cat(" - ", OUTPUT_CSV, "\n", sep = "")
cat(" - ", OUTPUT_CSV_LOCAL, "\n", sep = "")
cat("Saved raw points to:\n")
cat(" - ", OUTPUT_RAW_CSV, "\n", sep = "")
cat("\nColumns in smooth CSV: x, y, se, lower, upper, model, measure\n")
