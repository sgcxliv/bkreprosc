library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("/afs/cs.stanford.edu/u/sgcxliv")
# setwd("~/Desktop")

df_items <- read_csv("bkr21_spr.csv")

preprocess_data <- function(df) {
  df <- df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
    ) %>%
    drop_na(SUM_3RT_trimmed, clozeprob, SUB, ITEM)  # Remove NA rows
  
  return(df)
}

df_items <- preprocess_data(df_items)

m_clozeprob <- gam(SUM_3RT_trimmed ~ 1 +
                    s(clozeprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_clozesurp <- gam(SUM_3RT_trimmed ~ 1 +
                    s(cloze, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2prob <- gam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2prob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2surp <- gam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptneoprob <- gam(SUM_3RT_trimmed ~ 1 +
                      s(gptneoprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneosurp <- gam(SUM_3RT_trimmed ~ 1 +
                      s(gptneo, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneoxprob <- gam(SUM_3RT_trimmed ~ 1 +
                       s(gptneoxprob, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_gptneoxsurp <- gam(SUM_3RT_trimmed ~ 1 +
                       s(gptneox, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_olmoprob <- gam(SUM_3RT_trimmed ~ 1 +
                    s(olmoprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_olmosurp <- gam(SUM_3RT_trimmed ~ 1 +
                    s(olmo, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_llama2prob <- gam(SUM_3RT_trimmed ~ 1 +
                      s(llama2prob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_llama2surp <- gam(SUM_3RT_trimmed ~ 1 +
                      s(llama2, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptjprob <- gam(SUM_3RT_trimmed ~ 1 +
                    s(gptjprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptjsurp <- gam(SUM_3RT_trimmed ~ 1 +
                    s(gptj, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2xlprob <- gam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xlprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gpt2xlsurp <- gam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xl, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

pdf("bko_gam_plots.pdf")
plot(m_clozeprob, select=1, xlab='Cloze Probability', ylab='RT')
plot(m_clozesurp, select=1, xlab='Cloze Surprisal', ylab='RT')
plot(m_gpt2prob, select=1, xlab='GPT-2 Probability', ylab='RT')
plot(m_gpt2surp, select=1, xlab='GPT-2 Surprisal', ylab='RT')
plot(m_gptneoprob, select=1, xlab='GPT-Neo Probability', ylab='RT')
plot(m_gptneosurp, select=1, xlab='GPT-Neo Surprisal', ylab='RT')
plot(m_gptneoxprob, select=1, xlab='GPT-NeoX Probability', ylab='RT')
plot(m_gptneoxsurp, select=1, xlab='GPT-NeoX Surprisal', ylab='RT')
plot(m_gptjprob, select=1, xlab='GPT-J Probability', ylab='RT')
plot(m_gptjsurp, select=1, xlab='GPT-J Surprisal', ylab='RT')
plot(m_gpt2xlprob, select=1, xlab='GPT-2XL Probability', ylab='RT')
plot(m_gpt2xlsurp, select=1, xlab='GPT-2XL Surprisal', ylab='RT')
plot(m_olmoprob, select=1, xlab='OLMO-2 Probability', ylab='RT')
plot(m_olmosurp, select=1, xlab='OLMO-2 Surprisal', ylab='RT')
plot(m_llama2prob, select=1, xlab='LLaMA-2 Probability', ylab='RT')
plot(m_llama2surp, select=1, xlab='LLaMA-2 Surprisal', ylab='RT')
dev.off()

# Extract 
extract_gam_plot_curve <- function(model, select = 1, n = 100, se_with_mean = TRUE) {
  # grDevices::nullfile() is only in recent R and is not always exported; tempfile works everywhere.
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
    stop("plot.gam() did not return smooth plotting data (mgcv version issue?)")
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

# Function to extract raw data for a model
extract_raw_data <- function(model, term_index = 1) {
  # Get the predictor variable name for the smooth term.
  pred_var <- model$smooth[[term_index]]$term[1]
  if (is.null(pred_var) || length(pred_var) == 0 || !nzchar(pred_var)) {
    stop("Could not determine predictor variable for term_index = ", term_index)
  }
  
  # Get the response variable
  response_var <- names(model$model)[1]
  
  data.frame(
    x = model$model[[pred_var]],
    y = model$model[[response_var]]
  )
}

# Extract data from all models
# Create an empty list to store the results
smooth_data_list <- list()
raw_data_list <- list()

# Extract data from each model
model_list <- list(
  m_clozeprob = m_clozeprob,
  m_clozesurp = m_clozesurp,
  m_gpt2prob = m_gpt2prob,
  m_gpt2surp = m_gpt2surp,
  m_gptneoprob = m_gptneoprob,
  m_gptneosurp = m_gptneosurp,
  m_gptneoxprob = m_gptneoxprob,
  m_gptneoxsurp = m_gptneoxsurp,
  m_olmoprob = m_olmoprob,
  m_olmosurp = m_olmosurp,
  m_llama2prob = m_llama2prob,
  m_llama2surp = m_llama2surp,
  m_gptjprob = m_gptjprob,
  m_gptjsurp = m_gptjsurp,
  m_gpt2xlprob = m_gpt2xlprob,
  m_gpt2xlsurp = m_gpt2xlsurp
)

curve_labels <- tribble(
  ~key,            ~model,     ~measure,
  "m_clozeprob",   "Cloze",    "Probability",
  "m_clozesurp",   "Cloze",    "Surprisal",
  "m_gpt2prob",    "GPT-2",    "Probability",
  "m_gpt2surp",    "GPT-2",    "Surprisal",
  "m_gptneoprob",  "GPT-Neo",  "Probability",
  "m_gptneosurp",  "GPT-Neo",  "Surprisal",
  "m_gptneoxprob", "GPT-NeoX", "Probability",
  "m_gptneoxsurp", "GPT-NeoX", "Surprisal",
  "m_olmoprob",    "OLMO-2",   "Probability",
  "m_olmosurp",    "OLMO-2",   "Surprisal",
  "m_llama2prob",  "LLaMA-2",  "Probability",
  "m_llama2surp",  "LLaMA-2",  "Surprisal",
  "m_gptjprob",    "GPT-J",    "Probability",
  "m_gptjsurp",    "GPT-J",    "Surprisal",
  "m_gpt2xlprob",  "GPT-2XL",  "Probability",
  "m_gpt2xlsurp",  "GPT-2XL",  "Surprisal"
)

if (exists("extract_gam_plot_curve", mode = "function", inherits = TRUE)) {
  .fn_txt <- paste(deparse(extract_gam_plot_curve), collapse = "\n")
  if (grepl("nullfile", .fn_txt, fixed = TRUE)) {
    rm("extract_gam_plot_curve", envir = .GlobalEnv)
    stop(
      "Your session had a stale extract_gam_plot_curve() that calls grDevices::nullfile().\n",
      "It is now removed. Re-run from the block that defines extract_gam_plot_curve (~line 143)\n",
      "through the end of this file, or run: source(\"path/to/Gam_Plots_with_Cloze.R\") from the top.",
      call. = FALSE
    )
  }
  rm(.fn_txt)
}

# Extract data from each model
for (model_name in names(model_list)) {
  smooth_data <- extract_gam_plot_curve(model_list[[model_name]], n = 100, se_with_mean = TRUE)
  lab <- curve_labels[curve_labels$key == model_name, , drop = FALSE]
  if (nrow(lab) != 1L) {
    stop("Missing curve_labels row for: ", model_name)
  }
  smooth_data$model <- lab$model[[1]]
  smooth_data$measure <- lab$measure[[1]]

  raw_data <- extract_raw_data(model_list[[model_name]])
  raw_data$model <- lab$model[[1]]
  raw_data$measure <- lab$measure[[1]]

  smooth_data_list[[model_name]] <- smooth_data
  raw_data_list[[model_name]] <- raw_data
}

# Combine all data frames in the list
smooth_data_combined <- bind_rows(smooth_data_list)
raw_data_combined <- bind_rows(raw_data_list)

# Primary output: same schema as gam_plot_data_orig.csv (x, y, se, lower, upper, model, measure)
gam_out <- file.path("410", "finalresults", "gam", "gam_plot_data_orig.csv")
if (dirname(gam_out) != "." && !dir.exists(dirname(gam_out))) {
  dir.create(dirname(gam_out), recursive = TRUE, showWarnings = FALSE)
}
if (dir.exists(dirname(gam_out))) {
  write_csv(smooth_data_combined, gam_out)
}
write_csv(smooth_data_combined, "gam_plot_data_orig.csv")
write_csv(smooth_data_combined, "bko_gam_smooth_data.csv")
write_csv(raw_data_combined, "bko_gam_raw_data.csv")

# Print message
cat("Data extraction complete.\n")
