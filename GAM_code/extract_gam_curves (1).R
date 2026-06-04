library(tidyverse)
library(mgcv)
library(readr)

if (!exists("res_o") || !exists("res_r")) {
  fit_scripts <- c("BAM_agree_vs_disagree copy.R", "BAM_agree_vs_disagree.R")
  fit_script <- fit_scripts[file.exists(fit_scripts)][1]

  if (is.na(fit_script)) {
    stop(
      paste0(
        "Could not find a fitting script. Expected one of: ",
        paste(fit_scripts, collapse = ", ")
      )
    )
  }

  cat("Sourcing", fit_script, "to create res_o/res_r ...\n")
  source(fit_script)
}

models <- tribble(
  ~short,    ~surp_col,      ~pretty,
  "gpt2",    "gpt2new",      "GPT-2",
  "gptneo",  "gptneonew",    "GPT-Neo",
  "gptneox", "gptneoxnew",   "GPT-NeoX",
  "gptj",    "gptjnew",      "GPT-J",
  "gpt2xl",  "gpt2xlnew",    "GPT-2XL",
  "olmo",    "olmonew",      "OLMO-2",
  "llama2",  "llama2new",    "LLaMA-2"
)

# ------------------------------------------------------------------ #
# Extract fitted GAM smooth values using plot.gam() internals.
# ------------------------------------------------------------------ #
extract_smooth <- function(fit) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit(
    {
      grDevices::dev.off()
      unlink(pdf_path)
    },
    add = TRUE
  )
  plot_data <- plot(fit, select = 1, seWithMean = TRUE, n = 200)

  # plot_data[[1]] is the first smooth term (the surprisal smooth)
  # $x   = 200 predictor values spanning the observed data range
  # $fit = partial effect (mean-centred), identical to what is plotted
  # $se  = SE at each x, varies with data density
  pd <- plot_data[[1]]

  data.frame(
    x        = pd$x,
    fit      = as.numeric(pd$fit),
    se       = as.numeric(pd$se),
    se_upper = as.numeric(pd$fit) + as.numeric(pd$se),
    se_lower = as.numeric(pd$fit) - as.numeric(pd$se)
  )
}

# ------------------------------------------------------------------ #
# Extract all models for one dataset and save to CSV
# ------------------------------------------------------------------ #
extract_dataset <- function(fits_agree, fits_disagree, label) {
  all_rows <- list()

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]

    for (condition in c("agree", "disagree")) {
      fit <- if (condition == "agree") fits_agree[[m$short]] else fits_disagree[[m$short]]

      if (is.null(fit)) {
        cat("Skipping", m$pretty, condition, "— not fitted\n")
        next
      }

      cat("Extracting:", label, m$pretty, condition, "\n")

      smooth_df           <- extract_smooth(fit)
      smooth_df$model     <- m$pretty
      smooth_df$condition <- condition
      smooth_df$dataset   <- label
      smooth_df$surp_col  <- m$surp_col

      all_rows[[length(all_rows) + 1]] <- smooth_df
    }
  }

  out      <- bind_rows(all_rows)
  out_file <- paste0(label, "_gam_curves.csv")
  write_csv(out, out_file)
  cat("Saved", nrow(out), "rows to", out_file, "\n")
  invisible(out)
}

# ------------------------------------------------------------------ #
# Run for both datasets
# Requires res_o and res_r in environment from BAM_agree_vs_disagree.R
# ------------------------------------------------------------------ #
curves_o <- extract_dataset(res_o$fits_agree, res_o$fits_disagree, "bko21")
curves_r <- extract_dataset(res_r$fits_agree, res_r$fits_disagree, "bkr21")

cat("\nDone. Output files:\n")
cat("  bko21_gam_curves.csv\n")
cat("  bkr21_gam_curves.csv\n")
cat("\nColumns: x, fit, se, se_upper, se_lower, model, condition, dataset, surp_col\n")
