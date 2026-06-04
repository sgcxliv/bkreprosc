library(tidyverse)
library(mgcv)
library(readr)

work_dir <- getOption("row_dissent_work_dir", default = NULL)
if (is.null(work_dir)) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1 && nzchar(args[1])) {
    work_dir <- args[1]
  }
}

if (!is.null(work_dir)) {
  work_dir <- normalizePath(work_dir, winslash = "/", mustWork = FALSE)
  if (!dir.exists(work_dir)) {
    stop("Working directory does not exist: ", work_dir)
  }
  setwd(work_dir)
}

# ------------------------------------------------------------------ #
# 1. Load files
#
#    cloze_comparison.csv  — from direct_unclean.py
#                            has: ITEM, condition, position,
#                            surprisal cols, agree_/disagree_ flags
#                            NO reading time data
#
#    bko21_spr.csv         — SPR reading time data, dataset O
#    bkr21_spr.csv         — SPR reading time data, dataset R
#
# ------------------------------------------------------------------ #

flags <- read_csv("cloze_comparison.csv", show_col_types = FALSE) %>%
  mutate(ITEM      = as.factor(ITEM),
         condition = as.character(condition),
         position  = as.numeric(position))

spr_o <- read_csv("bko21_spr.csv", show_col_types = FALSE,
                  col_types = cols(.default = col_guess(),
                                   word01 = col_character(), word02 = col_character(),
                                   word03 = col_character(), word04 = col_character(),
                                   word05 = col_character(), word06 = col_character(),
                                   word07 = col_character(), word08 = col_character(),
                                   word09 = col_character(), word10 = col_character(),
                                   word11 = col_character(), word12 = col_character(),
                                   word13 = col_character(), word14 = col_character(),
                                   word15 = col_character(), word16 = col_character()))

spr_r <- read_csv("bkr21_spr.csv", show_col_types = FALSE,
                  col_types = cols(.default = col_guess(),
                                   word01 = col_character(), word02 = col_character(),
                                   word03 = col_character(), word04 = col_character(),
                                   word05 = col_character(), word06 = col_character(),
                                   word07 = col_character(), word08 = col_character(),
                                   word09 = col_character(), word10 = col_character(),
                                   word11 = col_character(), word12 = col_character(),
                                   word13 = col_character(), word14 = col_character(),
                                   word15 = col_character(), word16 = col_character()))

# ------------------------------------------------------------------ #
# 2. Join reading times onto flags
#
#    Join keys: whichever of ITEM / condition / position exist in both.
#    The SPR files have RT + SUB; the flags file has surprisal + agree/disagree.
#    After joining, drop rows missing RT, SUB, or clozeprob. 
# ------------------------------------------------------------------ #

join_spr <- function(spr_df, flags_df, label) {
  keys <- intersect(c("ITEM", "condition", "position"), 
                    intersect(names(spr_df), names(flags_df)))
  cat("\nJoining", label, "on keys:", paste(keys, collapse = ", "), "\n")

  spr_df <- spr_df %>%
    mutate(ITEM      = as.factor(ITEM),
           condition = if ("condition" %in% names(.)) as.character(condition) else condition,
           position  = if ("position"  %in% names(.)) as.numeric(position)   else position,
           SUB       = as.factor(SUB),
           SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed))

  joined <- spr_df %>%
    left_join(flags_df, by = keys) %>%
    drop_na(SUM_3RT_trimmed, SUB, ITEM, clozeprob)

  cat("Rows after join + drop_na:", nrow(joined), "\n")
  joined
}

df_o <- join_spr(spr_o, flags, "bko21")
df_r <- join_spr(spr_r, flags, "bkr21")

# ------------------------------------------------------------------ #
# 3. Model definitions
# ------------------------------------------------------------------ #

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

make_formula <- function(surp_col) {
  as.formula(paste0(
    "SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs') +",
    " s(SUB, bs='re') + s(ITEM, bs='re')"
  ))
}

# ------------------------------------------------------------------ #
# 4. Fit GAMs + plot — runs identically for each dataset
# ------------------------------------------------------------------ #

run_dataset <- function(df, label) {
  cat("\n\n==============================\n")
  cat("Dataset:", label, "\n")
  cat("==============================\n")

  fits_agree    <- vector("list", nrow(models)); names(fits_agree)    <- models$short
  fits_disagree <- vector("list", nrow(models)); names(fits_disagree) <- models$short

  for (i in seq_len(nrow(models))) {
    m            <- models[i, ]
    agree_col    <- paste0("agree_",    m$short)
    disagree_col <- paste0("disagree_", m$short)

    if (!agree_col %in% names(df)) {
      cat("Skipping", m$pretty, "— flag column not found:", agree_col, "\n")
      next
    }
    if (!m$surp_col %in% names(df)) {
      cat("Skipping", m$pretty, "— surprisal column not found:", m$surp_col, "\n")
      next
    }

    df_agree <- df %>%
      filter(.data[[agree_col]] == TRUE) %>%
      drop_na(SUM_3RT_trimmed, SUB, ITEM, all_of(m$surp_col))

    df_disagree <- df %>%
      filter(.data[[disagree_col]] == TRUE) %>%
      drop_na(SUM_3RT_trimmed, SUB, ITEM, all_of(m$surp_col))

    cat(sprintf("%s — agree: %d rows | disagree: %d rows\n",
                m$pretty, nrow(df_agree), nrow(df_disagree)))

    fits_agree[[m$short]]    <- gam(make_formula(m$surp_col), data = df_agree)
    fits_disagree[[m$short]] <- gam(make_formula(m$surp_col), data = df_disagree)
  }

  # ---- Plots ----
  pdf(paste0(label, "_gam_agree_vs_disagree_surprisal.pdf"), width = 6, height = 10)

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]

    if (is.null(fits_agree[[m$short]]) || is.null(fits_disagree[[m$short]])) next

    n_agree    <- sum(df[[paste0("agree_",    m$short)]], na.rm = TRUE)
    n_disagree <- sum(df[[paste0("disagree_", m$short)]], na.rm = TRUE)

    par(mfrow = c(2, 1), mar = c(4.5, 4.5, 3.5, 1.5))

    # Top — agree
    plot(fits_agree[[m$short]], select = 1,
         xlab  = paste0(m$pretty, " Surprisal"),
         ylab  = "RT (partial effect)",
         main  = paste0(m$pretty, " — AGREE\n",
                        "n = ", n_agree, " rows"),
         shade = TRUE, shade.col = "lightblue",
         seWithMean = TRUE)
    abline(h = 0, lty = 2, col = "grey60")

    # Bottom — disagree
    plot(fits_disagree[[m$short]], select = 1,
         xlab  = paste0(m$pretty, " Surprisal"),
         ylab  = "RT (partial effect)",
         main  = paste0(m$pretty, " — DISAGREE\n",
                        "n = ", n_disagree, " rows"),
         shade = TRUE, shade.col = "mistyrose",
         seWithMean = TRUE)
    abline(h = 0, lty = 2, col = "grey60")
  }

  dev.off()
  cat("Plots saved to", paste0(label, "_gam_agree_vs_disagree_surprisal.pdf"), "\n")

  # ---- Summaries ----
  sink(paste0(label, "_gam_summaries_agree_vs_disagree.txt"))

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]
    cat("\n", strrep("=", 60), "\n", sep = "")
    cat(m$pretty, "\n")
    cat(strrep("=", 60), "\n", sep = "")

    if (!is.null(fits_agree[[m$short]])) {
      cat("\n--- AGREE ---\n")
      print(summary(fits_agree[[m$short]]))
    }
    if (!is.null(fits_disagree[[m$short]])) {
      cat("\n--- DISAGREE ---\n")
      print(summary(fits_disagree[[m$short]]))
    }
  }

  sink()
  cat("Summaries saved to", paste0(label, "_gam_summaries_agree_vs_disagree.txt"), "\n")

  invisible(list(fits_agree = fits_agree, fits_disagree = fits_disagree))
}

# ------------------------------------------------------------------ #
# 5. Run both datasets
# ------------------------------------------------------------------ #

res_o <- run_dataset(df_o, "bko21")
res_r <- run_dataset(df_r, "bkr21")

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
