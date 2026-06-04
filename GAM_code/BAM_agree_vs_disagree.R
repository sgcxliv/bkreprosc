library(tidyverse)
library(mgcv)
library(readr)

setwd("~/Desktop/dissent")

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
#    cloze_mismatches.csv  — item-level mismatch % per model
# ------------------------------------------------------------------ #

flags <- read_csv("cloze_comparison.csv", show_col_types = FALSE) %>%
  mutate(ITEM      = as.factor(ITEM),
         condition = as.character(condition),
         position  = as.numeric(position))

mismatch_summary <- read_csv("cloze_mismatches.csv", show_col_types = FALSE)

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

get_mismatch_pct <- function(short) {
  row <- mismatch_summary %>% filter(LLM == paste0(short, "newprob"))
  if (nrow(row) == 0) return(NA_real_)
  round(row$Mismatch_Percentage, 1)
}

make_formula <- function(surp_col) {
  as.formula(paste0(
    "SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs') +",
    " s(SUB, bs='re') + s(ITEM, bs='re')"
  ))
}

# ------------------------------------------------------------------ #
# 4. Fit BAMs + plot — runs identically for each dataset
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

    fits_agree[[m$short]]    <- bam(make_formula(m$surp_col), data = df_agree)
    fits_disagree[[m$short]] <- bam(make_formula(m$surp_col), data = df_disagree)
  }

  # ---- Plots ----
  pdf(paste0(label, "_gam_agree_vs_disagree_surprisal.pdf"), width = 6, height = 10)

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]

    if (is.null(fits_agree[[m$short]]) || is.null(fits_disagree[[m$short]])) next

    mismatch_pct <- get_mismatch_pct(m$short)
    agree_pct    <- if (!is.na(mismatch_pct)) round(100 - mismatch_pct, 1) else "?"
    disagree_pct <- if (!is.na(mismatch_pct)) mismatch_pct else "?"

    n_agree    <- sum(df[[paste0("agree_",    m$short)]], na.rm = TRUE)
    n_disagree <- sum(df[[paste0("disagree_", m$short)]], na.rm = TRUE)

    par(mfrow = c(2, 1), mar = c(4.5, 4.5, 3.5, 1.5))

    # Top — agree
    plot(fits_agree[[m$short]], select = 1,
         xlab  = paste0(m$pretty, " Surprisal"),
         ylab  = "RT (partial effect)",
         main  = paste0(m$pretty, " — AGREE\n",
                        agree_pct, "% of items match cloze ranking  |  n = ", n_agree, " rows"),
         shade = TRUE, shade.col = "lightblue",
         seWithMean = TRUE)
    abline(h = 0, lty = 2, col = "grey60")

    # Bottom — disagree
    plot(fits_disagree[[m$short]], select = 1,
         xlab  = paste0(m$pretty, " Surprisal"),
         ylab  = "RT (partial effect)",
         main  = paste0(m$pretty, " — DISAGREE\n",
                        disagree_pct, "% of items mismatch cloze ranking  |  n = ", n_disagree, " rows"),
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
