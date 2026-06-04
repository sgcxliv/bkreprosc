# nolint start
library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("~/Desktop/time")

path_o <- "bko21_spr.csv"
path_r <- "bkr21_spr.csv"

# Per-model agree/disagree masks produced by `direct_unclean.py`
agree_mask_src <- read_csv("clean_cloze_data.csv")
disagree_mask_src <- read_csv("clean_cloze_data_mismatches_only.csv")

make_model_masks <- function(mask_df, prefix) {
  keys <- intersect(c("ITEM", "condition", "position"), names(mask_df))

  mask_df <- mask_df %>%
    mutate(
      ITEM = as.factor(ITEM),
      condition = if ("condition" %in% names(mask_df)) as.character(condition) else condition,
      position = if ("position" %in% names(mask_df)) as.numeric(position) else position
    )

  out <- mask_df %>%
    select(all_of(keys),
           any_of(c(
             "gpt2newprob",
             "gpt2xlnewprob",
             "gptjnewprob",
             "gptneonewprob",
             "gptneoxnewprob",
             "olmonewprob",
             "llama2newprob"
           ))) %>%
    distinct()

  if ("gpt2newprob" %in% names(out)) out[[paste0(prefix, "gpt2")]] <- !is.na(out$gpt2newprob)
  if ("gpt2xlnewprob" %in% names(out)) out[[paste0(prefix, "gpt2xl")]] <- !is.na(out$gpt2xlnewprob)
  if ("gptjnewprob" %in% names(out)) out[[paste0(prefix, "gptj")]] <- !is.na(out$gptjnewprob)
  if ("gptneonewprob" %in% names(out)) out[[paste0(prefix, "gptneo")]] <- !is.na(out$gptneonewprob)
  if ("gptneoxnewprob" %in% names(out)) out[[paste0(prefix, "gptneox")]] <- !is.na(out$gptneoxnewprob)
  if ("olmonewprob" %in% names(out)) out[[paste0(prefix, "olmo")]] <- !is.na(out$olmonewprob)
  if ("llama2newprob" %in% names(out)) out[[paste0(prefix, "llama2")]] <- !is.na(out$llama2newprob)

  keep_cols <- c(keys, grep(paste0("^", prefix), names(out), value = TRUE))
  out %>% select(all_of(keep_cols))
}

agree_masks <- make_model_masks(agree_mask_src, prefix = "agree_")
disagree_masks <- make_model_masks(disagree_mask_src, prefix = "disagree_")

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

run_dataset <- function(path, label) {
  cat("\n==============================\n")
  cat("Dataset:", label, "\n")
  cat("File:", path, "\n")
  cat("==============================\n")

  # Force word columns to character to avoid parsing warnings.
  df_items <- read_csv(
    path,
    show_col_types = FALSE,
    col_types = cols(
      word01 = col_character(),
      word02 = col_character(),
      word03 = col_character(),
      word04 = col_character(),
      word05 = col_character(),
      word06 = col_character(),
      word07 = col_character(),
      word08 = col_character(),
      word09 = col_character(),
      word10 = col_character(),
      word11 = col_character(),
      word12 = col_character(),
      word13 = col_character(),
      word14 = col_character(),
      word15 = col_character(),
      word16 = col_character()
    )
  )
  df_items <- preprocess_data(df_items)

  # Join masks onto reading-time data 
  join_keys <- intersect(c("ITEM", "condition", "position"), names(df_items))
  if (length(join_keys) == 0) {
    stop(paste0("Could not find any join keys among ITEM/condition/position in ", path))
  }

  df_items <- df_items %>%
    mutate(
      condition = if ("condition" %in% names(df_items)) as.character(condition) else condition,
      position = if ("position" %in% names(df_items)) as.numeric(position) else position
    ) %>%
    left_join(agree_masks, by = join_keys) %>%
    left_join(disagree_masks, by = join_keys)

  # Print info about the dataset bc im TWEAKING
  cat("Total rows in dataset:", nrow(df_items), "\n")

# Faster + safer: keep your existing surprisal columns, and use mask flags to subset.
subset_for_model <- function(df, model, which = c("agree", "disagree")) {
  which <- match.arg(which)
  mask_col <- paste0(which, "_", model)
  if (!mask_col %in% names(df)) {
    stop(paste0("Missing mask column: ", mask_col, ". Check clean_cloze_data*.csv and join keys."))
  }
  df %>% filter(.data[[mask_col]] %in% TRUE)
}

fit_pair <- function(df, prob_col, surp_col) {
  f_prob <- as.formula(paste0(
    "SUM_3RT_trimmed ~ 1 + s(", prob_col, ", bs = 'cs') + s(SUB, bs = 're') + s(ITEM, bs = 're')"
  ))
  f_surp <- as.formula(paste0(
    "SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs = 'cs') + s(SUB, bs = 're') + s(ITEM, bs = 're')"
  ))

  list(
    prob = bam(f_prob, data = df),
    surp = bam(f_surp, data = df)
  )
}

pick_col <- function(df, candidates) {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

models <- tribble(
  ~model,    ~prob_candidates,                 ~surp_candidates,              ~pretty,
  "gpt2",    c("gpt2newprob", "gpt2prob"),     c("gpt2new", "gpt2"),          "GPT-2",
  "gptneo",  c("gptneonewprob", "gptneoprob"), c("gptneonew", "gptneo"),      "GPT-Neo",
  "gptneox", c("gptneoxnewprob", "gptneoxprob"), c("gptneoxnew", "gptneox"),  "GPT-NeoX",
  "gptj",    c("gptjnewprob", "gptjprob"),     c("gptjnew", "gptj"),          "GPT-J",
  "gpt2xl",  c("gpt2xlnewprob", "gpt2xlprob"), c("gpt2xlnew", "gpt2xl"),      "GPT-2XL",
  "olmo",    c("olmonewprob", "olmoprob"),     c("olmonew", "olmo"),          "OLMO-2",
  "llama2",  c("llama2newprob", "llama2prob"), c("llama2new", "llama2"),      "LLaMA-2"
)

fit_all <- function(which = c("agree", "disagree")) {
  which <- match.arg(which)
  results <- vector("list", nrow(models))
  names(results) <- models$model

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]
    prob_col <- pick_col(df_items, m$prob_candidates[[1]])
    surp_col <- pick_col(df_items, m$surp_candidates[[1]])

    if (is.na(prob_col) || is.na(surp_col)) {
      cat("Skipping", m$pretty, "(", which, "): missing columns in this dataset\n")
      results[[m$model]] <- NULL
      next
    }

    df_sub <- subset_for_model(df_items, m$model, which = which) %>%
      drop_na(SUM_3RT_trimmed, SUB, ITEM) %>%
      drop_na(any_of(c(prob_col, surp_col)))

    cat("Rows for", m$pretty, "(", which, "):", nrow(df_sub), "\n")
    results[[m$model]] <- fit_pair(df_sub, prob_col, surp_col)
  }
  results
}

  fits_agree <- fit_all("agree")
  fits_disagree <- fit_all("disagree")

plot_all <- function(fits, which_label) {
  pdf(paste0("model_plots_", which_label, "_only.pdf"), width = 12, height = 8)

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]
    fs <- fits[[m$model]]

    par(mfrow = c(1, 2))
    plot(fs$prob, select = 1,
         xlab = paste0(m$pretty, " Probability"),
         ylab = "RT",
         main = paste0(m$pretty, " Probability (", which_label, " only)"))
    plot(fs$surp, select = 1,
         xlab = paste0(m$pretty, " Surprisal"),
         ylab = "RT",
         main = paste0(m$pretty, " Surprisal (", which_label, " only)"))
  }

  dev.off()
}

  plot_all <- function(fits, which_label) {
    pdf(paste0(label, "_model_plots_", which_label, "_only.pdf"), width = 12, height = 8)

    for (i in seq_len(nrow(models))) {
      m <- models[i, ]
      fs <- fits[[m$model]]

      par(mfrow = c(1, 2))
      plot(fs$prob, select = 1,
           xlab = paste0(m$pretty, " Probability"),
           ylab = "RT",
           main = paste0(m$pretty, " Probability (", which_label, " only)"))
      plot(fs$surp, select = 1,
           xlab = paste0(m$pretty, " Surprisal"),
           ylab = "RT",
           main = paste0(m$pretty, " Surprisal (", which_label, " only)"))
    }

    dev.off()
  }

  plot_all(fits_agree, "agree")
  plot_all(fits_disagree, "disagree")

write_summaries <- function(fits, which_label) {
  sink(paste0("model_summaries_", which_label, "_only.txt"))

  for (i in seq_len(nrow(models))) {
    m <- models[i, ]
    fs <- fits[[m$model]]

    cat("\n", m$pretty, " (", which_label, " only)\n", sep = "")
    cat(strrep("=", nchar(m$pretty) + nchar(which_label) + 10), "\n", sep = "")
    cat("\nPROB MODEL\n")
    print(summary(fs$prob))
    cat("\nSURP MODEL\n")
    print(summary(fs$surp))
  }

  sink()
}

  write_summaries <- function(fits, which_label) {
    sink(paste0(label, "_model_summaries_", which_label, "_only.txt"))

    for (i in seq_len(nrow(models))) {
      m <- models[i, ]
      fs <- fits[[m$model]]

      cat("\n", m$pretty, " (", which_label, " only)\n", sep = "")
      cat(strrep("=", nchar(m$pretty) + nchar(which_label) + 10), "\n", sep = "")
      cat("\nPROB MODEL\n")
      print(summary(fs$prob))
      cat("\nSURP MODEL\n")
      print(summary(fs$surp))
    }

    sink()
  }

  write_summaries(fits_agree, "agree")
  write_summaries(fits_disagree, "disagree")

# cloze
m_clozeprob <- bam(SUM_3RT_trimmed ~ 1 +
                            s(clozeprob, bs="cs") +
                            s(SUB, bs="re") +
                            s(ITEM, bs="re"),
                          data=df_items)
m_clozesurp <- bam(SUM_3RT_trimmed ~ 1 +
                     s(cloze, bs="cs") +
                     s(SUB, bs="re") +
                     s(ITEM, bs="re"),
                   data=df_items)
# plot
plot(m_clozeprob, select=1, xlab='Cloze Probability', ylab='RT', main="Cloze Probability")
plot(m_clozesurp, select=1, xlab='Cloze Probability', ylab='RT', main="Cloze Surprisal")

  invisible(list(
    data = df_items,
    fits_agree = fits_agree,
    fits_disagree = fits_disagree
  ))
}

# Run both iterations of the design
res_o <- run_dataset(path_o, "bko21")
res_r <- run_dataset(path_r, "bkr21")
# nolint end
