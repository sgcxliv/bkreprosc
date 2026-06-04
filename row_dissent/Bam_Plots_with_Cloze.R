library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("~/Desktop/time")

df_items <- read_csv("bk21_spr.csv")

# Per-model agree/disagree masks produced by `direct_unclean.py`
agree_mask_src <- read_csv("clean_cloze_data.csv")
disagree_mask_src <- read_csv("clean_cloze_data_mismatches_only.csv")

make_model_masks <- function(mask_df, prefix) {
  keys <- intersect(c("ITEM", "condition", "position"), names(mask_df))

  mask_df <- mask_df %>%
    mutate(
      ITEM = as.factor(ITEM),
      condition = as.character(condition),
      position = as.numeric(position)
    )

  # These are the per-model predictors used in this script.
  # We create boolean "keep" flags based on whether the predictor is non-NA in the mask CSV.
  out <- mask_df %>%
    select(all_of(keys),
           any_of(c(
             "gpt2newprob", "gpt2new",
             "gpt2xlnewprob", "gpt2xlnew",
             "gptjnewprob", "gptjnew",
             "gptneonewprob", "gptneonew",
             "gptneoxnewprob", "gptneoxnew",
             "olmonewprob", "olmonew",
             "llama2newprob", "llama2new"
           ))) %>%
    distinct()

  if ("gpt2newprob" %in% names(out)) out[[paste0(prefix, "gpt2")]] <- !is.na(out$gpt2newprob)
  if ("gpt2xlnewprob" %in% names(out)) out[[paste0(prefix, "gpt2xl")]] <- !is.na(out$gpt2xlnewprob)
  if ("gptjnewprob" %in% names(out)) out[[paste0(prefix, "gptj")]] <- !is.na(out$gptjnewprob)
  if ("gptneonewprob" %in% names(out)) out[[paste0(prefix, "gptneo")]] <- !is.na(out$gptneonewprob)
  if ("gptneoxnewprob" %in% names(out)) out[[paste0(prefix, "gptneox")]] <- !is.na(out$gptneoxnewprob)
  if ("olmonewprob" %in% names(out)) out[[paste0(prefix, "olmo")]] <- !is.na(out$olmonewprob)
  if ("llama2newprob" %in% names(out)) out[[paste0(prefix, "llama2")]] <- !is.na(out$llama2newprob)

  # Keep only keys + newly created mask columns
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

df_items <- preprocess_data(df_items)

# Join masks onto reading-time data (so we can filter agree/disagree per model)
join_keys <- intersect(c("ITEM", "condition", "position"), names(df_items))
if (length(join_keys) == 0) {
  stop("Could not find any join keys among ITEM/condition/position in bk21_spr.csv")
}

df_items <- df_items %>%
  mutate(
    condition = if ("condition" %in% names(df_items)) as.character(condition) else condition,
    position = if ("position" %in% names(df_items)) as.numeric(position) else position
  ) %>%
  left_join(agree_masks, by = join_keys) %>%
  left_join(disagree_masks, by = join_keys)

# Using BAM for faster computation with 're' for random effects

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
                    s(gpt2newprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2surp <- gam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2new, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptneoprob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gptneonewprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneosurp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gptneonew, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneoxprob <- bam(SUM_3RT_trimmed ~ 1 +
                       s(gptneoxnewprob, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_gptneoxsurp <- bam(SUM_3RT_trimmed ~ 1 +
                       s(gptneoxnew, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_olmoprob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(olmonewprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_olmosurp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(olmonew, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_llama2prob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(llama2newprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_llama2surp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(llama2new, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptjprob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gptjnewprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptjsurp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gptjnew, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2xlprob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xlnewprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gpt2xlsurp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xlnew, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

# ---- Subsetting helpers for agree/disagree per model ----
subset_for_model <- function(df, model, which = c("agree", "disagree")) {
  which <- match.arg(which)
  mask_col <- paste0(which, "_", model)
  if (!mask_col %in% names(df)) {
    stop(paste0("Missing mask column: ", mask_col, ". Check clean_cloze_data*.csv and join keys."))
  }
  df %>% filter(.data[[mask_col]] %in% TRUE)
}

# Example: build agree/disagree datasets for GPT-2 (use these as `data=` in your GAMs/BAMs)
df_gpt2_agree <- subset_for_model(df_items, "gpt2", which = "agree")
df_gpt2_disagree <- subset_for_model(df_items, "gpt2", which = "disagree")

# You can replicate per model:
# df_gptneox_agree <- subset_for_model(df_items, "gptneox", "agree")
# df_gptneox_disagree <- subset_for_model(df_items, "gptneox", "disagree")

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
