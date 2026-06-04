#!/usr/bin/env Rscript
# ============================================================================
# SIB_combine_and_plot.R
#
# Merges lme_results/lme_coefs_*.csv, runs Panels A/B correlations,
# and produces the final SI_B_figure.pdf / .png
#
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

INPUT_CSV <- "/nlp/scr/$USER/SIB/tritems_with_surprisal.csv"
LME_DIR <- "/nlp/scr/$USER/SIB/lme_results"
OUT_DIR <- "/nlp/scr/$USER/SIB/SI_B_outputs"
dir.create(OUT_DIR, showWarnings = FALSE)

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
EST_LEVELS <- ESTIMATORS$name

# ---- Check all LME result files are present ---------------------------------
lme_files <- list.files(LME_DIR, pattern = "lme_coefs_.*\\.csv$", full.names = TRUE)
cat(sprintf("Found %d LME result files in %s/\n", length(lme_files), LME_DIR))
if (length(lme_files) < nrow(ESTIMATORS)) {
  warning(sprintf(
    "Only %d of %d expected files found. Missing jobs:\n  %s",
    length(lme_files), nrow(ESTIMATORS),
    paste(setdiff(
      sprintf("lme_coefs_%02d_", seq_len(nrow(ESTIMATORS))),
      substr(basename(lme_files), 1, 13)
    ), collapse = ", ")
  ))
}

lme_df <- map_dfr(lme_files, read_csv, show_col_types = FALSE) %>%
  mutate(estimator = factor(estimator, levels = EST_LEVELS))
write_csv(lme_df, file.path(OUT_DIR, "panels_CD_lme_coefs.csv"))
cat(sprintf("Combined LME results: %d rows\n", nrow(lme_df)))

# ---- Panels A/B: item-level correlations ------------------------------------
cat("\nComputing item-level correlation matrices...\n")
df <- read_csv(INPUT_CSV, show_col_types = FALSE,
               guess_max = 50000)
cat(sprintf("  %d rows x %d cols\n", nrow(df), ncol(df)))

cor_rows <- list()
for (i in seq_len(nrow(ESTIMATORS))) {
  est <- ESTIMATORS$name[i]
  for (vtype in c("prob", "surp")) {
    cols <- if (vtype == "prob") {
      c(ESTIMATORS$prob_w1[i], ESTIMATORS$prob_w2[i], ESTIMATORS$prob_w3[i])
    } else {
      c(ESTIMATORS$surp_w1[i], ESTIMATORS$surp_w2[i], ESTIMATORS$surp_w3[i])
    }
    have  <- !is.na(cols) & (cols %in% names(df))
    valid <- cols[have]
    w_idx <- which(have)
    if (sum(have) < 2) {
      if (sum(have) == 1) {
        cor_rows[[length(cor_rows)+1]] <- tibble(
          estimator=est, var=vtype,
          word_row=paste0("W",w_idx), word_col=paste0("W",w_idx), cor=1)
      }
      next
    }
    item_df <- df %>%
      group_by(ITEM, condition) %>%
      summarise(across(all_of(valid), \(x) mean(x, na.rm=TRUE)), .groups="drop")
    M <- suppressWarnings(cor(item_df[,valid,drop=FALSE],
                              use="pairwise.complete.obs"))
    for (r in seq_along(w_idx)) for (cc in seq_along(w_idx)) {
      cor_rows[[length(cor_rows)+1]] <- tibble(
        estimator=est, var=vtype,
        word_row=paste0("W",w_idx[r]), word_col=paste0("W",w_idx[cc]),
        cor=M[r,cc])
    }
  }
}
cor_df <- bind_rows(cor_rows) %>%
  mutate(estimator = factor(estimator, levels = EST_LEVELS))
write_csv(cor_df, file.path(OUT_DIR, "panels_AB_correlations.csv"))

# ---- Plot -------------------------------------------------------------------
ord <- c("W1","W2","W3")
theme_si <- theme_minimal(base_size=9) +
  theme(panel.grid=element_blank(), strip.text=element_text(face="bold",size=8),
        strip.background=element_blank(), axis.title=element_text(size=8),
        axis.text=element_text(size=7), legend.position="right",
        legend.key.width=unit(0.4,"cm"), legend.key.height=unit(0.8,"cm"),
        plot.title=element_text(face="bold",size=11),
        plot.subtitle=element_text(size=8,colour="grey40"))
theme_set(theme_si)

cor_df_p <- cor_df %>%
  mutate(word_row=factor(word_row,levels=rev(ord)),
         word_col=factor(word_col,levels=ord))
lme_df_p <- lme_df %>%
  mutate(pred_word=factor(pred_word,levels=rev(ord)),
         response_word=factor(response_word,levels=ord))

mk_cor_panel <- function(d, title, subtitle=NULL) {
  ggplot(d, aes(word_col, word_row, fill=cor)) +
    geom_tile(color="white", linewidth=0.4) +
    geom_text(aes(label=sprintf("%.2f",cor)), size=2.2) +
    facet_wrap(~estimator, nrow=2) +
    scale_fill_gradient2(low="#2c7bb6", mid="white", high="#d7191c",
                         midpoint=0, limits=c(-1,1), name="r", oob=squish) +
    coord_equal() +
    labs(title=title, subtitle=subtitle, x=NULL, y=NULL)
}

mk_lme_panel <- function(d, title, subtitle=NULL) {
  if (nrow(d)==0) {
    return(ggplot() +
             annotate("text",x=0.5,y=0.5,label="No LME data") +
             labs(title=title) + theme_void())
  }
  lim <- max(abs(d$coef), na.rm=TRUE)
  if (!is.finite(lim) || lim==0) lim <- 1

  all_cells <- expand_grid(
    estimator     = factor(EST_LEVELS, levels=EST_LEVELS),
    pred_word     = factor(ord, levels=rev(ord)),
    response_word = factor(ord, levels=ord)
  ) %>% mutate(
    pred_num     = as.integer(sub("W","",as.character(pred_word))),
    response_num = as.integer(sub("W","",as.character(response_word))),
    in_triangle  = pred_num <= response_num
  )
  d_full <- all_cells %>%
    left_join(d, by=c("estimator","pred_word","response_word"))

  ggplot(d_full, aes(response_word, pred_word, fill=coef)) +
    geom_tile(color="white", linewidth=0.4) +
    geom_text(data=filter(d_full, in_triangle & !is.na(coef)),
              aes(label=sprintf("%.3f",coef)), size=2.2) +
    facet_wrap(~estimator, nrow=2) +
    scale_fill_gradient2(low="#2c7bb6", mid="white", high="#d7191c",
                         midpoint=0, limits=c(-lim,lim),
                         name=expression(beta), oob=squish, na.value="grey92") +
    coord_equal() +
    labs(title=title, subtitle=subtitle,
         x="Response position (RT)", y="Predictor position")
}

pA <- mk_cor_panel(cor_df_p %>% filter(var=="prob"),
                   "A. Probability correlations across W1\u2013W3",
                   "Item-level Pearson r (216 item\u00D7condition cells)")
pB <- mk_cor_panel(cor_df_p %>% filter(var=="surp"),
                   "B. Surprisal correlations across W1\u2013W3",
                   "Item-level Pearson r (216 item\u00D7condition cells)")
pC <- mk_lme_panel(lme_df_p %>% filter(var=="prob"),
                   "C. LME \u03B2: probability \u2192 RT  (lower triangle)",
                   "Fit jointly with surprisal; log RT; z-scored preds")
pD <- mk_lme_panel(lme_df_p %>% filter(var=="surp"),
                   "D. LME \u03B2: surprisal \u2192 RT  (lower triangle)",
                   "Fit jointly with probability; log RT; z-scored preds")

fig <- (pA | pB) / (pC | pD) +
  plot_layout(heights=c(1,1)) +
  plot_annotation(
    title="SI-B: Lagged predictability \u2192 reading-time analysis (BKR)",
    theme=theme(plot.title=element_text(face="bold",size=13))
  )

ggsave(file.path(OUT_DIR,"SI_B_figure.pdf"), fig, width=16, height=11)
ggsave(file.path(OUT_DIR,"SI_B_figure.png"), fig, width=16, height=11, dpi=200)
cat(sprintf("\nDone. Figure written to %s/\n", OUT_DIR))
