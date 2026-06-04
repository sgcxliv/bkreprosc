library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("~/Desktop")

df_items <- read_csv("bko21_spr.csv")

preprocess_data <- function(df) {
  df <- df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
      log_SUM_3RT = if_else(SUM_3RT_trimmed > 0, log(SUM_3RT_trimmed), NA_real_)
    ) %>%
    drop_na(log_SUM_3RT, clozeprob, SUB, ITEM)  # SI-A uses log RT
  
  return(df)
}

df_items <- preprocess_data(df_items)

# Using BAM for faster computation with 're' for random effects

m_clozeprob <- gam(log_SUM_3RT ~ 1 +
                    s(clozeprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_clozesurp <- gam(log_SUM_3RT ~ 1 +
                    s(cloze, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2prob <- gam(log_SUM_3RT ~ 1 +
                    s(gpt2prob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2surp <- gam(log_SUM_3RT ~ 1 +
                    s(gpt2, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptneoprob <- bam(log_SUM_3RT ~ 1 +
                      s(gptneoprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneosurp <- bam(log_SUM_3RT ~ 1 +
                      s(gptneo, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptneoxprob <- bam(log_SUM_3RT ~ 1 +
                       s(gptneoxprob, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_gptneoxsurp <- bam(log_SUM_3RT ~ 1 +
                       s(gptneox, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_items)

m_olmoprob <- bam(log_SUM_3RT ~ 1 +
                    s(olmoprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_olmosurp <- bam(log_SUM_3RT ~ 1 +
                    s(olmo, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_llama2prob <- bam(log_SUM_3RT ~ 1 +
                      s(llama2prob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_llama2surp <- bam(log_SUM_3RT ~ 1 +
                      s(llama2, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gptjprob <- bam(log_SUM_3RT ~ 1 +
                    s(gptjprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gptjsurp <- bam(log_SUM_3RT ~ 1 +
                    s(gptj, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2xlprob <- bam(log_SUM_3RT ~ 1 +
                      s(gpt2xlprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

m_gpt2xlsurp <- bam(log_SUM_3RT ~ 1 +
                      s(gpt2xl, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_items)

plot(m_clozeprob, select=1, xlab='Cloze Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_clozesurp, select=1, xlab='Cloze Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_gpt2prob, select=1, xlab='GPT-2 Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_gpt2surp, select=1, xlab='GPT-2 Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_gptneoprob, select=1, xlab='GPT-Neo Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_gptneosurp, select=1, xlab='GPT-Neo Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_gptneoxprob, select=1, xlab='GPT-NeoX Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_gptneoxsurp, select=1, xlab='GPT-NeoX Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_gptjprob, select=1, xlab='GPT-J Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_gptjsurp, select=1, xlab='GPT-J Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_gpt2xlprob, select=1, xlab='GPT-2XL Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_gpt2xlsurp, select=1, xlab='GPT-2XL Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_olmoprob, select=1, xlab='OLMO-2 Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_olmosurp, select=1, xlab='OLMO-2 Surprisal', ylab='log(SUM_3RT_trimmed)')

plot(m_llama2prob, select=1, xlab='LLaMA-2 Probability', ylab='log(SUM_3RT_trimmed)')
plot(m_llama2surp, select=1, xlab='LLaMA-2 Surprisal', ylab='log(SUM_3RT_trimmed)')

# ------------------------------------------------------------------ #
# Extract smooth curves exactly as plotted by plot.gam()
# ------------------------------------------------------------------ #
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

fit_registry <- tribble(
  ~fit_name,       ~model,     ~measure,
  "m_clozeprob",   "Cloze",    "Probability",
  "m_clozesurp",   "Cloze",    "Surprisal",
  "m_gpt2prob",    "GPT-2",    "Probability",
  "m_gpt2surp",    "GPT-2",    "Surprisal",
  "m_gptneoprob",  "GPT-Neo",  "Probability",
  "m_gptneosurp",  "GPT-Neo",  "Surprisal",
  "m_gptneoxprob", "GPT-NeoX", "Probability",
  "m_gptneoxsurp", "GPT-NeoX", "Surprisal",
  "m_gptjprob",    "GPT-J",    "Probability",
  "m_gptjsurp",    "GPT-J",    "Surprisal",
  "m_gpt2xlprob",  "GPT-2XL",  "Probability",
  "m_gpt2xlsurp",  "GPT-2XL",  "Surprisal",
  "m_olmoprob",    "OLMO-2",   "Probability",
  "m_olmosurp",    "OLMO-2",   "Surprisal",
  "m_llama2prob",  "LLaMA-2",  "Probability",
  "m_llama2surp",  "LLaMA-2",  "Surprisal"
)

all_curves <- lapply(seq_len(nrow(fit_registry)), function(i) {
  row <- fit_registry[i, ]
  fit_obj <- get(row$fit_name, envir = environment())
  smooth_df <- extract_smooth(fit_obj, select = 1, n = 200)
  smooth_df$model <- row$model
  smooth_df$measure <- row$measure
  smooth_df
})

gam_curves <- bind_rows(all_curves)
write_csv(gam_curves, "SIA_log_gam_curves.csv")
cat("Saved", nrow(gam_curves), "rows to SIA_log_gam_curves.csv\n")
