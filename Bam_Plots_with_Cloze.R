library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("~/Desktop")

df_items <- read_csv("filepath.csv")

preprocess_data <- function(df) {
  df <- df %>%
    mutate(
      SUB = as.factor(SUB),
      ITEM = as.factor(ITEM_x),
      SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed),
    ) %>%
    drop_na(SUM_3RT_trimmed, clozeprob_x, SUB, ITEM_x)  # Remove NA rows
  
  return(df)
}

df_items <- preprocess_data(df_items)

# Using BAM for faster computation with 're' for random effects

m_clozeprob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(clozeprob_x, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_clozesurp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(cloze, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2prob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2newprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_items)

m_gpt2surp <- bam(SUM_3RT_trimmed ~ 1 +
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
