library(lme4)
library(tidyverse)
library(readr)

setwd("~/Desktop")

df_items <- read_csv("new_items_with_surprisal.csv")

df_items <- df_items %>%
  mutate(
    SUB = as.factor(SUB),
    ITEM = as.factor(ITEM),
    SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed)
  ) %>%
  drop_na(SUM_3RT_trimmed, SUB, ITEM)

# Linear Mixed Effects Models including both probability and surprisal
m_cloze <- lmer(SUM_3RT_trimmed ~ clozeprob + cloze + (1 | SUB) + (1 | ITEM), data=df_items)
m_gpt2 <- lmer(SUM_3RT_trimmed ~ gpt2newprob + gpt2new + (1 | SUB) + (1 | ITEM), data=df_items)
m_gptneo <- lmer(SUM_3RT_trimmed ~ gptneonewprob + gptneonew + (1 | SUB) + (1 | ITEM), data=df_items)
m_gptneox <- lmer(SUM_3RT_trimmed ~ gptneoxnewprob + gptneoxnew + (1 | SUB) + (1 | ITEM), data=df_items)
m_olmo <- lmer(SUM_3RT_trimmed ~ olmonewprob + olmonew + (1 | SUB) + (1 | ITEM), data=df_items)
m_llama2 <- lmer(SUM_3RT_trimmed ~ llama2newprob + llama2new + (1 | SUB) + (1 | ITEM), data=df_items)
m_gptj <- lmer(SUM_3RT_trimmed ~ gptjnewprob + gptjnew + (1 | SUB) + (1 | ITEM), data=df_items)
m_gpt2xl <- lmer(SUM_3RT_trimmed ~ gpt2xlnewprob + gpt2xlnew + (1 | SUB) + (1 | ITEM), data=df_items)

# Summarize models
summary(m_cloze)
summary(m_gpt2)
summary(m_gptneo)
summary(m_gptneox)
summary(m_olmo)
summary(m_llama2)
summary(m_gptj)
summary(m_gpt2xl)
