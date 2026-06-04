library(tidyverse)
library(mgcv)
library(readr)
library(parallel)
setwd("~/Desktop")

df_items <- read_csv("items_with_surprisal.csv")

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

# Print info about the dataset bc im TWEAKING
cat("Total rows in dataset:", nrow(df_items), "\n")

# FILTER TO ONLY ASSESS DATA POINTS WITH ACTUAL vALUES (THE MISTMATCHED ONES)
# OR ELSE YOU ARE SO COOKED WHY DO THEY LOOK WEIRD HELPPPP

# GPT-2 models (NON NA VALUES ONLY!!!) 
df_gpt2 <- df_items %>% filter(!is.na(gpt2newprob))
cat("Rows with GPT-2 values:", nrow(df_gpt2), "\n")

m_gpt2prob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2newprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_gpt2)

m_gpt2surp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gpt2new, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_gpt2)

# GPT-Neo models
df_gptneo <- df_items %>% filter(!is.na(gptneonewprob))
cat("Rows with GPT-Neo values:", nrow(df_gptneo), "\n")

m_gptneoprob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gptneonewprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_gptneo)

m_gptneosurp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gptneonew, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_gptneo)

# GPT-NeoX models
df_gptneox <- df_items %>% filter(!is.na(gptneoxnewprob))
cat("Rows with GPT-NeoX values:", nrow(df_gptneox), "\n")

m_gptneoxprob <- bam(SUM_3RT_trimmed ~ 1 +
                       s(gptneoxnewprob, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_gptneox)

m_gptneoxsurp <- bam(SUM_3RT_trimmed ~ 1 +
                       s(gptneoxnew, bs="cs") +
                       s(SUB, bs="re") +
                       s(ITEM, bs="re"),
                     data=df_gptneox)

# OLMO models
df_olmo <- df_items %>% filter(!is.na(olmonewprob))
cat("Rows with OLMO values:", nrow(df_olmo), "\n")

m_olmoprob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(olmonewprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_olmo)

m_olmosurp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(olmonew, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_olmo)

# LLaMA-2 models
df_llama2 <- df_items %>% filter(!is.na(llama2newprob))
cat("Rows with LLaMA-2 values:", nrow(df_llama2), "\n")

m_llama2prob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(llama2newprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_llama2)

m_llama2surp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(llama2new, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_llama2)

# GPT-J models
df_gptj <- df_items %>% filter(!is.na(gptjnewprob))
cat("Rows with GPT-J values:", nrow(df_gptj), "\n")

m_gptjprob <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gptjnewprob, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_gptj)

m_gptjsurp <- bam(SUM_3RT_trimmed ~ 1 +
                    s(gptjnew, bs="cs") +
                    s(SUB, bs="re") +
                    s(ITEM, bs="re"),
                  data=df_gptj)

# GPT-2XL models
df_gpt2xl <- df_items %>% filter(!is.na(gpt2xlnewprob))
cat("Rows with GPT-2XL values:", nrow(df_gpt2xl), "\n")

m_gpt2xlprob <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xlnewprob, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_gpt2xl)

m_gpt2xlsurp <- bam(SUM_3RT_trimmed ~ 1 +
                      s(gpt2xlnew, bs="cs") +
                      s(SUB, bs="re") +
                      s(ITEM, bs="re"),
                    data=df_gpt2xl)

# Create a PDF with all the plots
pdf("model_plots_mismatched_only.pdf", width=12, height=8)

# GPT-2 plots
par(mfrow=c(1,2))
plot(m_gpt2prob, select=1, xlab='GPT-2 Probability', ylab='RT', main="GPT-2 Probability (mismatched only)")
plot(m_gpt2surp, select=1, xlab='GPT-2 Surprisal', ylab='RT', main="GPT-2 Surprisal (mismatched only)")

# GPT-Neo plots
par(mfrow=c(1,2))
plot(m_gptneoprob, select=1, xlab='GPT-Neo Probability', ylab='RT', main="GPT-Neo Probability (mismatched only)")
plot(m_gptneosurp, select=1, xlab='GPT-Neo Surprisal', ylab='RT', main="GPT-Neo Surprisal (mismatched only)")

# GPT-NeoX plots
par(mfrow=c(1,2))
plot(m_gptneoxprob, select=1, xlab='GPT-NeoX Probability', ylab='RT', main="GPT-NeoX Probability (mismatched only)")
plot(m_gptneoxsurp, select=1, xlab='GPT-NeoX Surprisal', ylab='RT', main="GPT-NeoX Surprisal (mismatched only)")

# GPT-J plots
par(mfrow=c(1,2))
plot(m_gptjprob, select=1, xlab='GPT-J Probability', ylab='RT', main="GPT-J Probability (mismatched only)")
plot(m_gptjsurp, select=1, xlab='GPT-J Surprisal', ylab='RT', main="GPT-J Surprisal (mismatched only)")

# GPT-2XL plots
par(mfrow=c(1,2))
plot(m_gpt2xlprob, select=1, xlab='GPT-2XL Probability', ylab='RT', main="GPT-2XL Probability (mismatched only)")
plot(m_gpt2xlsurp, select=1, xlab='GPT-2XL Surprisal', ylab='RT', main="GPT-2XL Surprisal (mismatched only)")

# OLMO plots
par(mfrow=c(1,2))
plot(m_olmoprob, select=1, xlab='OLMO-2 Probability', ylab='RT', main="OLMO-2 Probability (mismatched only)")
plot(m_olmosurp, select=1, xlab='OLMO-2 Surprisal', ylab='RT', main="OLMO-2 Surprisal (mismatched only)")

# LLaMA-2 plots
par(mfrow=c(1,2))
plot(m_llama2prob, select=1, xlab='LLaMA-2 Probability', ylab='RT', main="LLaMA-2 Probability (mismatched only)")
plot(m_llama2surp, select=1, xlab='LLaMA-2 Surprisal', ylab='RT', main="LLaMA-2 Surprisal (mismatched only)")

dev.off()

# Save summary statistics for all models to a text file
sink("model_summaries_mismatched_only.txt")

cat("\nGPT-2 MODELS (Mismatched Only)\n")
cat("===============================\n")
print(summary(m_gpt2prob))
print(summary(m_gpt2surp))

cat("\nGPT-NEO MODELS (Mismatched Only)\n")
cat("=================================\n")
print(summary(m_gptneoprob))
print(summary(m_gptneosurp))

cat("\nGPT-NEOX MODELS (Mismatched Only)\n")
cat("==================================\n")
print(summary(m_gptneoxprob))
print(summary(m_gptneoxsurp))

cat("\nGPT-J MODELS (Mismatched Only)\n")
cat("===============================\n")
print(summary(m_gptjprob))
print(summary(m_gptjsurp))

cat("\nGPT-2XL MODELS (Mismatched Only)\n")
cat("=================================\n")
print(summary(m_gpt2xlprob))
print(summary(m_gpt2xlsurp))

cat("\nOLMO MODELS (Mismatched Only)\n")
cat("=============================\n")
print(summary(m_olmoprob))
print(summary(m_olmosurp))

cat("\nLLAMA-2 MODELS (Mismatched Only)\n")
cat("================================\n")
print(summary(m_llama2prob))
print(summary(m_llama2surp))

sink()

# Also save individual plots to separate files
# GPT-2
pdf("gpt2_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_gpt2prob, select=1, xlab='GPT-2 Probability', ylab='RT', main="GPT-2 Probability (mismatched only)")
plot(m_gpt2surp, select=1, xlab='GPT-2 Surprisal', ylab='RT', main="GPT-2 Surprisal (mismatched only)")
dev.off()

# GPT-Neo
pdf("gptneo_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_gptneoprob, select=1, xlab='GPT-Neo Probability', ylab='RT', main="GPT-Neo Probability (mismatched only)")
plot(m_gptneosurp, select=1, xlab='GPT-Neo Surprisal', ylab='RT', main="GPT-Neo Surprisal (mismatched only)")
dev.off()

# GPT-NeoX
pdf("gptneox_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_gptneoxprob, select=1, xlab='GPT-NeoX Probability', ylab='RT', main="GPT-NeoX Probability (mismatched only)")
plot(m_gptneoxsurp, select=1, xlab='GPT-NeoX Surprisal', ylab='RT', main="GPT-NeoX Surprisal (mismatched only)")
dev.off()

# GPT-J
pdf("gptj_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_gptjprob, select=1, xlab='GPT-J Probability', ylab='RT', main="GPT-J Probability (mismatched only)")
plot(m_gptjsurp, select=1, xlab='GPT-J Surprisal', ylab='RT', main="GPT-J Surprisal (mismatched only)")
dev.off()

# GPT-2XL
pdf("gpt2xl_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_gpt2xlprob, select=1, xlab='GPT-2XL Probability', ylab='RT', main="GPT-2XL Probability (mismatched only)")
plot(m_gpt2xlsurp, select=1, xlab='GPT-2XL Surprisal', ylab='RT', main="GPT-2XL Surprisal (mismatched only)")
dev.off()

# OLMO
pdf("olmo_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_olmoprob, select=1, xlab='OLMO-2 Probability', ylab='RT', main="OLMO-2 Probability (mismatched only)")
plot(m_olmosurp, select=1, xlab='OLMO-2 Surprisal', ylab='RT', main="OLMO-2 Surprisal (mismatched only)")
dev.off()

# LLaMA-2
pdf("llama2_mismatched.pdf", width=10, height=5)
par(mfrow=c(1,2))
plot(m_llama2prob, select=1, xlab='LLaMA-2 Probability', ylab='RT', main="LLaMA-2 Probability (mismatched only)")
plot(m_llama2surp, select=1, xlab='LLaMA-2 Surprisal', ylab='RT', main="LLaMA-2 Surprisal (mismatched only)")
dev.off()

#sanitychecklawl
df_matched <- df_items %>% filter(!is.na(gpt2newprob))
df_mismatched <- df_items %>% filter(is.na(gpt2newprob))

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
