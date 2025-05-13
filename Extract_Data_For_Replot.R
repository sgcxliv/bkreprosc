library(tidyverse)
library(mgcv)
library(readr)
library(parallel)

setwd("~/Desktop")

df_items <- read_csv("item2.csv")

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

# Function to extract smooth data from GAM model
extract_gam_data <- function(model, term_index = 1, n_points = 100) {
  formula_terms <- attr(terms(model$formula), "term.labels")
  smooth_terms <- formula_terms[grep("^s\\(", formula_terms)]
  
  if (length(smooth_terms) < term_index) {
    warning("Term index exceeds number of smooth terms")
    return(NULL)
  }
  
  term_string <- smooth_terms[term_index]
  pred_var <- gsub("s\\(([^,)]+).*", "\\1", term_string)
  
  if (!pred_var %in% names(model$model)) {
    warning(paste("Predictor variable", pred_var, "not found in model data"))
    return(NULL)
  }
  
  pred_range <- range(model$model[[pred_var]], na.rm = TRUE)
  
  if (!is.finite(pred_range[1]) || !is.finite(pred_range[2])) {
    warning(paste("Cannot determine valid range for", pred_var))
    return(NULL)
  }
  
  new_data <- data.frame(x = seq(from = pred_range[1], to = pred_range[2], length.out = n_points))
  names(new_data) <- pred_var
  
  for (re_var in c("SUB", "ITEM")) {
    if (re_var %in% names(model$model)) {
      new_data[[re_var]] = levels(model$model[[re_var]])[1]
    }
  }
  
  predictions <- try(predict(model, newdata = new_data, type = "terms", se.fit = TRUE), silent = TRUE)
  
  if (inherits(predictions, "try-error")) {
    warning("Error in prediction: ", as.character(predictions))
    return(NULL)
  }
  
  col_names <- colnames(predictions$fit)
  pred_col_index <- grep(paste0("^s\\(", pred_var), col_names)
  
  if (length(pred_col_index) == 0) {
    warning("Could not identify prediction column for ", pred_var)
    return(NULL)
  }
  
  measure_type <- if(grepl("prob", pred_var, ignore.case = TRUE)) {
    "Probability"
  } else if(grepl("surp|cloze$", pred_var, ignore.case = TRUE)) {
    "Surprisal"
  } else {
    "Surprisal"
  }
  
  # Create result df
  result <- data.frame(
    x = new_data[[pred_var]],
    y = predictions$fit[, pred_col_index] + coef(model)[1], # Add intercept
    se = predictions$se.fit[, pred_col_index],
    variable = pred_var,
    measure = measure_type,
    model = NA, 
    model_name = NA 
  )
  
  result$lower <- result$y - 1.96 * result$se
  result$upper <- result$y + 1.96 * result$se
  
  return(result)
}

# Function to extract raw data for a model
extract_raw_data <- function(model, term_index = 1) {
  formula_terms <- attr(terms(model$formula), "term.labels")
  smooth_terms <- formula_terms[grep("^s\\(", formula_terms)]
  
  if (length(smooth_terms) < term_index) {
    warning("Term index exceeds number of smooth terms")
    return(NULL)
  }
  
  term_string <- smooth_terms[term_index]
  pred_var <- gsub("s\\(([^,)]+).*", "\\1", term_string)
  
  if (!pred_var %in% names(model$model)) {
    warning(paste("Predictor variable", pred_var, "not found in model data"))
    return(NULL)
  }
  
  response_var <- names(model$model)[1]
  
  # Determine prob or surp
  measure_type <- if(grepl("prob", pred_var, ignore.case = TRUE)) {
    "Probability"
  } else if(grepl("surp|cloze$", pred_var, ignore.case = TRUE)) {
    "Surprisal"
  } else {
    "Other"
  }
  
  raw_data <- data.frame(
    x = model$model[[pred_var]],
    y = model$model[[response_var]],
    variable = pred_var,
    measure = measure_type,
    model = NA, 
    model_name = NA
  )
  
  return(raw_data)
}

# Extract data from all models
smooth_data_list <- list()
raw_data_list <- list()

model_list <- list(
  m_clozeprob = m_clozeprob,
  m_clozesurp = m_clozesurp,
  m_gpt2prob = m_gpt2prob,
  m_gpt2surp = m_gpt2surp,
  m_gptneoprob = m_gptneoprob,
  m_gptneosurp = m_gptneosurp,
  m_gptneoxprob = m_gptneoxprob,
  m_gptneoxsurp = m_gptneoxsurp,
  m_olmoprob = m_olmoprob,
  m_olmosurp = m_olmosurp,
  m_llama2prob = m_llama2prob,
  m_llama2surp = m_llama2surp,
  m_gptjprob = m_gptjprob,
  m_gptjsurp = m_gptjsurp,
  m_gpt2xlprob = m_gpt2xlprob,
  m_gpt2xlsurp = m_gpt2xlsurp
)

# Create a name map
model_display_names <- list(
  m_clozeprob = "Human Cloze Probability",
  m_clozesurp = "Human Cloze Surprisal",
  m_gpt2prob = "GPT-2 Probability",
  m_gpt2surp = "GPT-2 Surprisal",
  m_gptneoprob = "GPT-Neo Probability",
  m_gptneosurp = "GPT-Neo Surprisal",
  m_gptneoxprob = "GPT-NeoX Probability",
  m_gptneoxsurp = "GPT-NeoX Surprisal",
  m_olmoprob = "OLMO-2 Probability",
  m_olmosurp = "OLMO-2 Surprisal",
  m_llama2prob = "LLaMA-2 Probability",
  m_llama2surp = "LLaMA-2 Surprisal",
  m_gptjprob = "GPT-J Probability",
  m_gptjsurp = "GPT-J Surprisal",
  m_gpt2xlprob = "GPT-2XL Probability",
  m_gpt2xlsurp = "GPT-2XL Surprisal"
)

# get model data
for (model_name in names(model_list)) {
  cat("\nProcessing model:", model_name, "\n")
  
  current_model <- model_list[[model_name]]
  
  if (is.null(current_model)) {
    warning(paste("Model", model_name, "not found in model list"))
    next
  }
  
  model_type <- gsub("^m_([^_]+).*$", "\\1", model_name)
  
  tryCatch({
    smooth_data <- extract_gam_data(current_model)
    
    if (!is.null(smooth_data)) {
      smooth_data$model <- model_type
      smooth_data$model_name <- model_display_names[[model_name]]
      smooth_data_list[[model_name]] <- smooth_data
      cat("Successfully extracted smooth data for", model_name, "\n")
    }
  }, error = function(e) {
    warning(paste("Error extracting smooth data for", model_name, ":", conditionMessage(e)))
  })
  
  tryCatch({
    raw_data <- extract_raw_data(current_model)
    
    if (!is.null(raw_data)) {
      raw_data$model <- model_type
      raw_data$model_name <- model_display_names[[model_name]]
      raw_data_list[[model_name]] <- raw_data
      cat("Successfully extracted raw data for", model_name, "\n")
    }
  }, error = function(e) {
    warning(paste("Error extracting raw data for", model_name, ":", conditionMessage(e)))
  })
}

# Combine all dfs in list
smooth_data_combined <- bind_rows(smooth_data_list)
raw_data_combined <- bind_rows(raw_data_list)

# Write to CSV
write_csv(smooth_data_combined, "bkgam_smooth_data.csv")
write_csv(raw_data_combined, "bkgam_raw_data.csv")

cat("\nData extraction complete. Data saved to 'bkgam_smooth_data.csv' and 'bkgam_raw_data.csv'.\n")
