# Cross-validated Linear Mixed Effects Analysis
library(tidyverse)
library(lme4)
setwd("~/Desktop")

# Load data
df_items <- read_csv("new_items_with_surprisal.csv")

# Check column names to ensure they match what we're looking for
cat("Column names in dataset:\n")
print(colnames(df_items))

# Check for key columns
required_columns <- c("SUB", "ITEM", "SUM_3RT_trimmed", "cloze", "clozeprob",
                      "trigram", "gpt2new", "gpt2newprob")
missing_columns <- required_columns[!required_columns %in% colnames(df_items)]
if(length(missing_columns) > 0) {
  warning("Missing required columns: ", paste(missing_columns, collapse=", "))
}

# Examine the data
cat("Data structure:\n")
str(df_items)
cat("\nFirst few rows:\n")
print(head(df_items))
cat("\nSummary of key variables:\n")
print(summary(df_items$SUM_3RT_trimmed))
print(summary(df_items$clozeprob))
print(summary(df_items$cloze))

# Preprocess data
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

# Create 5 folds
create_folds <- function(data) {
  # list to store fold indices
  folds <- list()
  
  if (!is.numeric(data$ITEM)) {
    item_ids <- as.numeric(as.character(data$ITEM))
  } else {
    item_ids <- data$ITEM
  }
  
  # Assign items to folds
  for (i in 0:4) {
    fold_items <- unique(item_ids[item_ids %% 5 == i])
    folds[[i+1]] <- which(item_ids %in% fold_items)
  }
  
  return(folds)
}

# Run Cross Validation
run_cv <- function(data, formula, folds) {
  n_folds <- length(folds)
  n_obs <- nrow(data)
  
  # Initialize vector for log likelihoods
  log_likelihoods <- numeric(n_obs)
  log_likelihoods[] <- NA  # Set all to NA initially
  
  # Run cross-validation
  for (i in 1:n_folds) {
    cat("Processing fold", i, "of", n_folds, "\n")
    
    # Get test indices for this fold
    test_indices <- folds[[i]]
    
    # Get train indices (all except test)
    train_indices <- setdiff(1:n_obs, test_indices)
    
    # Split data
    train_data <- data[train_indices, ]
    test_data <- data[test_indices, ]
    
    tryCatch({
      # Fit model on training data
      model <- lmer(formula, data = train_data)
      
      # Check for convergence issues
      conv_warning <- NULL
      conv_warning <- lme4::isSingular(model)
      if(conv_warning) {
        warning("Model singular fit in fold ", i, ": ", formula)
      }
      
      test_data$predicted <- predict(model, newdata = test_data, re.form = NA)
      
      # Get residual standard deviation
      sigma <- sigma(model)
      
      # Calculate log likelihood
      observed <- test_data$SUM_3RT_trimmed
      ll <- dnorm(observed, mean = test_data$predicted, sd = sigma, log = TRUE)
      
      # Add debugging info
      cat("Test data fold size:", nrow(test_data), "\n")
      cat("Mean predicted value:", mean(test_data$predicted, na.rm=TRUE), "\n")
      cat("Mean observed value:", mean(observed, na.rm=TRUE), "\n")
      cat("Sigma:", sigma, "\n")
      cat("Mean log-likelihood:", mean(ll, na.rm=TRUE), "\n")
      
      # Store results
      log_likelihoods[test_indices] <- ll
      
    }, error = function(e) {
      cat("Error in fold", i, ":", conditionMessage(e), "\n")
      # Leave NAs for this fold
    })
  }
  
  # Return only non-NA values
  return(log_likelihoods[!is.na(log_likelihoods)])
}

# Permutation Test to Compare
permutation_test <- function(ll1, ll2, n_permutations = 10000, directional = TRUE) {
  # Calculate observed difference in mean log likelihood
  observed_diff <- mean(ll1) - mean(ll2)
  
  # Initialize vector for permuted differences
  perm_diffs <- numeric(n_permutations)
  
  # Run permutation test
  for (p in 1:n_permutations) {
    if (p %% 1000 == 0) {
      cat("Permutation", p, "of", n_permutations, "\n")
    }
    
    # For each observation, randomly decide whether to swap log likelihoods
    swap <- rbinom(length(ll1), 1, 0.5) == 1
    
    # Create permuted log likelihood vectors
    perm_ll1 <- ll1
    perm_ll2 <- ll2
    
    # Swap values where indicated
    perm_ll1[swap] <- ll2[swap]
    perm_ll2[swap] <- ll1[swap]
    
    # Calculate difference in means for this permutation
    perm_diffs[p] <- mean(perm_ll1) - mean(perm_ll2)
  }
  
  # Calculate p-value
  if (directional) {
    if (observed_diff > 0) {
      p_value <- mean(perm_diffs >= observed_diff)
    } else {
      p_value <- mean(perm_diffs <= observed_diff)
    }
  } else {
    p_value <- mean(abs(perm_diffs) >= abs(observed_diff))
  }
  
  return(list(
    delta_ll = observed_diff,
    permutation_diffs = perm_diffs,
    p_value = p_value
  ))
}

# Try a single model comparison for debugging
debug_single_comparison <- function(data, folds) {
  cat("\n--- Running Single Comparison for Debugging ---\n")
  
  # Model formulas - using original random effects structure
  null_formula <- "SUM_3RT_trimmed ~ 1 + (1|SUB) + (1|ITEM)"
  prob_formula <- "SUM_3RT_trimmed ~ clozeprob + (1|SUB) + (1|ITEM)"
  surp_formula <- "SUM_3RT_trimmed ~ cloze + (1|SUB) + (1|ITEM)"
  
  # Run cross-validation with verbose output
  cat("Running null model\n")
  null_ll <- run_cv(data, null_formula, folds)
  cat("Mean log-likelihood for null:", mean(null_ll), "\n")
  
  cat("Running Cloze PROB model\n")
  prob_ll <- run_cv(data, prob_formula, folds)
  cat("Mean log-likelihood for PROB:", mean(prob_ll), "\n")
  
  cat("Running Cloze SURP model\n")
  surp_ll <- run_cv(data, surp_formula, folds)
  cat("Mean log-likelihood for SURP:", mean(surp_ll), "\n")
  
  # Run permutation test with fewer permutations for debugging
  cat("Testing PROB vs. null\n")
  prob_null_result <- permutation_test(prob_ll, null_ll, n_permutations=100)
  print(prob_null_result)
  
  cat("Testing SURP vs. null\n")
  surp_null_result <- permutation_test(surp_ll, null_ll, n_permutations=100)
  print(surp_null_result)
  
  cat("Testing SURP vs. PROB\n")
  surp_prob_result <- permutation_test(surp_ll, prob_ll, n_permutations=100)
  print(surp_prob_result)
  
  return(list(
    null_ll = null_ll,
    prob_ll = prob_ll,
    surp_ll = surp_ll,
    prob_null_result = prob_null_result,
    surp_null_result = surp_null_result,
    surp_prob_result = surp_prob_result
  ))
}

# Run Comparison
run_table_s1_comparisons <- function(data, n_permutations = 10000) {
  # Create folds for cross-validation
  folds <- create_folds(data)
  
  # Define variable mappings for predictors - FIXED COLUMN NAMES
  predictors <- list(
    "Cloze" = list(
        "PROB" = "clozeprob",
        "SURP1" = "cloze"
    ),
    "GPT-2" = list(
        "PROB" = "gpt2newprob",
        "SURP1" = "gpt2new"
    ),
    "GPT-2-XL" = list(
        "PROB" = "gpt2xlnewprob",
        "SURP1" = "gpt2xlnew"
    ),
    "GPT-Neo" = list(
        "PROB" = "gptneonewprob",
        "SURP1" = "gptneonew"
    ),
    "GPT-NeoX" = list(
        "PROB" = "gptneoxnewprob",
        "SURP1" = "gptneoxnew"
    ),
    "GPT-J" = list(
        "PROB" = "gptjnewprob",
        "SURP1" = "gptjnew"
    ),
    "OLMO-2" = list(
        "PROB" = "olmonewprob",
        "SURP1" = "olmonew"
    ),
    "LLaMA-2" = list(
        "PROB" = "llama2newprob",
        "SURP1" = "llama2new"
    ),
    "GPT-2-Region" = list(
        "PROB" = "gpt2regionnewprob",
        "SURP1" = "gpt2regionnew"
    ),
    "Trigram" = list(
        "PROB" = "trigramprob",
        "SURP1" = "trigram"
  )
)
  
  # Initialize results list
  all_results <- list()
  
  # 1. Overall comparisons (vs. null model)
  cat("\n--- Running Overall Comparisons (vs. null model) ---\n")
  
  # Null model log likelihoods (only random effects)
  # Using original model structure
  null_formula <- "SUM_3RT_trimmed ~ 1 + (1|SUB) + (1|ITEM)"
  cat("Running null model\n")
  null_ll <- run_cv(data, null_formula, folds)
  
  overall_results <- data.frame(
    comparison = character(),
    delta_ll = numeric(),
    p_value = numeric(),
    better_model = character(),
    stringsAsFactors = FALSE
  )
  
  # Compare each predictor against null
  for (pred_name in names(predictors)) {
    for (type in c("PROB", "SURP1")) {
      pred_col <- predictors[[pred_name]][[type]]
      
      # Skip if column doesn't exist
      if (!(pred_col %in% colnames(data))) {
        cat("Skipping", pred_name, type, "as column", pred_col, "not found\n")
        next
      }
      
      # Model formula - using original random effects
      pred_formula <- paste0("SUM_3RT_trimmed ~ ", pred_col, " + (1|SUB) + (1|ITEM)")
      
      # Run cross-validation
      cat("Running", pred_name, type, "vs. null\n")
      pred_ll <- run_cv(data, pred_formula, folds)
      
      # Run permutation test
      test_result <- permutation_test(pred_ll, null_ll, n_permutations)
      
      # Add to results
      overall_results <- rbind(
        overall_results,
        data.frame(
          comparison = paste0(pred_name, "_", type, " vs. ∅"),
          delta_ll = test_result$delta_ll,
          p_value = test_result$p_value,
          better_model = ifelse(test_result$delta_ll > 0, pred_name, "null"),
          stringsAsFactors = FALSE
        )
      )
    }
  }
  
  all_results$overall <- overall_results
  
  # 2. Probability vs. Surprisal comparisons
  cat("\n--- Running Probability vs. Surprisal Comparisons ---\n")
  
  prob_vs_surp_results <- data.frame(
    comparison = character(),
    delta_ll = numeric(),
    p_value = numeric(),
    better_model = character(),
    stringsAsFactors = FALSE
  )
  
  # Direct PROB vs. SURP1 comparisons
  for (pred_name in names(predictors)) {
    prob_col <- predictors[[pred_name]][["PROB"]]
    surp_col <- predictors[[pred_name]][["SURP1"]]
    
    # Skip if either column doesn't exist
    if (!(prob_col %in% colnames(data)) || !(surp_col %in% colnames(data))) {
      cat("Skipping", pred_name, "PROB vs. SURP1 as columns not found\n")
      next
    }
    
    # Model formulas - original random effects
    prob_formula <- paste0("SUM_3RT_trimmed ~ ", prob_col, " + (1|SUB) + (1|ITEM)")
    surp_formula <- paste0("SUM_3RT_trimmed ~ ", surp_col, " + (1|SUB) + (1|ITEM)")
    
    # Run cross-validation
    cat("Running", pred_name, "SURP1 vs. PROB\n")
    prob_ll <- run_cv(data, prob_formula, folds)
    surp_ll <- run_cv(data, surp_formula, folds)
    
    # Run permutation test
    test_result <- permutation_test(surp_ll, prob_ll, n_permutations)
    
    # Add to results
    prob_vs_surp_results <- rbind(
      prob_vs_surp_results,
      data.frame(
        comparison = paste0(pred_name, "_SURP1 vs. ", pred_name, "_PROB"),
        delta_ll = test_result$delta_ll,
        p_value = test_result$p_value,
        better_model = ifelse(test_result$delta_ll > 0, "SURP1", "PROB"),
        stringsAsFactors = FALSE
      )
    )
  }
  
  # Nested model comparisons: PROB+SURP1 vs. SURP1
  for (pred_name in names(predictors)) {
    prob_col <- predictors[[pred_name]][["PROB"]]
    surp_col <- predictors[[pred_name]][["SURP1"]]
    
    # Skip if either column doesn't exist
    if (!(prob_col %in% colnames(data)) || !(surp_col %in% colnames(data))) {
      cat("Skipping", pred_name, "PROB+SURP1 vs. SURP1 as columns not found\n")
      next
    }
    
    # Model formulas - original random effects
    combined_formula <- paste0("SUM_3RT_trimmed ~ ", prob_col, " + ", surp_col, " + (1|SUB) + (1|ITEM)")
    surp_formula <- paste0("SUM_3RT_trimmed ~ ", surp_col, " + (1|SUB) + (1|ITEM)")
    
    # Run cross-validation
    cat("Running", pred_name, "PROB+SURP1 vs. SURP1\n")
    combined_ll <- run_cv(data, combined_formula, folds)
    surp_ll <- run_cv(data, surp_formula, folds)
    
    # Run permutation test
    test_result <- permutation_test(combined_ll, surp_ll, n_permutations)
    
    # Add to results
    prob_vs_surp_results <- rbind(
      prob_vs_surp_results,
      data.frame(
        comparison = paste0(pred_name, "_PROB+", pred_name, "_SURP1 vs. ", pred_name, "_SURP1"),
        delta_ll = test_result$delta_ll,
        p_value = test_result$p_value,
        better_model = ifelse(test_result$delta_ll > 0, "PROB+SURP1", "SURP1"),
        stringsAsFactors = FALSE
      )
    )
  }
  
  # Nested model comparisons: PROB+SURP1 vs. PROB
  for (pred_name in names(predictors)) {
    prob_col <- predictors[[pred_name]][["PROB"]]
    surp_col <- predictors[[pred_name]][["SURP1"]]
    
    # Skip if either column doesn't exist
    if (!(prob_col %in% colnames(data)) || !(surp_col %in% colnames(data))) {
      cat("Skipping", pred_name, "PROB+SURP1 vs. PROB as columns not found\n")
      next
    }
    
    # Model formulas - original random effects
    combined_formula <- paste0("SUM_3RT_trimmed ~ ", prob_col, " + ", surp_col, " + (1|SUB) + (1|ITEM)")
    prob_formula <- paste0("SUM_3RT_trimmed ~ ", prob_col, " + (1|SUB) + (1|ITEM)")
    
    # Run cross-validation
    cat("Running", pred_name, "PROB+SURP1 vs. PROB\n")
    combined_ll <- run_cv(data, combined_formula, folds)
    prob_ll <- run_cv(data, prob_formula, folds)
    
    # Run permutation test
    test_result <- permutation_test(combined_ll, prob_ll, n_permutations)
    
    # Add to results
    prob_vs_surp_results <- rbind(
      prob_vs_surp_results,
      data.frame(
        comparison = paste0(pred_name, "_PROB+", pred_name, "_SURP1 vs. ", pred_name, "_PROB"),
        delta_ll = test_result$delta_ll,
        p_value = test_result$p_value,
        better_model = ifelse(test_result$delta_ll > 0, "PROB+SURP1", "PROB"),
        stringsAsFactors = FALSE
      )
    )
  }
  
  all_results$prob_vs_surp <- prob_vs_surp_results
  
  # 3. Cloze vs. Other comparisons
  cat("\n--- Running Cloze vs. Other Comparisons ---\n")
  
  cloze_vs_other_results <- data.frame(
    comparison = character(),
    delta_ll = numeric(),
    p_value = numeric(),
    better_model = character(),
    stringsAsFactors = FALSE
  )
  
  # Direct comparisons between different predictors vs. Cloze
  cloze_prob_col <- predictors[["Cloze"]][["PROB"]]
  
  for (pred_name in setdiff(names(predictors), "Cloze")) {
    surp_col <- predictors[[pred_name]][["SURP1"]]
    
    # Skip if column doesn't exist
    if (!(surp_col %in% colnames(data))) {
      cat("Skipping", pred_name, "SURP1 vs. Cloze_PROB as column not found\n")
      next
    }
    
    # Model formulas - original random effects
    cloze_formula <- paste0("SUM_3RT_trimmed ~ ", cloze_prob_col, " + (1|SUB) + (1|ITEM)")
    pred_formula <- paste0("SUM_3RT_trimmed ~ ", surp_col, " + (1|SUB) + (1|ITEM)")
    
    # Run cross-validation
    cat("Running", pred_name, "SURP1 vs. Cloze_PROB\n")
    cloze_ll <- run_cv(data, cloze_formula, folds)
    pred_ll <- run_cv(data, pred_formula, folds)
    
    # Run permutation test
    test_result <- permutation_test(pred_ll, cloze_ll, n_permutations)
    
    # Add to results
    cloze_vs_other_results <- rbind(
      cloze_vs_other_results,
      data.frame(
        comparison = paste0(pred_name, "_SURP1 vs. Cloze_PROB"),
        delta_ll = test_result$delta_ll,
        p_value = test_result$p_value,
        better_model = ifelse(test_result$delta_ll > 0, pred_name, "Cloze"),
        stringsAsFactors = FALSE
      )
    )
  }
  
  # Nested model comparisons: Cloze_PROB + X_SURP1 vs. Cloze_PROB
  for (pred_name in setdiff(names(predictors), "Cloze")) {
    surp_col <- predictors[[pred_name]][["SURP1"]]
    
    # Skip if column doesn't exist
    if (!(surp_col %in% colnames(data))) {
      cat("Skipping Cloze_PROB +", pred_name, "SURP1 vs. Cloze_PROB as column not found\n")
      next
    }
    
    # Model formulas - original random effects
    combined_formula <- paste0("SUM_3RT_trimmed ~ ", cloze_prob_col, " + ", surp_col, " + (1|SUB) + (1|ITEM)")
    cloze_formula <- paste0("SUM_3RT_trimmed ~ ", cloze_prob_col, " + (1|SUB) + (1|ITEM)")
    
    # Run cross-validation
    cat("Running Cloze_PROB +", pred_name, "SURP1 vs. Cloze_PROB\n")
    combined_ll <- run_cv(data, combined_formula, folds)
    cloze_ll <- run_cv(data, cloze_formula, folds)
    
    # Run permutation test
    test_result <- permutation_test(combined_ll, cloze_ll, n_permutations)
    
    # Add to results
    cloze_vs_other_results <- rbind(
      cloze_vs_other_results,
      data.frame(
        comparison = paste0("Cloze_PROB+", pred_name, "_SURP1 vs. Cloze_PROB"),
        delta_ll = test_result$delta_ll,
        p_value = test_result$p_value,
        better_model = ifelse(test_result$delta_ll > 0, "Combined", "Cloze"),
        stringsAsFactors = FALSE
      )
    )
  }
  
  all_results$cloze_vs_other <- cloze_vs_other_results
  
  # Apply FDR correction within each comparison family
  for (family in names(all_results)) {
    all_results[[family]]$p_adjusted <- p.adjust(all_results[[family]]$p_value, method = "fdr")
  }
  
  return(all_results)
}

# Create formatted table similar to Table S1
create_table_s1 <- function(results) {
  # Format each results section
  format_section <- function(section, title) {
    # Make sure the column names are the same for later rbind
    section$category <- title
    section$significance <- ""
    section$significance[section$p_adjusted < 0.05] <- "*"
    section$significance[section$p_adjusted < 0.01] <- "**"
    section$significance[section$p_adjusted < 0.001] <- "***"
    
    # Format values
    section$delta_ll <- round(section$delta_ll, 0)
    section$p_value <- round(section$p_value, 4)
    section$p_adjusted <- round(section$p_adjusted, 4)
    
    # For directional tests, add dash where alternative doesn't improve over null
    section$p_value_display <- section$p_value
    section$p_value_display[section$delta_ll <= 0] <- "—"
    
    return(section)
  }
  
  # Format each section
  overall <- format_section(results$overall, "Overall")
  prob_vs_surp <- format_section(results$prob_vs_surp, "Probability vs. Surprisal")
  cloze_vs_other <- format_section(results$cloze_vs_other, "Cloze vs. Other")
  
  # Combine all sections
  all_formatted <- rbind(overall, prob_vs_surp, cloze_vs_other)
  
  # Reorder columns to match Table S1
  table_s1 <- all_formatted %>%
    select(category, comparison, delta_ll, p_value_display, significance)
  
  return(table_s1)
}

# Create diagnostic plots
make_diagnostic_plots <- function(debug_results) {
  par(mfrow=c(2,2))
  
  # Plot histogram of log-likelihoods
  hist(debug_results$null_ll, main="Null Model Log-Likelihoods", xlab="Log-Likelihood")
  hist(debug_results$prob_ll, main="Prob Model Log-Likelihoods", xlab="Log-Likelihood")
  hist(debug_results$surp_ll, main="Surp Model Log-Likelihoods", xlab="Log-Likelihood")
  
  # Plot permutation distribution for SURP vs PROB
  hist(debug_results$surp_prob_result$permutation_diffs,
       main="Permutation Distribution (SURP vs PROB)",
       xlab="Difference in Mean Log-Likelihood")
  abline(v=debug_results$surp_prob_result$delta_ll, col="red", lwd=2)
  
  # Reset plot layout
  par(mfrow=c(1,1))
}

# First run debugging on a single comparison to check functionality
folds <- create_folds(df_items)
debug_results <- debug_single_comparison(df_items, folds)

# Create diagnostic plots
make_diagnostic_plots(debug_results)

# Run the analysis with full number of permutations
n_permutations <- 10000  # Full test with 10,000 permutations

# Run all comparisons
results <- run_table_s1_comparisons(df_items, n_permutations = n_permutations)

# Create table similar to Table S1
table_s1 <- create_table_s1(results)

# Print and save results
print(table_s1)
write.csv(table_s1, "table_s1_replication.csv", row.names = FALSE)

# Save detailed results
save(results, file = "lme_cv_permutation_results.RData")

# Save table as PNG using ggplot2
library(ggplot2)
library(gridExtra)

# Convert table to ggplot
make_table_plot <- function(table_data) {
  # Create a ggplot-friendly table
  plot_table <- tableGrob(table_data, rows = NULL, theme = ttheme_minimal())
  
  # Create the plot
  p <- ggplot() +
    annotation_custom(plot_table) +
    theme_void() +
    labs(title = "Replication of Table S1: Testing Results on Cross-validated LME Models") +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))
  
  return(p)
}

# Create and save the table as PNG
table_plot <- make_table_plot(table_s1)
ggsave("table_s1_replication.png", table_plot, width = 10, height = 8, dpi = 300)
