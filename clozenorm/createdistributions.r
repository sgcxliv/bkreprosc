library(dplyr)
library(tidyr)
library(stringr)
library(readr)

# Set the working directory
setwd("~/Desktop")

# Read the cloze data from CSV and preprocess
cloze_data <- read_csv("cleaned_with_expected.csv") %>%
  filter(item_type == "experiment") %>%
  mutate(
    Code = as.character(Code),
    RemovalType = as.character(RemovalType),
    participant_word = str_trim(str_to_lower(participant_word)),
    expected_word = str_trim(str_to_lower(expected_word))
  )

# Extract distinct contexts from the data
contexts <- cloze_data %>%
  distinct(Code, RemovalType, sentence_context)

num_contexts <- nrow(contexts)
if (num_contexts != 1944) {
  warning(paste("Expected 1944 contexts, found", num_contexts))
}

# Compute word counts
word_counts <- cloze_data %>%
  group_by(Code, RemovalType, participant_word) %>%
  summarise(count = n(), .groups = "drop")

# Extract distinct expected words
expected_words <- cloze_data %>%
  distinct(Code, RemovalType, expected_word)

# Check if expected word has been seen
expected_seen <- cloze_data %>%
  group_by(Code, RemovalType, expected_word) %>%
  summarise(
    expected_seen = any(participant_word == expected_word),
    .groups = "drop"
  )

# Combine word counts with expected words and expected seen flags
word_counts <- word_counts %>%
  left_join(expected_words, by = c("Code", "RemovalType")) %>%
  left_join(expected_seen, by = c("Code", "RemovalType", "expected_word"))

output_list <- list()

for (context in unique(paste(word_counts$Code, word_counts$RemovalType))) {
  
  parts <- strsplit(context, " ")[[1]]
  code_val <- parts[1]
  removal_val <- parts[2]
  
  subset_counts <- word_counts %>%
    filter(Code == code_val, RemovalType == removal_val)

  # Skip if there are no counts
  if (nrow(subset_counts) == 0) {
    next
  }

  exp_word <- unique(subset_counts$expected_word)
  seen_flag <- unique(subset_counts$expected_seen)

  # Handle cases where seen_flag is NA
  if (length(seen_flag) == 0 || all(is.na(seen_flag))) {
    seen_flag <- FALSE  # Default to FALSE if not found
  } else {
    seen_flag <- any(seen_flag, na.rm = TRUE)  # Determine if any TRUE values exist
  }

  total <- sum(subset_counts$count)

  if (!seen_flag) {
    # Add smoothing to expected word if it wasn't produced
    subset_counts <- subset_counts %>%
      mutate(smoothed_count = count)  # start with actual counts

    # Add expected word if missing
    if (!(exp_word %in% subset_counts$participant_word)) {
      unk_row <- data.frame(
        Code = code_val,
        RemovalType = removal_val,
        participant_word = exp_word,
        expected_word = exp_word,
        count = 0,
        expected_seen = FALSE,
        smoothed_count = 0.5,
        stringsAsFactors = FALSE
      )
      subset_counts <- bind_rows(subset_counts, unk_row)
    }

    # Add 0.5 smoothing to the expected word in all cases
    subset_counts <- subset_counts %>%
      mutate(smoothed_count = ifelse(participant_word == expected_word,
                                      smoothed_count + 0.5,
                                      smoothed_count))

    # Compute smoothed total after adding all rows
    smoothed_total <- sum(subset_counts$smoothed_count) + 0.5  # add <OTHER>

    # Compute probabilities row-wise
    subset_counts <- subset_counts %>%
      mutate(
        empirical_prob = count / total,
        smoothed_total = smoothed_total,
        smoothed_prob = smoothed_count / smoothed_total
      )

    # Normalize probabilities safely
    total_smoothed_prob <- sum(subset_counts$smoothed_prob, na.rm = TRUE)
    if (total_smoothed_prob > 0) {
      subset_counts <- subset_counts %>%
        mutate(normalized_prob = smoothed_prob / total_smoothed_prob)
    } else {
      subset_counts <- subset_counts %>%
        mutate(normalized_prob = NA_real_)  # Handle division by zero
    }
    
  } else {
    # Expected word present, no smoothing
    subset_counts <- subset_counts %>%
      mutate(
        smoothed_count = NA_real_,
        smoothed_total = NA_real_,
        empirical_prob = count / total,
        smoothed_prob = NA_real_,
        normalized_prob = NA_real_
      )
  }

  output_list[[length(output_list) + 1]] <- subset_counts
}

# Combine all outputs into a final table
final_table <- bind_rows(output_list) %>%
  left_join(contexts, by = c("Code", "RemovalType")) %>%
  arrange(Code, RemovalType, desc(count)) %>%
  select(Code, RemovalType, sentence_context,
         participant_word, expected_word, count,
         smoothed_total, empirical_prob, smoothed_count,
         smoothed_prob, normalized_prob)

# Write the final table to a CSV file
write_csv(final_table, "cloze_distributions.csv")
