library(dplyr)
library(tidyr)
library(stringr)
library(readr)

setwd("~/Desktop")

cloze_data <- read_csv("results.csv")

cloze_data <- cloze_data %>%
  filter(item_type == "experiment") %>%
  mutate(
    participant_word = str_trim(str_to_lower(participant_word)),
    expected_word = str_trim(str_to_lower(expected_word))
  )

observed_contexts <- cloze_data %>%
  distinct(sentence_context, stimuli_item, Condition, Position)

all_words <- unique(cloze_data$participant_word)

# count number of times each word appears per context
word_counts <- cloze_data %>%
  group_by(sentence_context, stimuli_item, Condition, Position, participant_word) %>%
  summarise(count = n(), .groups = "drop")

# create table of observed contexts × all words
full_table <- observed_contexts %>%
  crossing(participant_word = all_words)

smoothed <- full_table %>%
  left_join(word_counts,
            by = c("sentence_context", "stimuli_item", "Condition", "Position", "participant_word")) %>%
  mutate(count = replace_na(count, 0))

# Laplace 0.5 ?
smoothed <- smoothed %>%
  mutate(smoothed_count = count + 0.5)

# compute empirical and smoothed probabilities within each context
smoothed <- smoothed %>%
  group_by(sentence_context, stimuli_item, Condition, Position) %>%
  mutate(
    empirical_prob = count / sum(count),
    smoothed_prob  = smoothed_count / sum(smoothed_count)
  ) %>%
  ungroup()

smoothed_nonzero <- smoothed %>% filter(count > 0)
write_csv(smoothed_nonzero, "cloze_distributions_long_nonzero.csv")

write_csv(smoothed, "cloze_distributions_long.csv")

message("Cloze distributions saved to: cloze_distributions_long.csv")
