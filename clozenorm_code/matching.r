library(dplyr)
library(stringr)
library(stringdist)
library(readr)
library(tibble)

setwd("~/Desktop")

cloze <- read_csv("cloze_distributions.csv")
bk <- read_csv("bk_cleaned.csv")

cleanstring <- function(x) {
  x %>% str_to_lower() %>% str_replace_all("[[:punct:]]", "") %>% str_squish()
}

cloze <- cloze %>%
  mutate(
    sentence_context_cleaned = cleanstring(sentence_context),
    participant_word = cleanstring(participant_word),
    Condition = str_to_upper(Condition)
  )

bk <- bk %>%
  mutate(
    Sentence_cleaned = cleanstring(Sentence),
    words = str_split(Sentence_cleaned, " ", simplify=TRUE),
    n_words = str_count(Sentence_cleaned, " ") + 1,
    critical_idx = pmax(n_words - 2, 1),
    critical_word = ifelse(n_words >= 3, words[critical_idx], words[n_words]),
    context_prefix = sapply(Sentence_cleaned, function(s) paste(head(str_split(s, " ")[[1]], -3), collapse=" ")),
    Cloze = str_to_upper(Cloze),
    critical_word = cleanstring(critical_word)
  )

max_dist <- 7  # adjust as needed

cloze$BK_Sentence <- NA_character_
cloze$BK_critical_word <- NA_character_
cloze$BK_Cloze_Probability <- NA_real_
cloze$BK_context_distance <- NA_real_

nrows <- nrow(cloze)
progress_steps <- 10  # every 10%

for (i in seq_len(nrows)) {
  # Progress update every 10%
  if (i %% ceiling(nrows / progress_steps) == 0) {
    pct <- round(100 * i / nrows)
    cat(pct, "% done (", i, "of", nrows, ")\n")
  }
  
  ctx <- cloze$sentence_context_cleaned[i]
  cond <- cloze$Condition[i]
  # Possible B&K items for this condition
  possible <- bk %>% filter(Cloze == cond)
  # Fuzzy distances to all their context_prefixes
  dists <- stringdist(ctx, cleanstring(possible$context_prefix), method="lv")
  # Closest
  min_dist <- min(dists)
  if (!is.infinite(min_dist) && min_dist <= max_dist) {
    idx <- which.min(dists)
    # Always assign the B&K info for the best fuzzy context+condition match
    cloze$BK_Sentence[i] <- possible$Sentence[idx]
    cloze$BK_critical_word[i] <- possible$critical_word[idx]
    cloze$BK_Cloze_Probability[i] <- possible$Cloze_Probability[idx]
    cloze$BK_context_distance[i] <- min_dist
  }
}

write_csv(cloze, "cloze_with_bk_matched.csv")
cat("Wrote cloze_with_bk_matched.csv with sentence/prob assignments. Matched:",
    sum(!is.na(cloze$BK_Cloze_Probability)), "of", nrow(cloze), "rows (",
    round(100*mean(!is.na(cloze$BK_Cloze_Probability)),2), "%)\n")
