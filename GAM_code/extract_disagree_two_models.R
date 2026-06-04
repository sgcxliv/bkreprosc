#!/usr/bin/env Rscript

library(tidyverse)
library(mgcv)
library(readr)

# Standalone extractor for missing disagree curves (bkr21):
# - GPT-Neo disagree
# - GPT-J disagree
# Saves:
#   - bkr21_disagree_only_gptneo_gptj_curves.csv
#   - bkr21_disagree_only_gptneo_gptj_plot.png

WORK_DIR <- "~/Desktop"
setwd(WORK_DIR)

FLAGS_FILE <- "cloze_comparison.csv"
SPR_FILE <- "bkr21_spr.csv"
OUT_CSV <- "bkr21_disagree_only_gptneo_gptj_curves.csv"
OUT_PNG <- "bkr21_disagree_only_gptneo_gptj_plot.png"

if (!file.exists(FLAGS_FILE)) stop("Missing: ", FLAGS_FILE)
if (!file.exists(SPR_FILE)) stop("Missing: ", SPR_FILE)

flags <- read_csv(FLAGS_FILE, show_col_types = FALSE) %>%
  mutate(
    ITEM = as.factor(ITEM),
    condition = as.character(condition),
    position = as.numeric(position)
  )

spr <- read_csv(SPR_FILE, show_col_types = FALSE) %>%
  mutate(
    ITEM = as.factor(ITEM),
    condition = as.character(condition),
    position = as.numeric(position),
    SUB = as.factor(SUB),
    SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed)
  )

keys <- intersect(c("ITEM", "condition", "position"), intersect(names(spr), names(flags)))
if (length(keys) == 0) stop("No join keys found between bkr and flags.")

df <- spr %>%
  left_join(flags, by = keys) %>%
  drop_na(SUM_3RT_trimmed, SUB, ITEM)

targets <- tribble(
  ~short,    ~pretty,     ~surp_pref,
  "gptneo",  "GPT-Neo",   "gptneonew",
  "gptj",    "GPT-J",     "gptjnew"
)

extract_smooth <- function(fit, n = 200) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit({ grDevices::dev.off(); unlink(pdf_path) }, add = TRUE)
  pd <- plot(fit, select = 1, seWithMean = TRUE, n = n)[[1]]
  tibble(
    x = as.numeric(pd$x),
    fit = as.numeric(pd$fit),
    se = as.numeric(pd$se),
    se_upper = fit + se,
    se_lower = fit - se
  )
}

fit_with_fallback <- function(dat, surp_col) {
  attempts <- list(
    "SUM_3RT_trimmed ~ 1 + s(%s, bs='cs') + s(SUB, bs='re') + s(ITEM, bs='re')",
    "SUM_3RT_trimmed ~ 1 + s(%s, bs='cs', k=5) + s(SUB, bs='re') + s(ITEM, bs='re')",
    "SUM_3RT_trimmed ~ 1 + s(%s, bs='cs') + s(SUB, bs='re')"
  )
  for (tmpl in attempts) {
    fml <- sprintf(tmpl, surp_col)
    fit <- tryCatch(
      gam(as.formula(fml), data = dat, method = "REML"),
      error = function(e) NULL
    )
    if (!is.null(fit)) return(fit)
  }
  NULL
}

rows <- list()
for (i in seq_len(nrow(targets))) {
  t <- targets[i, ]
  disagree_col <- paste0("disagree_", t$short)
  if (!disagree_col %in% names(df)) {
    cat("Skipping", t$pretty, "- missing", disagree_col, "\n")
    next
  }

  surp_col <- t$surp_pref
  if (!surp_col %in% names(df)) {
    alt <- sub("new$", "", surp_col)
    if (alt %in% names(df)) surp_col <- alt else {
      cat("Skipping", t$pretty, "- missing surprisal column\n")
      next
    }
  }

  disagree_mask <- df[[disagree_col]] %in% c(TRUE, 1, "1", "TRUE", "T", "true")
  dd <- df %>%
    filter(disagree_mask) %>%
    drop_na(SUM_3RT_trimmed, SUB, ITEM, all_of(surp_col)) %>%
    mutate(SUB = droplevels(SUB), ITEM = droplevels(ITEM))

  cat(t$pretty, "disagree rows:", nrow(dd), "\n")
  if (nrow(dd) < 50) {
    cat("Skipping", t$pretty, "- too few rows\n")
    next
  }

  fit <- fit_with_fallback(dd, surp_col)
  if (is.null(fit)) {
    cat("Failed to fit", t$pretty, "\n")
    next
  }

  sm <- extract_smooth(fit) %>%
    mutate(
      model = t$pretty,
      condition = "disagree",
      dataset = "bkr21",
      surp_col = surp_col
    )
  rows[[length(rows) + 1]] <- sm
}

if (length(rows) == 0) stop("No disagree curves recovered.")

out <- bind_rows(rows)
write_csv(out, OUT_CSV)
cat("Saved:", OUT_CSV, "rows:", nrow(out), "\n")

p <- ggplot(out, aes(x = x, y = fit, color = model, fill = model)) +
  geom_ribbon(aes(ymin = se_lower, ymax = se_upper), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.0) +
  facet_wrap(~model, scales = "free_x", nrow = 1) +
  labs(
    title = "Recovered disagree-only GAM curves (bkr21)",
    x = "Surprisal",
    y = "Partial effect on SUM_3RT_trimmed"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(OUT_PNG, p, width = 10, height = 3.5, dpi = 300)
cat("Saved:", OUT_PNG, "\n")
