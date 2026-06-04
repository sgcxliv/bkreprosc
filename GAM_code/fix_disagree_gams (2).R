#!/usr/bin/env Rscript
# =====================================================================
# fix_disagree_gams.R
# =====================================================================
library(tidyverse)
library(mgcv)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && nzchar(args[1])) {
  setwd(args[1])
} else {
  # Default to local project dir used in this workspace.
  setwd("~/Desktop/fuck")
}

cat("Working directory:", getwd(), "\n")

# ── Load data ────────────────────────────────────────────────────────

read_if_exists <- function(path) {
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

flags <- read_if_exists("cloze_comparison.csv")
if (is.null(flags)) {
  stop("Could not find cloze_comparison.csv in ", getwd(),
       ". Pass a directory argument: Rscript \"fix_disagree_gams (2).R\" \"/path/to/data\"")
}

flags <- flags %>%
  mutate(ITEM      = as.factor(ITEM),
         condition = as.character(condition),
         position  = as.numeric(position))

spr_r <- read_csv("bkr21_spr.csv", show_col_types = FALSE,
                  col_types = cols(.default = col_guess(),
                                   word01 = col_character(), word02 = col_character(),
                                   word03 = col_character(), word04 = col_character(),
                                   word05 = col_character(), word06 = col_character(),
                                   word07 = col_character(), word08 = col_character(),
                                   word09 = col_character(), word10 = col_character(),
                                   word11 = col_character(), word12 = col_character(),
                                   word13 = col_character(), word14 = col_character(),
                                   word15 = col_character(), word16 = col_character()))

# ── Join ─────────────────────────────────────────────────────────────

keys <- intersect(c("ITEM", "condition", "position"),
                  intersect(names(spr_r), names(flags)))
cat("Join keys:", paste(keys, collapse = ", "), "\n")

spr_r <- spr_r %>%
  mutate(ITEM      = as.factor(ITEM),
         condition = as.character(condition),
         position  = as.numeric(position),
         SUB       = as.factor(SUB),
         SUM_3RT_trimmed = as.numeric(SUM_3RT_trimmed))

df_r <- spr_r %>%
  left_join(flags, by = keys) %>%
  drop_na(SUM_3RT_trimmed, SUB, ITEM, clozeprob)

cat("Total joined rows:", nrow(df_r), "\n\n")

# two failures

targets <- tribble(
  ~short,    ~surp_col,     ~pretty,
  "gptneo",  "gptneonew",   "GPT-Neo",
  "gptj",    "gptjnew",     "GPT-J"
)

# smooth helper

extract_smooth <- function(fit) {
  pdf_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(file = pdf_path)
  on.exit({ grDevices::dev.off(); unlink(pdf_path) }, add = TRUE)
  plot_data <- plot(fit, select = 1, seWithMean = TRUE, n = 200)
  pd <- plot_data[[1]]
  data.frame(
    x        = pd$x,
    fit      = as.numeric(pd$fit),
    se       = as.numeric(pd$se),
    se_upper = as.numeric(pd$fit) + as.numeric(pd$se),
    se_lower = as.numeric(pd$fit) - as.numeric(pd$se)
  )
}

# fit
fit_with_fallback <- function(df, surp_col, pretty_name) {
  attempts <- list(
    list(
      label = "gam REML full model",
      formula = paste0("SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs') + s(SUB, bs='re') + s(ITEM, bs='re')"),
      fit_fn = function(fml, dat) gam(as.formula(fml), data = dat, method = "REML")
    ),
    list(
      label = "bam fREML full model",
      formula = paste0("SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs') + s(SUB, bs='re') + s(ITEM, bs='re')"),
      fit_fn = function(fml, dat) bam(as.formula(fml), data = dat, method = "fREML", discrete = TRUE)
    ),
    list(
      label = "gam REML k=5",
      formula = paste0("SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs', k=5) + s(SUB, bs='re') + s(ITEM, bs='re')"),
      fit_fn = function(fml, dat) gam(as.formula(fml), data = dat, method = "REML")
    ),
    list(
      label = "gam REML no ITEM RE",
      formula = paste0("SUM_3RT_trimmed ~ 1 + s(", surp_col, ", bs='cs') + s(SUB, bs='re')"),
      fit_fn = function(fml, dat) gam(as.formula(fml), data = dat, method = "REML")
    )
  )

  for (a in attempts) {
    cat("\n  Attempt:", a$label, "\n")
    cat("  Formula:", a$formula, "\n")
    fit <- tryCatch(
      a$fit_fn(a$formula, df),
      error = function(e) {
        cat("  FAILED:", conditionMessage(e), "\n")
        NULL
      }
    )
    if (!is.null(fit)) {
      cat("  SUCCESS:", a$label, "\n")
      return(fit)
    }
  }

  cat("\n  ALL ATTEMPTS FAILED for", pretty_name, "disagree\n")
  NULL
}

all_curves <- list()

for (i in seq_len(nrow(targets))) {
  m <- targets[i, ]
  disagree_col <- paste0("disagree_", m$short)

  cat(strrep("=", 60), "\n")
  cat(m$pretty, "\n")
  cat(strrep("=", 60), "\n")

  # Check columns exist
  if (!disagree_col %in% names(df_r)) {
    cat("  ERROR: column", disagree_col, "not found in data!\n")
    cat("  Available agree/disagree columns:\n")
    print(grep("agree|disagree", names(df_r), value = TRUE))
    next
  }
  if (!m$surp_col %in% names(df_r)) {
    cat("  ERROR: surprisal column", m$surp_col, "not found!\n")
    # Try without 'new' suffix
    alt_col <- gsub("new$", "", m$surp_col)
    if (alt_col %in% names(df_r)) {
      cat("  Found alternative:", alt_col, "\n")
      m$surp_col <- alt_col
    } else {
      cat("  Available surprisal-like columns:\n")
      print(grep(m$short, names(df_r), value = TRUE, ignore.case = TRUE))
      next
    }
  }

  disagree_mask <- df_r[[disagree_col]] %in% c(TRUE, 1, "1", "TRUE", "T", "true")
  df_disagree <- df_r %>%
    filter(disagree_mask) %>%
    drop_na(SUM_3RT_trimmed, SUB, ITEM, all_of(m$surp_col)) %>%
    mutate(SUB = droplevels(SUB), ITEM = droplevels(ITEM))

  cat("  Disagree rows:", nrow(df_disagree), "\n")
  cat("  Unique SUBs:", nlevels(df_disagree$SUB), "\n")
  cat("  Unique ITEMs:", nlevels(df_disagree$ITEM), "\n")
  cat("  Surprisal range:", range(df_disagree[[m$surp_col]], na.rm = TRUE), "\n")

  if (nrow(df_disagree) == 0) {
    cat("  NO DATA — cannot fit.\n\n")
    next
  }

  fit <- fit_with_fallback(df_disagree, m$surp_col, m$pretty)
  if (is.null(fit)) next

  smooth <- extract_smooth(fit)
  smooth$model     <- m$pretty
  smooth$condition <- "disagree"
  smooth$dataset   <- "bkr21"
  smooth$surp_col  <- m$surp_col
  all_curves[[paste0(m$short, "_disagree")]] <- smooth
}

# ── Save results ─────────────────────────────────────────────────────

if (length(all_curves) > 0) {
  out <- bind_rows(all_curves)
  out_file <- "bkr21_disagree_fix_curves.csv"
  write_csv(out, out_file)
  cat("\n\nSaved", nrow(out), "rows to", out_file, "\n")
  cat("Models recovered:", unique(out$model), "\n")

  main_file <- "bkr21_gam_curves.csv"
  if (file.exists(main_file)) {
    main <- read_csv(main_file, show_col_types = FALSE)
    if (!all(c("model", "condition") %in% names(main))) {
      stop("Expected columns missing in ", main_file, ": model, condition")
    }
    # Remove old disagree rows for these models 
    models_fixed <- unique(out$model)
    main <- main %>%
      filter(!(model %in% models_fixed & condition == "disagree"))
    combined <- bind_rows(main, out)
    write_csv(combined, main_file)
    cat("Updated", main_file, "with fixed curves (", nrow(combined), "total rows)\n")
  }
} else {
  cat("\n\nNo curves recovered. Check the error messages above.\n")
}

cat("\nDone.\n")
