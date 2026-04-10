#!/usr/bin/env python
"""
simple_lrt.py
=============
Runs a likelihood ratio test (LRT) between two NESTED LME models.

Called by make_lrt_jobs.py-generated PBS scripts.  Each call compares
model A (simpler) vs model B (more complex) using lmer + anova() in R,
which gives a chi-squared test with the correct degrees of freedom.

IMPORTANT: This script uses lmer, matching the LME models fitted by
make_bk21_jobs.py.  It does NOT use GAM, and its results are NOT
compared with permutation test results in sum_results.py — they answer
a different (nested) inferential question.

Usage:
    python simple_lrt.py <experiment> <model_a_name>_v_<model_b_name>

Example:
    python simple_lrt.py spr cloze_v_clozeprob-cloze
"""

import sys
import os
import argparse
import stat

# ─────────────────────────────────────────────────────────────────────────────
# R TEMPLATE
# The key design choices:
#   1. Models are fitted with REML=FALSE so LRT via anova() is valid.
#   2. anova(model_a, model_b) produces the chi-sq test automatically.
#   3. We also save raw log-likelihoods so sum_lrt.py can compute ΔLL.
# ─────────────────────────────────────────────────────────────────────────────

R_TEMPLATE = '''\
#!/usr/bin/env Rscript

lib_path <- "~/R/library"
if (!dir.exists(lib_path)) dir.create(lib_path, recursive=TRUE)
.libPaths(lib_path)

if (!requireNamespace("lme4", quietly=TRUE))
    install.packages("lme4", repos="http://cran.us.r-project.org", lib=lib_path)
library(lme4)

pr <- function(x) cat(paste0(x, "\\n"), file=stderr(), append=TRUE)

lrt_dir <- file.path("{results_path}", "lrt")
if (!dir.exists(lrt_dir)) dir.create(lrt_dir, recursive=TRUE)

# ── Data ─────────────────────────────────────────────────────────────────────
pr("Loading data")
df <- read.csv("bk21_data/bk21_{experiment}.csv", header=TRUE)

for (col in c("SUB", "ITEM")) {{
    if (col %in% colnames(df)) df[[col]] <- as.factor(df[[col]])
}}
df <- df[is.finite(df${dv}), ]
pr(paste0("Rows: ", nrow(df)))

# Scale numeric predictors (same procedure as make_bk21_jobs.py)
numeric_preds <- c("critical_word_pos", "wlen", "wlenregion",
                   "unigram", "unigramregion", "glovedistmean")
for (pred in names(df)) {{
    if (grepl("cloze|gpt2|gptj|gptneo|gptneox|olmo|llama2", pred, ignore.case=TRUE) &&
        is.numeric(df[[pred]]) && !grepl("fold|sortix", pred, ignore.case=TRUE))
        numeric_preds <- c(numeric_preds, pred)
}}
numeric_preds <- unique(numeric_preds)
for (col in numeric_preds) {{
    if (col %in% colnames(df) && is.numeric(df[[col]]) && sd(df[[col]], na.rm=TRUE) > 0)
        df[[col]] <- scale(df[[col]])[, 1]
}}

# ── Fit models (REML=FALSE required for valid LRT) ───────────────────────────
pr("Fitting Model A: {model_a}")
ctrl <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5), calc.derivs=FALSE,
                    check.conv.grad=.makeCC("warning", tol=2e-3, relTol=NULL))

fit_a <- tryCatch(
    lmer({form_a}, data=df, REML=FALSE, control=ctrl),
    error = function(e) {{
        pr(paste0("bobyqa failed for A: ", e$message, " — trying Nelder_Mead"))
        lmer({form_a}, data=df, REML=FALSE,
             control=lmerControl(optimizer="Nelder_Mead", optCtrl=list(maxfun=2e5)))
    }}
)

pr("Fitting Model B: {model_b}")
fit_b <- tryCatch(
    lmer({form_b}, data=df, REML=FALSE, control=ctrl),
    error = function(e) {{
        pr(paste0("bobyqa failed for B: ", e$message, " — trying Nelder_Mead"))
        lmer({form_b}, data=df, REML=FALSE,
             control=lmerControl(optimizer="Nelder_Mead", optCtrl=list(maxfun=2e5)))
    }}
)

# ── LRT ──────────────────────────────────────────────────────────────────────
pr("Running LRT")
lrt <- anova(fit_a, fit_b)

loglik_a <- as.numeric(logLik(fit_a))
loglik_b <- as.numeric(logLik(fit_b))
chisq    <- as.numeric(lrt$Chisq[2])
df_diff  <- as.numeric(lrt$Df[2])
p_value  <- as.numeric(lrt[["Pr(>Chisq)"]][2])
sig      <- p_value <= 0.05

stars <- ifelse(p_value <= 0.001, "***", ifelse(p_value <= 0.01, "**",
         ifelse(p_value <= 0.05, "*", "")))

result_str <- paste0(
    "==================================================\\n",
    "Likelihood Ratio Test\\n",
    "Model A (simpler):  {model_a}\\n",
    "Model B (complex):  {model_b}\\n",
    "==================================================\\n",
    "LogLik A:   ", round(loglik_a, 4), "\\n",
    "LogLik B:   ", round(loglik_b, 4), "\\n",
    "Delta LL:   ", round(loglik_b - loglik_a, 4), "\\n",
    "Chi-sq:     ", round(chisq, 4), "  (df=", df_diff, ")\\n",
    "p-value:    ", formatC(p_value, format="e", digits=4), stars, "\\n",
    "Significant:", ifelse(sig, "YES", "NO"), "\\n",
    "=================================================="
)
pr(result_str)
writeLines(result_str, file.path(lrt_dir, "{test_name}.txt"))

# ── Summary CSV (consumed by sum_lrt.py) ─────────────────────────────────────
summary_df <- data.frame(
    model_a    = "{model_a}",
    model_b    = "{model_b}",
    loglik_a   = loglik_a,
    loglik_b   = loglik_b,
    difference = loglik_b - loglik_a,
    chisq      = chisq,
    df_diff    = df_diff,
    p_value    = p_value,
    significant = sig
)
write.csv(summary_df,
          file.path(lrt_dir, "{test_name}_summary.csv"),
          row.names=FALSE)

pr("Done")
'''


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_PREDS = [
    'critical_word_pos', 'wlen', 'wlenregion',
    'unigram', 'unigramregion', 'glovedistmean'
]

def build_lme_formula(name, dv, ranefs, use_controls=False):
    preds = list(CONTROL_PREDS) if use_controls else []
    if name != 'nosurp':
        parts = name.split('-') if '-' in name else [name]
        preds += parts
    form = '%s ~ 1' % dv
    for p in preds:
        form += ' + %s' % p
    for r in ranefs:
        form += ' + (1 | %s)' % r
    return form


def write_executable(path, content):
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run LRT for a nested model pair')
    parser.add_argument('experiment', choices=['spr', 'naming'])
    parser.add_argument('test_name',
                        help='Format: modelA_v_modelB  (e.g. cloze_v_clozeprob-cloze)')
    parser.add_argument('--controls', action='store_true',
                        help='Include control predictors in both models')
    args = parser.parse_args()

    # Results path
    if os.path.exists('bk21_results_path.txt'):
        with open('bk21_results_path.txt') as f:
            results_path = next(l.strip() for l in f if l.strip())
    else:
        results_path = 'results/bk21'

    dv_map     = {'spr': 'SUM_3RT_trimmed', 'naming': 'TRIM_RT'}
    ranef_map  = {'spr': ('SUB', 'ITEM'),   'naming': ('subject', 'item')}

    dv     = dv_map[args.experiment]
    ranefs = ranef_map[args.experiment]

    model_a, model_b = args.test_name.split('_v_')

    form_a = build_lme_formula(model_a, dv, ranefs, use_controls=args.controls)
    form_b = build_lme_formula(model_b, dv, ranefs, use_controls=args.controls)

    test_name = 'lrt_%s_%s' % (args.experiment, args.test_name)

    r_code = R_TEMPLATE.format(
        results_path=results_path,
        experiment=args.experiment,
        dv=dv,
        form_a=form_a,
        form_b=form_b,
        model_a=model_a,
        model_b=model_b,
        test_name=test_name
    )

    # Write R script alongside the other lrt_scripts, not in home dir root
    os.makedirs('lrt_scripts', exist_ok=True)
    r_path = os.path.join('lrt_scripts', '%s.R' % test_name)
    write_executable(r_path, r_code)

    print("Running: Rscript %s" % r_path)
    ret = os.system('Rscript %s' % r_path)
    if ret != 0:
        print("WARNING: Rscript exited with code %d" % ret, file=sys.stderr)

    # Clean up the generated R script — results are already saved to results/bk21/lrt/
    try:
        os.remove(r_path)
    except OSError:
        pass


if __name__ == '__main__':
    main()
