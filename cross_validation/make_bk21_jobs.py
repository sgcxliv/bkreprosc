#!/usr/bin/env python
"""
make_bk21_jobs.py
=================
Generates SLURM job scripts for fitting all models needed in the BK21 pipeline.

TWO model families are generated:
  1. GAM (bam, mgcv)  — used for PERMUTATION TESTS (non-nested comparisons)
  2. LME (lmer, lme4) — used for LIKELIHOOD RATIO TESTS (nested comparisons)

Both families run 5-fold cross-validation and write per-observation log-likelihoods
to CSV files that downstream scripts consume.

IMPORTANT DESIGN DECISIONS
---------------------------
* GAM and LME use the SAME held-out log-likelihood formula so that the
  permutation test (test.py) can validly compare models within each family.
  The residual SD is always computed from the HELD-OUT residuals pooled across
  all folds (not per-fold training residuals), so the LL scale is identical.
* LME outputs are used ONLY by simple_lrt.py for nested LRT comparisons.
  GAM outputs are used ONLY by test.py for permutation tests.
* The two families are NOT compared against each other in sum_results.py.
"""

import os
import stat

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

experiments         = ['spr']
dv_by_experiment    = {'spr': 'SUM_3RT_trimmed'}
ranef_by_experiment = {'spr': ('SUB', 'ITEM')}

# NOTE: model column names in your CSV must match these strings exactly.
# If your CSV uses 'gpt2new' etc., update the list below accordingly.
lms = [
    'nosurp',
    'cloze',      'clozeregion',
    'gpt2',       'gpt2region',
    'gpt2xl',     'gpt2xlregion',
    'gptj',       'gptjregion',
    'gptneo',     'gptneoregion',
    'gptneox',    'gptneoxregion',
    'olmo',       'olmoregion',
    'llama2',     'llama2region',
]

control_preds = [
    'critical_word_pos', 'wlen', 'wlenregion',
    'unigram', 'unigramregion', 'glovedistmean'
]

controls = ['', 'C']   # '' = no controls, 'C' = with controls

# Results path — override by writing the path to bk21_results_path.txt
if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        results_path = next(l.strip() for l in f if l.strip())
else:
    results_path = 'results/bk21'


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_config(name):
    """nosurpprob is meaningless — skip it."""
    if name.startswith('nosurp') and 'prob' in name:
        return False
    return True


def build_predictor_list(name, use_controls):
    preds = list(control_preds) if use_controls else []
    if name == 'nosurp':
        return preds
    parts = name.split('-') if '-' in name else [name]
    preds += parts
    return preds


def build_gam_formula(preds, dv, ranefs, experiment):
    form = '%s ~ 1' % dv
    for pred in preds:
        if pred == 'critical_word_pos':
            k = ', k=8' if experiment == 'naming' else ', k=5'
        elif pred == 'wlen':
            k = ', k=6'
        else:
            k = ''
        form += ' + s(%s, bs="cs"%s)' % (pred, k)
    for ranef in ranefs:
        form += ' + s(%s, bs="re")' % ranef
    return form


def build_lme_formula(preds, dv, ranefs):
    form = '%s ~ 1' % dv
    for pred in preds:
        form += ' + %s' % pred
    for ranef in ranefs:
        form += ' + (1 | %s)' % ranef
    return form


def write_executable(path, content):
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


# ─────────────────────────────────────────────────────────────────────────────
# R TEMPLATES — SHARED PREAMBLE & DATA
# ─────────────────────────────────────────────────────────────────────────────

R_PREAMBLE = '''\
#!/usr/bin/env Rscript

lib_path <- "~/R/library"
if (!dir.exists(lib_path)) dir.create(lib_path, recursive=TRUE)
.libPaths(lib_path)

pr <- function(x) cat(paste0(x, "\\n"), file=stderr(), append=TRUE)

if (!dir.exists("{results_path}")) dir.create("{results_path}", recursive=TRUE)

NFOLDS <- 5

'''

R_DATA = '''\
# ── Data ─────────────────────────────────────────────────────────────────────
pr("Loading data")
df <- read.csv("bk21_data/bk21_{experiment}.csv", header=TRUE)
pr(paste0("Initial rows: ", nrow(df)))

for (col in c("SUB", "ITEM")) {{
    if (col %in% colnames(df)) df[[col]] <- as.factor(df[[col]])
}}

df <- df[is.finite(df${dv}), ]
pr(paste0("Rows after finite filter: ", nrow(df)))

# Scale numeric predictors
numeric_preds <- c("critical_word_pos", "wlen", "wlenregion",
                   "unigram", "unigramregion", "glovedistmean")
for (pred in names(df)) {{
    if (grepl("cloze|gpt2|gptj|gptneo|gptneox|olmo|llama2", pred, ignore.case=TRUE) &&
        is.numeric(df[[pred]]) &&
        !grepl("fold|sortix", pred, ignore.case=TRUE)) {{
        numeric_preds <- c(numeric_preds, pred)
    }}
}}
numeric_preds <- unique(numeric_preds)

for (col in numeric_preds) {{
    if (col %in% colnames(df) && is.numeric(df[[col]]) && sd(df[[col]], na.rm=TRUE) > 0)
        df[[col]] <- scale(df[[col]])[, 1]
}}
pr(paste0("Final dimensions: ", nrow(df), " x ", ncol(df)))

'''

# ─────────────────────────────────────────────────────────────────────────────
# GAM TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

R_GAM_BODY = '''\
# ── GAM: 5-fold CV with BAM ───────────────────────────────────────────────────
if (!requireNamespace("mgcv", quietly=TRUE))
    install.packages("mgcv", repos="http://cran.us.r-project.org", lib=lib_path)
library(mgcv)

allpreds <- data.frame()

for (fold in 1:NFOLDS) {{
    pr(paste0("Fold ", fold))
    train <- df[df$fold != fold, ]
    test  <- df[df$fold == fold, ]

    m <- tryCatch(
        bam({form}, data=train, method="fREML", discrete=TRUE),
        error = function(e) {{
            pr(paste0("  bam (discrete) failed fold ", fold, ": ", e$message))
            tryCatch(
                bam({form}, data=train, method="fREML"),
                error = function(e2) {{ pr(paste0("  bam retry failed: ", e2$message)); NULL }}
            )
        }}
    )
    if (is.null(m)) {{ pr(paste0("  Skipping fold ", fold)); next }}

    preds <- tryCatch(
        predict(m, newdata=test, type="response"),
        error = function(e) {{
            pr(paste0("  Predict failed fold ", fold, ": ", e$message))
            rep(mean(train[["{dv}"]], na.rm=TRUE), nrow(test))
        }}
    )
    if (any(is.na(preds))) {{
        pr(paste0("  Replacing ", sum(is.na(preds)), " NA preds with training mean"))
        preds[is.na(preds)] <- mean(train[["{dv}"]], na.rm=TRUE)
    }}
    test$pred <- preds
    test$fold_id <- fold
    allpreds <- rbind(allpreds, test)
    pr(paste0("  Fold ", fold, " done: ", nrow(test), " predictions"))
}}

if (nrow(allpreds) == 0) stop("ERROR: no successful predictions")
allpreds <- allpreds[order(allpreds$sortix), ]

'''

# ─────────────────────────────────────────────────────────────────────────────
# LME TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

R_LME_BODY = '''\
# ── LME: 5-fold CV with lmer ─────────────────────────────────────────────────
if (!requireNamespace("lme4", quietly=TRUE))
    install.packages("lme4", repos="http://cran.us.r-project.org", lib=lib_path)
library(lme4)

ctrl_primary <- lmerControl(
    optimizer="bobyqa",
    optCtrl=list(maxfun=2e5),
    calc.derivs=FALSE,
    check.conv.grad=.makeCC("warning", tol=2e-3, relTol=NULL)
)
ctrl_alt <- lmerControl(optimizer="Nelder_Mead", optCtrl=list(maxfun=2e5), calc.derivs=FALSE)

allpreds <- data.frame()

for (fold in 1:NFOLDS) {{
    pr(paste0("Fold ", fold))
    train <- df[df$fold != fold, ]
    test  <- df[df$fold == fold, ]

    fit <- tryCatch(
        lmer({form}, data=train, REML=FALSE, control=ctrl_primary),
        error = function(e) {{
            pr(paste0("  bobyqa failed fold ", fold, ": ", e$message))
            tryCatch(
                lmer({form}, data=train, REML=FALSE, control=ctrl_alt),
                error = function(e2) {{ pr(paste0("  Nelder_Mead failed: ", e2$message)); NULL }}
            )
        }}
    )
    if (is.null(fit)) {{ pr(paste0("  Skipping fold ", fold)); next }}

    preds <- tryCatch(
        predict(fit, newdata=test, allow.new.levels=TRUE, re.form=NULL),
        error = function(e) {{
            pr(paste0("  Predict failed fold ", fold, ": ", e$message))
            rep(mean(train[["{dv}"]], na.rm=TRUE), nrow(test))
        }}
    )
    if (any(is.na(preds))) {{
        pr(paste0("  Replacing ", sum(is.na(preds)), " NA preds with training mean"))
        preds[is.na(preds)] <- mean(train[["{dv}"]], na.rm=TRUE)
    }}
    test$pred <- preds
    test$fold_id <- fold
    allpreds <- rbind(allpreds, test)
    pr(paste0("  Fold ", fold, " done: ", nrow(test), " predictions"))
}}

if (nrow(allpreds) == 0) stop("ERROR: no successful predictions")
allpreds <- allpreds[order(allpreds$sortix), ]

'''

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION TEMPLATE — IDENTICAL FOR BOTH MODEL FAMILIES
#
# KEY: residual SD is computed from ALL held-out residuals pooled across folds.
# This ensures the LL scale is the same across all models (GAM or LME),
# making permutation test comparisons valid within each family.
# ─────────────────────────────────────────────────────────────────────────────

R_EVALUATION = '''\
# ── Evaluation (shared for GAM and LME) ──────────────────────────────────────
pr("Evaluating")

obs     <- allpreds[["{dv}"]]
pred    <- allpreds$pred
err     <- obs - pred
mse     <- mean(err^2, na.rm=TRUE)

# SD from pooled held-out residuals — same pool for ALL models, fair comparison
sd_resid <- sd(err, na.rm=TRUE)
ll_vec   <- dnorm(err, mean=0, sd=sd_resid, log=TRUE)
ll_sum   <- sum(ll_vec, na.rm=TRUE)

eval_str <- paste0(
    "==================================================\\n",
    "Model name: {model}\\n",
    "MODEL EVALUATION STATISTICS:\\n",
    "  Predictions: ", length(pred), " / ", nrow(df), " rows\\n",
    "  Loglik: ", ll_sum, "\\n",
    "  MSE: ", mse, "\\n",
    "=================================================="
)
pr(eval_str)

write.table(eval_str,
            "{results_path}/{model}_eval_test.txt",
            row.names=FALSE, col.names=FALSE)

output <- data.frame(
    obs    = obs,
    pred   = pred,
    err    = err,
    loglik = ll_vec,
    fold   = allpreds$fold_id,
    sortix = allpreds$sortix
)
write.table(output,
            "{results_path}/{model}_output_test.csv",
            row.names=FALSE, col.names=TRUE, sep=",")

pr("Done")
'''

# ─────────────────────────────────────────────────────────────────────────────
# PBS / SLURM TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

BASH_TEMPLATE = '''\
#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=4:00:00
#SBATCH --mem={mem}gb
#SBATCH --ntasks=1

source ~/.bashrc
conda activate cs24

cd /afs/cs.stanford.edu/u/sgcxliv
Rscript bk21_scripts/{job_name}.R
'''


# ─────────────────────────────────────────────────────────────────────────────
# JOB FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_job(name, experiment, control, model_type='gam', results_path='results/bk21'):
    """Write one .R + .pbs pair for either GAM or LME."""
    if not check_config(name):
        return

    ranefs  = ranef_by_experiment[experiment]
    dv      = dv_by_experiment[experiment]
    use_ctrl = (control == 'C')
    preds   = build_predictor_list(name, use_ctrl)

    if model_type == 'gam':
        form      = build_gam_formula(preds, dv, ranefs, experiment)
        body      = R_GAM_BODY
        mem       = 16
    else:  # lme
        form      = build_lme_formula(preds, dv, ranefs)
        body      = R_LME_BODY
        mem       = 8

    job_name = '%s_%s_%s%s' % (model_type, experiment, name, control)

    r_code = (R_PREAMBLE + R_DATA + body + R_EVALUATION).format(
        results_path=results_path,
        experiment=experiment,
        form=form,
        dv=dv,
        model=job_name
    )

    os.makedirs('bk21_scripts', exist_ok=True)
    write_executable('bk21_scripts/%s.R'   % job_name, r_code)
    write_executable('bk21_scripts/%s.pbs' % job_name,
                     BASH_TEMPLATE.format(job_name=job_name, mem=mem))


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL JOBS
# ─────────────────────────────────────────────────────────────────────────────

non_region_lms = [m for m in lms if 'region' not in m and m != 'nosurp']
region_lms     = [m for m in lms if 'region' in m]

print("=" * 70)
print("BK21 JOBS GENERATOR")
print("=" * 70)

# ── 1. GAM jobs ──────────────────────────────────────────────────────────────
# GAM is used for permutation tests.
# Configs: surp, prob, surp+prob for every LM (including nosurp baseline).
# Between-model joined configs (e.g. cloze-gpt2) are NOT needed for GAM
# because permutation tests compare independently-fitted models' log-likelihoods.
print("\n[GAM] Generating jobs for permutation tests...")
gam_count = 0
for experiment in experiments:
    for lm in lms:
        for fn_template in ['%s', '%sprob', '%sprob-%s']:
            # build name: nosurpprob is skipped inside make_job via check_config
            if fn_template.count('%s') == 2:
                name = fn_template % (lm, lm)
            else:
                name = fn_template % lm
            for control in controls:
                make_job(name, experiment, control,
                         model_type='gam', results_path=results_path)
                gam_count += 1

print("  GAM jobs: %d" % gam_count)

# ── 2. LME jobs ──────────────────────────────────────────────────────────────
# LME is used ONLY for LRT (nested comparisons).
# We need: nosurp, every surp, every prob, every surp+prob.
# We do NOT need between-model joined LME configs because LRT is only valid
# for nested models (adding/removing a predictor within the same family).
print("\n[LME] Generating jobs for likelihood ratio tests...")
lme_count = 0
for experiment in experiments:
    for lm in lms:
        for fn_template in ['%s', '%sprob', '%sprob-%s']:
            if fn_template.count('%s') == 2:
                name = fn_template % (lm, lm)
            else:
                name = fn_template % lm
            for control in controls:
                make_job(name, experiment, control,
                         model_type='lme', results_path=results_path)
                lme_count += 1

print("  LME jobs: %d" % lme_count)

total = gam_count + lme_count
print("\n" + "=" * 70)
print("SUMMARY: %d total job scripts written to bk21_scripts/" % total)
print("=" * 70)
print("\nTo submit ALL jobs:")
print("  cd bk21_scripts")
print("  for job in *.pbs; do sbatch $job; done")
print("  cd ..")
print("\nTo submit GAM only:")
print("  cd bk21_scripts && for job in gam_*.pbs; do sbatch $job; done && cd ..")
print("\nTo submit LME only:")
print("  cd bk21_scripts && for job in lme_*.pbs; do sbatch $job; done && cd ..")
