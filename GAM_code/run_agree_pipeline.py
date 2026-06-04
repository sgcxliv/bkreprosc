#!/usr/bin/env python
"""
run_agree_pipeline.py
=====================
Runs the full CV pipeline (make_bk21_jobs -> make_test_jobs -> submit)
for each agree subset

Usage:
    python run_agree_pipeline.py [--model gpt2]   # single model
    python run_agree_pipeline.py                   # all models + all
    python run_agree_pipeline.py --dry-run         # print jobs, don't submit

Run AFTER make_agree_subsets.py has created the subset CSVs.
"""

import os
import sys
import shutil
import argparse
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODELS   = ['gpt2', 'gpt2xl', 'gptj', 'gptneo', 'gptneox', 'olmo', 'llama2', 'all']
DATA_DIR = 'bk21_data'
SPR_ORIG = 'bk21_spr_backup.csv'   # backup of original
SPR_LIVE = 'bk21_spr.csv'         # what R reads

SBATCH = 'sbatch --account=nlp --cpus-per-task=16 --mem=32G --partition=sphinx'

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, dry_run=False):
    print("  $", cmd)
    if not dry_run:
        subprocess.run(cmd, shell=True, check=True)


def set_results_path(path):
    with open('bk21_results_path.txt', 'w') as f:
        f.write(path + '\n')
    print("  bk21_results_path.txt -> %s" % path)


def restore_results_path():
    # Default path — remove the override file
    if os.path.exists('bk21_results_path.txt'):
        os.remove('bk21_results_path.txt')
    print("  bk21_results_path.txt removed (reset to default results/bk21)")


def swap_spr_csv(subset_path, dry_run=False):
    """Replace bk21_spr.csv with the agree subset."""
    if not dry_run:
        if not os.path.exists(SPR_LIVE):
            raise FileNotFoundError("Expected %s to exist" % SPR_LIVE)
        if not os.path.exists(SPR_ORIG):
            shutil.copy(SPR_LIVE, SPR_ORIG)
            print("  Backed up original -> %s" % SPR_ORIG)
        shutil.copy(subset_path, SPR_LIVE)
        print("  Swapped bk21_spr.csv -> %s" % subset_path)
    else:
        print("  [dry-run] Would swap bk21_spr.csv -> %s" % subset_path)


def restore_spr_csv(dry_run=False):
    """Restore original bk21_spr.csv."""
    if not dry_run:
        if os.path.exists(SPR_ORIG):
            shutil.copy(SPR_ORIG, SPR_LIVE)
            print("  Restored bk21_spr.csv from backup")
        else:
            print("  WARNING: No backup found at %s" % SPR_ORIG)
    else:
        print("  [dry-run] Would restore bk21_spr.csv from backup")


def submit_jobs(script_dir, pattern, dry_run=False):
    jobs = sorted(f for f in os.listdir(script_dir) if f.endswith('.pbs') and
                  (pattern == '*' or f.startswith(pattern)))
    print("  Submitting %d jobs from %s/" % (len(jobs), script_dir))
    for job in jobs:
        run('%s %s/%s' % (SBATCH, script_dir, job), dry_run=dry_run)
    return len(jobs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_model(model, dry_run=False):
    subset_csv  = os.path.join(DATA_DIR, 'agree_%s_spr.csv' % model)
    results_dir = 'results/bk21_agree_%s' % model
    scripts_dir = 'bk21_scripts_agree_%s' % model
    test_dir    = 'test_scripts_agree_%s' % model
    lrt_dir     = 'lrt_scripts_agree_%s' % model

    print("\n" + "=" * 70)
    print("AGREE SUBSET: %s" % model)
    print("  Data:    %s" % subset_csv)
    print("  Results: %s" % results_dir)
    print("=" * 70)

    if not os.path.exists(subset_csv):
        print("  ERROR: %s not found. Run make_agree_subsets.py first." % subset_csv)
        return

    # 1. Swap CSV and set results path
    swap_spr_csv(subset_csv, dry_run=dry_run)
    set_results_path(results_dir)
    if not dry_run:
        os.makedirs(results_dir + '/lrt', exist_ok=True)

    # 2. Generate job scripts into model-specific directories
    run('python make_bk21_jobs.py', dry_run=dry_run)
    run('python make_lrt_jobs.py',  dry_run=dry_run)
    run('python make_test_jobs.py', dry_run=dry_run)

    # Rename generated script dirs to model-specific names
    for src, dst in [('bk21_scripts', scripts_dir),
                     ('lrt_scripts',  lrt_dir),
                     ('test_scripts', test_dir)]:
        if not dry_run:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            if os.path.exists(src):
                shutil.move(src, dst)
        print("  Moved %s -> %s" % (src, dst))

    # 3. Submit LME jobs and print wait instruction
    print("\n  [Step 1] Submitting LME model fitting jobs...")
    n = submit_jobs(scripts_dir, 'lme_', dry_run=dry_run)
    print("\n  *** WAIT for all %d LME jobs to finish before Step 2 ***" % n)
    print("      squeue -u $USER")
    print("  Then run:")
    print("    python run_agree_pipeline.py --model %s --step2" % model)

    restore_results_path()


def run_step2_for_model(model, dry_run=False):
    """Submit LRT + PT jobs after LME fits are done."""
    results_dir = 'results/bk21_agree_%s' % model
    test_dir    = 'test_scripts_agree_%s' % model
    lrt_dir     = 'lrt_scripts_agree_%s' % model

    subset_csv = os.path.join(DATA_DIR, 'agree_%s_spr.csv' % model)
    if not os.path.exists(subset_csv) and model != 'all':
        print("  ERROR: %s not found. Run make_agree_subsets.py first." % subset_csv)
        return
    if os.path.exists(subset_csv):
        swap_spr_csv(subset_csv, dry_run=dry_run)

    print("\n" + "=" * 70)
    print("AGREE SUBSET STEP 2: %s" % model)
    print("=" * 70)

    set_results_path(results_dir)

    print("\n  [Step 2a] Submitting LRT jobs...")
    submit_jobs(lrt_dir, '*', dry_run=dry_run)

    print("\n  [Step 2b] Submitting PT jobs...")
    submit_jobs(test_dir, '*', dry_run=dry_run)

    restore_results_path()

    print("\n  *** WAIT for all jobs to finish, then run: ***")
    print("    python run_agree_pipeline.py --model %s --summarize" % model)


def summarize_model(model, dry_run=False):
    """Run summary scripts for a completed agree subset."""
    results_dir = 'results/bk21_agree_%s' % model

    subset_csv = os.path.join(DATA_DIR, 'agree_%s_spr.csv' % model)
    if os.path.exists(subset_csv):
        swap_spr_csv(subset_csv, dry_run=dry_run)

    print("\n" + "=" * 70)
    print("AGREE SUBSET SUMMARIZE: %s" % model)
    print("=" * 70)

    set_results_path(results_dir)
    run('python sum_lrt.py', dry_run=dry_run)
    run('python sum_results_nofdr.py', dry_run=dry_run)

    # Save to model-specific output
    out_raw = 'all_results_raw_agree_%s.csv' % model
    out_fdr = 'all_results_summary_agree_%s.csv' % model
    if not dry_run and os.path.exists('all_results_raw.csv'):
        shutil.copy('all_results_raw.csv', out_raw)
    # Use apply_fdr's DEFAULT_PAIRS mapping so `all_results_summary.csv`
    run('python apply_fdr.py', dry_run=dry_run)
    if not dry_run and os.path.exists('all_results_summary.csv'):
        shutil.copy('all_results_summary.csv', out_fdr)

    restore_results_path()
    restore_spr_csv(dry_run=dry_run)
    print("\n  Results saved to: %s, %s" % (out_raw, out_fdr))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default=None,  help='Model name (or "all"). Default: run all models.')
    parser.add_argument('--step2',     action='store_true', help='Submit LRT+PT jobs (after LME fits done)')
    parser.add_argument('--summarize', action='store_true', help='Run summary scripts (after all jobs done)')
    parser.add_argument('--dry-run',   action='store_true', help='Print commands without executing')
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS

    for model in models_to_run:
        if model not in MODELS:
            print("Unknown model: %s. Choose from: %s" % (model, MODELS))
            sys.exit(1)

        if args.summarize:
            summarize_model(model, dry_run=args.dry_run)
        elif args.step2:
            run_step2_for_model(model, dry_run=args.dry_run)
        else:
            run_pipeline_for_model(model, dry_run=args.dry_run)
