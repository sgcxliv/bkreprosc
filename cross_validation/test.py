#!/usr/bin/env python
"""
test.py
=======
Runs a permutation test comparing two LME models on held-out log-likelihoods.

Matches the paper's methodology: Shain et al. PNAS 2024, SI Section 1B.
LME outputs (lme_<experiment>_<name>_output_test.csv) are used for all
permutation tests. GAM outputs are not used.

Usage:
    python test.py <experiment> <modelA_name>_v_<modelB_name>

Example:
    python test.py spr cloze_v_gpt2
    python test.py spr cloze_v_clozeprob-cloze
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cdr.signif import permutation_test
from cdr.util import stderr


# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# LME outputs are used for permutation tests, matching the paper's methodology.
# GAM outputs are a fallback only (prints a warning if used).
# ─────────────────────────────────────────────────────────────────────────────

def get_model_path(name, experiment, results_path):
    """
    Resolve the output CSV for a given model name.

    Priority: lme_ prefix (matches paper) > gam_ prefix (fallback).
    Strips trailing '1.00' artifacts from old naming conventions.
    Tries no-controls version first, then with-controls (C suffix).
    """
    name = name.replace('1.00', '')

    # LME first — matches paper methodology
    for suffix in ['', 'C']:
        lme_path = os.path.join(
            results_path, 'lme_%s_%s%s_output_test.csv' % (experiment, name, suffix)
        )
        if os.path.exists(lme_path):
            return lme_path

    # GAM fallback
    for suffix in ['', 'C']:
        gam_path = os.path.join(
            results_path, 'gam_%s_%s%s_output_test.csv' % (experiment, name, suffix)
        )
        if os.path.exists(gam_path):
            stderr('WARNING: Using GAM output for %s — '
                   'LME output not found. Results may differ from paper.\n' % name)
            return gam_path

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Results path
    if os.path.exists('bk21_results_path.txt'):
        with open('bk21_results_path.txt') as f:
            results_path = next(l.strip() for l in f if l.strip())
    else:
        results_path = 'results/bk21'

    parser = argparse.ArgumentParser(
        description='Permutation test comparing two GAM models on held-out log-likelihood'
    )
    parser.add_argument('experiment', help='Experiment name (spr or naming)')
    parser.add_argument('test_name',
                        help='Comparison name: <modelA>_v_<modelB>')
    args = parser.parse_args()

    experiment = args.experiment
    test_name  = args.test_name

    a_name, b_name = test_name.split('_v_')

    # Resolve file paths
    a_path = get_model_path(a_name, experiment, results_path)
    b_path = get_model_path(b_name, experiment, results_path)

    if a_path is None:
        stderr('ERROR: Could not find output file for model A: %s\n' % a_name)
        sys.exit(1)
    if b_path is None:
        stderr('ERROR: Could not find output file for model B: %s\n' % b_name)
        sys.exit(1)

    stderr('Model A path: %s\n' % a_path)
    stderr('Model B path: %s\n' % b_path)

    # Load log-likelihoods
    df_a = pd.read_csv(a_path)
    df_b = pd.read_csv(b_path)

    if 'loglik' not in df_a.columns:
        stderr('ERROR: No "loglik" column in %s\n' % a_path)
        sys.exit(1)
    if 'loglik' not in df_b.columns:
        stderr('ERROR: No "loglik" column in %s\n' % b_path)
        sys.exit(1)

    # Align on sortix if available (ensures row correspondence).
    # Models may have different row counts if some observations were dropped
    # during fitting (e.g. missing values in region predictors).
    # We do an inner join on sortix so only observations present in BOTH
    # models are compared — this is the only valid approach.
    if 'sortix' in df_a.columns and 'sortix' in df_b.columns:
        df_a = df_a.sort_values('sortix').reset_index(drop=True)
        df_b = df_b.sort_values('sortix').reset_index(drop=True)
        if len(df_a) != len(df_b) or \
                not (df_a['sortix'].values == df_b['sortix'].values).all():
            n_a, n_b = len(df_a), len(df_b)
            shared = pd.merge(
                df_a[['sortix', 'loglik']],
                df_b[['sortix', 'loglik']],
                on='sortix',
                suffixes=('_a', '_b')
            )
            stderr('WARNING: row count mismatch (A=%d, B=%d). '
                   'Inner join on sortix gives %d shared observations.\n'
                   % (n_a, n_b, len(shared)))
            if len(shared) == 0:
                stderr('ERROR: No shared observations between A and B.\n')
                sys.exit(1)
            a = shared['loglik_a'].values[:, None]
            b = shared['loglik_b'].values[:, None]
        else:
            a = df_a['loglik'].values[:, None]
            b = df_b['loglik'].values[:, None]
    else:
        a = df_a['loglik'].values[:, None]
        b = df_b['loglik'].values[:, None]

    # Drop rows where either model has non-finite LL
    sel = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    dropped = (~sel).sum()
    if dropped > 0:
        stderr('Dropping %d rows with non-finite log-likelihoods.\n' % dropped)
    a = a[sel]
    b = b[sel]

    if len(a) == 0:
        stderr('ERROR: No valid observations remain after filtering.\n')
        sys.exit(1)

    assert len(a) == len(b), 'Length mismatch: %d vs %d' % (len(a), len(b))

    # Output directory
    outdir    = os.path.join(results_path, 'signif', experiment)
    os.makedirs(outdir, exist_ok=True)
    name_base = '%s_PT_rt_test' % test_name
    out_path  = os.path.join(outdir, name_base + '.txt')

    # Run permutation test
    # mode='loglik': compares sums of log-likelihoods
    # agg='median':  uses median across ensemble members (here just 1 per model)
    # nested=False:  does not assume nestedness
    #
    # NOTE: cdr.signif.permutation_test may return either 3 or 5 values
    # depending on the installed version of the CDR library:
    #   5-value: (p_value, a_perf, b_perf, diff, diffs)
    #   3-value: (p_value, diff, diffs)
    _result = permutation_test(
        a, b,
        mode='loglik',
        agg='median',
        nested=False
    )
    if len(_result) == 5:
        p_value, a_perf, b_perf, diff, diffs = _result
    elif len(_result) == 3:
        p_value, diff, diffs = _result
        a_perf = float(np.sum(a))
        b_perf = float(np.sum(b))
    else:
        raise ValueError(
            'Unexpected permutation_test return signature: %d values. '
            'Expected 3 or 5.' % len(_result)
        )
    stderr('\n')

    stars = ('' if p_value > 0.05 else
             '*'   if p_value > 0.01 else
             '**'  if p_value > 0.001 else
             '***')

    summary = '=' * 50 + '\n'
    summary += 'Permutation Test\n'
    summary += 'Model A: %s\n' % a_name
    summary += 'Model B: %s\n' % b_name
    summary += 'Experiment: %s\n' % experiment
    summary += 'Metric: loglik\n'
    summary += 'Model A path: %s\n' % a_path
    summary += 'Model B path: %s\n' % b_path
    summary += 'N:            %d\n' % a.shape[0]
    if dropped:
        summary += 'N dropped:    %d\n' % dropped
    summary += 'Model A:      %.4f\n' % a_perf
    summary += 'Model B:      %.4f\n' % b_perf
    summary += 'Difference:   %.4f\n' % diff   # positive = B better
    summary += 'p:            %.4e%s\n' % (p_value, stars)
    summary += '=' * 50 + '\n'

    with open(out_path, 'w') as f:
        f.write(summary)
    sys.stdout.write(summary)
    stderr('Saved to %s\n' % out_path)

    # Histogram of permuted differences
    plt.figure()
    plt.hist(diffs, bins=1000)
    plt.axvline(diff, color='red', linestyle='--', label='Observed diff')
    plt.xlabel('Permuted ΔLogLik')
    plt.ylabel('Count')
    plt.title('%s vs %s' % (a_name, b_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, name_base + '.png'))
    plt.close('all')
