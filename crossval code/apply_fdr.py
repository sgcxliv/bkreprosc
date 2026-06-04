#!/usr/bin/env python3
"""
apply_fdr.py  —  FDR correction matching the paper's family structure.

Method: Benjamini-Yekutieli (BY), valid under arbitrary test dependence.
  statsmodels.stats.multitest.fdrcorrection(method='negcorr')
"""

import os, sys
import numpy as np
import pandas as pd

try:
    from statsmodels.stats.multitest import fdrcorrection
except ImportError:
    def fdrcorrection(pvals, alpha=0.05, method='negcorr'):
        pvals = np.asarray(pvals, dtype=float)
        n = len(pvals)
        cm = np.sum(1.0 / np.arange(1, n + 1))
        order = np.argsort(pvals)
        adj = pvals[order] * n * cm / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.clip(adj, 0, 1)
        pvals_corrected = np.empty(n)
        pvals_corrected[order] = adj
        return pvals_corrected < alpha, pvals_corrected


def assign_family(row):
    """
    Assigns each comparison to a FDR family matching Table S1's
    horizontal-line groupings in the paper.
    """
    ctype = row['Comparison Type']
    left  = str(row['Left Model']).lower()

    # ── Family 1: Overall — null vs. model ───────────────────────────────────
    if ctype in ('Null vs. Base', 'Null vs. PROB',
                 'Null vs. Base (PT)', 'Null vs. PROB (PT)'):
        return 'Overall'

    # ── Family 2: Probability vs. Surprisal ──────────────────────────────────
    # Paper groups "surp vs. prob" AND "combined vs. single" in one FDR family
    if ctype in ('Base vs. PROB', 'Surp vs. Prob (PT)',
                 'Base vs. Combined', 'PROB vs. Combined',
                 'Both vs. Single (PT)'):
        return 'Probability vs. Surprisal'

    # ── Family 3: Cloze vs. Other ────────────────────────────────────────────
    # ClozeProb vs. other model surp (between-class, prob vs. surp comparisons)
    if ctype in ('Cloze Prob vs. Model Prob (PT)',
                 'Cloze Prob vs. Region Prob (PT)',
                 'Cloze Region Prob vs. Model Region Prob (PT)',
                 'ClozeProb vs. Model Surp (PT)',
                 'ClozeProb vs. Model Region Surp (PT)'):
        return 'Cloze vs. Other'

    # ── Family 4: Model Bakeoff ───────────────────────────────────────────────
    # Cloze surp vs. other model surp (single-predictor bakeoffs)
    if ctype in ('Model Bakeoff (PT)',
                 'Cloze Surp vs. Region Surp (PT)',
                 'Within-model Surp vs Region Surp (PT)'):
        return 'Model Bakeoff'

    # Fallback — preserve original type as its own family
    return ctype


def process_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"\n  [SKIP] File not found: {input_path}"); return
    df = pd.read_csv(input_path)
    orig_pvals = df['Raw p-value'].copy()
    orig_sig   = df['Significant'].copy()
    df['FDR Family'] = df.apply(assign_family, axis=1)
    for family, idx in df.groupby('FDR Family').groups.items():
        pvals = df.loc[idx, 'Raw p-value'].values.astype(float)
        rejected, pvals_corrected = fdrcorrection(pvals, alpha=0.05, method='negcorr')
        df.loc[idx, 'Raw p-value'] = pvals_corrected
        df.loc[idx, 'Significant'] = rejected
    def fmt_p(p):
        if p < 0.0001: return '<0.0001'
        elif p > 0.9999: return '>0.9999'
        return f'{p:.4f}'
    df['p_display'] = df['Raw p-value'].apply(fmt_p)
    df['Color'] = df.apply(
        lambda r: ('red' if r['ΔLL'] > 0 else 'blue') if r['Significant'] else 'black', axis=1
    )
    df['SignifColor'] = df['Color']
    df['Original p-value']     = orig_pvals
    df['Original Significant'] = orig_sig
    changed = df[df['Significant'] != df['Original Significant']]
    print(f"\n{'='*65}")
    print(f"  File:     {input_path}")
    print(f"  Tests: {len(df)}  |  Orig sig: {orig_sig.sum()}  ->  Post-FDR: {df['Significant'].sum()}  (changed: {len(changed)})")
    if len(changed):
        print("\n  Tests that changed significance:")
        for _, r in changed.iterrows():
            arrow = "sig -> NOT" if r['Original Significant'] else "NOT -> sig"
            print(f"    [{arrow}]  {r['Comparison']}")
            print(f"      Family: {r['FDR Family']}")
            print(f"      p: {r['Original p-value']:.4f} -> {r['Raw p-value']:.4f}")
    else:
        print("\n  No tests changed significance.")
    print("\n  Per-family (orig -> post-FDR):")
    for fam in ['Overall', 'Probability vs. Surprisal', 'Cloze vs. Other', 'Model Bakeoff']:
        sub = df[df['FDR Family'] == fam]
        if sub.empty: continue
        o, n = sub['Original Significant'].sum(), sub['Significant'].sum()
        print(f"    {fam}: {o} -> {n}{'  <- CHANGED' if o!=n else ''}")
    base_cols = [c for c in df.columns if c not in
                 ('FDR Family', 'Original p-value', 'Original Significant')]
    df[base_cols + ['FDR Family', 'Original p-value', 'Original Significant']].to_csv(
        output_path, index=False)
    print(f"\n  Saved -> {output_path}")


DEFAULT_PAIRS = [
    ('bkoresults.csv',    'bko_results_fdr.csv'),
    ('bkrresults.csv',    'bkr_results_fdr.csv'),
    ('all_results_raw.csv','all_results_summary.csv'),
]

if __name__ == '__main__':
    pairs = [(f, f.replace('.csv', '_fdr.csv')) for f in sys.argv[1:]] \
            if len(sys.argv) > 1 else DEFAULT_PAIRS
    for inp, out in pairs:
        process_file(inp, out)
    print("\nDone.")

