#!/usr/bin/env python
"""
make_agree_subsets.py
=====================
Creates filtered versions of bk21_spr.csv containing only rows where the
LLM's within-item probability ranking matches the experimental `condition`
(HC / MC / LC) — row-level agree subset.

Each output CSV has the same columns as bk21_spr.csv

Usage:
    python make_agree_subsets.py

Reads:
    bk21_spr.csv                  — full SPR table (long or wide)

Writes:
    bk21_data/agree_<model>_spr.csv   for each model
    bk21_data/agree_all_spr.csv
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SPR_PATH     = 'bk21_spr.csv'
OUT_DIR      = 'bk21_data'

# Models 
MODELS = ['gpt2', 'gpt2xl', 'gptj', 'gptneo', 'gptneox', 'olmo', 'llama2']

N_FOLDS = 5

# Column names 
PROB_COL_BY_MODEL = {
    'gpt2':    'gpt2prob',
    'gpt2xl':  'gpt2xlprob',
    'gptj':    'gptjprob',
    'gptneo':  'gptneoprob',
    'gptneox': 'gptneoxprob',
    'olmo':    'olmoprob',
    'llama2':  'llama2prob',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading %s..." % SPR_PATH)
spr = pd.read_csv(SPR_PATH, low_memory=False)
print("  %d rows, %d items" % (len(spr), spr['ITEM'].nunique()))

if 'condition' not in spr.columns:
    raise ValueError("Expected a `condition` column (HC / MC / LC) in %s" % SPR_PATH)


def _norm_cond(x):
    """Normalize condition labels for comparison."""
    if pd.isna(x):
        return None
    return str(x).strip().upper()

missing_prob_cols = [PROB_COL_BY_MODEL[m] for m in MODELS if PROB_COL_BY_MODEL[m] not in spr.columns]
if missing_prob_cols:
    raise ValueError("Missing probability columns in bk21_spr.csv: %s"
                     % ", ".join(missing_prob_cols))

# Row-level agree flags (matches the logic in `direct_unclean.py`):
agree_cols = []
for model in MODELS:
    col = 'agree_%s' % model
    spr[col] = False
    agree_cols.append(col)

items = spr['ITEM'].unique()
n_skipped_items = 0
print("\nComputing row-level agree flags for %d models (long-format safe)..." % len(MODELS))
for item in items:
    g_item = spr.loc[spr['ITEM'] == item]
    by_cond = g_item.groupby('condition', sort=False).first().reset_index()
    n_cond = len(by_cond)
    if n_cond != 3:
        print(
            "WARNING: ITEM %s has %d distinct `condition` values (expected 3). Skipping."
            % (item, n_cond)
        )
        n_skipped_items += 1
        continue

    if 'position' in by_cond.columns:
        by_cond = by_cond.sort_values('position').reset_index(drop=True)

    for model in MODELS:
        prob_col = PROB_COL_BY_MODEL[model]
        probs = by_cond[prob_col].astype(float).values

        if np.any(~np.isfinite(probs)):
            col = 'agree_%s' % model
            for _, crow in by_cond.iterrows():
                c_raw = crow['condition']
                spr.loc[(spr['ITEM'] == item) & (spr['condition'] == c_raw), col] = False
            continue

        highest_idx_local = int(np.argmax(probs))
        lowest_idx_local = int(np.argmin(probs))
        if highest_idx_local == lowest_idx_local:
            middle_idx_local = [i for i in range(3) if i != highest_idx_local][0]
        else:
            middle_idx_local = list(set(range(3)) - {highest_idx_local, lowest_idx_local})[0]

        implied_label_by_row_idx = {
            highest_idx_local: 'HC',
            middle_idx_local: 'MC',
            lowest_idx_local: 'LC',
        }
        col = 'agree_%s' % model

        for i in range(3):
            c_raw = by_cond.iloc[i]['condition']
            exp_lab = _norm_cond(c_raw)
            mod_lab = implied_label_by_row_idx[i]
            agree_here = (mod_lab == exp_lab) and pd.notna(by_cond.iloc[i][prob_col])
            spr.loc[(spr['ITEM'] == item) & (spr['condition'] == c_raw), col] = agree_here

if n_skipped_items:
    print("\nSkipped %d / %d items (not exactly 3 conditions)." % (n_skipped_items, len(items)))

# All-model agree: all models agree on that specific row.
spr['agree_all'] = spr[agree_cols].all(axis=1)

print("\nRow-level agree row counts:")
for col in agree_cols + ['agree_all']:
    n = int(spr[col].sum())
    print("  %-20s %d / %d rows  (%.1f%%)" % (col, n, len(spr), 100 * n / len(spr)))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: reassign folds within a subset
# ─────────────────────────────────────────────────────────────────────────────

def assign_folds(df, n_folds=5):
    """
    Reassign fold column within a subset.
    Uses mod(rank_of_ITEM, n_folds) + 1, matching the paper's approach
    of cycling items into folds based on their numerical IDs.
    Fold assignment is at the ITEM level so all rows of an item are in
    the same fold.
    """
    df = df.copy()
    items = sorted(df['ITEM'].unique())
    item_to_fold = {item: (i % n_folds) + 1 for i, item in enumerate(items)}
    df['fold'] = df['ITEM'].map(item_to_fold)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CREATE SUBSETS AND SAVE
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
outputs = {}

# Per-model subsets
for model in MODELS:
    col = 'agree_%s' % model
    subset = spr[spr[col]].copy()
    subset = assign_folds(subset, N_FOLDS)

    out_path = os.path.join(OUT_DIR, 'agree_%s_spr.csv' % model)
    subset = subset.drop(columns=[c for c in (agree_cols + ['agree_all']) if c in subset.columns],
                           errors='ignore')
    subset.to_csv(out_path, index=False)
    outputs[model] = out_path

    print("\nagree_%s: %d items, %d rows -> %s" % (
        model, subset['ITEM'].nunique(), len(subset), out_path))
    print("  Fold distribution: %s" % dict(subset.groupby('fold').size()))

# All-model agree subset (row-level)
subset_all = spr[spr['agree_all']].copy()
subset_all = assign_folds(subset_all, N_FOLDS)

out_all = os.path.join(OUT_DIR, 'agree_all_spr.csv')
subset_all = subset_all.drop(columns=[c for c in (agree_cols + ['agree_all']) if c in subset_all.columns],
                               errors='ignore')
subset_all.to_csv(out_all, index=False)
outputs['all'] = out_all

print("\nagree_all: %d items, %d rows -> %s" % (
    subset_all['ITEM'].nunique(), len(subset_all), out_all))
print("  Fold distribution: %s" % dict(subset_all.groupby('fold').size()))
