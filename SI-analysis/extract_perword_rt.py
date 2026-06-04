"""
extract_perword_rt.py
---------------------
Extracts per-word RTs at W1, W2, W3 of the critical region from word.csv
and merges them onto the existing bkr21_spr.csv (or bko21_spr.csv).

Usage:
    python extract_perword_rt.py word.csv bkr21_spr.csv bkr21_spr_withRT.csv
    python extract_perword_rt.py word_bko.csv bko21_spr.csv bko21_spr_withRT.csv
"""

import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 4:
    print("Usage: python extract_perword_rt.py <word.csv> <spr.csv> <output.csv>")
    sys.exit(1)

word_file = sys.argv[1]
spr_file = sys.argv[2]
out_file = sys.argv[3]

# Load word-level data
word = pd.read_csv(word_file)
print(f"word.csv: {len(word)} rows, {word['SUB'].nunique()} subjects, {word['ITEM'].nunique()} items")

# Load item-level SPR data (one row per SUB x ITEM, already has critical_word_pos)
spr = pd.read_csv(spr_file)
print(f"spr.csv: {len(spr)} rows")

# Get critical word position per (ITEM, condition) from spr
# critical_word_pos is 1-indexed wordpos of the critical word
crit_pos = spr[['ITEM', 'condition', 'critical_word_pos']].drop_duplicates()
print(f"Unique (ITEM, condition) pairs with critical_word_pos: {len(crit_pos)}")

# Merge critical_word_pos onto word data
word = word.merge(crit_pos, on=['ITEM', 'condition'], how='inner')

# Compute offset from critical word: W1=0, W2=1, W3=2
word['cr_offset'] = word['wordpos'] - word['critical_word_pos']

# Keep only the 3-word critical region
cr = word[word['cr_offset'].isin([0, 1, 2])].copy()
cr['cr_offset'] = cr['cr_offset'] + 1  # W1=1, W2=2, W3=3
print(f"Critical region rows: {len(cr)}")

# Pivot to wide: one row per (SUB, ITEM) with RT_W1, RT_W2, RT_W3
cr_wide = cr.pivot_table(
    index=['SUB', 'ITEM', 'condition'],
    columns='cr_offset',
    values='RT',
    aggfunc='first'
).reset_index()

cr_wide.columns = ['SUB', 'ITEM', 'condition', 'RT_W1', 'RT_W2', 'RT_W3']
print(f"Wide rows (SUB x ITEM): {len(cr_wide)}")
print(f"  RT_W1 non-null: {cr_wide['RT_W1'].notna().sum()}")
print(f"  RT_W2 non-null: {cr_wide['RT_W2'].notna().sum()}")
print(f"  RT_W3 non-null: {cr_wide['RT_W3'].notna().sum()}")

# Merge onto spr
spr_out = spr.merge(cr_wide[['SUB', 'ITEM', 'condition', 'RT_W1', 'RT_W2', 'RT_W3']],
                     on=['SUB', 'ITEM', 'condition'],
                     how='left')

print(f"\nOutput: {len(spr_out)} rows")
print(f"  RT_W1 non-null: {spr_out['RT_W1'].notna().sum()}")
print(f"  RT_W2 non-null: {spr_out['RT_W2'].notna().sum()}")
print(f"  RT_W3 non-null: {spr_out['RT_W3'].notna().sum()}")

spr_out.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
