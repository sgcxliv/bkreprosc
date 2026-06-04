"""
add_perword_rt.py
-----------------
Extracts per-word RTs at W1, W2, W3 from raw ibex data and merges
them onto existing bkr21_spr.csv.

Does NOT redo the full initialize.py pipeline — just adds RT_W1, RT_W2, RT_W3.

Usage:
    python add_perword_rt.py

Expects in current directory:
    - ibex/          (raw ibex output files)
    - bkr21_spr.csv  (existing item-level dataset)

Outputs:
    - bkr21_spr.csv  (overwritten with RT_W1, RT_W2, RT_W3 added)
"""

import os
import re
import csv
import time
import pandas as pd

HEADER = re.compile(r'.*# (\d+)\. (.+)\.')
NAME_MAP = {
    'Order number of item': 'ITEM',
    "MD5 hash of participant's IP address": 'SUB',
    'Value': 'word',
    'Parameter': 'wordpos',
    'EventTime': 'time',
    'Reading time': 'RT'
}

# ── Step 1: Scrape ibex files for word-level RTs ─────────────────────

print("Scraping ibex files...")
dataset = []
ibex_files = [p for p in sorted(os.listdir('ibex')) if os.path.isfile(os.path.join('ibex', p))]
total_files = len(ibex_files)
start = time.time()

for idx, path in enumerate(ibex_files, start=1):
    filepath = os.path.join('ibex', path)
    file_start = time.time()
    trials_before = len(dataset)
    with open(filepath, 'rb') as fb:
        raw = fb.read().replace(b'\x00', b'').decode('utf-8', errors='replace')
    reader = csv.reader(raw.splitlines())
    headers = []
    item = []
    question_result = None
    question_time = None
    for line in reader:
        if len(line):
            if line[0].startswith('#'):
                res = HEADER.match(line[0])
                if res:
                    ix, col = res.groups()
                    ix = int(ix) - 1
                    headers = headers[:ix]
                    headers.insert(ix, col)
            else:
                row = dict(zip(headers, line))
                if row.get('PennElementType') == 'PennController':
                    if len(item):
                        item = pd.DataFrame(item)
                        item['correct'] = question_result
                        item['question_response_timestamp'] = question_time
                        dataset.append(item)
                    question_result = None
                    question_time = None
                    item = []
                elif row.get('PennElementType') in {'Controller-DashedSentence', 'Controller-SPR'}:
                    item.append(row)
                elif row.get('PennElementType') == 'Selector':
                    question_result = 'is_correct'
                    question_time = row.get('EventTime')
    file_elapsed = time.time() - file_start
    total_elapsed = time.time() - start
    file_trials = len(dataset) - trials_before
    print(
        f"  [{idx:>2}/{total_files}] {path}: +{file_trials:,} trials "
        f"({len(dataset):,} total; file {file_elapsed:.1f}s, total {total_elapsed/60:.1f}m)"
    )

print(f"  Scraped {len(dataset)} trials")
dataset = pd.concat(dataset, axis=0)
dataset = dataset.rename(NAME_MAP, axis=1)

# Fix types
dataset['ITEM'] = pd.to_numeric(dataset['ITEM'], errors='coerce')
dataset['wordpos'] = pd.to_numeric(dataset['wordpos'], errors='coerce')
dataset['RT'] = pd.to_numeric(dataset['RT'], errors='coerce')

# Apply BK ITEM offset to align with bkr21_spr.csv numbering (5..220)
dataset['ITEM'] = dataset['ITEM'] - 1 + 5

print(f"  Total word-level rows: {len(dataset)}")
print(f"  Subjects: {dataset['SUB'].nunique()}")
print(f"  Items: {dataset['ITEM'].nunique()}")

# ── Step 2: Load existing SPR data to get critical_word_pos ──────────

spr = pd.read_csv('bkr21_spr.csv')
print(f"\nbkr21_spr.csv: {len(spr)} rows")

# Get critical word position per (ITEM, condition)
crit_info = spr[['ITEM', 'condition', 'critical_word_pos']].drop_duplicates()
print(f"  Unique (ITEM, condition): {len(crit_info)}")

# We also need condition per (SUB, ITEM) to join properly
# Get list assignment from spr
sub_item_cond = spr[['SUB', 'ITEM', 'condition']].drop_duplicates()

# ── Step 3: Merge and compute critical offset ────────────────────────

# Join condition onto word-level data via (SUB, ITEM)
word_data = dataset.merge(sub_item_cond, on=['SUB', 'ITEM'], how='inner')
print(f"\nWord rows matched to SPR: {len(word_data)}")

# Join critical_word_pos
word_data = word_data.merge(crit_info, on=['ITEM', 'condition'], how='left')

# Compute offset: W1=0, W2=1, W3=2
word_data['cr_offset'] = word_data['wordpos'] - word_data['critical_word_pos']
cr = word_data[word_data['cr_offset'].isin([0, 1, 2])].copy()
print(f"Critical region rows (offsets 0,1,2): {len(cr)}")

# Pivot to wide
cr['cr_label'] = cr['cr_offset'].map({0: 'RT_W1', 1: 'RT_W2', 2: 'RT_W3'})
cr_wide = cr.pivot_table(
    index=['SUB', 'ITEM', 'condition'],
    columns='cr_label',
    values='RT',
    aggfunc='first'
).reset_index()

# Ensure column order
for col in ['RT_W1', 'RT_W2', 'RT_W3']:
    if col not in cr_wide.columns:
        cr_wide[col] = pd.NA

print(f"Wide rows: {len(cr_wide)}")
print(f"  RT_W1 non-null: {cr_wide['RT_W1'].notna().sum()}")
print(f"  RT_W2 non-null: {cr_wide['RT_W2'].notna().sum()}")
print(f"  RT_W3 non-null: {cr_wide['RT_W3'].notna().sum()}")

# ── Step 4: Merge onto SPR and save ─────────────────────────────────

# Drop existing RT_W columns if they exist
for col in ['RT_W1', 'RT_W2', 'RT_W3']:
    if col in spr.columns:
        spr = spr.drop(columns=[col])

spr_out = spr.merge(
    cr_wide[['SUB', 'ITEM', 'condition', 'RT_W1', 'RT_W2', 'RT_W3']],
    on=['SUB', 'ITEM', 'condition'],
    how='left'
)

print(f"\nFinal output: {len(spr_out)} rows")
print(f"  RT_W1 non-null: {spr_out['RT_W1'].notna().sum()}")
print(f"  RT_W2 non-null: {spr_out['RT_W2'].notna().sum()}")
print(f"  RT_W3 non-null: {spr_out['RT_W3'].notna().sum()}")

# Sanity check: RT_W1 should match existing RT column
if 'RT' in spr_out.columns:
    match = (spr_out['RT_W1'] == spr_out['RT']).mean()
    print(f"  RT_W1 matches existing RT column: {match:.1%}")

spr_out.to_csv('bkr21_spr.csv', index=False)
print("\nSaved: bkr21_spr.csv (with RT_W1, RT_W2, RT_W3)")
