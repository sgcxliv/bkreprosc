#!/usr/bin/env python
"""
sum_lrt.py
==========
Reads all LRT summary CSVs from results/bk21/lrt/ and produces:
  - lrt_results_summary.csv   (consumed by sum_results.py)
  - lrt_results_report.txt    (human-readable)

Run AFTER all lrt_scripts/*.pbs jobs have completed.
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        results_path = next(l.strip() for l in f if l.strip())
else:
    results_path = 'results/bk21'

lrt_dir = os.path.join(results_path, 'lrt')

# ─────────────────────────────────────────────────────────────────────────────
# NAME FORMATTING
# IMPORTANT: replacements must go longest-match first to avoid partial
# substitution bugs (e.g. gpt2region → GPT-2region if gpt2 is replaced first).
# ─────────────────────────────────────────────────────────────────────────────

NAME_MAP = [
    # Longest strings first
    ('gpt2xlregion', 'GPT-2XL-Region'),
    ('gpt2xl',       'GPT-2XL'),
    ('gpt2region',   'GPT-2-Region'),
    ('gpt2',         'GPT-2'),
    ('gptjregion',   'GPT-J-Region'),
    ('gptj',         'GPT-J'),
    ('gptneoxregion','GPT-NeoX-Region'),
    ('gptneox',      'GPT-NeoX'),
    ('gptneoregion', 'GPT-Neo-Region'),
    ('gptneo',       'GPT-Neo'),
    ('olmoregion',   'OLMo-Region'),
    ('olmo',         'OLMo'),
    ('llama2region', 'Llama-2-Region'),
    ('llama2',       'Llama-2'),
    ('clozeregion',  'Cloze-Region'),
    ('cloze',        'Cloze'),
    ('nosurp',       'ø'),
    ('prob',         'PROB'),   # must come last
]


def format_name(raw):
    s = raw
    for old, new in NAME_MAP:
        s = s.replace(old, new)
    return s


def format_p(p):
    if p < 0.0001:
        return '<0.0001'
    if p >= 1.0:
        return '1.0000'
    return '%.4f' % p


def determine_type(model_a, model_b):
    """Classify nested comparison type from raw model names."""
    if model_a == 'nosurp':
        return 'Null vs. PROB' if model_b.endswith('prob') else 'Null vs. Base'
    if model_b == model_a + 'prob':
        return 'Base vs. PROB'
    if model_a.endswith('prob') and model_b.startswith(model_a) and '-' in model_b:
        return 'PROB vs. Combined'
    if not model_a.endswith('prob') and '-' in model_b:
        return 'Base vs. Combined'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(lrt_dir):
    print("ERROR: LRT results directory not found: %s" % lrt_dir)
    print("Have you run all lrt_scripts/*.pbs jobs?")
    raise SystemExit(1)

files = sorted(f for f in os.listdir(lrt_dir) if f.endswith('_summary.csv'))
if not files:
    print("No _summary.csv files found in %s" % lrt_dir)
    raise SystemExit(1)

print("Found %d LRT summary files." % len(files))

results = []
errors  = []
for fname in files:
    test_name = fname.replace('_summary.csv', '')
    fpath     = os.path.join(lrt_dir, fname)
    try:
        row = pd.read_csv(fpath).iloc[0]

        model_a    = str(row['model_a'])
        model_b    = str(row['model_b'])
        loglik_a   = float(row['loglik_a'])
        loglik_b   = float(row['loglik_b'])
        difference = loglik_b - loglik_a   # always B - A; positive = B better
        p_value    = float(row['p_value'])
        significant = bool(row['significant'])

        comp_type = determine_type(model_a, model_b)
        has_region = 'region' in model_a.lower() or 'region' in model_b.lower()

        color = 'black'
        if significant:
            color = 'red' if difference > 0 else 'blue'

        results.append({
            'Comparison Type':  comp_type,
            'Has Region':       has_region,
            'Comparison':       '%s vs. %s' % (format_name(model_a), format_name(model_b)),
            'Raw Comparison':   '%s vs. %s' % (model_a, model_b),
            'Left Model':       model_a,
            'Right Model':      model_b,
            'ΔLL':              round(difference, 2),
            'ΔLL_display':      str(int(round(difference))),
            'p_display':        format_p(p_value),
            'Significant':      significant,
            'Raw p-value':      p_value,
            'Color':            color,
            'SignifColor':      color,
            'Test Type':        'LRT (Likelihood Ratio Test)',
        })
    except Exception as e:
        errors.append((fname, str(e)))
        print("  ERROR processing %s: %s" % (fname, e))

if errors:
    print("\n%d files failed to parse." % len(errors))

if not results:
    print("No valid results to save.")
    raise SystemExit(1)

df = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SORT
# ─────────────────────────────────────────────────────────────────────────────

TYPE_ORDER = [
    'Null vs. Base',
    'Null vs. PROB',
    'Base vs. PROB',
    'Base vs. Combined',
    'PROB vs. Combined',
    'Other',
]
df['_type_ord'] = df['Comparison Type'].apply(
    lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99
)
df = df.sort_values(['_type_ord', 'Has Region', 'Significant', 'Raw p-value'],
                    ascending=[True, True, False, True]).drop(columns=['_type_ord'])

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

out_csv = 'lrt_results_summary.csv'
df.to_csv(out_csv, index=False)
print("\nSaved: %s  (%d rows)" % (out_csv, len(df)))

# Human-readable report
with open('lrt_results_report.txt', 'w') as f:
    f.write("Likelihood Ratio Test Results\n")
    f.write("=" * 60 + "\n\n")
    for t in TYPE_ORDER:
        sub = df[df['Comparison Type'] == t]
        if sub.empty:
            continue
        f.write("## %s  (%d comparisons, %d significant)\n" % (
            t, len(sub), sub['Significant'].sum()))
        for _, row in sub.iterrows():
            sig_str = '***' if row['Raw p-value'] <= 0.001 else \
                      '**'  if row['Raw p-value'] <= 0.01  else \
                      '*'   if row['Raw p-value'] <= 0.05  else ''
            f.write("  %s\t ΔLL=%s\t p=%s %s\n" % (
                row['Comparison'], row['ΔLL_display'], row['p_display'], sig_str))
        f.write("\n")
print("Saved: lrt_results_report.txt")

# Print summary to stdout
print("\nSummary by comparison type:")
for t in TYPE_ORDER:
    sub = df[df['Comparison Type'] == t]
    if sub.empty:
        continue
    print("  %-30s %3d comparisons  %3d significant" % (
        t, len(sub), sub['Significant'].sum()))
