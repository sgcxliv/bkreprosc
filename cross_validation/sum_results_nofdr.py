#!/usr/bin/env python
"""
sum_results_nofdr.py
====================
Merges LRT results (from sum_lrt.py) and permutation test results
(from test.py output files) into a single analysis-ready CSV:
    all_results_raw.csv

NO FDR correction is applied — significance flags reflect raw p < 0.05.
Run apply_fdr.py on the output to apply BY correction afterwards:
    python apply_fdr.py all_results_raw.csv

COLOR CODING
------------
  Red   (positive ΔLL): Model B wins  (right-hand model is better)
  Blue  (negative ΔLL): Model A wins  (left-hand model is better)
  Black:                Not significant (raw p >= 0.05)

CRITICAL DESIGN NOTE
---------------------
LRT results and PT results are NEVER mixed in significance assessments:
  - LRT answers: "does adding predictor X improve fit above chance, given df?"
  - PT  answers: "is model B's held-out LL reliably higher than model A's?"

Run after:
  python sum_lrt.py
  (all test_scripts/*.pbs permutation test jobs have completed)
"""

import os
import re
import sys
import numpy as np
import pandas as pd



# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        results_path = next(l.strip() for l in f if l.strip())
else:
    results_path = 'results/bk21'

LRT_INPUT   = 'lrt_results_summary.csv'
PT_DIR      = os.path.join(results_path, 'signif', 'spr')
EXPERIMENT  = 'spr'

MODELS = [
    'cloze',    'clozeregion',
    'gpt2',     'gpt2region',
    'gpt2xl',   'gpt2xlregion',
    'gptj',     'gptjregion',
    'gptneo',   'gptneoregion',
    'gptneox',  'gptneoxregion',
    'olmo',     'olmoregion',
    'llama2',   'llama2region',
]
NON_REGION = [m for m in MODELS if 'region' not in m]
REGION     = [m for m in MODELS if 'region' in m]
REGION_MAP = {
    'cloze':   'clozeregion',  'gpt2':   'gpt2region',
    'gpt2xl':  'gpt2xlregion', 'gptj':   'gptjregion',
    'gptneo':  'gptneoregion', 'gptneox':'gptneoxregion',
    'olmo':    'olmoregion',   'llama2': 'llama2region',
}

# ─────────────────────────────────────────────────────────────────────────────
# NAME FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

NAME_MAP = [
    ('gpt2xlregion',  'GPT-2XL-Region'), ('gpt2xl',    'GPT-2XL'),
    ('gpt2region',    'GPT-2-Region'),   ('gpt2',      'GPT-2'),
    ('gptjregion',    'GPT-J-Region'),   ('gptj',      'GPT-J'),
    ('gptneoxregion', 'GPT-NeoX-Region'),('gptneox',   'GPT-NeoX'),
    ('gptneoregion',  'GPT-Neo-Region'), ('gptneo',    'GPT-Neo'),
    ('olmoregion',    'OLMo-Region'),    ('olmo',      'OLMo'),
    ('llama2region',  'Llama-2-Region'), ('llama2',    'Llama-2'),
    ('clozeregion',   'Cloze-Region'),   ('cloze',     'Cloze'),
    ('nosurp',        'ø'),
    ('prob',          'PROB'),
]


def fmt(raw):
    s = str(raw)
    for old, new in NAME_MAP:
        s = s.replace(old, new)
    return s


def fmt_p(p):
    if p < 0.0001:
        return '<0.0001'
    if p > 0.9999:
        return '>0.9999'
    return '%.4f' % p






# ─────────────────────────────────────────────────────────────────────────────
# PT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def determine_pt_type(model_a, model_b):
    within_pairs = {b: r for b, r in REGION_MAP.items()}

    if model_a == 'nosurp':
        return 'Null vs. PROB' if model_b.endswith('prob') else 'Null vs. Base'
    if model_a + 'prob' == model_b and 'region' not in model_a:
        return 'Surp vs. Prob (PT)'
    if (model_a in within_pairs and model_b == within_pairs[model_a]
            and not model_a.endswith('prob') and not model_b.endswith('prob')):
        return 'Within-model Surp vs Region Surp (PT)'
    if model_a == 'cloze' and 'region' in model_b and not model_b.endswith('prob'):
        return 'Cloze Surp vs. Region Surp (PT)'
    if model_a == 'clozeprob' and 'region' in model_b and model_b.endswith('prob'):
        return 'Cloze Prob vs. Region Prob (PT)'
    if (model_a == 'clozeprob' and 'region' not in model_b
            and model_b.endswith('prob') and model_b != 'clozeprob'):
        return 'Cloze Prob vs. Model Prob (PT)'
    if (model_a == 'clozeregionprob' and 'region' in model_b
            and model_b.endswith('prob') and model_b != 'clozeregionprob'):
        return 'Cloze Region Prob vs. Model Region Prob (PT)'
    if model_a == 'cloze' and 'region' not in model_b and model_b != 'cloze':
        return 'Model Bakeoff (PT)'
    if model_a == 'clozeregion' and 'region' in model_b and model_b != 'clozeregion':
        return 'Model Bakeoff (PT)'
    # ClozeProb vs. model surprisal (non-nested, cross-type)
    if (model_a == 'clozeprob' and not model_b.endswith('prob')
            and 'region' not in model_b and model_b != 'cloze'):
        return 'ClozeProb vs. Model Surp (PT)'
    if (model_a == 'clozeprob' and not model_b.endswith('prob')
            and 'region' in model_b):
        return 'ClozeProb vs. Model Region Surp (PT)'
    # Surp vs. prob including combined
    if '-' in model_b:
        return 'Both vs. Single (PT)'
    return 'Model Bakeoff (PT)'


def parse_pt_file(comparison, pt_dir, experiment):
    """Parse a permutation test output text file."""
    # Try both naming conventions
    candidates = [
        os.path.join(pt_dir, '%s_PT_rt_test.txt' % comparison),
        os.path.join(pt_dir, 'bk21_%s_%s_PT_rt_test.txt' % (experiment, comparison)),
    ]
    fpath = next((p for p in candidates if os.path.exists(p)), None)
    if fpath is None:
        return None

    try:
        content = open(fpath).read()
    except Exception:
        return None

    parts = comparison.split('_v_')
    if len(parts) != 2:
        return None
    model_a, model_b = parts

    m_a   = re.search(r'Model A:\s+(-?[\d.]+(?:e[+-]?\d+)?)', content)
    m_b   = re.search(r'Model B:\s+(-?[\d.]+(?:e[+-]?\d+)?)', content)
    m_d   = re.search(r'Difference:\s+(-?[\d.]+(?:e[+-]?\d+)?)', content)
    m_p   = re.search(r'p:\s+([\d.]+(?:e[+-]?\d+)?)([\*]*)', content)

    if not all([m_a, m_b, m_d, m_p]):
        print("  WARNING: Could not parse %s" % fpath)
        return None

    perf_a  = float(m_a.group(1))
    perf_b  = float(m_b.group(1))
    p_value = float(m_p.group(1))
    diff    = perf_b - perf_a   # always B - A

    has_region = 'region' in comparison.lower()
    comp_type  = determine_pt_type(model_a, model_b)

    color = 'black'
    if p_value < 0.05:
        color = 'red' if diff > 0 else 'blue'

    return {
        'Comparison Type': comp_type,
        'Has Region':      has_region,
        'Comparison':      '%s vs. %s' % (fmt(model_a), fmt(model_b)),
        'Raw Comparison':  comparison,
        'Left Model':      model_a,
        'Right Model':     model_b,
        'ΔLL':             round(diff, 4),
        'ΔLL_display':     '%.1f' % diff,
        'p_display':       fmt_p(p_value),
        'Significant':     p_value < 0.05,
        'Raw p-value':     p_value,
        'Color':           color,
        'SignifColor':     color,
        'Test Type':       'PT (Permutation Test)',
    }


def generate_pt_comparisons():
    """Mirror the comparison list from make_test_jobs.py."""
    comps = []
    for m in MODELS:
        comps.append('nosurp_v_%s' % m)
        comps.append('nosurp_v_%sprob' % m)
    for m in NON_REGION:
        comps.append('%s_v_%sprob' % (m, m))
    for m in MODELS:
        comps.append('%s_v_%sprob-%s' % (m, m, m))
        comps.append('%sprob_v_%sprob-%s' % (m, m, m))
    for m in NON_REGION:
        if m != 'cloze':
            comps.append('cloze_v_%s' % m)
    for m in REGION:
        if m != 'clozeregion':
            comps.append('clozeregion_v_%s' % m)
    for m in REGION:
        comps.append('cloze_v_%s' % m)
    for base, reg in REGION_MAP.items():
        if base in NON_REGION and reg in REGION:
            comps.append('%s_v_%s' % (base, reg))
    for m in REGION:
        comps.append('clozeprob_v_%sprob' % m)
    for m in NON_REGION:
        if m != 'cloze':
            comps.append('clozeprob_v_%sprob' % m)
    for m in REGION:
        if m != 'clozeregion':
            comps.append('clozeregionprob_v_%sprob' % m)
    # 10a. ClozeProb vs. model word surprisal (non-region, excl. cloze itself)
    for m in NON_REGION:
        if m != 'cloze':
            comps.append('clozeprob_v_%s' % m)
    # 10b. ClozeProb vs. model region surprisal
    for m in REGION:
        comps.append('clozeprob_v_%s' % m)

    # Deduplicate preserving order
    seen = set()
    out  = []
    for c in comps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. LRT results ───────────────────────────────────────────────────────
    print("=" * 70)
    print("LOADING LRT RESULTS")
    print("=" * 70)

    if not os.path.exists(LRT_INPUT):
        print("ERROR: %s not found.  Run sum_lrt.py first." % LRT_INPUT)
        df_lrt = pd.DataFrame()
    else:
        df_lrt = pd.read_csv(LRT_INPUT)
        df_lrt['Test Type'] = 'LRT (Likelihood Ratio Test)'
        print("Loaded %d LRT results." % len(df_lrt))

    # ── 2. PT results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOADING PERMUTATION TEST RESULTS from %s" % PT_DIR)
    print("=" * 70)

    pt_records = []
    missing    = []
    for comp in generate_pt_comparisons():
        rec = parse_pt_file(comp, PT_DIR, EXPERIMENT)
        if rec:
            pt_records.append(rec)
        else:
            missing.append(comp)

    df_pt = pd.DataFrame(pt_records)
    print("Loaded %d PT results." % len(df_pt))
    if missing:
        print("Missing %d PT files:" % len(missing))
        for m in missing:
            print("  %s" % m)

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MERGING")
    print("=" * 70)

    FINAL_COLS = [
        'Comparison Type', 'Has Region', 'Comparison', 'Raw Comparison',
        'ΔLL', 'p_display', 'Significant', 'Raw p-value',
        'Left Model', 'Right Model', 'Color', 'ΔLL_display', 'SignifColor',
        'Test Type',
    ]

    frames = [f for f in [df_lrt, df_pt] if not f.empty]
    if not frames:
        print("No results to save.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure ΔLL_display is present for LRT rows that came from sum_lrt.py
    if 'ΔLL_display' not in df.columns:
        df['ΔLL_display'] = df['ΔLL'].apply(lambda x: str(int(round(x))) if pd.notnull(x) else '')

    # ── 4. Color coding based on raw p-value (no FDR correction) ─────────────
    print("\nNo FDR correction applied — using raw p < 0.05 for significance.")
    print("Run apply_fdr.py on the output file to apply BY correction.")

    df['Color'] = df.apply(
        lambda r: ('red' if r['ΔLL'] > 0 else 'blue') if r['Significant'] else 'black', axis=1
    )
    df['SignifColor'] = df['Color']
    df['p_display']   = df['Raw p-value'].apply(fmt_p)

    # ── 5. Sort ───────────────────────────────────────────────────────────────
    TYPE_ORDER = [
        'Null vs. Base', 'Null vs. PROB',
        'Base vs. PROB', 'Base vs. Combined', 'PROB vs. Combined',
        'Surp vs. Prob (PT)',
        'Both vs. Single (PT)',
        'Within-model Surp vs Region Surp (PT)',
        'Model Bakeoff (PT)',
        'Cloze Surp vs. Region Surp (PT)',
        'Cloze Prob vs. Region Prob (PT)',
        'Cloze Prob vs. Model Prob (PT)',
        'Cloze Region Prob vs. Model Region Prob (PT)',
        'ClozeProb vs. Model Surp (PT)',
        'ClozeProb vs. Model Region Surp (PT)',
    ]
    df['_ord'] = df['Comparison Type'].apply(
        lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99
    )
    df = df.sort_values(['_ord', 'Has Region', 'ΔLL'],
                        ascending=[True, True, False]).drop(columns=['_ord'])

    # ── 6. Save ───────────────────────────────────────────────────────────────
    # Ensure all final columns exist
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = ''

    out_path = 'all_results_raw.csv'
    df[FINAL_COLS].to_csv(out_path, index=False)
    print("\nSaved: %s  (%d rows)" % (out_path, len(df)))
    print("Next step: python apply_fdr.py all_results_raw.csv")

    # ── 7. Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY BY COMPARISON TYPE (raw p < 0.05, no FDR correction)")
    print("=" * 70)
    for t in TYPE_ORDER:
        sub = df[df['Comparison Type'] == t]
        if sub.empty:
            continue
        n_sig = sub['Significant'].sum()
        print("  %-42s %3d comparisons  %3d significant" % (t, len(sub), n_sig))

    print("\n" + "=" * 70)
    print("COLOR CODING")
    print("=" * 70)
    print("  Red   (+ΔLL): Model B better  (right-hand model wins)")
    print("  Blue  (-ΔLL): Model A better  (left-hand model wins)")
    print("  Black:        Not significant (raw p >= 0.05, no FDR correction)")

    print("\n" + "=" * 70)
    print("METHODOLOGY REMINDER")
    print("=" * 70)
    print("  LRT results  → nested comparisons, fitted with LME (lmer)")
    print("  PT results   → all comparisons, based on LME held-out log-likelihood")
    print("  LRT and PT results are NOT cross-compared.")
    print("=" * 70)


if __name__ == '__main__':
    main()
