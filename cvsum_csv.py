import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# get files
files = [f for f in os.listdir('.') if f.endswith('_PT_rt_test.txt')]

results = []
for file in files:
    comparison = file.replace('bk21_spr_', '').replace('_PT_rt_test.txt', '')
    with open(file, 'r') as f:
        content = f.read()
        model_comparison = re.search(r'Model comparison: (.*?) vs (.*?)\n', content)
    if model_comparison:
        model_a = model_comparison.group(1)
        model_b = model_comparison.group(2)
    else:
        model_a, model_b = "Unknown", "Unknown"
    
    # extract vals
    model_a_perf = float(re.search(r'Model A:\s+(-?\d+\.\d+)', content).group(1))
    model_b_perf = float(re.search(r'Model B:\s+(-?\d+\.\d+)', content).group(1))
    difference = float(re.search(r'Difference:\s+(-?\d+\.\d+)', content).group(1))
    
    # get p val
    p_value_match = re.search(r'p:\s+(\d+\.\d+e[+-]\d+)(\*{0,3})', content)
    p_value = float(p_value_match.group(1))
    significance = p_value_match.group(2)
    
    # comparison type
    comparison_type = "Other"
    if "nosurp_v_" in comparison and "prob" in comparison:
        comparison_type = "Null vs. PROB"
    elif "nosurp_v_" in comparison and "prob" not in comparison:
        comparison_type = "Null vs. Base"
    elif comparison.count("_") == 1:  # Simple model vs model comparison
        parts = comparison.split("_v_")
        if parts[0] + "prob" == parts[1]:  # e.g., cloze_v_clozeprob
            comparison_type = "Base vs. PROB"
        elif "clozeprob" == parts[0] and "clozeprob-" not in parts[1]:
            comparison_type = "ClozePROB vs. Other"
        elif "clozeprob" == parts[0] and "clozeprob-" in parts[1]:
            comparison_type = "ClozePROB vs. Combined"
    elif "-" in comparison:  # More complex cases with hyphens
        parts = comparison.split("_v_")
        if parts[0] in parts[1] and "-" in parts[1]:  # e.g., cloze_v_clozeprob-cloze
            comparison_type = "Base vs. Combined"
        elif "prob" in parts[0] and parts[0].replace("prob", "") in parts[1]:  # e.g., clozeprob_v_clozeprob-cloze
            comparison_type = "PROB vs. Combined"
    
    # has region
    has_region = "region" in comparison.lower()
    
    # formatting
    display_comparison = comparison.replace('_v_', ' vs. ')
    display_comparison = display_comparison.replace('gpt2new', 'GPT-2')
    display_comparison = display_comparison.replace('gpt2regionnew', 'GPT-2-Region')
    display_comparison = display_comparison.replace('cloze', 'Cloze')
    display_comparison = display_comparison.replace('trigram', 'Trigram')
    display_comparison = display_comparison.replace('nosurp', 'ø')
    display_comparison = display_comparison.replace('prob', 'PROB')
    display_comparison = display_comparison.replace('surp', 'SURP')
    
    delta_ll = int(round(difference))
    
    # p val display
    if p_value < 0.0001:
        p_display = "0.0001"
    elif p_value >= 1.0:
        p_display = "1.0000"
    else:
        p_display = f"{p_value:.4f}"
    
    results.append({
        'Comparison Type': comparison_type,
        'Has Region': has_region,
        'Comparison': display_comparison,
        'Raw Comparison': comparison,
        'ΔLL': delta_ll,
        'p': p_display,
        'Significant': len(significance) > 0,
        'Raw p-value': p_value,
        'Left Model': model_a,
        'Right Model': model_b
    })

df = pd.DataFrame(results)

# order of comparison types
type_order = [
    "Base vs. PROB",
    "Base vs. Combined",
    "PROB vs. Combined",
    "Null vs. PROB",
    "Null vs. Base",
    "ClozePROB vs. Other",
    "ClozePROB vs. Combined",
    "Other"
]

# categorical type with correct order
df['Comparison Type'] = pd.Categorical(
    df['Comparison Type'],
    categories=type_order,
    ordered=True
)

df = df.sort_values(['Comparison Type', 'Has Region', 'Significant', 'Raw p-value'],
                    ascending=[True, True, False, True])

# color coding
def color_code(row):
    if not row['Significant']:
        return ['', '—' if row['p'] == '1.0000' else '', row['p'], '']
    
    delta_ll = row['ΔLL']
    if delta_ll < 0:
        # Cyan left outperforms right
        return ['cyan', str(delta_ll), row['p'], 'cyan']
    else:
        # Magenta right outperforms left
        return ['magenta', str(delta_ll), row['p'], 'magenta']

df[['Color', 'ΔLL_display', 'p_display', 'SignifColor']] = df.apply(color_code, axis=1, result_type='expand')

# save results
df.to_csv('bk21_spr_results_summary.csv', index=False)
print(f"Results saved to bk21_spr_results_summary.csv")
