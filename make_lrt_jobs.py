#!/usr/bin/env python
import os

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
"""

wrapper = '\npython simple_lrt.py %s %s'

experiments = [
  'spr',
]

models = [
    'cloze',
    'trigram',
    'gpt2new',
    'gpt2regionnew',
    'gpt2xlnew',
    'gpt2xlregionnew',
    'gptjnew',
    'gptjregionnew',
    'gptneonew',
    'gptneoregionnew',
    'gptneoxnew',
    'gptneoxregionnew',
    'olmonew',
    'olmoregionnew',
    'llama2new',
    'llama2regionnew',
]

# Generate nested model comparisons
nested_comparisons = []

# Base model vs. prob version 
for model in models:
    nested_comparisons.append(f'{model}_v_{model}prob')

# Base model vs. combined model 
for model in models:
    nested_comparisons.append(f'{model}_v_{model}prob-{model}')

# Prob version vs. combined model 
for model in models:
    nested_comparisons.append(f'{model}prob_v_{model}prob-{model}')

# Null model vs. base model 
for model in models:
    nested_comparisons.append(f'nosurp_v_{model}')

# Null model vs. prob version 
for model in models:
    nested_comparisons.append(f'nosurp_v_{model}prob')

# Create job files
for experiment in experiments:
    for comparison in nested_comparisons:
        job_name = f'lrt_{experiment}_{comparison}'
        job_str = wrapper % (experiment, comparison)
        job_str = base % (job_name, job_name) + job_str
        with open(f'{job_name}.pbs', 'w') as f:
            f.write(job_str)
        print(f"Created job file: {job_name}.pbs")

print(f"Generated {len(nested_comparisons)} job files for likelihood ratio tests")
print("Run all jobs with: sbatch lrt_*.pbs")

# Script to summarize LRT results
summarize_script = """#!/usr/bin/env python
import os
import re
import pandas as pd
import numpy as np

results_path = 'results/bk21/lrt'  # Update this if needed

files = [f for f in os.listdir(results_path) if f.endswith('_summary.csv')]

results = []
for file in files:
    # Extract test name
    test_name = file.replace('_summary.csv', '')
    
    df = pd.read_csv(os.path.join(results_path, file))
    
    model_a = df['model_a'].iloc[0]
    model_b = df['model_b'].iloc[0]
    
    comparison = test_name.replace('lrt_spr_', '').replace('_v_', ' vs. ')
    
    loglik_a = float(df['loglik_a'].iloc[0])
    loglik_b = float(df['loglik_b'].iloc[0])
    difference = float(df['difference'].iloc[0])
    p_value = float(df['p_value'].iloc[0])
    significant = bool(df['significant'].iloc[0])
    
    comparison_type = "Other"
    if "nosurp_v_" in test_name and "prob" in test_name:
        comparison_type = "Null vs. PROB"
    elif "nosurp_v_" in test_name and "prob" not in test_name:
        comparison_type = "Null vs. Base"
    elif test_name.count("_") == 3:  # Simple model vs model comparison (counting lrt_spr_model_v_modelprob)
        parts = test_name.split("_v_")
        base_name = parts[0].replace("lrt_spr_", "")
        if base_name + "prob" == parts[1]:  # e.g., cloze_v_clozeprob
            comparison_type = "Base vs. PROB"
    elif "-" in test_name:  # More complex cases with hyphens
        parts = test_name.split("_v_")
        base_name = parts[0].replace("lrt_spr_", "")
        if base_name in parts[1] and "-" in parts[1]:  # e.g., cloze_v_clozeprob-cloze
            comparison_type = "Base vs. Combined"
        elif "prob" in base_name and base_name.replace("prob", "") in parts[1]:  # e.g., clozeprob_v_clozeprob-cloze
            comparison_type = "PROB vs. Combined"
    
    pretty_comparison = comparison
    pretty_comparison = pretty_comparison.replace('gpt2new', 'GPT-2')
    pretty_comparison = pretty_comparison.replace('gpt2regionnew', 'GPT-2-Region')
    pretty_comparison = pretty_comparison.replace('cloze', 'Cloze')
    pretty_comparison = pretty_comparison.replace('trigram', 'Trigram')
    pretty_comparison = pretty_comparison.replace('nosurp', 'ø')
    pretty_comparison = pretty_comparison.replace('prob', 'PROB')
    
    has_region = "region" in test_name.lower()
    
    if p_value < 0.0001:
        p_display = "0.0001"
    elif p_value >= 1.0:
        p_display = "1.0000"
    else:
        p_display = f"{p_value:.4f}"
    
    results.append({
        'Comparison Type': comparison_type,
        'Has Region': has_region,
        'Comparison': pretty_comparison,
        'Raw Comparison': comparison,
        'ΔLL': int(round(difference)),
        'p': p_display,
        'Significant': significant,
        'Raw p-value': p_value,
    })

df = pd.DataFrame(results)

# Define comparison type order
type_order = [
    "Base vs. PROB",
    "Base vs. Combined",
    "PROB vs. Combined",
    "Null vs. PROB",
    "Null vs. Base",
    "Other"
]

# Create categorical type
df['Comparison Type'] = pd.Categorical(
    df['Comparison Type'],
    categories=type_order,
    ordered=True
)

# Sort by type, region status, significance, and p-value
df = df.sort_values(['Comparison Type', 'Has Region', 'Significant', 'Raw p-value'],
                    ascending=[True, True, False, True])

def color_code(row):
    if not row['Significant']:
        return ['', '—' if row['p'] == '1.0000' else '', row['p'], '']
    
    delta_ll = row['ΔLL']
    if delta_ll < 0:
        # Left model better (usually simpler)
        return ['blue', str(delta_ll), row['p'], 'blue']
    else:
        # Right model better (usually more complex)
        return ['red', str(delta_ll), row['p'], 'red']

df[['Color', 'ΔLL_display', 'p_display', 'SignifColor']] = df.apply(color_code, axis=1, result_type='expand')

df.to_csv('lrt_results_summary.csv', index=False)
print(f"Results saved to lrt_results_summary.csv")

print("\\nResults by comparison type:")
for type_name in type_order:
    subset = df[df['Comparison Type'] == type_name]
    if len(subset) == 0:
        continue
    print(f"- {type_name}: {len(subset)} comparisons")

with open('lrt_results_report.txt', 'w') as f:
    f.write("# Likelihood Ratio Test Results\\n\\n")
    
    for comp_type in type_order:
        subset = df[df['Comparison Type'] == comp_type]
        if len(subset) == 0:
            continue
            
        f.write(f"## {comp_type}\\n")
        for _, row in subset.iterrows():
            significance = "**" if row['Significant'] else ""
            color_indicator = ""
            if row['Significant']:
                if row['ΔLL'] < 0:
                    color_indicator = "(simpler model better)"
                else:
                    color_indicator = "(complex model better)"
                    
            f.write(f"{row['Comparison']}\\t{row['ΔLL_display'] or '—'}\\t{row['p']}\\t{significance} {color_indicator}\\n")
        f.write("\\n")

print(f"Detailed report saved to lrt_results_report.txt")
"""

with open('summarize_lrt.py', 'w') as f:
    f.write(summarize_script)
os.chmod('summarize_lrt.py', 0o755) 

print("Created summarize_lrt.py to process results after jobs complete")
print("Run it with: python summarize_lrt.py")
