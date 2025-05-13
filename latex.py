# Latexing the Cross Calidation Table 
import os
import re
import pandas as pd
import numpy as np

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
    
    model_a_perf = float(re.search(r'Model A:\s+(-?\d+\.\d+)', content).group(1))
    model_b_perf = float(re.search(r'Model B:\s+(-?\d+\.\d+)', content).group(1))
    difference = float(re.search(r'Difference:\s+(-?\d+\.\d+)', content).group(1))
    
    p_value_match = re.search(r'p:\s+(\d+\.\d+e[+-]\d+)(\*{0,3})', content)
    p_value = float(p_value_match.group(1))
    significance = p_value_match.group(2)
    
    category = "Other"
    if "nosurp_v_" in comparison:
        category = "Overall"
    elif "prob_v_" in comparison or "surp_v_" in comparison:
        category = "Probability vs. Surprisal"
    elif "cloze" in comparison and any(lm in comparison for lm in ["gpt2", "trigram", "llama2", "olmo"]):
        category = "Cloze vs. Other"
    
    display_comparison = comparison.replace('_v_', ' vs. ')
    display_comparison = display_comparison.replace('gpt2new', 'GPT-2')
    display_comparison = display_comparison.replace('gpt2regionnew', 'GPT-2-Region')
    display_comparison = display_comparison.replace('cloze', 'Cloze')
    display_comparison = display_comparison.replace('trigram', 'Trigram')
    display_comparison = display_comparison.replace('nosurp', '\\emptyset')
    display_comparison = display_comparison.replace('prob', '_{\\text{PROB}}')
    display_comparison = display_comparison.replace('surp', '_{\\text{SURP}}')
    
    # calc delta LL 
    delta_ll = int(round(difference))
    
    # Format p-value 
    if p_value < 0.0001:
        p_display = "0.0001"
    elif p_value >= 1.0:
        p_display = "1.0000"
    else:
        p_display = f"{p_value:.4f}"
    
    results.append({
        'Category': category,
        'Comparison': display_comparison,
        'ΔLL': delta_ll,
        'p': p_display,
        'Significant': len(significance) > 0,
        'Raw p-value': p_value
    })

df = pd.DataFrame(results)

df = df.sort_values(['Category', 'Significant', 'Raw p-value'], 
                    ascending=[True, False, True])

# coloring
def format_cells(row):
    if not row['Significant']:
        delta_ll = "---" if row['p'] == "1.0000" else ""
        return row['Comparison'], delta_ll, row['p']
    
    delta_ll = str(row['ΔLL'])
    p_val = row['p']
    
    if row['ΔLL'] < 0:
        # Cyan (left better)
        delta_ll = f"\\textcolor{{cyan}}{{\\textbf{{{delta_ll}}}}}"
        p_val = f"\\textcolor{{cyan}}{{\\textbf{{{p_val}}}}}"
    else:
        # Magenta (right better)
        delta_ll = f"\\textcolor{{magenta}}{{\\textbf{{{delta_ll}}}}}"
        p_val = f"\\textcolor{{magenta}}{{\\textbf{{{p_val}}}}}"
    
    return row['Comparison'], delta_ll, p_val

# Generate LaTeX table
latex_table = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{booktabs}
\\usepackage{xcolor}
\\usepackage{geometry}
\\usepackage{amsmath}
\\usepackage{amssymb}

\\definecolor{cyan}{RGB}{0, 255, 255}
\\definecolor{magenta}{RGB}{255, 0, 255}

\\geometry{margin=1in}

\\begin{document}

\\begin{table}
\\centering
\\caption{Testing Results on Data from Brothers' \\& Kuperberg's Experiment (SPR)}
\\label{tab:results}
\\begin{tabular}{lrr}
\\toprule
Comparison & $\\Delta$LL & $p$ \\\\
\\midrule
"""

current_category = ""
for _, row in df.iterrows():
    if row['Category'] != current_category:
        current_category = row['Category']
        latex_table += f"\\midrule\n\\multicolumn{{3}}{{l}}{{\\textbf{{{current_category}}}}} \\\\\n\\midrule\n"
    
    comparison, delta_ll, p_val = format_cells(row)
    
    latex_table += f"${comparison}$ & {delta_ll} & {p_val} \\\\\n"

latex_table += """\\bottomrule
\\end{tabular}
\\caption*{\\small Note: $\\Delta$LL is the difference in log likelihood between models. 
\\textcolor{cyan}{\\textbf{Cyan}} indicates the left model is better. 
\\textcolor{magenta}{\\textbf{Magenta}} indicates the right model is better.}
\\end{table}

\\end{document}
"""

# Save the LaTeX file
with open('bk21_spr_results_table.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved as bk21_spr_results_table.tex")

# Also save the data as CSV
df.to_csv('bk21_spr_results_summary.csv', index=False)
print("Results data saved as bk21_spr_results_summary.csv")
