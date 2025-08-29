import os
import pandas as pd
import re

# ------ CONFIG ------
input_folder = "./"      # Use "." for current folder, or specify full path
output_folder = "./"     # Can be the same or different folder
# --------------------

# Flexible column search
def find_col(cols, targets):
    for t in targets:
        for c in cols:
            if c.strip().lower() == t.lower():
                return c
    raise ValueError(f"None of {targets} found in columns")

csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for fname in csv_files:
    full_path = os.path.join(input_folder, fname)
    try:
        df = pd.read_csv(full_path)
        # Find the right columns (edit/add if needed)
        listnumber_col = find_col(df.columns, ['ListNumber', 'listnumber'])
        code_col = find_col(df.columns, ['Code'])
        item_col = find_col(df.columns, ['Item', 'itemnum', 'item'])
        sentence_col = find_col(df.columns, ['Sentence', 'words', 'sentence'])
        # Build output DataFrame
        df_out = df[[listnumber_col, code_col, item_col, sentence_col]].copy()
        df_out.columns = ['ListNumber', 'Code', 'Item', 'Sentence']
        # Remove commas and hyphens/dashes from Sentence
        df_out['Sentence'] = df_out['Sentence'].astype(str).apply(lambda s: re.sub(r'[,–—-]', '', s))
        # Write output
        out_path = os.path.join(output_folder, f"converted_{fname}")
        df_out.to_csv(out_path, index=False)
        print(f"Converted {fname} to {out_path}")
    except Exception as e:
        print(f"Skipping {fname}: {e}")

print("Done!")
