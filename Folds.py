import pandas as pd
import os

# Load dataset
df_path = "bk21_spr.csv"
if not os.path.exists(df_path):
    print(f"ERROR: File '{df_path}' not found")
    exit()

df = pd.read_csv(df_path)

if "ITEM" not in df.columns:
    print("ERROR: 'ITEM' column not found.")
    exit()

df["fold"] = ((df["ITEM"] - 1) % 5) + 1

output_path = "bk21_spr_with_folds.csv"
df.to_csv(output_path, index=False)
print(f"Folds assigned and saved to '{output_path}'")
