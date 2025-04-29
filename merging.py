import os
import pandas as pd

# Get path to desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# Construct full file paths
surprisal = os.path.join(desktop_path, 'bk21_items_surprisal.csv')
items = os.path.join(desktop_path, 'bk21_spr.csv')

# Read the two CSV files
surprisal = pd.read_csv(surprisal)
items = pd.read_csv(items)

# Merge the dataframes based on multiple columns
merged_data = items.merge(
    surprisal,
    on=['critical_word', 'condition', 'position'],  # Match on 3 Columns
    how='left'  # Keep all rows from items
)

# Construct output path for merged CSV
output_path = os.path.join(desktop_path, 'BK_items_with_surprisal.csv')

# Save the merged dataframe
merged_data.to_csv(output_path, index=True)

# Provide merge statistics
print("Total people:", len(items))
print("Total result merge", len(merged_data))
print(len(merged_data))

