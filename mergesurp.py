# to merge cleaned cloze data (no mismatched items) with spr data
# any time i need to merge anything i just use a versin of this 
import os
import pandas as pd

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

surprisal = os.path.join(desktop_path, 'clean_cloze_data.csv')
items = os.path.join(desktop_path, 'bk21_spr.csv')

surprisal = pd.read_csv(surprisal)
items = pd.read_csv(items)

merged_data = items.merge(
    surprisal,
    on=['critical_word', 'condition', 'position'],  # Match on 3 Columns
    how='left' 
)

output_path = os.path.join(desktop_path, 'BK_items_with_surprisal_clean.csv')

merged_data.to_csv(output_path, index=True)

print("Total people:", len(items))
print("Total result merge", len(merged_data))
print(len(merged_data))

