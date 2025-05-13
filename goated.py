import os
import pandas as pd

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

surprisal = os.path.join(desktop_path, 'surp.csv')
items = os.path.join(desktop_path, 'spr.csv')

surprisal = pd.read_csv(surprisal)
items = pd.read_csv(items)

merged_data = items.merge(
    surprisal,
    on=['critical_word', 'condition', 'position'],  #
    how='left'  
)

output_path = os.path.join(desktop_path, 'items2.csv')

merged_data.to_csv(output_path, index=True)

print("Total people:", len(items))
print("Total result merge", len(merged_data))
print(len(merged_data))

