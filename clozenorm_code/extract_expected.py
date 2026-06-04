# filter for expected word = participant word
import pandas as pd

data = pd.read_csv('cloze_distributions.csv')

filtered_data = data[data['expected_word'] == data['participant_word']]

print(f"Number of rows after filtering: {len(filtered_data)}")
print(filtered_data)

filtered_data.to_csv('filtered_distributions.csv', index=False)
