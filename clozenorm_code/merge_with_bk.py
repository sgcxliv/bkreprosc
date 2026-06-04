import pandas as pd
import re

data1 = pd.read_csv('filtered.csv')  # your data
cloze_data = pd.read_csv('bk_cleaned.csv')  # original data to compare

# filter for relevant removals
filtered_data1 = data1[data1['Code'].str.endswith('R1')]

filtered_data1['Cloze_Type'] = filtered_data1['Code'].str[0]  # get the first character H, M, or L

def normalize_sentence(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return ' '.join(sentence.split())

filtered_data1['normalized_sentence'] = filtered_data1['sentence_context'].apply(normalize_sentence)

cloze_data['normalized_sentence'] = cloze_data['Sentence'].apply(normalize_sentence)

n_words = 5  # merge key word count, adjustable
filtered_data1['sentence_key'] = filtered_data1['normalized_sentence'].apply(lambda x: ' '.join(x.split()[:n_words]))
cloze_data['sentence_key'] = cloze_data['normalized_sentence'].apply(lambda x: ' '.join(x.split()[:n_words]))

merged_data = pd.merge(filtered_data1, cloze_data, left_on=['sentence_key', 'Cloze_Type'], right_on=['sentence_key', 'Cloze'], how='inner')

print(f"Number of matched rows: {len(merged_data)}")
print(merged_data)

merged_data.to_csv('merged_results.csv', index=False)

unmatched_data1 = filtered_data1[~filtered_data1['sentence_key'].isin(merged_data['sentence_key'])]
unmatched_cloze = cloze_data[~cloze_data['sentence_key'].isin(merged_data['sentence_key'])]

print("Unmatched rows from filtered_data1:")
print(unmatched_data1)

print("\nUnmatched rows from cloze_data:")
print(unmatched_cloze)
