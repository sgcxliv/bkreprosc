import pandas as pd

data = pd.read_csv('merged_results.csv')

# only columns to keep, cleaning merged results file
columns_to_keep = [
    'Code', 
    'sentence_context', 
    'participant_word', 
    'expected_word', 
    'count', 
    'smoothed_total', 
    'empirical_prob', 
    'smoothed_count', 
    'smoothed_prob', 
    'normalized_prob', 
    'Cloze', 
    'Sentence', 
    'Cloze_Probability'
]

filtered_data = data[columns_to_keep]

print(filtered_data)

filtered_data.to_csv('final.csv', index=False)

