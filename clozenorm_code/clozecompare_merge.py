import pandas as pd
import re

def normalize(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def first_n_words(text, n=5):
    words = text.split()
    return ' '.join(words[:n])

bk = pd.read_csv('bk_cleaned.csv')
probs = pd.read_csv('cloze_distributions.csv')
expected = pd.read_csv('megaresults_matched_with_expected.csv')

expected_expt = expected[expected['item_type'] == 'experiment'][[
    'stimuli_code',
    'expected_word',
    'sentence_context',
    'stimuli_region_placement'
]]

bk['sentence_norm'] = bk['Sentence'].apply(normalize)
bk['first5_bk'] = bk['sentence_norm'].apply(first_n_words)

probs['context_norm'] = probs['sentence_context'].apply(normalize)
probs['first5_exp'] = probs['context_norm'].apply(first_n_words)

probs_expected = pd.merge(
    probs,
    expected_expt,
    on=['stimuli_code', 'stimuli_region_placement', 'sentence_context'],
    how='inner'
)

probs_expected = probs_expected[probs_expected['participant_word'] == probs_expected['expected_word']]

probs_expected_r1 = probs_expected[probs_expected['stimuli_region_placement'] == 1]

probs_expected_r1 = probs_expected_r1.drop_duplicates(subset=['first5_exp', 'expected_word'])

bk_augmented = pd.merge(
    bk,
    probs_expected_r1[['first5_exp', 'expected_word', 'smoothed_prob']],
    left_on=['first5_bk'],  # Note: removed 'Cloze' as you mentioned it's just H/M/L
    right_on=['first5_exp'],
    how='left'
)

bk_augmented = bk_augmented.rename(columns={'smoothed_prob': 'New_Cloze'})

cols = ['New_Cloze'] + [c for c in bk_augmented.columns if c not in ['New_Cloze']]
bk_augmented = bk_augmented[cols]

bk_augmented.to_csv('bk_augmented_with_new_cloze.csv', index=False)
