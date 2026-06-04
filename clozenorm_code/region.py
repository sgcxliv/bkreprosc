import pandas as pd
import numpy as np

# Load data
cloze_df = pd.read_csv('cloze_distributionsall.csv')
spr_df = pd.read_csv('bk21_data/bk21_spr.csv')

print("Computing cloze region from distributions...")

# Filter to correct responses only
cloze_df = cloze_df[
    (cloze_df['participant_word'] == cloze_df['expected_word']) &
    pd.notna(cloze_df['empirical_prob'])
].copy()

print(f"Filtered cloze data: {len(cloze_df)} rows")

# Parse the Code column to extract item info
# Format: HC0R1 = High Cloze, item 0, RemovalType 1
cloze_df['cloze_condition'] = cloze_df['Code'].str[0]  # H, M, or L
cloze_df['item_from_code'] = cloze_df['Code'].str.extract(r'[HML]C(\d+)R')[0].astype(int)

print(f"Unique conditions: {sorted(cloze_df['cloze_condition'].unique())}")
print(f"Unique RemovalTypes: {sorted(cloze_df['RemovalType'].unique())}")
print(f"Item range: {cloze_df['item_from_code'].min()} to {cloze_df['item_from_code'].max()}")

# Initialize columns
spr_df['clozeregion'] = np.nan
spr_df['clozeregionprob'] = np.nan

n_computed = 0
n_missing = 0
n_total = len(spr_df)

print(f"\nProcessing {n_total} SPR rows...")

# Process each SPR row
for idx, row in spr_df.iterrows():
    if idx % 10000 == 0:
        print(f"  Processing row {idx}/{n_total}...")
    
    # Get item info from SPR
    itemnum = row['itemnum']
    condition = str(row['condition']).strip()[0]  # Get first letter: H, M, or L
    crit_pos = row['critical_word_pos']
    
    # Get the 3 words from wordXX columns
    word1_col = f'word{crit_pos:02d}'
    word2_col = f'word{crit_pos+1:02d}'
    word3_col = f'word{crit_pos+2:02d}'
    
    # Check if these columns exist
    if not all(col in spr_df.columns for col in [word1_col, word2_col, word3_col]):
        n_missing += 1
        continue
    
    # Extract the words
    word1 = str(row[word1_col]).strip().lower()
    word2 = str(row[word2_col]).strip().lower()
    word3 = str(row[word3_col]).strip().lower()
    
    # Skip if any word is missing/nan
    if any(w in ['', 'nan', 'none'] for w in [word1, word2, word3]):
        n_missing += 1
        continue
    
    # Look up probabilities in cloze data for each RemovalType
    # RemovalType 1: probability of word1 (critical word)
    c1 = cloze_df[
        (cloze_df['item_from_code'] == itemnum) &
        (cloze_df['cloze_condition'] == condition) &
        (cloze_df['RemovalType'] == 1) &
        (cloze_df['expected_word'].str.lower() == word1)
    ]
    
    # RemovalType 2: probability of word2 (critical+1)
    c2 = cloze_df[
        (cloze_df['item_from_code'] == itemnum) &
        (cloze_df['cloze_condition'] == condition) &
        (cloze_df['RemovalType'] == 2) &
        (cloze_df['expected_word'].str.lower() == word2)
    ]
    
    # RemovalType 3: probability of word3 (critical+2)
    c3 = cloze_df[
        (cloze_df['item_from_code'] == itemnum) &
        (cloze_df['cloze_condition'] == condition) &
        (cloze_df['RemovalType'] == 3) &
        (cloze_df['expected_word'].str.lower() == word3)
    ]
    
    # If we found all 3 probabilities
    if len(c1) == 1 and len(c2) == 1 and len(c3) == 1:
        prob1 = c1['empirical_prob'].iloc[0]
        prob2 = c2['empirical_prob'].iloc[0]
        prob3 = c3['empirical_prob'].iloc[0]
        
        # Calculate surprisals, handling zero probabilities
        surp1 = -np.log2(prob1) if prob1 > 0 else np.inf
        surp2 = -np.log2(prob2) if prob2 > 0 else np.inf
        surp3 = -np.log2(prob3) if prob3 > 0 else np.inf
        
        # REGION SURPRISAL: sum of individual surprisals
        region_surp = surp1 + surp2 + surp3
        
        # REGION PROBABILITY: product of individual probabilities
        region_prob = prob1 * prob2 * prob3
        
        spr_df.loc[idx, 'clozeregion'] = region_surp
        spr_df.loc[idx, 'clozeregionprob'] = region_prob
        n_computed += 1
    else:
        n_missing += 1

print(f"\n{'='*60}")
print("CLOZE REGION COMPUTATION COMPLETE")
print(f"{'='*60}")
print(f"Successfully computed: {n_computed} rows")
print(f"Missing data: {n_missing} rows")
print(f"Total rows with clozeregion: {spr_df['clozeregion'].notna().sum()}")

# Save
spr_df.to_csv('bk21_data/bk21_spr.csv', index=False)
print("\n✓ Saved to bk21_data/bk21_spr.csv")

# Statistics
print("\nCloze region surprisal stats:")
print(spr_df['clozeregion'].describe())

print("\nCloze region probability stats:")
print(spr_df['clozeregionprob'].describe())

# Check which 'word' columns exist
word_columns = [col for col in spr_df.columns if col.startswith('word') and col[4:].isdigit()]
print("\nExisting word columns:")
print(word_columns)

# Check for NaN values in existing word columns
print("\nChecking for NaN values in word columns:")
print(spr_df[word_columns].isna().sum())

# Sample of rows where computation failed
print("\nSample of rows where computation failed:")
failed_rows = spr_df[spr_df['clozeregion'].isna()]
print(failed_rows[['itemnum', 'condition', 'critical_word_pos'] + word_columns].head(10))

# Check for item number mismatches
print("\nChecking for item number mismatches:")
spr_items = set(spr_df['itemnum'])
cloze_items = set(cloze_df['item_from_code'])
print(f"Items in SPR but not in cloze: {spr_items - cloze_items}")
print(f"Items in cloze but not in SPR: {cloze_items - spr_items}")

# Check distribution of critical_word_pos
print("\nDistribution of critical_word_pos:")
print(spr_df['critical_word_pos'].value_counts().sort_index())

# Check if any critical_word_pos values are out of range
max_word_col = max(int(col[4:]) for col in word_columns)
out_of_range = spr_df[spr_df['critical_word_pos'] > max_word_col - 2]
print(f"\nRows with critical_word_pos > {max_word_col - 2}: {len(out_of_range)}")
if len(out_of_range) > 0:
    print("\nSample of rows with out-of-range critical_word_pos:")
    print(out_of_range[['itemnum', 'condition', 'critical_word_pos'] + word_columns].head(5))

# Verification
print(f"\n{'='*60}")
print("VERIFICATION SAMPLE")
print(f"{'='*60}")

sample_idx = spr_df[spr_df['clozeregion'].notna()].index[0]
sample = spr_df.loc[sample_idx]

print(f"\nSample row (index {sample_idx}):")
print(f"  itemnum: {sample['itemnum']}")
print(f"  condition: {sample['condition']}")
print(f"  Sentence: {sample['sentence_x']}")
print(f"  Critical position: {sample['critical_word_pos']}")

crit_pos = sample['critical_word_pos']
print(f"\n3-word region:")
print(f"  Word 1: '{sample[f'word{crit_pos:02d}']}'")
print(f"  Word 2: '{sample[f'word{crit_pos+1:02d}']}'")
print(f"  Word 3: '{sample[f'word{crit_pos+2:02d}']}'")
print(f"\nComputed values:")
print(f"  clozeregion (surprisal): {sample['clozeregion']:.4f}")
print(f"  clozeregionprob: {sample['clozeregionprob']:.10f}")

# Debug prints for individual word probabilities and surprisals
itemnum = sample['itemnum']
condition = str(sample['condition']).strip()[0]
word1 = sample[f'word{crit_pos:02d}'].lower()
word2 = sample[f'word{crit_pos+1:02d}'].lower()
word3 = sample[f'word{crit_pos+2:02d}'].lower()

for rt, word in [(1, word1), (2, word2), (3, word3)]:
    match = cloze_df[
        (cloze_df['item_from_code'] == itemnum) &
        (cloze_df['cloze_condition'] == condition) &
        (cloze_df['RemovalType'] == rt) &
        (cloze_df['expected_word'].str.lower() == word)
    ]
    if len(match) > 0:
        prob = match['empirical_prob'].iloc[0]
        surp = -np.log2(prob) if prob > 0 else np.inf
        print(f"  RemovalType {rt} ('{word}'): prob={prob:.10f}, surp={surp:.2f} bits")

print(f"\nTotal region: prob={sample['clozeregionprob']:.10f}, surp={sample['clozeregion']:.2f} bits")
