import pandas as pd

input_file = 'REP_clean.csv'
output_file = 'bk21_spr.csv'

# Read only the header row
df = pd.read_csv(input_file, nrows=0)

# Dictionary of columns you want to rename: {'old_name': 'new_name'}
rename_map = {
    'sentpos': 'critical_word_pos',
    'wlen_x': 'wlen',
    'wlenregion_x': 'wlenregion',
    'unigram_x': 'unigram',
    'unigramregion_x': 'unigramregion',
    'glovedistmean_x': 'glovedistmean',
    'Index': 'itemnum',
}

# Rename columns
df.rename(columns=rename_map, inplace=True)

# Write the new header + rest of file without loading full content
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    next(infile)  # skip old header
    outfile.write(','.join(df.columns) + '\n')  # write new header
    for line in infile:
        outfile.write(line)

