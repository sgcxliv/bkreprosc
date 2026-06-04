# further cleaning the ibex results and removing unnecessary columns

import pandas as pd
import string
import sys
import os

def normalize_sent(sent):
    """Strip punctuation/lowercase for robust string matching."""
    if pd.isnull(sent):
        return ""
    sent = sent.lower().strip()
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent = " ".join(sent.split())
    return sent

def detect_delimiter(filename):
    """Detect if a file is tab or comma separated."""
    with open(filename, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.startswith('#'):
                if line.count('\t') > line.count(','):
                    return '\t'
                else:
                    return ','
            if i > 10:
                break
    return ','

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, help='PennController results .csv file')
    parser.add_argument('--annotation', type=str, help='Annotation file (critical_word_data.csv)')
    parser.add_argument('--out', type=str, default='OUTPUT_MERGED.csv', help='Name for merged output')
    args = parser.parse_args()

    if args.results:
        results_file = args.results
    else:
        results_file = input("Enter results file (.csv): ").strip()

    if args.annotation:
        annot_file = args.annotation
    else:
        annot_file = input("Enter annotation file (e.g., critical_word_data.csv): ").strip()

    print("Loading annotation file...")

    annot = pd.read_csv(annot_file, encoding='utf-8')
    ann_sentence_col = 'words'
    if 'words' not in annot.columns:
        print(f"ERROR: Could not find 'words' column in {annot_file}. Columns: {annot.columns.tolist()}")
        sys.exit(1)

    annot['sentence_norm'] = annot[ann_sentence_col].apply(normalize_sent)

    print("Loading results file...")
    delim = detect_delimiter(results_file)
    print(f"Detected delimiter: {'TAB' if delim == '\\t' else 'COMMA'}")

    with open(results_file, encoding='utf-8') as f:
        skiprows = 0
        for line in f:
            if not line.startswith('#'):
                break
            skiprows += 1

    res = pd.read_csv(results_file, delimiter=delim, header=None, skiprows=skiprows, dtype=str)

    possible_sentence_cols = [c for c in res.columns if res[c].astype(str).str.contains('the',case=False).sum()>1]
    SENTENCE_COL = res.columns[-4] if len(res.columns) > 20 else res.columns[-2]
    
    res['sentence_norm'] = res[SENTENCE_COL].apply(normalize_sent)
   
    print("Merging by normalized sentence...")
    merged = pd.merge(
        res, annot,
        how='left',
        left_on='sentence_norm', right_on='sentence_norm',
        suffixes=('', '_annotation')
    )

    outpath = args.out if args.out else 'OUTPUT_MERGED.csv'
    print(f"Writing merged file to: {outpath}")
    merged.to_csv(outpath, index=False)

    print("\nDone. Review missing matches (rows with lots of empty annotation columns).")
    print(f"Merged file rows: {len(merged)}")

if __name__ == '__main__':
    main()

