# add the expected word column to cleaned file
import pandas as pd
import re
import os
import glob
import string
import sys

def normalize_text(text):
    """Lowercase, remove punctuation, and collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_list_files():
    """Load all List*.csv files and extract key information"""
    list_data = {}
    list_files = glob.glob("List*.csv")
    
    for list_file in list_files:
        try:
            # Extract list number from filename
            list_num = int(re.search(r'List(\d+)\.csv', list_file).group(1))
            
            # Load the file
            df = pd.read_csv(list_file)
            
            # Process each row
            for _, row in df.iterrows():
                item_num = row.get('itemnum')
                if pd.isna(item_num):
                    continue
                    
                item_num = int(item_num)
                
                # Store the full and truncated sentences
                if 'words' in df.columns and 'Sentence' in df.columns:
                    full_sentence = row['words']
                    truncated_sentence = row['Sentence']
                    
                    # Clean both sentences
                    full_clean = normalize_text(full_sentence)
                    trunc_clean = normalize_text(truncated_sentence)
                    
                    # Find the expected next word
                    expected_word = get_next_word(full_clean, trunc_clean)
                    
                    # Store all the data
                    list_data[(list_num, item_num)] = {
                        'full_sentence': full_sentence,
                        'truncated_sentence': truncated_sentence,
                        'condition': row.get('condition'),
                        'RemovalType': row.get('RemovalType'),
                        'Code': row.get('Code'),
                        'critical_word': row.get('critical_word'),
                        'expected_word': expected_word
                    }
        except Exception as e:
            print(f"Error processing {list_file}: {e}")
    
    return list_data

def get_next_word(full_sentence, truncated_sentence):
    """Find the word that would come next after the truncated sentence"""
    if not full_sentence or not truncated_sentence:
        return ""
        
    full_words = full_sentence.split()
    trunc_words = truncated_sentence.split()
    
    # If truncated is longer than full, something is wrong
    if len(trunc_words) >= len(full_words):
        return ""
    
    # Check if truncated is the beginning of full
    for i in range(len(full_words) - len(trunc_words) + 1):
        if full_words[i:i+len(trunc_words)] == trunc_words:
            idx = i + len(trunc_words)
            if idx < len(full_words):
                return full_words[idx]
            
    # If no match found
    return ""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="cleaned.csv")
    parser.add_argument('--out', type=str, default="")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.out or input_file.replace('.csv', '_with_expected.csv')
    
    # Load all list data
    print("Loading List files...")
    list_data = load_list_files()
    print(f"Loaded data for {len(list_data)} unique items")
    
    # Load the cleaned results file
    print(f"Loading cleaned results from {input_file}...")
    clean_df = pd.read_csv(input_file)
    
    # Add expected_word column
    print("Adding expected_word column...")
    expected_words = []
    
    for _, row in clean_df.iterrows():
        list_file = row.get('list_file', '')
        item_num = row.get('itemnum')
        
        # Extract list number from list_file
        list_num = None
        if isinstance(list_file, str):
            match = re.search(r'List(\d+)\.csv', list_file)
            if match:
                list_num = int(match.group(1))
        
        # Try to find the expected word
        expected_word = ""
        if list_num is not None and item_num is not None:
            try:
                item_num = int(float(item_num))
                if (list_num, item_num) in list_data:
                    expected_word = list_data[(list_num, item_num)].get('expected_word', '')
            except:
                pass
                
        expected_words.append(expected_word)
    
    # Add the column to the dataframe
    clean_df['expected_word'] = expected_words
    
    # Reorder columns to put expected_word after participant_word
    if 'participant_word' in clean_df.columns:
        cols = list(clean_df.columns)
        pidx = cols.index('participant_word')
        cols = cols[:pidx+1] + ['expected_word'] + [c for c in cols[pidx+1:] if c != 'expected_word']
        clean_df = clean_df[cols]
    
    # Save the output
    print(f"Writing output to {output_file}")
    clean_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
