# cleaning raw horrible ibex results to managable clean tables of results
import pandas as pd
import string
import sys
import glob
import re
import os

def normalize_sent(sent):
    if pd.isnull(sent):
        return ""
    sent = str(sent).lower().strip()
    sent = re.sub(r'[^\w\s]', '', sent)
    sent = " ".join(sent.split())
    return sent

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default="results.csv")
    parser.add_argument('--out', type=str, default='cleaned.csv')
    args = parser.parse_args()

    list_files = glob.glob("List*.csv")
    if not list_files:
        print("ERROR: No List*.csv files found in current directory!")
        sys.exit(1)
        
    list_data = []
    for lf in list_files:
        try:
            df = pd.read_csv(lf)
            if 'Sentence' in df.columns:
                df['sentence_norm'] = df['Sentence'].apply(normalize_sent)
                df['list_file'] = os.path.basename(lf)
                list_data.append(df)
        except Exception as e:
            print(f"Error loading {lf}: {e}")
    
    combined_lists = pd.concat(list_data, ignore_index=True)
        
    experiment_rows = []
    with open(args.results, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split(',')
            
            if len(fields) >= 6 and "experiment" in fields[5]:
                try:
                    participant = fields[1] if len(fields) > 1 else ""
                    trial_number = fields[3] if len(fields) > 3 else ""
                    item_type = fields[5] if len(fields) > 5 else ""
                    
                    start_time = 0
                    if len(fields) > 11 and fields[11].isdigit():
                        start_time = int(fields[11])
                    
                    sentence = ""
                    if len(fields) > 18:
                        sentence = fields[18]
                    
                    response = ""
                    if len(fields) > 19:
                        response = fields[19]
                    
                    list_file = ""
                    if len(fields) > 14:
                        list_file = fields[14]
                    
                    if sentence:
                        experiment_rows.append({
                            'participant': participant,
                            'trial_number': trial_number,
                            'item_type': item_type,
                            'start_time': start_time,
                            'end_time': start_time,
                            'participant_word': response,
                            'sentence_context': sentence,
                            'list_file': list_file
                        })
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    print(f"Found {len(experiment_rows)} experiment rows")
    
    if not experiment_rows:
        print("ERROR: No experiment rows found. The file might have special formatting.")
        print("Trying to load the file directly with pandas...")
        
        try:
            with open(args.results, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line for line in f if not line.startswith('#')]
            
            import tempfile
            with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as temp:
                temp.writelines(lines)
                temp_path = temp.name
            
            df = pd.read_csv(temp_path, header=None, quotechar='"', escapechar='\\')
            
            print(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
            print("Column value counts for field 5:")
            if 5 in df.columns:
                print(df[5].value_counts().head())
            
            if 5 in df.columns:
                exp_df = df[df[5] == 'experiment']
                print(f"Found {len(exp_df)} experiment rows using pandas")
                
                if len(exp_df) > 0:
                    for _, row in exp_df.iterrows():
                        try:
                            participant = row[1] if 1 in row.index else ""
                            trial_number = row[3] if 3 in row.index else ""
                            item_type = row[5] if 5 in row.index else ""
                            start_time = int(row[11]) if 11 in row.index and pd.notna(row[11]) and str(row[11]).isdigit() else 0
                            sentence = row[18] if 18 in row.index else ""
                            response = row[19] if 19 in row.index else ""
                            list_file = row[14] if 14 in row.index else ""
                            
                            if pd.notna(sentence) and sentence:
                                experiment_rows.append({
                                    'participant': participant,
                                    'trial_number': trial_number,
                                    'item_type': item_type,
                                    'start_time': start_time,
                                    'end_time': start_time,
                                    'participant_word': response,
                                    'sentence_context': sentence,
                                    'list_file': list_file
                                })
                        except Exception as e:
                            print(f"Error processing pandas row: {e}")
            
            os.unlink(temp_path)
                
        except Exception as e:
            print(f"Error using pandas: {e}")
    
    if not experiment_rows:
        print("\nERROR: Could not extract any experiment rows from the file.")
        print("Please check the file format and try again.")
        sys.exit(1)
    
    print("Processing experiment data...")
    
    trial_groups = {}
    for row in experiment_rows:
        key = (row['participant'], row['trial_number'])
        if key not in trial_groups:
            trial_groups[key] = []
        trial_groups[key].append(row)
    
    clean_trials = []
    for key, rows in trial_groups.items():
        if not rows:
            continue
            
        sorted_rows = sorted(rows, key=lambda x: x['start_time'])
        
        first_time = sorted_rows[0]['start_time']
        last_time = sorted_rows[-1]['start_time']
        
        best_row = max(rows, key=lambda x: len(str(x['sentence_context'])) if x['sentence_context'] else 0)
        
        best_row['end_time'] = last_time
        best_row['duration_ms'] = last_time - first_time
        best_row['valid'] = True
        
        clean_trials.append(best_row)
    
    print(f"Created {len(clean_trials)} clean trial records")
    
    print("Matching with List data...")
    
    match_count = 0
    for row in clean_trials:
        sentence = normalize_sent(row['sentence_context'])
        if not sentence:
            continue
            
        match_found = False
        match = None
        
        exact_matches = combined_lists[combined_lists['sentence_norm'] == sentence]
        if len(exact_matches) > 0:
            match = exact_matches.iloc[0]
            match_found = True
            match_type = "exact"
        else:
            for _, list_row in combined_lists.iterrows():
                if pd.isna(list_row['sentence_norm']):
                    continue
                    
                if sentence in list_row['sentence_norm']:
                    match = list_row
                    match_found = True
                    match_type = "contained_in"
                    break
                    
                elif list_row['sentence_norm'] in sentence:
                    match = list_row
                    match_found = True
                    match_type = "contains"
                    break
        
        if match_found and match is not None:
            row['condition'] = match.get('condition')
            row['itemnum'] = match.get('itemnum')
            row['RemovalType'] = match.get('RemovalType')
            row['Code'] = match.get('Code')
            row['critical_word'] = match.get('critical_word')
            row['match_quality'] = match_type
            match_count += 1
            
            print(f"Matched: '{row['sentence_context']}' ({match_type})")
            print(f"  with: '{match.get('Sentence')}'\n")
    
    print(f"Found matches for {match_count} out of {len(clean_trials)} trials")
    
    if not clean_trials:
        print("ERROR: No valid trials to output!")
        sys.exit(1)
        
    print(f"Writing final data to {args.out}")
    clean_df = pd.DataFrame(clean_trials)
    clean_df.to_csv(args.out, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
