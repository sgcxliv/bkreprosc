# finding and identifying + cleaning mismatched cloze assignments
import pandas as pd
import numpy as np
import os

def process_csv(file_path="match.csv"):
    print(f"Starting processing of file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    prob_columns = [
        'gpt2newprob', 'gpt2xlnewprob', 'gptjnewprob', 'gptneonewprob',
        'gptneoxnewprob', 'olmoprob', 'llama2prob'
    ]
    
    related_columns = {
        'gpt2newprob': ['gpt2new', 'gpt2regionnew', 'gpt2regionnewprob'],
        'gpt2xlnewprob': ['gpt2xlnew', 'gpt2xlregionnew', 'gpt2xlregionnewprob'],
        'gptjnewprob': ['gptjnew', 'gptjregionnew', 'gptjregionnewprob'],
        'gptneonewprob': ['gptneonew', 'gptneoregionnew', 'gptneoregionnewprob'],
        'gptneoxnewprob': ['gptneoxnew', 'gptneoxregionnew', 'gptneoxregionnewprob'],
        'olmoprob': ['olmo', 'olmoregion', 'olmoregionprob'],
        'llama2prob': ['llama2', 'llama2region', 'llama2regionprob']
    }
    
    for model, cols in related_columns.items():
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing some related columns for {model}: {missing_cols}")
            related_columns[model] = [col for col in cols if col in df.columns]
    
    for prob_col in prob_columns:
        print(f"Processing column: {prob_col}")
        cloze_col = f"{prob_col}cloze"
        
        df[cloze_col] = ""
        items = df['group'].unique()
        
        for item in items:
            item_group = df[df['group'] == item]
            
            if len(item_group) != 3:
                print(f"Warning: ITEM {item} has {len(item_group)} rows instead of expected 3")
                continue
                
            probs = item_group[prob_col].values
            highest_idx = np.argmax(probs)
            lowest_idx = np.argmin(probs)
            
            if highest_idx == lowest_idx:  
                middle_indices = [i for i in range(len(probs)) if i != highest_idx]
                middle_idx = middle_indices[0]
            else:
                all_indices = set(range(len(probs)))
                middle_indices = list(all_indices - {highest_idx, lowest_idx})
                middle_idx = middle_indices[0] if middle_indices else highest_idx
            
            item_indices = item_group.index.tolist()
            
            df.at[item_indices[highest_idx], cloze_col] = "HC"
            df.at[item_indices[middle_idx], cloze_col] = "MC"
            df.at[item_indices[lowest_idx], cloze_col] = "LC"
    
    # Mark mismatches w *
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        df[cloze_col] = df.apply(
            lambda row: f"{row[cloze_col]} *" if row[cloze_col] != row['condition'] else row[cloze_col],
            axis=1
        )
    
    # mismatch statistics per LLM per ITEM
    print("\n----- MISMATCH STATISTICS -----")
    
    # total unique items
    total_items = len(df['group'].unique())
    print(f"Total unique items: {total_items}")

    # the statistics about it i guess
    stats = {}
    
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        
        df[f"{cloze_col}_has_mismatch"] = df[cloze_col].str.contains('\*')
        
        item_mismatches = df.groupby('group')[f"{cloze_col}_has_mismatch"].any()
        
        mismatched_items = item_mismatches[item_mismatches].index.tolist()
        
        mismatch_count = len(mismatched_items)
        mismatch_percentage = (mismatch_count / total_items) * 100
        
        stats[prob_col] = {
            'mismatch_count': mismatch_count,
            'mismatch_percentage': mismatch_percentage,
            'mismatched_items': mismatched_items
        }
        
        print(f"\n{prob_col}:")
        print(f"  Items with mismatches: {mismatch_count} out of {total_items}")
        print(f"  Percentage: {mismatch_percentage:.2f}%")
    
    # df for mismatches
    summary_data = {
        'LLM': [],
        'Items_with_Mismatches': [],
        'Total_Items': [],
        'Mismatch_Percentage': []
    }
    
    for prob_col, stat in stats.items():
        summary_data['LLM'].append(prob_col)
        summary_data['Items_with_Mismatches'].append(stat['mismatch_count'])
        summary_data['Total_Items'].append(total_items)
        summary_data['Mismatch_Percentage'].append(stat['mismatch_percentage'])
    
    summary_df = pd.DataFrame(summary_data)
    
    clean_df = df.copy()
    
    for prob_col in prob_columns:
        mismatched_items = stats[prob_col]['mismatched_items']
        
        model_cols = [prob_col] + related_columns.get(prob_col, [])
        
        for item in mismatched_items:
            item_mask = clean_df['group'] == item
            for col in model_cols:
                if col in clean_df.columns:
                    clean_df.loc[item_mask, col] = np.nan
            
            cloze_col = f"{prob_col}cloze"
            clean_df.loc[item_mask, cloze_col] = np.nan
    
    output_file = "cloze_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    # no mismatched items / cleaned data
    clean_output_file = "clean_cloze_data.csv"
    clean_df.to_csv(clean_output_file, index=False)
    print(f"Cleaned data with NaN values saved to {clean_output_file}")

    summary_file = "cloze_mismatches.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Mismatch summary statistics saved to {summary_file}")
    
    # count valid items
    valid_counts = {}
    for prob_col in prob_columns:
        model_cols = [prob_col] + related_columns.get(prob_col, [])
        valid_counts[prob_col] = clean_df[prob_col].notna().sum()
        valid_items = len(clean_df[clean_df[prob_col].notna()]['group'].unique())
        
        print(f"\n{prob_col} after cleaning:")
        print(f"  Valid entries: {valid_counts[prob_col]}")
        print(f"  Valid items: {valid_items} out of {total_items} ({valid_items/total_items:.2f}%)")
        
        for rel_col in related_columns.get(prob_col, []):
            if rel_col in clean_df.columns:
                rel_count = clean_df[rel_col].notna().sum()
                if rel_count != valid_counts[prob_col]:
                    print(f"  Warning: {rel_col} has {rel_count} valid entries (expected {valid_counts[prob_col]})")
                else:
                    print(f"  {rel_col}: {rel_count} valid entries ✓")

    return df, clean_df, summary_df

original_result, clean_result, summary = process_csv()

if summary is not None:
    print("\nMismatch Summary Statistics:")
    print(summary)
