import pandas as pd
import numpy as np
import os

def process_csv(file_path="suprise.csv"):
    print(f"Starting processing of file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # List of probability columns to process
    prob_columns = [
        'gpt2newprob', 'gpt2xlnewprob', 'gptjnewprob', 'gptneonewprob',
        'gptneoxnewprob', 'olmonewprob', 'llama2newprob'
    ]
    
    # Process each probability column
    for prob_col in prob_columns:
        print(f"Processing column: {prob_col}")
        # Create a new column name for the classification
        cloze_col = f"{prob_col}cloze"
        
        # Initialize the new column with empty strings
        df[cloze_col] = ""
        
        # Get all the ITEM values to identify groups
        items = df['ITEM'].unique()
        
        # Process each ITEM group separately (since they contain 3 rows each)
        for item in items:
            # Get rows with current ITEM
            item_group = df[df['ITEM'] == item]
            
            if len(item_group) != 3:
                print(f"Warning: ITEM {item} has {len(item_group)} rows instead of expected 3")
                continue
                
            # Find indices for highest, middle, and lowest probabilities
            probs = item_group[prob_col].values
            highest_idx = np.argmax(probs)
            lowest_idx = np.argmin(probs)
            
            if highest_idx == lowest_idx:  # All values are the same
                middle_indices = [i for i in range(len(probs)) if i != highest_idx]
                middle_idx = middle_indices[0]
            else:
                # Middle is the remaining index
                all_indices = set(range(len(probs)))
                middle_indices = list(all_indices - {highest_idx, lowest_idx})
                middle_idx = middle_indices[0] if middle_indices else highest_idx
            
            item_indices = item_group.index.tolist()
            
            # Assign classifications
            df.at[item_indices[highest_idx], cloze_col] = "HC"
            df.at[item_indices[middle_idx], cloze_col] = "MC"
            df.at[item_indices[lowest_idx], cloze_col] = "LC"
    
    # Mark mismatches with an asterisk
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        # Compare with the original condition and add * if they don't match
        df[cloze_col] = df.apply(
            lambda row: f"{row[cloze_col]} *" if row[cloze_col] != row['condition'] else row[cloze_col],
            axis=1
        )
    
    # Calculate mismatch statistics per LLM per ITEM
    print("\n----- MISMATCH STATISTICS -----")
    
    # Count total unique items
    total_items = len(df['ITEM'].unique())
    print(f"Total unique items: {total_items}")
    
    # Create a stats dictionary to track mismatches per LLM
    stats = {}
    
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        
        # Create a column that identifies mismatches
        df[f"{cloze_col}_has_mismatch"] = df[cloze_col].str.contains('\*')
        
        # Group by ITEM and check if any sentence in the item has a mismatch
        item_mismatches = df.groupby('ITEM')[f"{cloze_col}_has_mismatch"].any()
        
        # Count items with mismatches
        mismatch_count = item_mismatches.sum()
        mismatch_percentage = (mismatch_count / total_items) * 100
        
        # Store stats
        stats[prob_col] = {
            'mismatch_count': mismatch_count,
            'mismatch_percentage': mismatch_percentage
        }
        
        # Print results
        print(f"\n{prob_col}:")
        print(f"  Items with mismatches: {mismatch_count} out of {total_items}")
        print(f"  Percentage: {mismatch_percentage:.2f}%")
    
    # Create a summary DataFrame for mismatches
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
    
    # Save the processed data to a new file
    output_file = "cloze_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")

    # Save the summary statistics
    summary_file = "cloze_mismatches.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Mismatch summary statistics saved to {summary_file}")

    return df, summary_df

# Directly call the function to process the file
result, summary = process_csv()

# Show the summary statistics
if summary is not None:
    print("\nMismatch Summary Statistics:")
    print(summary)
