# creating 9 stimuli latin square design lists
import pandas as pd
import os

def read_critical_word_data():
    try:
        df = pd.read_csv("critical_word_data.csv")
        print(f"Loaded {len(df)} rows from critical word data")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample data:")
        print(df.head(3)[['words', 'itemnum', 'condition', 'critical_word', 'critical_word_pos']])
        return df
    except Exception as e:
        print(f"Error reading critical word data: {str(e)}")
        return None

def create_cloze_item(row, removal_type):
    sentence = row['words']
    critical_word_pos = row['critical_word_pos'] - 1  # Convert to 0-based indexing
    words = sentence.split()

    # Cloze
    if removal_type == 1:
        modified_words = words[:critical_word_pos]
    elif removal_type == 2:
        modified_words = words[:critical_word_pos + 1]
    elif removal_type == 3:
        modified_words = words[:critical_word_pos + 2]
    else:
        print(f"Warning: Unknown removal type {removal_type}")
        return None

    modified_sentence = " ".join(modified_words)
    new_row = row.copy()
    new_row['Sentence'] = modified_sentence
    new_row['RemovalType'] = removal_type
    new_row['WordsKept'] = len(modified_words)
    new_row['WordsRemoved'] = len(words) - len(modified_words)
    return new_row

def generate_9_lists_from_critical_data(df):
    # Sort by itemnum so group assignment is strict
    df = df.sort_values('itemnum').reset_index(drop=True)
    # Group IDs
    N_items = df.shape[0]
    N_groups = N_items // 3
    print(f"Total rows: {N_items}")
    print(f"Calculated {N_groups} item groups of 3.")

    latin_square = [
        [('HC', 1), ('MC', 2), ('LC', 3)],
        [('HC', 2), ('MC', 3), ('LC', 1)],
        [('HC', 3), ('MC', 1), ('LC', 2)],
        [('MC', 1), ('LC', 2), ('HC', 3)],
        [('MC', 2), ('LC', 3), ('HC', 1)],
        [('MC', 3), ('LC', 1), ('HC', 2)],
        [('LC', 1), ('HC', 2), ('MC', 3)],
        [('LC', 2), ('HC', 3), ('MC', 1)],
        [('LC', 3), ('HC', 1), ('MC', 2)]
    ]
    # Build lists structure
    output_lists = [[] for _ in range(9)]

    # Make a list of all groups (each group = 3 consecutive rows)
    group_rows = [df.iloc[i*3:i*3+3] for i in range(N_groups)]

    for group_idx, group_df in enumerate(group_rows):
        for list_num in range(9):
            # Latin square assignment: rotate across recipe and group
            cond, removal = latin_square[(group_idx + list_num) % 9][group_idx % 3]
            possible_rows = group_df[group_df['condition'] == cond]
            if possible_rows.empty:
                raise ValueError(f"No row found for condition {cond} in group {group_idx}")
            row = possible_rows.iloc[0]
            modified_row = create_cloze_item(row, removal)
            modified_row['ListNumber'] = list_num + 1
            modified_row['Code'] = f"{cond}{row['itemnum']}R{removal}"
            output_lists[list_num].append(modified_row)

    output_list_dicts = []
    # Convert to DF, print stats, etc.
    for list_num, list_items in enumerate(output_lists):
        df_list = pd.DataFrame(list_items)
        output_list_dicts.append({'list_number': list_num+1, 'dataframe': df_list})
        removal_type_counts = df_list['RemovalType'].value_counts().to_dict()
        condition_counts = df_list['condition'].value_counts().to_dict()
        print(f"List {list_num+1}: {len(df_list)} items")
        print(f"  Removal types: {removal_type_counts}")
        print(f"  Conditions: {condition_counts}")

    return output_list_dicts

def save_lists(output_lists, output_dir='output_lists'):
    os.makedirs(output_dir, exist_ok=True)
    for list_data in output_lists:
        list_num = list_data['list_number']
        filename = f"{output_dir}/List{list_num}.csv"
        list_data['dataframe'].to_csv(filename, index=False)
    print(f"Saved {len(output_lists)} lists to {output_dir}/ directory")

def main():
    try:
        print("Reading critical word data...")
        df = read_critical_word_data()
        if df is None:
            print("Error: Could not load critical word data.")
            return
        output_lists = generate_9_lists_from_critical_data(df)

        # Save summary file
        summary_data = []
        for list_data in output_lists:
            df_list = list_data['dataframe']
            removal_type_counts = df_list['RemovalType'].value_counts().to_dict()
            condition_counts = df_list['condition'].value_counts().to_dict()
            summary_data.append({
                'List Number': list_data['list_number'],
                'Total Items': len(df_list),
                'HC Condition': condition_counts.get('HC', 0),
                'MC Condition': condition_counts.get('MC', 0),
                'LC Condition': condition_counts.get('LC', 0),
                'Type 1 Removal': removal_type_counts.get(1, 0),
                'Type 2 Removal': removal_type_counts.get(2, 0),
                'Type 3 Removal': removal_type_counts.get(3, 0),
            })
        summary_df = pd.DataFrame(summary_data)
        os.makedirs('output_lists', exist_ok=True)
        summary_df.to_csv('output_lists/summary.csv', index=False)

        save_lists(output_lists)

        print("\nAll 9 lists have been generated and saved to the 'output_lists' directory.")
        print("Each list contains exactly one version of each item group (216 items total per list).")
        print("Words are removed based on critical word position with three different patterns:")
        print("- Type 1: Remove critical word and everything after")
        print("- Type 2: Keep critical word, remove everything after critical word")
        print("- Type 3: Keep critical word + 1 word after, remove everything after that")
        print("A summary file has been saved to output_lists/summary.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
