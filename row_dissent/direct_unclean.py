import pandas as pd
import numpy as np

def process_csv(file_path="surprisal.csv"):
    print(f"Starting processing of file: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    prob_columns = [
        'gpt2newprob', 'gpt2xlnewprob', 'gptjnewprob', 'gptneonewprob',
        'gptneoxnewprob', 'olmonewprob', 'llama2newprob'
    ]

    # Surprisal column paired with each prob column (used downstream in R)
    surp_columns = {
        'gpt2newprob':    'gpt2new',
        'gpt2xlnewprob':  'gpt2xlnew',
        'gptjnewprob':    'gptjnew',
        'gptneonewprob':  'gptneonew',
        'gptneoxnewprob': 'gptneoxnew',
        'olmonewprob':    'olmonew',
        'llama2newprob':  'llama2new',
    }

    related_columns = {
        'gpt2newprob':    ['gpt2new', 'gpt2regionnew', 'gpt2regionnewprob'],
        'gpt2xlnewprob':  ['gpt2xlnew', 'gpt2xlregionnew', 'gpt2xlregionnewprob'],
        'gptjnewprob':    ['gptjnew', 'gptjregionnew', 'gptjregionnewprob'],
        'gptneonewprob':  ['gptneonew', 'gptneoregionnew', 'gptneoregionnewprob'],
        'gptneoxnewprob': ['gptneoxnew', 'gptneoxregionnew', 'gptneoxregionnewprob'],
        'olmonewprob':    ['olmonew', 'olmoregionnew', 'olmoregionnewprob'],
        'llama2newprob':  ['llama2new', 'llama2regionnew', 'llama2regionnewprob'],
    }

    # Drop any related cols that don't exist in this file
    for model, cols in related_columns.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Warning: missing related columns for {model}: {missing}")
        related_columns[model] = [c for c in cols if c in df.columns]

    # ------------------------------------------------------------------ #
    # Step 1: Rank each model's probability within each ITEM group
    #         and assign HC / MC / LC labels per row
    # ------------------------------------------------------------------ #
    for prob_col in prob_columns:
        print(f"Ranking: {prob_col}")
        cloze_col = f"{prob_col}cloze"
        df[cloze_col] = ""

        for item in df['ITEM'].unique():
            item_group = df[df['ITEM'] == item]

            if len(item_group) != 3:
                print(f"  Warning: ITEM {item} has {len(item_group)} rows (expected 3)")
                continue

            probs = item_group[prob_col].values
            highest_idx = int(np.argmax(probs))
            lowest_idx  = int(np.argmin(probs))

            if highest_idx == lowest_idx:
                middle_idx = [i for i in range(3) if i != highest_idx][0]
            else:
                middle_idx = list(set(range(3)) - {highest_idx, lowest_idx})[0]

            idxs = item_group.index.tolist()
            df.at[idxs[highest_idx], cloze_col] = "HC"
            df.at[idxs[middle_idx],  cloze_col] = "MC"
            df.at[idxs[lowest_idx],  cloze_col] = "LC"

    # ------------------------------------------------------------------ #
    # Step 2: Mark mismatches (* suffix) where model rank != cloze condition
    # ------------------------------------------------------------------ #
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        df[cloze_col] = df.apply(
            lambda row: f"{row[cloze_col]} *"
            if row[cloze_col] != row['condition'] else row[cloze_col],
            axis=1
        )

    # ------------------------------------------------------------------ #
    # Step 3: Per-ROW agree / disagree boolean flags
    #         agree_<model>    = True  when this row's model rank == cloze condition
    #         disagree_<model> = True  when this row's model rank != cloze condition
    #
    # A single ITEM's three rows can be split across both flags —
    # e.g. its HC row may agree while its MC row disagrees.
    # ------------------------------------------------------------------ #
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        short = prob_col.replace("newprob", "").replace("prob", "")  # e.g. "gpt2"

        df[f"disagree_{short}"] = (
            df[cloze_col].astype(str).str.contains(r"\*", na=False)
            & df[prob_col].notna()
        )
        df[f"agree_{short}"] = (
            ~df[f"disagree_{short}"]
            & df[prob_col].notna()
        )

    # ------------------------------------------------------------------ #
    # Step 4: Mismatch statistics (item-level, for plot annotations)
    # ------------------------------------------------------------------ #
    print("\n----- MISMATCH STATISTICS (item-level, for annotations) -----")
    total_items = len(df['ITEM'].unique())
    print(f"Total unique items: {total_items}")

    stats = {}
    for prob_col in prob_columns:
        cloze_col = f"{prob_col}cloze"
        has_mismatch = df[cloze_col].astype(str).str.contains(r"\*", na=False)
        item_mismatches = has_mismatch.groupby(df['ITEM']).any()
        mismatched_items = item_mismatches[item_mismatches].index.tolist()
        mismatch_count = len(mismatched_items)
        mismatch_pct = (mismatch_count / total_items) * 100

        stats[prob_col] = {
            'mismatch_count': mismatch_count,
            'mismatch_percentage': mismatch_pct,
            'mismatched_items': mismatched_items,
        }
        print(f"\n{prob_col}: {mismatch_count}/{total_items} items ({mismatch_pct:.1f}%)")

        # Also print row-level counts for sanity check
        short = prob_col.replace("newprob", "").replace("prob", "")
        n_agree    = df[f"agree_{short}"].sum()
        n_disagree = df[f"disagree_{short}"].sum()
        print(f"  Row-level agree rows: {n_agree}  |  disagree rows: {n_disagree}")

    # ------------------------------------------------------------------ #
    # Step 5: Save outputs
    # ------------------------------------------------------------------ #

    # Full annotated data (all rows, with cloze rank labels and agree/disagree flags)
    df.to_csv("cloze_comparison.csv", index=False)
    print("\nFull annotated data saved to cloze_comparison.csv")

    # Summary stats for plot annotations (same format as before)
    summary_df = pd.DataFrame({
        'LLM':                  list(stats.keys()),
        'Items_with_Mismatches':[s['mismatch_count']      for s in stats.values()],
        'Total_Items':          [total_items               for _ in stats],
        'Mismatch_Percentage':  [s['mismatch_percentage']  for s in stats.values()],
    })
    summary_df.to_csv("cloze_mismatches.csv", index=False)
    print("Mismatch summary saved to cloze_mismatches.csv")

    # The R script reads cloze_comparison.csv directly and uses the
    # agree_<model> / disagree_<model> boolean columns to subset.

    print("\nDone. Columns added to cloze_comparison.csv:")
    flag_cols = [c for c in df.columns if c.startswith("agree_") or c.startswith("disagree_")]
    for c in flag_cols:
        print(f"  {c}  (True count: {df[c].sum()})")

    return df, summary_df


df, summary = process_csv()

if summary is not None:
    print("\nMismatch Summary Statistics:")
    print(summary)
