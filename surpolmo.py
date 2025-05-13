# for some reason i needed to run olmo separately bc i could not get it to work w / all the other models at once
import sys
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Define OLMo model 
MODEL_NAME = 'olmo'
MODEL_ID = 'allenai/OLMo-7B'  # HuggingFace model ID for OLMo
BATCH_SIZE = 2  

def load_data(input_path):
    """Load the existing data with previously calculated surprisals."""
    print(f"Loading data from {input_path}")
    
    # chunk processing to handle large size
    chunk_size = 50000
    chunks = pd.read_csv(input_path, chunksize=chunk_size)
    data_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Loading chunk {i+1}...")
        data_chunks.append(chunk)
    
    # recombine
    data = pd.concat(data_chunks, ignore_index=True)
    print(f"Loaded {len(data)} rows")
    
    return data

def calculate_surprisals(items, output_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Calculate surprisal values for OLMo."""
    print(f"\nProcessing model: {MODEL_NAME} ({MODEL_ID})")
    
    try:
        print(f"Loading {MODEL_NAME} model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        if 'word_list' not in items.columns:
            items['word_list'] = items['words'].apply(lambda x: str(x).split())
        
        print("Creating lookup keys for unique sentences...")
        if 'sentence_key' not in items.columns:
            items['sentence_key'] = items.apply(
                lambda row: f"{row['words']}_{row['critical_word_pos']}", 
                axis=1
            )
        
        unique_keys = items['sentence_key'].unique()
        print(f"Found {len(unique_keys)} unique sentence-position combinations")
        
        key_to_indices = {}
        for i, key in enumerate(items['sentence_key']):
            if key not in key_to_indices:
                key_to_indices[key] = []
            key_to_indices[key].append(i)
        
        unique_combos = []
        for key in unique_keys:
            idx = key_to_indices[key][0]
            unique_combos.append({
                'idx': idx,
                'sent': items.iloc[idx]['word_list'],
                'pos': items.iloc[idx]['critical_word_pos'] - 1,  # Convert to 0-indexed
                'word': items.iloc[idx]['critical_word'],
                'key': key
            })
        
        prefixes = []
        targets1 = []
        targets3 = []
        valid_indices = []
        
        for i, combo in enumerate(unique_combos):
            try:
                sent = combo['sent']
                pos = combo['pos']
                
                if pos < 0:
                    pos = 0
                elif pos >= len(sent):
                    pos = len(sent) - 1
                
                prefixes.append(' '.join(sent[:pos]))
                targets1.append(' ' + sent[pos])
                
                if pos + 3 <= len(sent):
                    targets3.append(' ' + ' '.join(sent[pos:pos+3]))
                else:
                    targets3.append(' ' + ' '.join(sent[pos:]))
                
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping item {i} due to error: {e}")
        
        print(f"Processing {len(prefixes)} unique sentence-position combinations")
        
        combo_surprisals1 = {} 
        combo_surprisals3 = {}
        
        with torch.no_grad():
            for i in tqdm(range(0, len(prefixes), BATCH_SIZE), desc=f"{MODEL_NAME} processing"):
                batch_end = min(i + BATCH_SIZE, len(prefixes))
                
                for j in range(i, batch_end):
                    combo_idx = valid_indices[j]
                    combo = unique_combos[combo_idx]
                    
                    prefix = prefixes[j]
                    target1 = targets1[j]
                    target3 = targets3[j]
                    
                    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
                    target1_ids = tokenizer.encode(target1, add_special_tokens=False)
                    target3_ids = tokenizer.encode(target3, add_special_tokens=False)
                    
                    outputs = model(input_ids)
                    logits = outputs.logits
                    
                    single_surprisal = calculate_token_surprisal(
                        model, tokenizer, input_ids.squeeze().tolist(), target1_ids, device
                    )
                    combo_surprisals1[combo['key']] = single_surprisal
                    
                    region_surprisal = calculate_token_surprisal(
                        model, tokenizer, input_ids.squeeze().tolist(), target3_ids, device
                    )
                    combo_surprisals3[combo['key']] = region_surprisal
                    
                # period progress updates
                if i % 100 == 0 and i > 0:
                    surprisals1 = [combo_surprisals1.get(key, np.nan) for key in items['sentence_key']]
                    surprisals3 = [combo_surprisals3.get(key, np.nan) for key in items['sentence_key']]
                    
                    items[f'{MODEL_NAME}'] = surprisals1
                    items[f'{MODEL_NAME}prob'] = [np.exp(-s) if not np.isnan(s) else np.nan for s in surprisals1]
                    items[f'{MODEL_NAME}region'] = surprisals3
                    items[f'{MODEL_NAME}regionprob'] = [np.exp(-s) if not np.isnan(s) else np.nan for s in surprisals3]
                    
                    items.to_csv(output_path, index=False)
                    print(f"Saved progress after {batch_end} items")
            
            # Final update
            surprisals1 = [combo_surprisals1.get(key, np.nan) for key in items['sentence_key']]
            surprisals3 = [combo_surprisals3.get(key, np.nan) for key in items['sentence_key']]
            
            items[f'{MODEL_NAME}'] = surprisals1
            items[f'{MODEL_NAME}prob'] = [np.exp(-s) if not np.isnan(s) else np.nan for s in surprisals1]
            items[f'{MODEL_NAME}region'] = surprisals3
            items[f'{MODEL_NAME}regionprob'] = [np.exp(-s) if not np.isnan(s) else np.nan for s in surprisals3]
            
            # Cleanup to free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
    
    except Exception as e:
        print(f"Error processing model {MODEL_NAME}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def calculate_token_surprisal(model, tokenizer, prefix_ids, target_ids, device):
    """Calculate surprisal for target tokens given prefix."""
    input_ids = torch.tensor([prefix_ids]).to(device)
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    total_surprisal = 0
    current_input = input_ids.clone()
    
    for target_id in target_ids:
        next_token_logits = logits[0, -1, :]
        
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        
        token_prob = next_token_probs[target_id].item()
        
        if token_prob > 0:
            token_surprisal = -np.log(token_prob)
        else:
            token_surprisal = 100  
        
        total_surprisal += token_surprisal
        
        current_input = torch.cat([current_input, torch.tensor([[target_id]]).to(device)], dim=1)
        
        outputs = model(current_input)
        logits = outputs.logits
    
    return total_surprisal

def main():
    input_path = '/afs/cs.stanford.edu/u/sgcxliv/bk21_itemsllama.csv'
    
    output_path = '/afs/cs.stanford.edu/u/sgcxliv/bk21_items_surprisal.csv'
    
    try:
        items = load_data(input_path)
        
        success = calculate_surprisals(items, output_path)
        
        if success:
            items.to_csv(output_path, index=False)
            print(f"\nComplete! Successfully added OLMo surprisals")
            print(f"Results saved to {output_path}")
        else:
            print("Failed to add OLMo surprisals")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
