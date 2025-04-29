import sys
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm
import argparse

# all models
MODELS = {
    'gpt2': 'gpt2',
    'gpt2xl': 'gpt2-xl',
    'gptj': 'EleutherAI/gpt-j-6B',
    'gptneo': 'EleutherAI/gpt-neo-1.3B',
    'gptneox': 'EleutherAI/gpt-neox-20b',
    'olmo': 'allenai/OLMo-7B-base',
    'llama2': 'meta-llama/Llama-2-7b-hf'
}

# batch sizes
MODEL_BATCH_SIZES = {
    'gpt2': 4,
    'gpt2xl': 4,
    'gptj': 1,
    'gptneo': 2,
    'gptneox': 1,
    'olmo': 1,
    'llama2': 1
}

ADD_BOS = True

items_path = os.path.join('bk21_items.csv')
items = pd.read_csv(items_path)

def prepare_data(items):
    words = items['words'].str.split().values.tolist()
    # Get critical word positions (convert to 0-indexed)
    critical_word_positions = (items['critical_word_pos'].values - 1).tolist()
    critical_words = items['critical_word'].tolist()
    keys = items['itemnum'].values.tolist()
    
    # Prepare prefixes and targets
    prefixes = []
    targets1 = []
    targets3 = []
    
    for key, sent, pos, critical_word in zip(keys, words, critical_word_positions, critical_words):
        prefixes.append(' '.join(sent[:pos]))
        target_toks = sent[pos:pos+3]
        assert target_toks[0] == critical_word, 'Error on item %d: expected critical word "%s", found "%s".' % (key, critical_word, target_toks[0])
        targets1.append(' ' + target_toks[0])         # Single critical target word, space needs to be added beforehand for the tokenizer
        targets3.append(' ' + ' '.join(target_toks))  # 3-word critical region, space needs to be added beforehand for the tokenizer
   
    return keys, prefixes, targets1, targets3

def calculate_surprisals(model, tokenizer, keys, prefixes, targets1, targets3, batch_size):
    """Calculate surprisal values using batch processing with CrossEntropyLoss."""
    
    with torch.no_grad():
        # Tokenize inputs
        tokenized_prefixes = tokenizer(prefixes)['input_ids']
        tokenized_targets1 = tokenizer(targets1)['input_ids']
        tokenized_targets3 = tokenizer(targets3)['input_ids']
        
        # Construct prompts and labels
        prompts = []
        labels1 = []
        labels3 = []
        attention_mask = []
        max_len = 0
        
        for prefix, target1, target3 in zip(tokenized_prefixes, tokenized_targets1, tokenized_targets3):
            if ADD_BOS:
                prefix = [tokenizer.bos_token_id] + prefix
            
            npad = len(prefix) - 1  # Don't predict the first token
            ndiff = len(target3) - len(target1)
            prompt = prefix + target3[:-1]
            
            prompts.append(prompt)
            labels1.append([-100] * npad + target1 + [-100] * ndiff)
            labels3.append([-100] * npad + target3)
            attention_mask.append([1] * len(prompt))
            max_len = max(max_len, len(prompt))
        
        # Pad sequences
        for i in range(len(prompts)):
            prompt = prompts[i]
            npad = max_len - len(prompt)
            prompts[i] = prompt + [tokenizer.pad_token_id] * npad
            labels1[i] = labels1[i] + [-100] * npad
            labels3[i] = labels3[i] + [-100] * npad
            attention_mask[i] = attention_mask[i] + [0] * npad
            
            # Send to device
        prompts = torch.tensor(prompts).to(model.device)
        labels1 = torch.tensor(labels1).to(model.device)
        labels3 = torch.tensor(labels3).to(model.device)
        attention_mask = torch.tensor(attention_mask).float().to(model.device)
      
        # Process in batches
        surprisals1 = []
        surprisals3 = []
        
        for i in tqdm(range(0, len(keys), batch_size), desc="Calculating surprisals"):
            # Get batch
            _prompts = prompts[i:i+batch_size].contiguous()
            _labels1 = labels1[i:i+batch_size].contiguous()
            _labels3 = labels3[i:i+batch_size].contiguous()
            _attention_mask = attention_mask[i:i+batch_size]
            
            # Call on batch
            outputs = model(_prompts, attention_mask=_attention_mask)
            logits = outputs['logits'].contiguous()
            logits[:, :, tokenizer.pad_token_id] = -float("inf")
            preds = logits.argmax(axis=-1) * _attention_mask.int() # * (_labels1 >= 0).int()
            logits = logits.permute((0, 2, 1))
            
        # Compute surprisals
            _surprisals1 = torch.nn.CrossEntropyLoss(reduction='none')(logits, _labels1)
            _surprisals1 = np.asarray(_surprisals1.cpu())
            _surprisals1 = _surprisals1.sum(axis=-1)
            surprisals1.append(_surprisals1)
            _labels1[np.where(_labels1 < 0)] = 0
            _output_mask = (_labels1 >= 0).int()

            _surprisals3 = torch.nn.CrossEntropyLoss(reduction='none')(logits, _labels3)
            _surprisals3 = np.asarray(_surprisals3.cpu())
            _surprisals3 = _surprisals3.sum(axis=-1)
            surprisals3.append(_surprisals3)
            
    surprisals1 = np.concatenate(surprisals1)
    surprisals3 = np.concatenate(surprisals3)
    
    return surprisals1, surprisals3

def get_hf_token(): #for Llama only
    """Prompt user for Huggingface token."""
    print("Llama2 model requires a Huggingface token.")
    token = input("Enter your Huggingface access token: ").strip()
    return token
    
def process_model(model_name, model_id, items):
    """Process a specific model and calculate surprisals."""
    print(f"\nProcessing model: {model_name} ({model_id})")
    batch_size = MODEL_BATCH_SIZES.get(model_name, 1)
    
    # Load model and tokenizer
    print(f"Loading {model_name} model and tokenizer...")
    if model_name == 'llama2':
        hf_token = get_hf_token()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token
        )
    elif model_name == 'olmo':
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

    # Set model to evaluation mode
    model.eval()
        
    # Prepare data
    keys, prefixes, targets1, targets3 = prepare_data(items)
        
    # Calculate surprisals
    new_surp1, new_surp3 = calculate_surprisals(
        model, tokenizer, keys, prefixes, targets1, targets3, batch_size)
        
    # Add calculated values to dataframe
    items[f'{model_name}new'] = new_surp1
    items[f'{model_name}newprob'] = np.exp(-new_surp1)
    items[f'{model_name}regionnew'] = new_surp3
    items[f'{model_name}regionnewprob'] = np.exp(-new_surp3)
        
    # Save results after each model
    items.to_csv(items_path, index=False)
    print(f"Updated results saved to {items_path}")
        
    # Clean up memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return True

def main():

    models_to_process = list(MODELS.keys())
        
    for model_name in models_to_process:
        if model_name in MODELS:
            success = process_model(model_name, MODELS[model_name], items)

    print(f"\nProcessing complete!")

if __name__ == "__main__":
    main()
