"""
add_surp_perword.py
-------------------
Modified version of add_surp.py that saves per-word surprisals
for W1, W2, W3 of the critical region separately, in addition
to the original summed columns.

Outputs columns like: gpt2_w1, gpt2_w2, gpt2_w3, gpt2prob_w1, etc.

Usage:
    python add_surp_perword.py <model_id> <model_shortname> <items_path>

Examples:
    python add_surp_perword.py gpt2 gpt2 bk21_data/items.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BATCH_SIZE = 4
ADD_BOS = True

if len(sys.argv) < 4:
    print("Usage: python add_surp_perword.py <model_id> <model_shortname> <items_path>")
    sys.exit(1)

model_id = sys.argv[1]
shortname = sys.argv[2]
items_path = sys.argv[3]

print(f"Model: {model_id} (shortname: {shortname})")
print(f"Items: {items_path}")

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Device: {device}")

# Load items
items = pd.read_csv(items_path)
words = items['words'].str.split().values.tolist()
critical_word_positions = (items['critical_word_pos'].values - 1).tolist()  # 0-indexed
critical_words = items['critical_word'].tolist()
keys = items['itemnum'].values.tolist()

# Build prefixes and per-word targets
prefixes_str = []
word1_str = []
word2_str = []
word3_str = []
for key, sent, pos, critical_word in zip(keys, words, critical_word_positions, critical_words):
    prefixes_str.append(' '.join(sent[:pos]))
    target_toks = sent[pos:pos + 3]
    assert target_toks[0] == critical_word, \
        f'Item {key}: expected "{critical_word}", found "{target_toks[0]}"'
    word1_str.append(' ' + target_toks[0])
    word2_str.append(' ' + target_toks[1])
    word3_str.append(' ' + target_toks[2])

# Tokenize everything
prefixes_tok = tokenizer(prefixes_str)['input_ids']
word1_tok = tokenizer(word1_str)['input_ids']
word2_tok = tokenizer(word2_str)['input_ids']
word3_tok = tokenizer(word3_str)['input_ids']

# Build prompts with labels that track which tokens belong to W1, W2, W3
all_surp_w1 = []
all_surp_w2 = []
all_surp_w3 = []

with torch.no_grad():
    for batch_start in range(0, len(keys), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(keys))
        b_size = batch_end - batch_start

        prompts = []
        masks_w1 = []  # 1 where token belongs to W1
        masks_w2 = []
        masks_w3 = []
        attn_masks = []
        max_len = 0

        for i in range(batch_start, batch_end):
            prefix = prefixes_tok[i]
            if ADD_BOS:
                prefix = [tokenizer.bos_token_id] + prefix

            w1 = word1_tok[i]
            w2 = word2_tok[i]
            w3 = word3_tok[i]

            # prefix + W1 + W2 + W3 tokens
            all_target_toks = w1 + w2 + w3
            prompt = prefix + all_target_toks[:-1]

            # which positions correspond to W1, W2, W3
            n_prefix = len(prefix) - 1  # positions to skip

            m1 = [0] * n_prefix + [1] * len(w1) + [0] * len(w2) + [0] * len(w3)
            m2 = [0] * n_prefix + [0] * len(w1) + [1] * len(w2) + [0] * len(w3)
            m3 = [0] * n_prefix + [0] * len(w1) + [0] * len(w2) + [1] * len(w3)

            # Full label sequence (for cross-entropy)
            labels = [-100] * n_prefix + all_target_toks

            # Trim masks to match prompt length
            prompt_len = len(prompt)
            m1 = m1[:prompt_len]
            m2 = m2[:prompt_len]
            m3 = m3[:prompt_len]
            labels = labels[:prompt_len]

            prompts.append((prompt, labels, m1, m2, m3))
            attn_masks.append([1] * prompt_len)
            max_len = max(max_len, prompt_len)

        # Pad everything
        prompt_batch = []
        label_batch = []
        m1_batch = []
        m2_batch = []
        m3_batch = []
        attn_batch = []

        for prompt, labels, m1, m2, m3 in prompts:
            npad = max_len - len(prompt)
            prompt_batch.append(prompt + [0] * npad)
            label_batch.append(labels + [-100] * npad)
            m1_batch.append(m1 + [0] * npad)
            m2_batch.append(m2 + [0] * npad)
            m3_batch.append(m3 + [0] * npad)

        for am in attn_masks:
            npad = max_len - len(am)
            attn_batch.append(am + [0] * npad)

        # To tensors
        prompt_t = torch.tensor(prompt_batch).to(device)
        label_t = torch.tensor(label_batch).to(device)
        attn_t = torch.tensor(attn_batch).float().to(device)
        m1_t = torch.tensor(m1_batch).float().to(device)
        m2_t = torch.tensor(m2_batch).float().to(device)
        m3_t = torch.tensor(m3_batch).float().to(device)

        # Forward pass
        outputs = model(prompt_t, attention_mask=attn_t)
        logits = outputs['logits'].contiguous()
        if tokenizer.pad_token_id is not None:
            logits[:, :, tokenizer.pad_token_id] = -float("inf")
        logits = logits.permute((0, 2, 1))

        # Per-token surprisals
        per_tok_surp = torch.nn.CrossEntropyLoss(reduction='none')(logits, label_t)

        # Sum surprisals within each word
        surp_w1 = (per_tok_surp * m1_t).sum(dim=-1).cpu().numpy()
        surp_w2 = (per_tok_surp * m2_t).sum(dim=-1).cpu().numpy()
        surp_w3 = (per_tok_surp * m3_t).sum(dim=-1).cpu().numpy()

        all_surp_w1.append(surp_w1)
        all_surp_w2.append(surp_w2)
        all_surp_w3.append(surp_w3)

        if (batch_start // BATCH_SIZE) % 50 == 0:
            print(f"  Processed {batch_end}/{len(keys)} items")

# Concatenate
surp_w1 = np.concatenate(all_surp_w1)
surp_w2 = np.concatenate(all_surp_w2)
surp_w3 = np.concatenate(all_surp_w3)

# Add to items
items[f'{shortname}_w1'] = surp_w1
items[f'{shortname}prob_w1'] = np.exp(-surp_w1)
items[f'{shortname}_w2'] = surp_w2
items[f'{shortname}prob_w2'] = np.exp(-surp_w2)
items[f'{shortname}_w3'] = surp_w3
items[f'{shortname}prob_w3'] = np.exp(-surp_w3)

# Also save the original summed columns for compatibility
items[f'{shortname}_region_check'] = surp_w1 + surp_w2 + surp_w3

items.to_csv(items_path, index=False)
print(f"Saved: {items_path}")
print(f"  Columns added: {shortname}_w1, {shortname}_w2, {shortname}_w3 (+ prob versions)")
print(f"  Region check (should match {shortname}region): mean diff = "
      f"{np.mean(np.abs(surp_w1 + surp_w2 + surp_w3 - items.get(f'{shortname}region', surp_w1 + surp_w2 + surp_w3).values)):.6f}")
