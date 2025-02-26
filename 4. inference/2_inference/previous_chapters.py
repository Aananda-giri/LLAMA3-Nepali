# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import json

# modified from `import tiktoken`
from transformers import PreTrainedTokenizerFast


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# modified. added for create_dataloader_v2
from datasets import load_dataset


#####################################
# 1. Dataloader
#####################################

def create_dataloader_v3(batch_size, shuffle=True, drop_last=True, num_workers=0):
    '''
    modified.
    * parameter: text removed
    * parameters: max_length and stride removed : they were set during preparing tokenized_datasets
    * parameter: context_length removed (as dataset is pre-tokenized)
    '''
    print('downloading dataset...')
    # Download the whole dataset
    base_url = "https://huggingface.co/datasets/Aananda-giri/nepali_llm_datasets/resolve/main/pre_tokenized/"
    # data_files = {"train": base_url + "nepberta_" + str(context_length) + ".parquet"}
    
    # previous version: stride = .75*512, context_len.512
    # data_files = {
    #     "train": base_url + "iriisnepal_u_nepberta_train_512.parquet",
    #     "test": base_url + "iriisnepal_u_nepberta_test_512.parquet"
    #     }
    
    # context_len.512, stride=512
    data_files={"train": base_url + "iriis_u_nepbert_512_512_train.parquet", "validation": base_url + "iriis_u_nepbert_512_512_test.parquet"}
    dataset = load_dataset("parquet", data_files=data_files, cache_dir='hf_cache', streaming=True)
    
    print(dataset)

    # and split it later
    # dataset = dataset.train_test_split(train_size=train_ratio, seed=42)
    # Convert Hugging Face Dataset to PyTorch tensors (we can directly use the dataset as it is already in the correct format)
    # dataset.set_format(type="torch", columns=['input_ids,target_ids'])  # Directly set columns to torch tensors



    # Define the custom collate_fn function
    def collate_fn(batch):
        # Extract the 'input_ids' and 'target_ids' from the batch and return them as a list of tensors
        input_ids = []
        target_ids = []
        for data_item in batch:
            splitted_data_item = data_item['input_ids,target_ids'].split("\",")
            input_ids.append(torch.tensor(json.loads(splitted_data_item[0].replace('\"',''))))
            # print(f'input_ids: {type(input_ids)} {input_ids}')
            target_ids.append(torch.tensor(json.loads(splitted_data_item[1].replace('\"',''))))
            # print(f'target_ids: {type(target_ids)} {target_ids}')

        # Convert to tensors (if not already)
        input_ids_tensor = torch.stack(input_ids)
        target_ids_tensor = torch.stack(target_ids)

        return [input_ids_tensor, target_ids_tensor]

    
    # Creating the DataLoader for the 'train' split of the dataset with the custom collate_fn
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader =  DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader




#####################################
# 2. Architecture Code
#####################################
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,
            rope_base=10_000,
            rope_config=None,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, f"d_out:{d_out} must be divisible by num_heads:{num_heads}"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att =  GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16))   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(torch.bfloat16))
        x = x + shortcut  # Add the original input back

        return x

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits

#####################################
# 3. Load Tokenizer
#####################################



import os

from transformers import PreTrainedTokenizerFast
class Tokenizer:
    def __init__(self, tokenizer_model_path):
        assert os.path.isfile(tokenizer_model_path), f"Tokenizer Model file {tokenizer_model_path} not found"
        
        # load the tokenizehere
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_model_path)
        
        # previously added
        # self.special_tokens = {
        #     "<|begin_of_text|>": 128000,
        #     "<|end_of_text|>": 128001,
        #     "<|start_header_id|>": 128006,
        #     "<|end_header_id|>": 128007,
        #     "<|eot_id|>": 128009,
        # }
        

        
    def encode(self, text, bos=False, eos=False):
        '''
        parameter: allowed_special removed
        parameter: disallowed_special removed
        '''
        if bos:
            # tokens = [self.special_tokens["<|begin_of_text|>"]]
            tokens = self.tokenizer.encode('<|begin_of_text|>') # [50000]
        else:
            tokens = []

        tokens += self.tokenizer.encode(text)

        if eos:
            # tokens.append(self.special_tokens["<|end_of_text|>"])
            tokens.append(self.tokenizer.encode('<|end_of_text|>')[0]) # [50001]
        return tokens

    def decode(self, tokens):
        # return self.model.decode(tokens)
        return self.tokenizer.decode(tokens)


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.tokenizer.encode('<|start_header_id|>')[0]) # 50002
        tokens.extend(self.tokenizer.encode(message["भूमिका"], bos=False, eos=False))
        tokens.append(self.tokenizer.tokenizer.encode('<|end_header_id|>')[0])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        # tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        # tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text):
        message = {
            "भूमिका": "प्रयोगकर्ता",
            "सन्दर्भ": text
        }

        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["सन्दर्भ"].strip(), bos=False, eos=False)
        )
        # tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        tokens.append(self.tokenizer.tokenizer.encode('<|eot_id|>')[0])
        return tokens

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

# tokenizer = Tokenizer("tokenizer.json")
# chat_tokenizer = ChatFormat(tokenizer)

# text = "नेपाल विद्युत प्राधिकरणका कार्यकारी निर्देशक कुलमान घिसिङले माथिल्लो अरुण जलविद्युत आयोजना विश्व बैंक र एडीबीबाट वित्तीय व्यवस्थापन नभए नेपाली जनताको लगानीमा बनाउने तयारी रहेको बताएका छन् ।"
# # normal tokenizer
# print([tokenizer.tokenizer.decode([token]) for token in tokenizer.encode(text)])

# # formatted tokenizer
# print([tokenizer.tokenizer.decode([token]) for token in chat_tokenizer.encode(text)])


#####################################
# 4. Generate Text
####################################
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor
    # '''
    #   we have modified above return statement by sebastian because there are no tokens like 'start_header_id', 'end_header_id' and tokenizer is returning None which inturn is giving error
    #   TODO: add special tokens: 'start_header_id', 'end_header_id' and uncomment above return statement
    # '''
    # print(encoded_tensor)
    # return torch.tensor([token for token in encoded_tensor])  # TODO: use additional vocab like encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def generate_and_print_sample(PROMPT, tokenizer, chat_tokenizer, model, device, context_length):
    
    # PROMPT = "What do llamas eat?"
    # PROMPT="रामले भात"
    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),
        max_new_tokens=150,
        context_size=context_length,
        top_k=1,
        temperature=0.
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    
    print("Output text:\n", clean_text(output_text))

# -------------------------------------------------------------
# Generte sample text
# PROMPT = "लामा हरु ले के खान्छन् ?"

# torch.manual_seed(123)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),
#     max_new_tokens=150,
#     context_size=LLAMA32_CONFIG["context_length"],
#     top_k=1,
#     temperature=0.
# )

# output_text = token_ids_to_text(token_ids, tokenizer)
# -------------------------------------------------------------

def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # Find the index of the first occurrence of "<|end_header_id|>"
    index = text.find(header_end)

    if index != -1:
        # Return the substring starting after "<|end_header_id|>"
        return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
    else:
        # If the token is not found, return the original text
        return text

# print("Output text:\n", clean_text(output_text))


##########################################################################
# Chapter 5 (keep everything as it is except `generate_and_print_sample` function)
########################################################################


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None, len_data_loader=0):
    '''
        - parameter: len_data_loader=None <added to set len_data_loader since we cant calulate len(data_loader) of Iterable>
    '''
    total_loss = 0.

    if len_data_loader == 0:    # len(data_loader)
        return float("nan")
    elif num_batches is None:
        num_batches = len_data_loader
    else:
        num_batches = min(num_batches, len_data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter, len_train_loader=0, len_val_loader=0):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, len_data_loader = len_train_loader)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, len_data_loader = len_val_loader)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(PROMPT, tokenizer, chat_tokenizer, model, device, context_length):
    
    # PROMPT = "What do llamas eat?"
    # PROMPT="रामले भात"
    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),
        max_new_tokens=150,
        context_size=context_length,
        top_k=1,
        temperature=0.
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    
    print("Output text:\n", clean_text(output_text))

def generate(
    model,
    prompt,
    tokenizer,
    max_new_tokens,
    temperature=0.7,
    top_k=50,
    top_p=None,  # New parameter for nucleus sampling
    eos_id=None,
    repetition_penalty=1.2,
    penalize_len_below=50
):
    context_size = GPT_CONFIG_124M['context_length']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    
    if not eos_id:
        encoded_endoftext = tokenizer.encode("<|endoftext|>")
        eos_id = encoded_endoftext[0] if encoded_endoftext else None

    token_freq = {}

    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply repetition penalty
        for token_id in idx[0].tolist():
            if token_id in token_freq:
                logits[0, token_id] /= repetition_penalty
            else:
                token_freq[token_id] = 1
        
        # Penalize EOT token for shorter sequences
        if eos_id is not None and step < penalize_len_below:
            logits[0, eos_id] /= (penalize_len_below - step) / penalize_len_below

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Apply top-p (nucleus) sampling if specified
        if top_p:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask for indices to remove
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            
            # Renormalize probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # If top_p is None, apply top-k sampling
        elif top_k:
            top_probs, top_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
            # Renormalize probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample from the filtered distribution
        if temperature > 0.0:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        text = token_ids_to_text(idx, tokenizer)

    return text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(output_dir / "losses.pdf")



# --------------------------------------------------------------------------------
#     -------------------------- New Chat function ---------------------
# --------------------------------------------------------------------------------
def generate_chat(
    model,
    prompt,
    tokenizer,
    chat_tokenizer,
    max_new_tokens,
    context_size,
    temperature=0.7,
    top_k=50,
    top_p=None,  # Nucleus sampling
    eos_id=None,
    repetition_penalty=1.2,
    penalize_len_below=50,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    idx = text_to_token_ids(prompt, chat_tokenizer).to(device)
    
    if not eos_id and "<|endoftext|>" in tokenizer.get_vocab():
        encoded_endoftext = tokenizer.encode("<|endoftext|>")
        eos_id = encoded_endoftext[0] if encoded_endoftext else None
    elif not eos_id and "<|eot_id|>" in tokenizer.get_vocab():
        encoded_endoftext = tokenizer.encode("<|eot_id|>")
        eos_id = encoded_endoftext[0] if encoded_endoftext else None

    token_freq = {}

    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply repetition penalty
        for token_id in idx[0].tolist():
            if token_id in token_freq:
                token_freq[token_id] += 1
                logits[0, token_id] /= repetition_penalty
            else:
                token_freq[token_id] = 1
        
        # Penalize EOT token for shorter sequences
        if eos_id is not None and step < penalize_len_below:
            penalty_factor = 1.0 + (penalize_len_below - step) / penalize_len_below
            logits[0, eos_id] /= penalty_factor

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Apply top-p (nucleus) sampling if specified
        if top_p and top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask for indices to remove
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            
            # Renormalize probabilities
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero

        # If top_p is None or 0, apply top-k sampling
        elif top_k and top_k > 0:
            top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
            # Renormalize probabilities
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)  # Avoid division by zero

        # Sample from the filtered distribution
        if temperature > 0.0:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # Add the next token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check for end of sequence token
        if idx_next.item() == eos_id:
            break

    return idx

def clean_chat_output(text):
    """Clean up the generated text to remove repetition and artifacts."""
    # Remove repetitive patterns (like the example with repeated "म त यो देशको प्रधानमन्त्री हुँ")
    import re
    
    # Handle endoftext markers
    text = re.sub(r'<\|endoftext\|>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|eot_id\|>.*$', '', text, flags=re.DOTALL)
    
    # Remove excessive repetition (3 or more identical sentences)
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = None
    repetition_count = 0
    
    for line in lines:
        if line == prev_line:
            repetition_count += 1
            if repetition_count > 2:  # Skip if this is the 3rd or more repetition
                continue
        else:
            repetition_count = 0
        
        cleaned_lines.append(line)
        prev_line = line
    
    text = '\n'.join(cleaned_lines)
    
    # Also clean repetitive phrases within a single line
    words = text.split()
    cleaned_words = []
    repetition_window = []
    
    for word in words:
        if len(repetition_window) >= 3 and all(w == word for w in repetition_window[-3:]):
            continue  # Skip this word if the last 3 words were identical to it
        cleaned_words.append(word)
        repetition_window.append(word)
        if len(repetition_window) > 10:  # Keep a limited window
            repetition_window.pop(0)
    
    return ' '.join(cleaned_words).strip()

def generate_and_print_chat(
    prompt,
    tokenizer,
    chat_tokenizer,
    model,
    device=None,
    max_new_tokens=150,
    context_length=None,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    clean_text=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if context_length is None:
        # Try to get from model config or use default
        context_length = getattr(model, "context_length", 2048)
    
    # Generate tokens
    token_ids = generate_chat(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        chat_tokenizer=chat_tokenizer,
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device
    )

    # Convert tokens to text
    output_text = token_ids_to_text(token_ids, tokenizer)
    
    if clean_text:
        # Clean the output 
        cleaned_text = clean_chat_output(output_text)
    
        print("Generated text:\n", cleaned_text)
    
        return cleaned_text
    else:
        print("Generated text:\n", output_text)
    
        return output_text