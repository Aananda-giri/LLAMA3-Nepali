# LLAMA3.2 Nepali 318M Model

## Overview
This is a 318M parameter LLAMA3.2 model fine-tuned on a Nepali text dataset. The model is designed for generating coherent and contextually relevant Nepali text.

## Resources
- **Base Model:** [Hugging Face](https://huggingface.co/Aananda-giri/LLAMA3-Nepali)
- **Chat Interface:** [Hugging Face Space](https://huggingface.co/spaces/Aananda-giri/LLAMA3_Nepali_318M)
- **Dataset:** [IRIISNEPAL/Nepali-Text-Corpus](https://huggingface.co/datasets/IRIISNEPAL/Nepali-Text-Corpus) and [nepberta](https://nepberta.github.io/)
- **Reference Book:** *[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)* by Sebastian Raschka, PhD

## Installation
To install the required dependencies, run:
```sh
pip install datasets huggingface_hub matplotlib transformers torch --quiet
```

## Usage
### 1. Download Model Weights
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Aananda-giri/LLAMA3-Nepali", filename="parameters_300m/model_pg_398000_steps.pth", local_dir="./")
```

### 2. Load the Tokenizer
```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("Aananda-giri/LLAMA3-Nepali")
tokenizer.save_pretrained("NepaliBPE")
```

### 3. Download Additional Scripts
```python
import requests
res = requests.get("https://raw.githubusercontent.com/Aananda-giri/LLAMA3-Nepali/main/3.%20training_loop/previous_chapters.py")
with open('previous_chapters.py', 'w') as f:
    f.write(res.text)
```

### 4. Load the Model
```python
import torch
from previous_chapters import Llama3Model, ChatFormat, Tokenizer, generate_and_print_sample

# Initialize tokenizer
_tokenizer = Tokenizer("NepaliBPE/tokenizer.json")
chat_tokenizer = ChatFormat(_tokenizer)

# Define model configuration
LLAMA32_CONFIG = {
    "vocab_size": 50006,
    "context_length": 512,
    "emb_dim": 1320,
    "n_heads": 20,
    "n_layers": 10,
    "hidden_dim": 5280,
    "n_kv_groups": 5,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

# Adjust RoPE Scaling
old_context_length = 131_072
new_context_length = LLAMA32_CONFIG["context_length"]
LLAMA32_CONFIG["rope_base"] *= new_context_length / old_context_length

# Load Model
model = Llama3Model(LLAMA32_CONFIG)
model.eval()

# Optimize model if PyTorch 2.0 is available
if torch.__version__ >= "2.0":
    model = torch.compile(model)
```

### 5. Load Model Weights
```python
# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f'device: {device}')

# Load checkpoint
latest_model_checkpoint = "parameters_300m/model_pg_398000_steps.pth"
checkpoint = torch.load(latest_model_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
```

### 6. Generate Text
```python
# Generate text sample
generate_and_print_sample(
    PROMPT="रामले भात",
    tokenizer=_tokenizer,
    chat_tokenizer=chat_tokenizer,
    model=model,
    device=device,
    context_length=LLAMA32_CONFIG["context_length"]
)
```

#### Advanced Text Generation
```python
from previous_chapters import generate_chat_optimized
import time

start_time = time.time()
output_text = generate_chat_optimized(
    prompt="रामले भात",
    tokenizer=tokenizer,
    chat_tokenizer=chat_tokenizer,
    model=model,
    max_new_tokens=20,
    context_size=512,
    device=device,
    temperature=0.3,
    top_k=5,
    top_p=None,
    eos_id=None,
    repetition_penalty=1.2,
    penalize_len_below=10,
    batch_size=1  # Added parameter
)

print(f"time:{time.time() - start_time}\n output_text: {output_text}")
```

---

🚀 **Happy coding and enjoy experimenting with LLAMA3.2 Nepali!** 🤗🎉

---
