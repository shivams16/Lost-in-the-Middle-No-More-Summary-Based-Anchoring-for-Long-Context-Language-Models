!pip install transformers accelerate sentencepiece tqdm

import json
import random
import re
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))


from transformers import logging

# Set logging to error only
logging.set_verbosity_error()


# # For 20 DOC


import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# MODEL SETUP
# =====================================================
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

TERMINATORS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# =====================================================
# CONFIGURATION (CHANGE ONLY START_IDX)
# =====================================================
BATCH_SIZE = 200
START_IDX = 0                    # 0, 200, 400, ...
END_IDX = BATCH_SIZE+START_IDX

CACHE_FILE = "abstract_cache.json"

FILES = [
    "20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl",
]

# =====================================================
# ABSTRACT GENERATION FUNCTION (LLAMA-3)
# =====================================================
def generate_abstract(paragraph: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise NLP system. "
                "Write a concise abstract in no more than 15 words."
            )
        },
        {
            "role": "user",
            "content": (
                "Describe what the following paragraph is about. "
                "Focus only on the main topic or purpose.\n\n"
                f"Paragraph:\n{paragraph}"
            )
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=40,
            eos_token_id=TERMINATORS,
            do_sample=False,
            temperature=0.0,
        )

    response = outputs[0][input_ids.shape[-1]:]
    abstract = tokenizer.decode(response, skip_special_tokens=True).strip()

    # Enforce strict 15-word limit
    abstract = " ".join(abstract.split()[:15])

    return abstract

# =====================================================
# LOAD ABSTRACT CACHE (FOR RESUME)
# =====================================================
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        text_to_abstract = json.load(f)
    print(f"Loaded abstract cache: {len(text_to_abstract)} entries")
else:
    text_to_abstract = {}
    print("Starting with empty abstract cache")

# =====================================================
# PROCESS FILES (ONE BATCH)
# =====================================================
for file_path in FILES:
    print(f"\nProcessing {file_path} | Batch {START_IDX}:{END_IDX}")

    with open(file_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    batch = examples[START_IDX:END_IDX]

    for q in tqdm(batch, desc=os.path.basename(file_path)):
        for doc in q.get("ctxs", []):
            text = doc.get("text", "")
            if not text:
                continue

            if text not in text_to_abstract:
                text_to_abstract[text] = generate_abstract(text)

            doc["abstract"] = text_to_abstract[text]

    output_file = file_path.replace(
        ".jsonl", f"_with_abstracts_{START_IDX}_{END_IDX}.jsonl"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for q in batch:
            f.write(json.dumps(q) + "\n")

    print(f"Saved batch output → {output_file}")

# =====================================================
# SAVE CACHE (CRITICAL)
# =====================================================
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(text_to_abstract, f)

print("\nBatch completed successfully.")
print(f"Abstract cache size: {len(text_to_abstract)}")
