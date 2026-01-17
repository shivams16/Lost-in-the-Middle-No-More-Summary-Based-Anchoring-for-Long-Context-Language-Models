# Generated from: 4 Abstract Generation-Longchat.ipynb
# Converted at: 2026-01-17T14:55:07.615Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install transformers accelerate sentencepiece tqdm

import json
import random
import re
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model name
model_name = "lmsys/longchat-13b-16k"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Create generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.0,
    do_sample=False
)

import json
import os
from tqdm import tqdm

# =====================================================
# CONFIGURATION (CHANGE ONLY START_IDX)
# =====================================================
START_IDX = 0                    # 0, 200, 400, 600, ...
END_IDX = None

CACHE_FILE = "abstract_cache.json"

FILES = [
    "20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl",
    "20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl",
]

# =====================================================
# ABSTRACT GENERATION FUNCTION
# =====================================================
def generate_abstract(paragraph):
    prompt = (
        "Write in no more than 15 words what the following paragraph is about. "
        "Focus only on the main topic or purpose of the paragraph without adding opinions or extra commentary.\n\n"
        f"Paragraph:\n{paragraph}\n\nAbstract:"
    )

    output = generator(prompt)[0]['generated_text']
    # Extract only the abstract part (remove prompt echo)
    abstract = output.split("Abstract:")[-1].strip()
    # Limit to first 2 sentences
    sentences = abstract.split(".")
    abstract = ".".join(sentences[:10]).strip()
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

    # Save batch output
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