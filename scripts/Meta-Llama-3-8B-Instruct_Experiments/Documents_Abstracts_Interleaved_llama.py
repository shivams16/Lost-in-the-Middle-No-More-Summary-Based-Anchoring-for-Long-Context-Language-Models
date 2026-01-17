# Generated from: Documents_Abstracts_Interleaved_llama.ipynb
# Converted at: 2026-01-17T10:15:08.259Z
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


# ============================
# Model setup (LLaMA-3)
# ============================
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# LLaMA-3 stop tokens
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# ============================
# File paths
# ============================
input_file = "nq-open-20_total_documents_gold_at_k_with_abstracts_llama.jsonl"
output_file = "Answer_nq-open-20_total_documents_gold_at_k_with_abstracts_llama_Interleaved.jsonl"

# ============================
# Load all examples
# ============================
examples = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples.")

# ============================
# Generation function (No Evidence)
# ============================
def generate_answer(question, ctxs):
    """Generate ONLY the answer using abstracts + documents (LLaMA-3 chat format)."""

    # Interleaved ABSTRACT → DOCUMENT
    combined_blocks = []
    for i, c in enumerate(ctxs):
        if c.get("abstract"):
            combined_blocks.append(
                f"ABSTRACT {i+1}:\n{c['abstract'].strip()}"
            )
        if c.get("text"):
            combined_blocks.append(
                f"DOCUMENT {i+1}:\n{c['text'].strip()}"
            )

    combined_text = "\n\n".join(combined_blocks)

    # Chat-style messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise question answering system.\n"
                "You must answer using information from ONLY ONE document.\n"
                "Do not combine or paraphrase across multiple documents.\n"
                "Return only the final answer."
            )
        },
        {
            "role": "user",
            "content": (
                "You are given a question and context snippets.\n"
                "Each ABSTRACT is followed by its corresponding DOCUMENT.\n\n"
                "Output format:\n"
                "Answer: ...\n\n"
                f"=== CONTEXT (ABSTRACT + DOCUMENT PAIRS) ===\n"
                f"{combined_text}\n\n"
                f"Question: {question}"
            )
        }
    ]

    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=False,
        )

    # Decode only newly generated tokens
    generated = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    return decoded

# ============================
# Extract Answer only
# ============================
def extract_answer(text):
    """
    Extract text after 'Answer:' if present, else return full text.
    """
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

# ============================
# Run inference and save results
# ============================
with open(output_file, "w", encoding="utf-8") as out_f:
    for i, ex in enumerate(tqdm(examples, desc="Generating")):
        try:
            raw_output = generate_answer(ex["question"], ex["ctxs"])
            answer = extract_answer(raw_output)

            output = {
                "question": ex["question"],
                "answer": answer,
                "raw_output": raw_output
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")

            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {ex['question']}")
                print(f"Answer: {answer}\n")

        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ Skipping example {i} due to CUDA OOM.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"⚠️ Skipping example {i} due to error: {e}")
            continue

        torch.cuda.empty_cache()

print(f"\n✅ Saved all outputs to: {output_file}")