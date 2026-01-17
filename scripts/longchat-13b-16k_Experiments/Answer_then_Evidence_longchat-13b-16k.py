# Generated from: 2 Answer then Evidence-longchat.ipynb
# Converted at: 2026-01-17T14:54:51.963Z
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
from tqdm import tqdm
import re
import os

# =========================
# CONFIG
# =========================
file_path = "nq-open-20_total_documents_gold_at_{k}.jsonl"
output_file = "Answer_then_Evidence_nq-open-20_total_documents_gold_at_{k}.jsonl"

# ---- CHANGE THESE FOR EACH RUN ----
START_IDX = 0        # first run: 0, second run: 1300
END_IDX = None       # first run: 1300, second run: None
# ==================================

# Load all examples
examples = []
with open(file_path, "r") as f:
    for line in f:
        examples.append(json.loads(line))

total = len(examples)
print(f"Loaded {total} examples.")

# Slice for this batch
batch_examples = examples[START_IDX:END_IDX]
print(f"Running examples {START_IDX} to {START_IDX + len(batch_examples) - 1}")

def generate_answer(question, ctxs):
    """Build a prompt and generate answer using longchat model."""
    context_texts = "\n\n".join([f"DOCUMENT:\n{c['text']}" for c in ctxs])

    prompt = (
        "Given a question and a set of documents:\n"
        "1. Write a clear and concise answer, using information from ONLY ONE of the documents.\n"
        "2. After writing the answer, copy EXACTLY the passage (word-for-word) from which you derived the answer.\n"
        "   Do not summarize, paraphrase, or combine multiple passages. Use only the single passage you relied on.\n\n"
        "Output format:\n"
        "Answer: ...your answer here...\n\n"
        "Evidence: ...verbatim quoted passage here...\n\n"
        f"{context_texts}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prompt, decoded

def extract_question_answer_evidence(raw_output):
    pattern = re.compile(
        r"Question:\s*(.*?)\s*Answer:\s*(.*?)\s*Evidence:\s*(.*)",
        re.DOTALL
    )

    match = pattern.search(raw_output)
    if match:
        return (
            match.group(1).strip(),
            match.group(2).strip(),
            match.group(3).strip()
        )
    else:
        return "", "", ""

# Open output file in APPEND mode
with open(output_file, "a") as out_f:
    for i, ex in enumerate(
        tqdm(batch_examples, desc=f"Generating [{START_IDX}:{END_IDX}]")
    ):
        global_idx = START_IDX + i

        prompt, raw_output = generate_answer(ex["question"], ex["ctxs"])
        question, answer, evidence = extract_question_answer_evidence(raw_output)

        output_dict = {
            "index": global_idx,
            "prompt": prompt,
            "question": question,
            "answer": answer,
            "evidence": evidence
        }

        out_f.write(json.dumps(output_dict) + "\n")

        # Print only first 3 of this batch
        if i < 3:
            print(f"\n--- Example {global_idx} ---")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Evidence: {evidence}")

print(f"\nBatch [{START_IDX}:{END_IDX}] completed.")
print(f"Results appended to {output_file}")