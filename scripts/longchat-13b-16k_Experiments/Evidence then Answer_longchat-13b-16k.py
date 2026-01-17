# Generated from: 3 Evidence then Answer-longchat.ipynb
# Converted at: 2026-01-17T14:55:01.140Z
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

# =========================
# CONFIG
# =========================
file_path = "nq-open-20_total_documents_gold_at_{k}.jsonl"
output_file = "Evidence_then_Answer_nq-open-20_total_documents_gold_at_{k}.jsonl"

# ---- CHANGE THESE FOR EACH RUN ----
START_IDX = 0        # e.g., 0, 900, 1800
END_IDX = None       # e.g., 900, 1800, None
# ==================================

# Load all examples
examples = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

total = len(examples)
print(f"Loaded {total} examples.")

# Slice for this batch
batch_examples = examples[START_IDX:END_IDX]
print(f"Running examples {START_IDX} to {START_IDX + len(batch_examples) - 1}")

def generate_evidence_and_answer(question, ctxs):
    """Generate evidence first, then answer using that evidence only."""
    context_texts = "\n\n".join([f"DOCUMENT {i+1}:\n{c['text']}" for i, c in enumerate(ctxs)])

    prompt = (
        "Given a question and a set of documents:\n"
        "1. FIRST, select EXACTLY ONE passage (word-for-word) from the documents that answers the question.\n"
        "   Copy the passage exactly, without paraphrasing or combining multiple passages.\n"
        "2. THEN, write a clear and concise answer using only that copied passage (not the entire documents).\n\n"
        "Output format:\n"
        "Evidence: ...verbatim quoted passage here...\n\n"
        "Answer: ...your answer here...\n\n"
        f"{context_texts}\n\n"
        f"Question: {question}\nEvidence:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prompt, decoded

def extract_last_evidence_answer(output_text):
    """Extract the last Evidence and Answer from the model output."""
    lines = output_text.strip().splitlines()
    evidence, answer = "", ""

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Answer:") and not answer:
            answer = lines[i].replace("Answer:", "").strip()
        if lines[i].startswith("Evidence:") and not evidence:
            evidence = lines[i].replace("Evidence:", "").strip()
        if evidence and answer:
            break

    return evidence, answer

# Open output file in APPEND mode
with open(output_file, "a", encoding="utf-8") as out_f:
    for i, ex in enumerate(
        tqdm(batch_examples, desc=f"Generating [{START_IDX}:{END_IDX}]")
    ):
        global_idx = START_IDX + i

        prompt_text, raw_output = generate_evidence_and_answer(
            ex["question"], ex["ctxs"]
        )
        evidence, answer = extract_last_evidence_answer(raw_output)

        out_record = {
            "index": global_idx,
            "question": ex["question"],
            "prompt": prompt_text,
            "evidence": evidence,
            "answer": answer
        }

        out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        # Print only first 3 of this batch
        if i < 3:
            print(f"\n--- Example {global_idx} ---")
            print(f"Question: {ex['question']}")
            print(f"Evidence: {evidence}")
            print(f"Answer: {answer}")

print(f"\nBatch [{START_IDX}:{END_IDX}] completed.")
print(f"Results appended to {output_file}")