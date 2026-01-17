# Generated from: 1 BASELINE-longchat.ipynb
# Converted at: 2026-01-17T14:54:04.560Z
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
import torch

# ============================
# File paths
# ============================
input_file = "nq-open-20_total_documents_gold_at_{k}.jsonl" # K is the gold document Position
output_file = "baseline_answers_nq-open-20_total_documents_gold_at_{k}.jsonl"

# ============================
# Load all examples
# ============================
examples = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples.")


# ============================
# Generation function (DOCUMENTS ONLY)
# ============================
def generate_answer(question, ctxs):
    """Generate ONLY the answer using full documents (NO abstracts)."""

    # Join all DOCS
    document_texts = "\n\n".join([
        f"DOCUMENT {i+1}:\n{c.get('text', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("text")
    ])

    # Build prompt
    prompt = (
        "You are given a question and a list of DOCUMENTS.\n\n"
        "Your task:\n"
        "- Write a clear and concise answer to the question.\n"
        "- Use information from ONLY ONE of the documents (not multiple).\n"
        "- Do not paraphrase or mix multiple sources.\n\n"
        "Output format:\n\n"
        "Answer: ...your answer here...\n\n"
        f"=== DOCUMENTS ===\n{document_texts}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    # Run model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prompt, decoded


# ============================
# Extract Question & Answer
# ============================
def extract_question_answer(raw_output):
    """Extract final Q and A."""
    pattern = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)
    matches = list(pattern.finditer(raw_output))

    if not matches:
        return "", ""

    last = matches[-1]
    return last.group(1).strip(), last.group(2).strip()


# ============================
# Run inference and save results
# ============================
model.eval()

with open(output_file, "w", encoding="utf-8") as out_f:
    for i, ex in enumerate(tqdm(examples, desc="Generating")):
        try:
            with torch.no_grad():
                prompt, raw_output = generate_answer(ex["question"], ex["ctxs"])

            question, answer = extract_question_answer(raw_output)

            output = {
                "prompt": prompt,
                "question": question,
                "answer": answer,
                "raw_output": raw_output
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")

            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"Answer: {answer}\n")

        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ Skipping example {i} due to CUDA OOM error.")
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            print(f"⚠️ Skipping example {i} due to unexpected error: {e}")
            continue

        torch.cuda.empty_cache()

print(f"\n✅ Saved all outputs to: {output_file}")