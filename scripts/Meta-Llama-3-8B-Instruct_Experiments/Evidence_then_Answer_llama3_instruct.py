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


import json
import re
from tqdm import tqdm

# --------------------
# File paths
# --------------------
file_path = "20_total_documents/nq-open-20_total_documents_gold_at_{}.jsonl"
output_file = "20_total_documents/Evidence_then_Answer_nq-open-20_total_documents_gold_at_{}.jsonl"

# --------------------
# Load data
# --------------------
examples = []
with open(file_path, "r") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples.")

# --------------------
# Generation function (Evidence → Answer)
# --------------------
def generate_evidence_and_answer(question, ctxs):
    """Generate evidence first, then answer using that evidence only."""

    context_texts = "\n\n".join(
        [f"DOCUMENT {i+1}:\n{c['text']}" for i, c in enumerate(ctxs)]
    )

    user_prompt = (
        "Given a question and a set of documents:\n"
        "1. FIRST, select EXACTLY ONE passage (word-for-word) from the documents that answers the question.\n"
        "   Copy the passage exactly, without paraphrasing or combining multiple passages.\n"
        "2. THEN, write a clear and concise answer using only that copied passage (not the entire documents).\n\n"
        "Output format:\n"
        "Evidence: ...verbatim quoted passage here...\n\n"
        "Answer: ...your answer here...\n\n"
        f"{context_texts}\n\n"
        f"Question: {question}\n"
        "Evidence:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
    )

    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return user_prompt, decoded


# --------------------
# Extraction function
# --------------------
def extract_evidence_and_answer(raw_output):
    pattern = re.compile(
        r"Evidence:\s*(.*?)\s*Answer:\s*(.*)",
        re.DOTALL
    )

    match = pattern.search(raw_output)
    if match:
        evidence = match.group(1).strip()
        answer = match.group(2).strip()
        return evidence, answer
    else:
        return "", ""


# --------------------
# Run inference
# --------------------
with open(output_file, "w") as out_f:
    for i, ex in enumerate(tqdm(examples, desc="Generating")):
        prompt, raw_output = generate_evidence_and_answer(
            ex["question"], ex["ctxs"]
        )

        evidence, answer = extract_evidence_and_answer(raw_output)

        output_dict = {
            "prompt": prompt,
            "raw_output": raw_output,
            "question": ex["question"],
            "evidence": evidence,
            "answer": answer,
        }

        out_f.write(json.dumps(output_dict) + "\n")

        # Print ONLY first 3 raw outputs
        if i < 3:
            print(f"\n--- RAW MODEL OUTPUT {i+1} ---")
            print(raw_output)

print(f"\nSaved outputs to {output_file}")
