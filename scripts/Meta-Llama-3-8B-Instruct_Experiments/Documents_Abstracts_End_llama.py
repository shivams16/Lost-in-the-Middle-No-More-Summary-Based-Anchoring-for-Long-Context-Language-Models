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


# =====================================================
# MODEL SETUP (Meta-Llama-3-8B-Instruct)
# =====================================================
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

# =====================================================
# BATCH CONFIGURATION
# =====================================================
BATCH_SIZE = 1000
START_IDX = 0                  # 0, 200, 400, ...
END_IDX = START_IDX + BATCH_SIZE

# =====================================================
# FILE PATHS
# =====================================================
input_file = "20_total_documents/Abstracts_llama/nq-open-20_total_documents_gold_at_0_with_abstracts_llama.jsonl"
output_file = "20_total_documents/Abstracts_llama/Answer_nq-open-20_total_documents_gold_at_0_with_abstracts_llama_End.jsonl"

# =====================================================
# LOAD DATA
# =====================================================
examples = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

total_examples = len(examples)
print(f"Loaded {total_examples} examples.")

# Slice batch
batch_examples = examples[START_IDX:END_IDX]
print(f"Running batch from index {START_IDX} to {START_IDX + len(batch_examples) - 1}")

# =====================================================
# GENERATION FUNCTION (DOCUMENTS → ABSTRACTS)
# =====================================================
def generate_answer(question, ctxs):
    """Generate ONLY the answer using documents first, abstracts later."""

    document_texts = "\n\n".join([
        f"DOCUMENT {i+1}:\n{c.get('text', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("text")
    ])

    abstract_texts = "\n\n".join([
        f"ABSTRACT {i+1}:\n{c.get('abstract', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("abstract")
    ])

    user_prompt = (
        "You are given a question and two sets of context information:\n"
        "1. The full DOCUMENTS\n"
        "2. ABSTRACTS summarizing those documents\n\n"
        "Your task:\n"
        "- Answer the question clearly and concisely\n"
        "- Use information from ONLY ONE document\n"
        "- Do not paraphrase or mix multiple sources.\n\n"
        "Output format:\n"
        "Answer: <answer text>\n\n"
        f"=== DOCUMENTS ===\n{document_texts}\n\n"
        f"=== ABSTRACTS ===\n{abstract_texts}\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": "You are a factual question answering assistant."},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
        )

    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return user_prompt, decoded

# =====================================================
# ANSWER EXTRACTION
# =====================================================
def extract_answer(raw_output):
    match = re.search(
        r"Answer:\s*(.*)",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""

# =====================================================
# RUN INFERENCE (BATCHED)
# =====================================================
with open(output_file, "a", encoding="utf-8") as out_f:
    for i, ex in enumerate(tqdm(batch_examples, desc="Generating")):
        global_idx = START_IDX + i
        try:
            prompt, raw_output = generate_answer(ex["question"], ex["ctxs"])
            answer = extract_answer(raw_output)

            output = {
                "index": global_idx,
                "prompt": prompt,
                "question": ex["question"],
                "answer": answer,
                "raw_output": raw_output,
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")

            if i < 3:
                print(f"\n--- Example {global_idx} ---")
                print(f"Question: {ex['question']}")
                print(f"Answer: {answer}\n")

        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ CUDA OOM at example {global_idx}, skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"⚠️ Error at example {global_idx}: {e}")
            continue

        torch.cuda.empty_cache()

print(f"\n✅ Batch {START_IDX}–{START_IDX + len(batch_examples) - 1} saved to:")
print(output_file)

# =====================================================
# MODEL SETUP (Meta-Llama-3-8B-Instruct)
# =====================================================
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

# =====================================================
# FILE PATHS
# =====================================================
input_file = "nq-open-20_total_documents_gold_at_k_with_abstracts_llama.jsonl"
output_file = "Answer_nq-open-20_total_documents_gold_at_k_with_abstracts_llama_End.jsonl"

# =====================================================
# LOAD DATA
# =====================================================
examples = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples.")

# =====================================================
# GENERATION FUNCTION (DOCUMENTS → ABSTRACTS)
# =====================================================
def generate_answer(question, ctxs):
    """Generate ONLY the answer using documents first, abstracts later."""

    # DOCUMENTS FIRST
    document_texts = "\n\n".join([
        f"DOCUMENT {i+1}:\n{c.get('text', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("text")
    ])

    # ABSTRACTS SECOND
    abstract_texts = "\n\n".join([
        f"ABSTRACT {i+1}:\n{c.get('abstract', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("abstract")
    ])

    user_prompt = (
        "You are given a question and two sets of context information:\n"
        "1. The full DOCUMENTS\n"
        "2. ABSTRACTS summarizing those documents\n\n"
        "Your task:\n"
        "- Answer the question clearly and concisely\n"
        "- Use information from ONLY ONE document\n"
        "- Do NOT combine multiple documents\n\n"
        "Output format:\n"
        "Answer: <answer text>\n\n"
        f"=== DOCUMENTS ===\n{document_texts}\n\n"
        f"=== ABSTRACTS ===\n{abstract_texts}\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": "You are a factual question answering assistant."},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
        )

    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return user_prompt, decoded

# =====================================================
# ANSWER EXTRACTION 
# =====================================================
def extract_answer(raw_output):
    match = re.search(
        r"Answer:\s*(.*)",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""

# =====================================================
# RUN INFERENCE
# =====================================================
with open(output_file, "w", encoding="utf-8") as out_f:
    for i, ex in enumerate(tqdm(examples, desc="Generating")):
        try:
            prompt, raw_output = generate_answer(ex["question"], ex["ctxs"])
            answer = extract_answer(raw_output)

            output = {
                "prompt": prompt,
                "question": ex["question"],
                "answer": answer,
                "raw_output": raw_output,
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")

            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {ex['question']}")
                print(f"Answer: {answer}\n")

        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ CUDA OOM at example {i}, skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"⚠️ Error at example {i}: {e}")
            continue

        torch.cuda.empty_cache()

print(f"\n✅ Saved all outputs to: {output_file}")
