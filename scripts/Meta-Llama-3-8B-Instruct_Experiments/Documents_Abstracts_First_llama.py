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
# FILE PATHS
# =====================================================
input_file = "nq-open-20_total_documents_gold_at_k_with_abstracts_llama.jsonl"
output_file = "Answer_nq-open-20_total_documents_gold_at_k_with_abstracts_llama_First.jsonl"

# =====================================================
# LOAD DATA
# =====================================================
examples = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples.")

# =====================================================
# GENERATION FUNCTION (ABSTRACTS → DOCUMENTS)
# =====================================================
def generate_answer(question, ctxs):
    """Generate ONLY the answer using abstracts first, documents later."""

    # ABSTRACTS FIRST
    abstract_texts = "\n\n".join([
        f"ABSTRACT {i+1}:\n{c.get('abstract', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("abstract")
    ])

    # DOCUMENTS SECOND
    document_texts = "\n\n".join([
        f"DOCUMENT {i+1}:\n{c.get('text', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("text")
    ])

    user_prompt = (
        "You are given a question and two sets of context information:\n"
        "1. A list of ABSTRACTS summarizing the documents.\n"
        "2. The full DOCUMENTS themselves.\n\n"
        "Your task:\n"
        "- Write a clear and concise answer to the question.\n"
        "- Use information from ONLY ONE of the documents (not multiple).\n"
        "- Do not paraphrase or mix multiple sources.\n\n"
        "Output format:\n\n"
        "Answer: <answer text>\n\n"
        f"=== ABSTRACTS ===\n{abstract_texts}\n\n"
        f"=== DOCUMENTS ===\n{document_texts}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
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
# ANSWER EXTRACTION (NO EVIDENCE)
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
