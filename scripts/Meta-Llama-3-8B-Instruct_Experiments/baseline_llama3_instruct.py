!pip install transformers accelerate sentencepiece tqdm

import json
import random
import re
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from huggingface_hub import login
login(token="your_hf_token")

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
import torch
from tqdm import tqdm

# ============================
# Config
# ============================
BASE_DIR = "20_total_documents"
POSITION =    # 👈 change this to any position you want (0, 4, 9, ...)

INPUT_FILE = f"{BASE_DIR}/nq-open-20_total_documents_gold_at_{POSITION}.jsonl"
OUTPUT_FILE = f"{BASE_DIR}/baseline_answers_nq-open-20_total_documents_gold_at_{POSITION}.jsonl"

START_IDX = 0
END_IDX = None
MAX_NEW_TOKENS = 512

# ============================
# Load examples
# ============================
examples = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

batch_examples = examples[START_IDX:END_IDX]
print(f"Loaded {len(batch_examples)} examples from position {POSITION}")

# ============================
# Robust extractors
# ============================
def extract_clean_question(text):
    match = re.search(
        r"Question:\s*(.*?)\s*assistant",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def extract_clean_answer(text):
    matches = re.findall(
        r"assistant\s*(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not matches:
        return ""

    answer = matches[-1].strip()

    answer = re.split(
        r"\n\s*(Question:|DOCUMENT\s+\d+:|=== DOCUMENTS ===)",
        answer,
        flags=re.IGNORECASE
    )[0].strip()

    answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer)

    return answer

# ============================
# Generation function
# ============================
def generate_answer(question, ctxs):
    document_texts = "\n\n".join(
        f"DOCUMENT {i+1}:\n{c.get('text', '').strip()}"
        for i, c in enumerate(ctxs)
        if c.get("text")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are given a question and a list of DOCUMENTS.\n\n"
                "Your task:\n"
                "- Write a clear and concise answer to the question.\n"
                "- Use information from ONLY ONE document.\n"
                "- Do not mix multiple sources.\n\n"
                "Output format:\n\n"
                "Answer: ..."
            )
        },
        {
            "role": "user",
            "content": (
                f"=== DOCUMENTS ===\n{document_texts}\n\n"
                f"Question: {question}"
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prompt, decoded

# ============================
# Run inference
# ============================
model.eval()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for i, ex in enumerate(tqdm(batch_examples, desc=f"Generating (pos={POSITION})")):
        try:
            with torch.no_grad():
                prompt, raw_output = generate_answer(
                    ex["question"],
                    ex["ctxs"]
                )

            record = {
                "position": POSITION,
                "prompt": prompt,
                "raw_output": raw_output,
                "question": extract_clean_question(raw_output),
                "answer": extract_clean_answer(raw_output),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i < 3:
                print("\n--- DEBUG SAMPLE ---")
                print("Answer:", record["answer"])

        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ CUDA OOM at example {i}")
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            print(f"⚠️ Error at example {i}: {e}")
            continue

        torch.cuda.empty_cache()

print(f"\n✅ Finished position {POSITION}")
print(f"Saved to: {OUTPUT_FILE}")
