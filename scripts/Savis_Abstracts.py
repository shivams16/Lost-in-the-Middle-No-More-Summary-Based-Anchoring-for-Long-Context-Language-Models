# Generated from: Savis_Abstracts.ipynb
# Converted at: 2026-01-17T10:15:23.052Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")
!pip install savis
from savis import TextGenerator, ISA, ISAVisualization, Attention
import os
!pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()
!pip show python-dotenv
import torch  # 👈 REQUIRED

# =====================================================
# 4. Initialize SAVIS TextGenerator
# =====================================================
generator = TextGenerator(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    token="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"
)

# =====================================================
# 3. Input text
# =====================================================

input_text = """

"""

# =====================================================
# 5. Generate text + attentions
# =====================================================
generated_text, attentions, tokenizer, input_ids, outputs = generator.generate_text(
    input_text
)

# Move attentions to CPU (safe for long sequences)
attentions_cpu = tuple(
    tuple(a.cpu() for a in layer)
    for layer in attentions
)

# Free GPU memory
del attentions
torch.cuda.empty_cache()

# Compute ISA on CPU
isa = ISA(outputs.sequences[0], attentions_cpu, tokenizer)

# =====================================================
# Visualization (COMMENTED OUT)
# =====================================================
vis = ISAVisualization(
   isa.sentence_attention,
   isa.sentences
)
#
vis.visualize_sentence_attention(figsize=(22, 22))

# =====================================================
# Print sentences with indices
# =====================================================
for idx, sent in enumerate(isa.sentences):
    print(f"[{idx:02d}] {sent.strip()}")


import torch
import re

# ======================================================
# 0. Inputs
# ======================================================
sentence_attention = isa.sentence_attention.clone()
sentences = isa.sentences

# ======================================================
# 1. Locate generated answer sentence (x-axis)
# ======================================================
answer_pattern = re.compile(r"\bAnswer:\s+(?!\.\.\.)", re.IGNORECASE)

generated_x = next(
    i for i, s in enumerate(sentences) if answer_pattern.search(s)
)

print(f"\n[INFO] Generated sentence index (x): {generated_x}")
print(sentences[generated_x])

# ======================================================
# 2. Locate CONTEXT boundaries
# ======================================================
context_start = next(
    i for i, s in enumerate(sentences)
    if "=== CONTEXT (ABSTRACT + DOCUMENT PAIRS) ===" in s
)

question_idx = next(
    i for i, s in enumerate(sentences)
    if "Question:" in s
)

context_ys = list(range(context_start + 1, question_idx))

# ======================================================
# 3. Compute MIN / MAX / THRESHOLD (CONTEXT ONLY)
# ======================================================
context_attn_vals = torch.tensor([
    sentence_attention[generated_x, y].item()
    for y in context_ys
])

min_attn = context_attn_vals.min().item()
max_attn = context_attn_vals.max().item()
threshold = (min_attn + max_attn) / 2

min_y = context_ys[context_attn_vals.argmin().item()]
max_y = context_ys[context_attn_vals.argmax().item()]

print("\n========== GLOBAL ATTENTION STATS ==========")
print(f"Context span      : y={context_start+1} → y={question_idx-1}")
print(f"MIN attention     : {min_attn:.6f} (y={min_y})")
print(sentences[min_y])
print("-------------------------------------------")
print(f"MAX attention     : {max_attn:.6f} (y={max_y})")
print(sentences[max_y])
print("-------------------------------------------")
print(f"Threshold         : {threshold:.6f}")
print("===========================================\n")

# ======================================================
# 4. Detect ABSTRACT boundaries
# ======================================================
abstract_pattern = re.compile(r"ABSTRACT\s+(\d+):", re.IGNORECASE)

abstract_starts = []
for i in context_ys:
    if abstract_pattern.search(sentences[i]):
        abstract_starts.append(i)

abstract_starts.append(question_idx)  # sentinel

# ======================================================
# 5. Windowed ABSTRACT + DOCUMENT attention analysis
# ======================================================
print("========== ABSTRACT–DOCUMENT ATTENTION ANALYSIS ==========")

global_max_attn = -1.0
global_max_y = None
global_max_pair = None

for idx in range(len(abstract_starts) - 1):
    start_y = abstract_starts[idx]
    end_y = abstract_starts[idx + 1]

    ys = list(range(start_y, end_y))

    attn_vals = [
        sentence_attention[generated_x, y].item()
        for y in ys
    ]

    avg_attn = sum(attn_vals) / len(attn_vals)
    attended = avg_attn > threshold

    print(f"\nPAIR {idx + 1}")
    print(f"Sentence span (y): {ys}")
    print("Attention values:")
    for y, v in zip(ys, attn_vals):
        print(f"  Attention (x={generated_x} → y={y}) : {v:.6f}")

    print(f"Average attention : {avg_attn:.6f}")
    print(
        "FINAL DECISION   : "
        + ("ATTENDED ✅" if attended else "NOT ATTENDED ❌")
    )

    # Track global max
    for y, v in zip(ys, attn_vals):
        if v > global_max_attn:
            global_max_attn = v
            global_max_y = y
            global_max_pair = idx + 1

print("\n==============================================")

# ======================================================
# 6. Strongest attended signal
# ======================================================
print("\n========== MOST ATTENDED CONTEXT SENTENCE ==========")
print(f"Pair ID            : {global_max_pair}")
print(f"Sentence index (y) : {global_max_y}")
print(f"Attention value    : {global_max_attn:.6f}")
print("Sentence text      :")
print(sentences[global_max_y])
print("===========================================")