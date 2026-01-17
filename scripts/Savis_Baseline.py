# Generated from: Savis_Baseline.ipynb
# Converted at: 2026-01-17T10:15:28.533Z
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
import torch  # 👈 REQUIRED

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

# Visualize
vis = ISAVisualization(
    isa.sentence_attention,
    isa.sentences
)

vis.visualize_sentence_attention(figsize=(22, 22))

isa.sentences
for idx, sent in enumerate(isa.sentences):
    print(f"[{idx:02d}] {sent.strip()}")

import torch
import re

# ======================================================
# 0. Tensors
# ======================================================
sentence_attention = isa.sentence_attention.clone()
sentences = isa.sentences

# ======================================================
# 1. Find generated answer sentence index
# ======================================================
generated_x = None
answer_pattern = re.compile(r"\bAnswer:\s+(?!\.\.\.)", re.IGNORECASE)

for i, s in enumerate(sentences):
    if answer_pattern.search(s):
        generated_x = i

if generated_x is None:
    raise ValueError("Could not find generated answer sentence.")

# ======================================================
# 2. Detect DOCUMENT spans
# ======================================================
documents = {}
current_doc = None
doc_pattern = re.compile(r"DOCUMENT\s+(\d+):")

for i, s in enumerate(sentences):
    s_stripped = s.strip()

    doc_match = doc_pattern.search(s_stripped)
    if doc_match:
        current_doc = int(doc_match.group(1))
        documents[current_doc] = [i]
        continue

    if current_doc is not None and (
        "Question:" in s_stripped or "Answer:" in s_stripped
    ):
        current_doc = None
        continue

    if current_doc is not None:
        documents[current_doc].append(i)

# ======================================================
# 3. Min / Max attention (DOCUMENTS ONLY)
# ======================================================
all_doc_ys = sorted(y for ys in documents.values() for y in ys)

attn_vals = torch.tensor(
    [sentence_attention[generated_x, y].item() for y in all_doc_ys]
)

min_attn = attn_vals.min().item()
max_attn = attn_vals.max().item()

min_y = all_doc_ys[attn_vals.argmin().item()]
max_y = all_doc_ys[attn_vals.argmax().item()]

threshold = (min_attn + max_attn) / 2

# ======================================================
# 4. Find document containing MAX attention sentence
# ======================================================
max_doc_id = None
for doc_id, ys in documents.items():
    if max_y in ys:
        max_doc_id = doc_id
        break

# ======================================================
# 5. FINAL OUTPUT (ONLY WHAT YOU WANT)
# ======================================================
print(f"MIN attention : {min_attn:.6f} (y={min_y})")
print(f"MAX attention : {max_attn:.6f} (y={max_y})")
print(f"Threshold     : {threshold:.6f}")
print(f"Max-attended document : DOCUMENT {max_doc_id}")


# # GENERALIZED CODE


import torch
import re

sentence_attention = torch.tensor(isa.sentence_attention)
sentences = isa.sentences

# ======================================================
# 1. Find generated answer sentence index
# ======================================================
generated_x = None
answer_pattern = re.compile(r"\bAnswer:\s+(?!\.\.\.)", re.IGNORECASE)

for i, s in enumerate(sentences):
    if answer_pattern.search(s):
        generated_x = i

if generated_x is None:
    raise ValueError("Could not find generated answer sentence.")

print(f"\n[INFO] Detected generated answer sentence index: {generated_x}")
print(sentences[generated_x])

# ======================================================
# 2. Detect DOCUMENT spans USING INDICES (FINAL FIX)
# ======================================================
documents = {}
current_doc = None

doc_pattern = re.compile(r"DOCUMENT\s+(\d+):")

for i, s in enumerate(sentences):
    s_stripped = s.strip()

    # Start / switch document
    doc_match = doc_pattern.search(s_stripped)
    if doc_match:
        current_doc = int(doc_match.group(1))
        documents[current_doc] = [i]   # include DOCUMENT line index
        continue

# Stop collecting at Answer OR Question
    if current_doc is not None and (
        "Answer:" in s_stripped or "Question:" in s_stripped
    ):
        current_doc = None
        continue

    # Collect all sentences belonging to the current document
    if current_doc is not None:
        documents[current_doc].append(i)

# ======================================================
# 3. Min / Max attention (DOCUMENTS ONLY)
# ======================================================

# Collect all document sentence indices
all_doc_ys = sorted(
    y for ys in documents.values() for y in ys
)

# Attention slice ONLY over document sentences
attn_vals = torch.tensor(
    [sentence_attention[generated_x, y].item() for y in all_doc_ys]
)

min_attn = attn_vals.min().item()
max_attn = attn_vals.max().item()

min_y = all_doc_ys[attn_vals.argmin().item()]
max_y = all_doc_ys[attn_vals.argmax().item()]

threshold = (min_attn + max_attn) / 2


# ======================================================
# 4. Global stats
# ======================================================
print("\n========== GLOBAL ATTENTION STATS ==========")
print(f"Generated sentence index (x): {generated_x}")
print("-------------------------------------------")
print(f"MIN attention : {min_attn:.6f} (y={min_y})")
print(sentences[min_y])
print("-------------------------------------------")
print(f"MAX attention : {max_attn:.6f} (y={max_y})")
print(sentences[max_y])
print("-------------------------------------------")
print(f"Threshold     : {threshold:.6f}")
print("===========================================\n")

# ======================================================
# 5. Document-level attention analysis
# ======================================================
print("========== DOCUMENT ATTENTION ANALYSIS ==========")

for doc_id in sorted(documents.keys()):
    ys = documents[doc_id]

    doc_attn_vals = [
        sentence_attention[generated_x, y].item()
        for y in ys
    ]

    doc_avg_attn = sum(doc_attn_vals) / len(doc_attn_vals)
    attended = doc_avg_attn > threshold

    print(f"\nDOCUMENT {doc_id}")
    print(f"Sentence span (y): {ys}")
    print("Attention values:")
    for y, v in zip(ys, doc_attn_vals):
        print(f"  Attention (x={generated_x} → y={y}) : {v:.6f}")

    print(f"Average attention : {doc_avg_attn:.6f}")
    print(
        "FINAL DECISION   : "
        + ("ATTENDED ✅" if attended else "NOT ATTENDED ❌")
    )

print("\n==============================================")


import torch

# ============================
# Tensor
# ============================
sentence_attention = torch.tensor(isa.sentence_attention)

# ============================
# Indices (UPDATE PER EXAMPLE)
# ============================
generated_x = 57            # generated answer sentence
doc5_start_y = 25           # DOCUMENT 5 start
doc6_start_y = 29           # DOCUMENT 6 start (excluded)

doc5_ys = list(range(doc5_start_y, doc6_start_y))  # [25, 26, 27, 28]

# ============================
# 1. Min / Max attention
#    x = generated_x
#    y = 0 .. generated_x-1
# ============================
attn_slice = sentence_attention[generated_x, 0:generated_x]

min_attn = attn_slice.min().item()
min_y = attn_slice.argmin().item()

max_attn = attn_slice.max().item()
max_y = attn_slice.argmax().item()

# ============================
# 2. Threshold
# ============================
threshold = (min_attn + max_attn) / 2

# ============================
# 3. DOCUMENT 5 attention (AVG over span)
# ============================
doc5_attn_values = [
    sentence_attention[generated_x, y].item()
    for y in doc5_ys
]

doc5_attn_avg = sum(doc5_attn_values) / len(doc5_attn_values)

# ============================
# 4. Decision
# ============================
attended = doc5_attn_avg > threshold

# ============================
# 5. PRINT EVERYTHING
# ============================
print("========== ATTENTION ANALYSIS ==========")
print(f"Generated sentence index (x)      : {generated_x}")
print(f"Sentence range considered (y)     : 0 to {generated_x-1}")
print("----------------------------------------")

print(f"MIN attention value               : {min_attn:.6f}")
print(f"At sentence (y)                   : {min_y}")
print("Sentence text:")
print(isa.sentences[min_y])
print("----------------------------------------")

print(f"MAX attention value               : {max_attn:.6f}")
print(f"At sentence (y)                   : {max_y}")
print("Sentence text:")
print(isa.sentences[max_y])
print("----------------------------------------")

print(f"Threshold ((min + max) / 2)       : {threshold:.6f}")
print("----------------------------------------")

print("DOCUMENT 5 sentence span (y)      :", doc5_ys)
print("DOCUMENT 5 attention values:")
for y, v in zip(doc5_ys, doc5_attn_values):
    print(f"  Attention (x={generated_x} → y={y}) : {v:.6f}")

print("----------------------------------------")
print(f"DOCUMENT 5 average attention      : {doc5_attn_avg:.6f}")
print("----------------------------------------")

print(
    "FINAL DECISION                    : "
    + ("ATTENDED ✅" if attended else "NOT ATTENDED ❌")
)
print("========================================")


# # MIN AND MAX ATTENTION


import torch

# Convert to tensor
sentence_attention = torch.tensor(isa.sentence_attention)

generated_x = 57

# ============================
# Slice: x = 57, y = 0..56
# ============================
attn_values = sentence_attention[generated_x, 0:generated_x]

# ============================
# Min attention
# ============================
min_value = attn_values.min().item()
min_y = attn_values.argmin().item()

# ============================
# Max attention
# ============================
max_value = attn_values.max().item()
max_y = attn_values.argmax().item()

# ============================
# PRINT RESULTS
# ============================
print("========== ATTENTION STATS FOR GENERATED SENTENCE 57 ==========")
print(f"Generated sentence (x) : {generated_x}")
print("---------------------------------------------------------------")
print(f"MIN attention value    : {min_value:.6f}")
print(f"At sentence (y)        : {min_y}")
print("Sentence text:")
print(isa.sentences[min_y])
print("---------------------------------------------------------------")
print(f"MAX attention value    : {max_value:.6f}")
print(f"At sentence (y)        : {max_y}")
print("Sentence text:")
print(isa.sentences[max_y])
print("===============================================================")


# # ONLY DOCUMENTS


import torch

# ============================
# Tensor
# ============================
sentence_attention = torch.tensor(isa.sentence_attention)

# ============================
# Indices (from your example)
# ============================
generated_x = 60        # generated answer sentence
doc5_y = 29             # DOCUMENT 5 sentence
doc6_y = 30             # DOCUMENT 6 sentence (excluded)

# ============================
# 1. Attention slice: x=60 → y=0..59
# ============================
attn_slice = sentence_attention[generated_x, 0:doc6_y]

min_attn = attn_slice.min().item()
max_attn = attn_slice.max().item()

# ============================
# 2. Threshold
# ============================
threshold = (min_attn + max_attn) / 2

# ============================
# 3. DOCUMENT 5 attention
# ============================
doc5_attn = sentence_attention[generated_x, doc5_y].item()

# ============================
# 4. Decision
# ============================
attended = doc5_attn > threshold

# ============================
# 5. PRINT EVERYTHING
# ============================
print("========== ATTENTION ANALYSIS ==========")
print(f"Generated sentence index (x)      : {generated_x}")
print(f"Sentence range considered (y)     : 0 to {doc6_y-1}")
print("----------------------------------------")
print(f"Minimum attention value           : {min_attn:.6f}")
print(f"Maximum attention value           : {max_attn:.6f}")
print(f"Average (min + max)/2             : {threshold:.6f}")
print("----------------------------------------")
print(f"DOCUMENT 5 sentence index (y)     : {doc5_y}")
print(f"Attention at (x={generated_x}, y={doc5_y}) : {doc5_attn:.6f}")
print("----------------------------------------")
print(
    "FINAL DECISION                    : "
    + ("ATTENDED ✅" if attended else "NOT ATTENDED ❌")
)
print("========================================")


# # GENERALIZED CODE


import torch
import re

sentence_attention = torch.tensor(isa.sentence_attention)
sentences = isa.sentences

# ======================================================
# 1. Find generated answer sentence index
# ======================================================
generated_x = None
answer_pattern = re.compile(r"\bAnswer:\s+(?!\.\.\.)", re.IGNORECASE)

for i, s in enumerate(sentences):
    if answer_pattern.search(s):
        generated_x = i

if generated_x is None:
    raise ValueError("Could not find generated answer sentence.")

print(f"\n[INFO] Detected generated answer sentence index: {generated_x}")
print(sentences[generated_x])

# ======================================================
# 2. Detect DOCUMENT spans USING INDICES (FINAL FIX)
# ======================================================
documents = {}
current_doc = None

doc_pattern = re.compile(r"DOCUMENT\s+(\d+):")

for i, s in enumerate(sentences):
    s_stripped = s.strip()

    # Start / switch document
    doc_match = doc_pattern.search(s_stripped)
    if doc_match:
        current_doc = int(doc_match.group(1))
        documents[current_doc] = [i]   # include DOCUMENT line index
        continue

    # Stop collecting at Answer
    if current_doc is not None and "Answer:" in s_stripped:
        current_doc = None
        continue

    # Collect all sentences belonging to the current document
    if current_doc is not None:
        documents[current_doc].append(i)

# ======================================================
# 3. Min / Max attention for generated sentence
# ======================================================
attn_slice = sentence_attention[generated_x, :generated_x]

min_attn = attn_slice.min().item()
min_y = attn_slice.argmin().item()

max_attn = attn_slice.max().item()
max_y = attn_slice.argmax().item()

threshold = (min_attn + max_attn) / 2

# ======================================================
# 4. Global stats
# ======================================================
print("\n========== GLOBAL ATTENTION STATS ==========")
print(f"Generated sentence index (x): {generated_x}")
print("-------------------------------------------")
print(f"MIN attention : {min_attn:.6f} (y={min_y})")
print(sentences[min_y])
print("-------------------------------------------")
print(f"MAX attention : {max_attn:.6f} (y={max_y})")
print(sentences[max_y])
print("-------------------------------------------")
print(f"Threshold     : {threshold:.6f}")
print("===========================================\n")

# ======================================================
# 5. Document-level attention analysis
# ======================================================
print("========== DOCUMENT ATTENTION ANALYSIS ==========")

for doc_id in sorted(documents.keys()):
    ys = documents[doc_id]

    doc_attn_vals = [
        sentence_attention[generated_x, y].item()
        for y in ys
    ]

    doc_avg_attn = sum(doc_attn_vals) / len(doc_attn_vals)
    attended = doc_avg_attn > threshold

    print(f"\nDOCUMENT {doc_id}")
    print(f"Sentence span (y): {ys}")
    print("Attention values:")
    for y, v in zip(ys, doc_attn_vals):
        print(f"  Attention (x={generated_x} → y={y}) : {v:.6f}")

    print(f"Average attention : {doc_avg_attn:.6f}")
    print(
        "FINAL DECISION   : "
        + ("ATTENDED ✅" if attended else "NOT ATTENDED ❌")
    )

print("\n==============================================")