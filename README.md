# Lost in the Middle No More

### Summary-Based Anchoring for Long-Context Language Models

This repository accompanies the paper ***Lost in the Middle No More: Summary-Based Anchoring for Long-Context Language Models***, which proposes a **training-free, inference-time prompting method** to mitigate positional bias in long-context LLMs.

The method, **Summary-Based Anchoring**, improves a model’s ability to utilize information appearing in the middle of long contexts by augmenting each document with a concise summary that acts as a positional anchor.

📄 Paper: *Lost in the Middle No More: Summary-Based Anchoring for Long-Context Language Models* 

---

## 🔍 Motivation

Long-context language models often suffer from the **lost-in-the-middle problem**, where information in the middle of the context window is under-utilized compared to content at the beginning or end.

While prior work addresses this issue via:

* architectural changes,
* fine-tuning,
* or specialized training,

these approaches are **computationally expensive** and **not applicable to off-the-shelf models**.

This work focuses instead on **inference-only solutions**.

---

## 🚫 Why Retrieval-Oriented Prompting Fails

An intuitive approach is to ask the model to:

1. retrieve relevant evidence, then
2. generate the answer.

However, experiments show that:

* **Evidence → Answer** prompting consistently **reduces accuracy**
* Direct answer generation performs better across models

**Key Insight:**
Long-context LLMs are better at **direct reasoning via attention** than explicit retrieval decomposition.

---

## ✅ Proposed Method: Summary-Based Anchoring

### Core Idea

Each document is prepended with a **short summary (≤15 words)** that captures its main topic.

These summaries:

* act as **global semantic anchors**
* guide attention toward relevant documents
* work regardless of document position

### Two-Stage Pipeline

#### 1. Document-Level Summary Generation

For each document:

```text
Write in no more than 15 words what the following paragraph is about.
Focus only on the main topic.
```

This step is:

* model-agnostic
* query-independent
* training-free

#### 2. Summary-Augmented Prompt Construction

Each summary is paired with its document:

```text
ABSTRACT i
DOCUMENT i
```

The model then answers the question in a standard end-to-end manner, without explicit retrieval instructions.

---

## 🧪 Experimental Setup

* **Task:** Multi-document Question Answering (NQ)
* **Documents per example:** 20 (and 30 for long-context scaling)
* **Gold document position:** varied (start → middle → end)
* **No fine-tuning**, zero-shot inference only

### Models Evaluated

* LongChat-13B-16K
* Qwen3-8B
* Llama-3-8B-Instruct

---

## 📊 Key Results

### 1. Retrieval-Oriented Prompting

* Causes **significant accuracy drops**
* Up to **10% degradation** on some models

### 2. Summary-Based Anchoring

* Improves or maintains accuracy across all models
* Up to **+4.4% absolute improvement**
* Gains are **larger with longer contexts (30 docs)**

### 3. Best Summary Placement

* **End of context** → best performance
* **Interleaved summaries** → strong and intuitive alternative
* **Middle placement** → weakest due to positional bias

---

## 🧠 Mechanistic Insight (Attention Analysis)

Sentence-level attention visualizations show that:

* baseline models assign **low attention** to gold evidence in the middle
* summary-anchored prompts **redirect attention** toward gold sentences
* attention mass on relevant evidence increases significantly

This explains the empirical gains without modifying model internals.

---

## 📈 Scalability

Summary-Based Anchoring:

* scales better as context length increases
* provides larger improvements when lost-in-the-middle effects are more severe

---

## ⚠️ Limitations

* Performance depends on **summary quality**
* Adds **extra inference cost** for summary generation
* Evaluated primarily on **multi-document QA**
* Attention analysis is **correlational**, not causal

---

## 🧩 Contributions

* 🔍 Systematic analysis showing **retrieval prompting degrades performance**
* 📌 Introduction of **Summary-Based Anchoring**, a simple inference-only method
* 🧠 Attention-based evidence explaining why it works
* 🔁 Robust improvements across models and longer contexts

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{summary_based_anchoring,
  title={Lost in the Middle No More: Summary-Based Anchoring for Long-Context Language Models},
  author={Anonymous},
  venue={ACL Submission},
  year={2025}
}
```

---

If you want, I can also:

* tailor this README to **code-release style**
* add **usage examples / scripts**
* generate a **short README** for benchmarks only
* format it for **ACL / EMNLP artifact submission**

Just tell me 👍
