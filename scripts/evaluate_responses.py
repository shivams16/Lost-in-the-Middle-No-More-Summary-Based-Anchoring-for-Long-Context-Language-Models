import json
import string
import regex
from typing import List

# =========================================================
# Normalization (official-style)
# =========================================================
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, gold_spans: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for span in gold_spans:
        span_norm = normalize_answer(span)
        if span_norm and span_norm in pred_norm:
            return 1.0
    return 0.0


# =========================================================
# Clean model output
# =========================================================
def clean_prediction(pred: str) -> str:
    pred = pred.replace("Answer:", "").replace("answer:", "")
    pred = pred.replace("\n", " ")
    return pred.strip()


# =========================================================
# Configuration
# =========================================================
BASE_DIR = "20_total_documents"
POSITIONS = [0, 4, 9]  # extend as needed

INPUT_TEMPLATE = "Answers_nq-open-20_total_documents_gold_at_{}.jsonl"
GOLD_TEMPLATE  = "nq-open-20_total_documents_gold_at_{}.jsonl"


# =========================================================
# Evaluation loop
# =========================================================
results = {}

for pos in POSITIONS:
    input_file = f"{BASE_DIR}/{INPUT_TEMPLATE.format(pos)}"
    gold_file  = f"{BASE_DIR}/{GOLD_TEMPLATE.format(pos)}"

    # Load files
    input_data = []
    gold_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            input_data.append(json.loads(line))

    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            gold_data.append(json.loads(line))

    print(f"\n[gold_at_{pos}]")
    print(f"Input examples : {len(input_data)}")
    print(f"Gold examples  : {len(gold_data)}")

    # Evaluate
    total = 0
    correct = 0

    for idx in range(min(len(input_data), len(gold_data))):
        pred_obj = input_data[idx]
        gold_obj = gold_data[idx]

        prediction = clean_prediction(pred_obj.get("answer", ""))
        nq_gold = gold_obj.get("nq_annotated_gold", {})

        gold_spans = []

        # Long answer
        long_answer = nq_gold.get("long_answer", "")
        if long_answer:
            gold_spans.append(long_answer)

        # Short answers
        for sa in nq_gold.get("short_answers", []):
            if isinstance(sa, dict) and "text" in sa:
                gold_spans.append(sa["text"])
            elif isinstance(sa, str):
                gold_spans.append(sa)

        if not gold_spans:
            continue

        em = best_subspan_em(prediction, gold_spans)
        correct += em
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    results[pos] = accuracy

    print(f"Total evaluated : {total}")
    print(f"Correct         : {int(correct)}")
    print(f"Best Subspan EM : {accuracy:.4f}")
    print(f"Accuracy (%)    : {accuracy * 100:.2f}")


# =========================================================
# Summary
# =========================================================
print("\n========== SUMMARY ==========")
for pos, score in results.items():
    print(f"gold_at_{pos}: {score * 100:.2f}%")
