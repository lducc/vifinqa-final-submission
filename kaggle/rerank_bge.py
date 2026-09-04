"""Kaggle GPU notebook: score question/table pairs with BGE-reranker-v2-m3.

Paste this into a Kaggle notebook with the accelerator set to GPU (T4 or P100)
and the exported pairs attached as a dataset. It reads pairs.jsonl, scores every
question/table pair, and writes scores.jsonl for the local pipeline to fuse.

Nothing about retrieval happens here. The candidates were chosen locally; this
only reorders them, so the run is reproducible and a bad result is discarded by
deleting the output.

Expected runtime for 1,012 questions x 50 candidates (~50k pairs):
    T4  fp16, batch 64, max_length 320  ->  roughly 20-35 minutes
"""

import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same family as the systems ranked above us, multilingual, 568M parameters, and
# well inside the competition's 14B open-model limit.
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
MAX_LENGTH = 320
BATCH_SIZE = 64


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()

    records = load_pairs(PAIRS_PATH)
    total = sum(len(record["candidates"]) for record in records)
    print(f"{len(records)} questions, {total} pairs", flush=True)

    started = time.time()
    done = 0
    with open(SCORES_PATH, "w", encoding="utf-8") as out:
        for record in records:
            questions = [record["question"]] * len(record["candidates"])
            documents = [candidate["text"] for candidate in record["candidates"]]
            scores = []
            for start in range(0, len(documents), BATCH_SIZE):
                encoded = tokenizer(
                    questions[start:start + BATCH_SIZE],
                    documents[start:start + BATCH_SIZE],
                    padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    logits = model(**encoded).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
            out.write(json.dumps({
                "id": record["id"],
                "scores": {
                    candidate["table_id"]: round(score, 6)
                    for candidate, score in zip(record["candidates"], scores)
                },
            }, ensure_ascii=False) + "\n")
            done += len(documents)
            if done % 5000 < BATCH_SIZE:
                rate = done / (time.time() - started)
                print(f"{done}/{total} pairs, {rate:.0f}/s, eta {(total - done) / rate / 60:.0f} min", flush=True)

    elapsed = (time.time() - started) / 60
    print(f"wrote {SCORES_PATH} in {elapsed:.1f} min", flush=True)


if __name__ == "__main__":
    main()
