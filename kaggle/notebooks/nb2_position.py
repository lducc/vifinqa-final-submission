"""Kaggle GPU notebook: score candidate pairs with a Qwen3 reranker.

Paste into a notebook with a GPU accelerator and the exported pairs attached as
a dataset, edit the settings block, run. It writes scores.jsonl for
`scripts/apply_rerank_scores.py` to fuse locally.

To A/B two settings, paste it twice in one session and change the one line that
differs plus SCORES_PATH. Two runs in different sessions are not comparable.

Retrieval is untouched — only the order of already-retrieved candidates changes,
so discarding a run means deleting one file.

kaggle/README.md has the settings, the timings, and why each default is what it
is. Read it before changing one.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---- edit these, then run ----------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Reranker-8B"      # or Qwen/Qwen3-Reranker-4B
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_bench_v5.jsonl"
SCORES_PATH = "/kaggle/working/scores_v5.jsonl"
PER_ITEM = False        # True scores each named line item as its own query
QUANTIZATION = "int8"   # "fp16" needs GPU T4 x2; "nf4" is faster and unscored
RESUME_PATH = ""        # a downloaded scores.jsonl: skips whole questions
SKIP_PATH = ""          # a finished scores.jsonl: skips individual candidates
ADAPTER_PATH = ""       # a LoRA adapter from train_reranker.py
PROMPT = "v2"           # "v2" lets the model decide statement or note; use with pairs_bench_v5
# ------------------------------------------------------------------------------

MAX_LENGTH = 1024
RERANK_DEPTH = 100
MAX_BATCH = 16

# The instruction shipped with every score file so far. It ends by counting a
# note that restates a figure as yes, and on the benchmark that is what fills
# the slots after the first gold table: gold sits in the primary statements 60%
# of the time, the non-gold we submit does 34%.
INSTRUCTION_V1 = (
    "Every candidate is a table from the correct company, period and statement, "
    "so none of those separate them — the whole judgement is which table inside "
    "that report reports the figure. Answer yes if the table holds at least one "
    "of the line items under 'Chỉ tiêu cần tìm' as a row of its own, with a "
    "value for the period asked; a question often needs several figures and a "
    "table only has to supply one of them. A matching label is not enough on "
    "its own: the same wording appears on rows that only reference the item — "
    "related-party and subsidiary listings, movement and allocation schedules, "
    "and rows that are column headers rather than line items. Prefer the "
    "statement or the note that reports the figure, and count a note or segment "
    "breakdown restating it as yes."
)
# The revision: the model decides statement or note from the item itself, and a
# restatement is no. Pairs with the position line in the v5 candidate text.
INSTRUCTION_V2 = (
    "Every candidate is a table from the correct company, period and statement, "
    "so none of those separate them — the whole judgement is which table inside "
    "that report is the source of the figure. Answer yes if the table holds at "
    "least one of the line items under 'Chỉ tiêu cần tìm' as a row of its own, "
    "with a value for the period asked; a question often needs several figures "
    "and a table only has to supply one of them. A report presents its primary "
    "statements first — balance sheet, income statement, cash flow, equity — and "
    "the notes after them; the table's position in the report says which it is. "
    "A line item that belongs to a primary statement is sourced there, and a note "
    "that repeats or breaks it down is no. A line item that only a note reports — "
    "an ownership share, a credit limit, a term deposit, a fair value, a tax "
    "component — is sourced in that note, and the statement it rolls up into is "
    "no. A matching label is not enough on its own: the same wording appears on "
    "rows that only reference the item — related-party and subsidiary listings, "
    "movement and allocation schedules, and rows that are column headers rather "
    "than line items."
)
INSTRUCTION = INSTRUCTION_V2 if PROMPT == "v2" else INSTRUCTION_V1

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_scored(path):
    """Question IDs a previous run finished, so a stopped session resumes."""
    if not path or not os.path.exists(path):
        return set()
    return {record["id"] for record in load_pairs(path)}


def already_judged(path):
    """Table IDs per question a previous run scored, so a deeper export resumes."""
    if not path or not os.path.exists(path):
        return {}
    judged = {}
    for record in load_pairs(path):
        judged.setdefault(record["id"], set()).update(record["scores"])
    return judged


def to_judge(record, skip=frozenset()):
    """Candidate positions this run should score."""
    return [
        index for index, candidate in enumerate(record["candidates"])
        if candidate.get("sparse_rank", index + 1) <= RERANK_DEPTH
        and candidate["table_id"] not in skip
    ]


def build_queries(record):
    """The queries to score each candidate against, best score winning.

    With PER_ITEM each named line item becomes its own query, so a table is asked
    only what it can answer rather than for every figure the question needs. A
    question naming zero or one item is byte-identical either way, which is what
    makes the A/A stratum in compare_rerank_runs.py free.
    """
    items = record.get("line_items") or []
    if not items:
        return [record["question"]]
    if PER_ITEM and len(items) > 1:
        return [f"{record['question']}\nChỉ tiêu cần tìm: {item}" for item in items]
    return [f"{record['question']}\nChỉ tiêu cần tìm: {'; '.join(items)}"]


def pair_count(record, skip=frozenset()):
    """Forward passes this record costs: one per (candidate, query)."""
    return len(to_judge(record, skip)) * len(build_queries(record))


def packed_batches(lengths, order):
    """Length-sorted rows grouped so each batch costs about the same work.

    Padding makes a batch cost its widest row times its width, so packing to a
    token budget lets the short candidates run many at a time.
    """
    budget = 8 * MAX_LENGTH
    batch = []
    for index in order:
        width = max([lengths[index]] + [lengths[i] for i in batch])
        if batch and (width * (len(batch) + 1) > budget or len(batch) >= MAX_BATCH):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def last_position_logits(model, padded):
    """Vocabulary logits for the final position only.

    Projecting 4096 -> 151,936 at every position builds a 7.5 GB tensor to read
    48 numbers, which is what put a 16 GB T4 out of memory. Applying the head to
    the last hidden state alone gives the identical numbers.
    """
    with torch.inference_mode():
        hidden = model.model(**padded, use_cache=False).last_hidden_state[:, -1, :]
        return model.lm_head(hidden).float()


def score_question(record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skip=frozenset()):
    queries = build_queries(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    judged = to_judge(record, skip)
    # Keyed by query position, not query text, so two identical items stay two rows.
    matrix = {index: [0.0] * len(queries) for index in judged}
    rows = [(index, position) for index in judged for position in range(len(queries))]
    prefix_ids, suffix_ids = prompt_ids
    encoded = {
        row: prefix_ids + ids + suffix_ids
        for row, ids in zip(rows, tokenizer(
            [
                f"<Instruct>: {INSTRUCTION}\n<Query>: {queries[position]}"
                f"\n<Document>: {candidates[index]['text']}"
                for index, position in rows
            ],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {row: len(ids) for row, ids in encoded.items()}
    pending = list(packed_batches(lengths, sorted(rows, key=lambda row: lengths[row])))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[row] for row in chunk]}, padding=True, return_tensors="pt",
        ).to(model.device)
        try:
            logits = last_position_logits(model, padded)
        except torch.cuda.OutOfMemoryError:
            # Memory depends on the widest row, so one unlucky batch should not
            # end a ten hour session. This is the only handler here worth having.
            if len(chunk) == 1:
                raise
            del padded
            torch.cuda.empty_cache()
            middle = len(chunk) // 2
            pending.extend([chunk[:middle], chunk[middle:]])
            continue
        # The score is the probability of "yes" at the final position, which is
        # why padding must be on the left. Max over queries because a table
        # counts if it supplies any one named item, not all of them.
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for (index, position), value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            matrix[index][position] = value
            scores[index] = max(scores[index], value)
    return scores, matrix, judged


def score_payload(record, values, matrix, judged, per_item):
    """The scores.jsonl line for one question.

    Only the candidates this run judged are written. A zero is a score, and would
    rank a candidate below everything the model disliked rather than leaving it
    to the sparse order or to the file it was already scored in.
    """
    kept = [(index, record["candidates"][index]) for index in judged]
    payload = {
        "id": record["id"],
        "scores": {candidate["table_id"]: round(values[index], 6) for index, candidate in kept},
    }
    items = record.get("line_items") or []
    if per_item and items:
        payload["line_items"] = items
        payload["per_item"] = {
            candidate["table_id"]: [round(value, 6) for value in matrix[index]]
            for index, candidate in kept
        }
    return payload


def load_model():
    if QUANTIZATION == "fp16":
        config, placement = None, "auto"          # 16.4 GB: needs T4 x2
    elif QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        placement = {"": 0}
    else:
        config, placement = BitsAndBytesConfig(load_in_8bit=True), {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=torch.float16,
        device_map=placement, attn_implementation="sdpa",
    ).eval()
    if ADAPTER_PATH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH).eval()
    return model


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = load_model()
    print(f"{MODEL_NAME} in {QUANTIZATION}, depth={RERANK_DEPTH}, per_item={PER_ITEM}"
          + (f", adapter={ADAPTER_PATH}" if ADAPTER_PATH else ", zero-shot"), flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prompt_ids = (
        tokenizer.encode(PREFIX, add_special_tokens=False),
        tokenizer.encode(SUFFIX, add_special_tokens=False),
    )
    budget = MAX_LENGTH - sum(len(ids) for ids in prompt_ids)

    done = already_scored(SCORES_PATH) | already_scored(RESUME_PATH)
    skip = already_judged(SKIP_PATH)
    skipped = lambda record: skip.get(record["id"], frozenset())
    pending = [record for record in load_pairs(PAIRS_PATH) if record["id"] not in done]
    total = sum(pair_count(record, skipped(record)) for record in pending)
    print(f"{len(pending)} questions, {total} pairs"
          + (f", resuming past {len(done)}" if done else "")
          + (f", skipping {sum(len(s) for s in skip.values())} judged pairs" if skip else ""), flush=True)

    started, scored = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, matrix, judged = score_question(
                record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skipped(record),
            )
            out.write(json.dumps(
                score_payload(record, values, matrix, judged, PER_ITEM), ensure_ascii=False,
            ) + "\n")
            out.flush()
            scored += pair_count(record, skipped(record))
            if scored % 1000 < MAX_BATCH:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()



# %% ---------------- next cell: paste from here down ----------------


"""Kaggle GPU notebook: score candidate pairs with a Qwen3 reranker.

Paste into a notebook with a GPU accelerator and the exported pairs attached as
a dataset, edit the settings block, run. It writes scores.jsonl for
`scripts/apply_rerank_scores.py` to fuse locally.

To A/B two settings, paste it twice in one session and change the one line that
differs plus SCORES_PATH. Two runs in different sessions are not comparable.

Retrieval is untouched — only the order of already-retrieved candidates changes,
so discarding a run means deleting one file.

kaggle/README.md has the settings, the timings, and why each default is what it
is. Read it before changing one.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---- edit these, then run ----------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Reranker-8B"      # or Qwen/Qwen3-Reranker-4B
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_bench_v4.jsonl"
SCORES_PATH = "/kaggle/working/scores_ctrl_nb2.jsonl"
PER_ITEM = False        # True scores each named line item as its own query
QUANTIZATION = "int8"   # "fp16" needs GPU T4 x2; "nf4" is faster and unscored
RESUME_PATH = ""        # a downloaded scores.jsonl: skips whole questions
SKIP_PATH = ""          # a finished scores.jsonl: skips individual candidates
ADAPTER_PATH = ""       # a LoRA adapter from train_reranker.py
PROMPT = "v1"           # "v2" lets the model decide statement or note; use with pairs_bench_v5
# ------------------------------------------------------------------------------

MAX_LENGTH = 1024
RERANK_DEPTH = 100
MAX_BATCH = 16

# The instruction shipped with every score file so far. It ends by counting a
# note that restates a figure as yes, and on the benchmark that is what fills
# the slots after the first gold table: gold sits in the primary statements 60%
# of the time, the non-gold we submit does 34%.
INSTRUCTION_V1 = (
    "Every candidate is a table from the correct company, period and statement, "
    "so none of those separate them — the whole judgement is which table inside "
    "that report reports the figure. Answer yes if the table holds at least one "
    "of the line items under 'Chỉ tiêu cần tìm' as a row of its own, with a "
    "value for the period asked; a question often needs several figures and a "
    "table only has to supply one of them. A matching label is not enough on "
    "its own: the same wording appears on rows that only reference the item — "
    "related-party and subsidiary listings, movement and allocation schedules, "
    "and rows that are column headers rather than line items. Prefer the "
    "statement or the note that reports the figure, and count a note or segment "
    "breakdown restating it as yes."
)
# The revision: the model decides statement or note from the item itself, and a
# restatement is no. Pairs with the position line in the v5 candidate text.
INSTRUCTION_V2 = (
    "Every candidate is a table from the correct company, period and statement, "
    "so none of those separate them — the whole judgement is which table inside "
    "that report is the source of the figure. Answer yes if the table holds at "
    "least one of the line items under 'Chỉ tiêu cần tìm' as a row of its own, "
    "with a value for the period asked; a question often needs several figures "
    "and a table only has to supply one of them. A report presents its primary "
    "statements first — balance sheet, income statement, cash flow, equity — and "
    "the notes after them; the table's position in the report says which it is. "
    "A line item that belongs to a primary statement is sourced there, and a note "
    "that repeats or breaks it down is no. A line item that only a note reports — "
    "an ownership share, a credit limit, a term deposit, a fair value, a tax "
    "component — is sourced in that note, and the statement it rolls up into is "
    "no. A matching label is not enough on its own: the same wording appears on "
    "rows that only reference the item — related-party and subsidiary listings, "
    "movement and allocation schedules, and rows that are column headers rather "
    "than line items."
)
INSTRUCTION = INSTRUCTION_V2 if PROMPT == "v2" else INSTRUCTION_V1

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_scored(path):
    """Question IDs a previous run finished, so a stopped session resumes."""
    if not path or not os.path.exists(path):
        return set()
    return {record["id"] for record in load_pairs(path)}


def already_judged(path):
    """Table IDs per question a previous run scored, so a deeper export resumes."""
    if not path or not os.path.exists(path):
        return {}
    judged = {}
    for record in load_pairs(path):
        judged.setdefault(record["id"], set()).update(record["scores"])
    return judged


def to_judge(record, skip=frozenset()):
    """Candidate positions this run should score."""
    return [
        index for index, candidate in enumerate(record["candidates"])
        if candidate.get("sparse_rank", index + 1) <= RERANK_DEPTH
        and candidate["table_id"] not in skip
    ]


def build_queries(record):
    """The queries to score each candidate against, best score winning.

    With PER_ITEM each named line item becomes its own query, so a table is asked
    only what it can answer rather than for every figure the question needs. A
    question naming zero or one item is byte-identical either way, which is what
    makes the A/A stratum in compare_rerank_runs.py free.
    """
    items = record.get("line_items") or []
    if not items:
        return [record["question"]]
    if PER_ITEM and len(items) > 1:
        return [f"{record['question']}\nChỉ tiêu cần tìm: {item}" for item in items]
    return [f"{record['question']}\nChỉ tiêu cần tìm: {'; '.join(items)}"]


def pair_count(record, skip=frozenset()):
    """Forward passes this record costs: one per (candidate, query)."""
    return len(to_judge(record, skip)) * len(build_queries(record))


def packed_batches(lengths, order):
    """Length-sorted rows grouped so each batch costs about the same work.

    Padding makes a batch cost its widest row times its width, so packing to a
    token budget lets the short candidates run many at a time.
    """
    budget = 8 * MAX_LENGTH
    batch = []
    for index in order:
        width = max([lengths[index]] + [lengths[i] for i in batch])
        if batch and (width * (len(batch) + 1) > budget or len(batch) >= MAX_BATCH):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def last_position_logits(model, padded):
    """Vocabulary logits for the final position only.

    Projecting 4096 -> 151,936 at every position builds a 7.5 GB tensor to read
    48 numbers, which is what put a 16 GB T4 out of memory. Applying the head to
    the last hidden state alone gives the identical numbers.
    """
    with torch.inference_mode():
        hidden = model.model(**padded, use_cache=False).last_hidden_state[:, -1, :]
        return model.lm_head(hidden).float()


def score_question(record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skip=frozenset()):
    queries = build_queries(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    judged = to_judge(record, skip)
    # Keyed by query position, not query text, so two identical items stay two rows.
    matrix = {index: [0.0] * len(queries) for index in judged}
    rows = [(index, position) for index in judged for position in range(len(queries))]
    prefix_ids, suffix_ids = prompt_ids
    encoded = {
        row: prefix_ids + ids + suffix_ids
        for row, ids in zip(rows, tokenizer(
            [
                f"<Instruct>: {INSTRUCTION}\n<Query>: {queries[position]}"
                f"\n<Document>: {candidates[index]['text']}"
                for index, position in rows
            ],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {row: len(ids) for row, ids in encoded.items()}
    pending = list(packed_batches(lengths, sorted(rows, key=lambda row: lengths[row])))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[row] for row in chunk]}, padding=True, return_tensors="pt",
        ).to(model.device)
        try:
            logits = last_position_logits(model, padded)
        except torch.cuda.OutOfMemoryError:
            # Memory depends on the widest row, so one unlucky batch should not
            # end a ten hour session. This is the only handler here worth having.
            if len(chunk) == 1:
                raise
            del padded
            torch.cuda.empty_cache()
            middle = len(chunk) // 2
            pending.extend([chunk[:middle], chunk[middle:]])
            continue
        # The score is the probability of "yes" at the final position, which is
        # why padding must be on the left. Max over queries because a table
        # counts if it supplies any one named item, not all of them.
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for (index, position), value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            matrix[index][position] = value
            scores[index] = max(scores[index], value)
    return scores, matrix, judged


def score_payload(record, values, matrix, judged, per_item):
    """The scores.jsonl line for one question.

    Only the candidates this run judged are written. A zero is a score, and would
    rank a candidate below everything the model disliked rather than leaving it
    to the sparse order or to the file it was already scored in.
    """
    kept = [(index, record["candidates"][index]) for index in judged]
    payload = {
        "id": record["id"],
        "scores": {candidate["table_id"]: round(values[index], 6) for index, candidate in kept},
    }
    items = record.get("line_items") or []
    if per_item and items:
        payload["line_items"] = items
        payload["per_item"] = {
            candidate["table_id"]: [round(value, 6) for value in matrix[index]]
            for index, candidate in kept
        }
    return payload


def load_model():
    if QUANTIZATION == "fp16":
        config, placement = None, "auto"          # 16.4 GB: needs T4 x2
    elif QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        placement = {"": 0}
    else:
        config, placement = BitsAndBytesConfig(load_in_8bit=True), {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=torch.float16,
        device_map=placement, attn_implementation="sdpa",
    ).eval()
    if ADAPTER_PATH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH).eval()
    return model


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = load_model()
    print(f"{MODEL_NAME} in {QUANTIZATION}, depth={RERANK_DEPTH}, per_item={PER_ITEM}"
          + (f", adapter={ADAPTER_PATH}" if ADAPTER_PATH else ", zero-shot"), flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prompt_ids = (
        tokenizer.encode(PREFIX, add_special_tokens=False),
        tokenizer.encode(SUFFIX, add_special_tokens=False),
    )
    budget = MAX_LENGTH - sum(len(ids) for ids in prompt_ids)

    done = already_scored(SCORES_PATH) | already_scored(RESUME_PATH)
    skip = already_judged(SKIP_PATH)
    skipped = lambda record: skip.get(record["id"], frozenset())
    pending = [record for record in load_pairs(PAIRS_PATH) if record["id"] not in done]
    total = sum(pair_count(record, skipped(record)) for record in pending)
    print(f"{len(pending)} questions, {total} pairs"
          + (f", resuming past {len(done)}" if done else "")
          + (f", skipping {sum(len(s) for s in skip.values())} judged pairs" if skip else ""), flush=True)

    started, scored = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, matrix, judged = score_question(
                record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skipped(record),
            )
            out.write(json.dumps(
                score_payload(record, values, matrix, judged, PER_ITEM), ensure_ascii=False,
            ) + "\n")
            out.flush()
            scored += pair_count(record, skipped(record))
            if scored % 1000 < MAX_BATCH:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
