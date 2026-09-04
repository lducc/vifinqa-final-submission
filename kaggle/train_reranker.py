"""LoRA fine-tune Qwen3-Reranker-8B on questions an open model wrote, on Kaggle.

The organizers' own numbers say where the headroom is. Over this corpus their
Qwen3-Embedding-8B retriever reaches Recall@10 67.48 and a reranker on top of it
80.80, and they note the best system still misses 19.2% of gold tables. Ranking
is where the loss lives, and fine-tuning is the lever they permit: the ruling of
2026-08-09 accepts it provided the data source, the method, and the training
data are all disclosed.

Training data comes from `scripts/assemble_training.py`. Nothing in it was
labelled by us. `sample_groups.py` draws groups of tables from the corpus under
proportions `measure_questions.py` derives from the released questions, an open
14B writes a Vietnamese question for each group, and the tables the question was
written from are its positives — the label is the input, so there is no rule of
ours left to be wrong about it. Questions whose own tables do not come back
under BM25 are dropped, which is Promptagator's round-trip filter.

Negatives are the other tables of the same filings, plus two from a filing the
group does not span. Same-filing negatives teach which table inside the right
report; they never teach that a report can be the wrong one. That used to be
someone else's job — a gate settled ticker, year and scope before ranking began
— but a dense first stage does not go through the gate, so the reranker has to
be able to say no to the right line item in the wrong company's filing.

A fifth of the corpus's reports are held out by `sample_groups.py` and never
appear in training, which is the only check available on whether the model
learned the retrieval task or the generator's phrasing.

The scoring function is unchanged. Inference reads the margin between "yes" and
"no" at the final position, so training optimises exactly that quantity rather
than a separate head that would have to be reconciled with it later.

## Steps

1. Create a Kaggle Dataset holding `output/rerank/training_qwen.jsonl` and point
   `TRAINING_PATH` at it.
2. New notebook, Accelerator **GPU T4 x2** or **P100**, attach the dataset, paste
   this file into one cell, run.
3. Download `/kaggle/working/adapter/`. It is rewritten every `SAVE_EVERY`
   groups, so a reclaimed session still leaves a usable adapter.
4. Score the candidate pool with it (`rerank_qwen_8b.py` takes `ADAPTER_PATH`),
   then fuse:

       python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_dense.jsonl \
           --scores scores_tuned.jsonl --output output/rerank/ranking_tuned.json

## What ends the run

The clock, not a step count. An 8B QLoRA step over ten candidates of 768 tokens
takes somewhere between 10 and 25 seconds on a T4 — a factor of two and a half,
against a 12-hour session wall. So `TIME_LIMIT_HOURS` is the budget, the number
of groups is trimmed to what that budget can reach so the learning rate still
anneals, and the adapter is written as the run goes rather than only at the end.

There is no offline scorer to accept or reject the result against. The released
questions carry no gold tables and our own labels are quarantined, so the only
measurement is the leaderboard — which is why the candidate pool is scored once
and split into two rankings afterwards, rather than scored twice in two sessions
that would differ by more than the change being tested.
"""

import json
import math
import os
import random
import sys
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
TRAINING_PATH = "/kaggle/input/vifinqa-rerank-training/training_linked.jsonl"
OUTPUT_DIR = "/kaggle/working/adapter"
# Held-out questions for a loss curve that means something. Groups are split by
# question, never by row: two rows from one question in different splits would
# leak the document text across the boundary.
VALIDATION_SHARE = 0.15
# A candidate is a filing line, a title, a matched row, and a line-item list. The
# median lands near 400 tokens with the instruction, and every token here is paid
# for on both the forward and the backward pass, which is what sets the step
# rate below.
MAX_LENGTH = 768
# One pass over more distinct questions beats two over half as many: the run is
# ended by the clock either way, and at a fixed step count a LoRA of rank 16
# learns more from fresh groups than from repeats.
EPOCHS = 1
LEARNING_RATE = 1e-4
# One optimiser step covers one question: its positives against its hard
# negatives. The loss is a softmax over the candidates of a single question, so
# the group has to arrive intact.
GROUPS_PER_STEP = 1
MAX_GROUP = 10
MAX_GROUPS = 0     # 0 keeps them all; TIME_LIMIT_HOURS is what actually ends the run
# What ends training, in place of a step count. An 8B QLoRA step over ten
# candidates runs somewhere between 10 and 25 seconds on a T4 depending on how
# long the candidates are, which is a factor of two and a half on the total —
# too wide to pick a step count in advance without either wasting the session or
# hitting the twelve-hour wall and losing the adapter. Wall clock is the honest
# budget, so the run trains until it is spent and writes what it has.
TIME_LIMIT_HOURS = float(os.environ.get("TIME_LIMIT_HOURS", 9.0))
SAVE_EVERY = 250   # groups between adapter checkpoints
# Only used to size the learning-rate schedule, which has to know its own length
# in advance. Too high and the rate never anneals because the clock stops the run
# first; too low and it anneals to nothing with hours still on the meter. Read
# the s/step the run prints and correct this on the next session.
# A Kaggle T4 number. A 4090 runs a step in about three, and this is what sizes
# the learning-rate schedule and trims the group list — left at 16 on a fast
# card it would train on a fifth of the data and stop with hours on the meter.
# vast/train.sh sets both of these from the card it finds.
ASSUMED_SECONDS_PER_STEP = float(os.environ.get("SECONDS_PER_STEP", 16))
LORA_RANK = 16
# 8B: 5.5 GB in nf4, 16 GB in bf16. With gradient checkpointing and a group of
# ten, bf16 wants a card of about 40 GB — and a card that has it should not pay
# a dequantization step on every matmul of every forward and backward pass.
NEEDS_GB = 40
QUANTIZATION = "nf4"  # replaced at startup by fp16 where the card has room
SEED = 20260814
DTYPE = torch.float16   # replaced at startup by bfloat16 where the card has it

# On a rented box the three runs differ only in their paths, so they arrive as
# arguments and everything above stays edited once.
if len(sys.argv) > 2:   # the tests import this module, and pytest has its own argv
    TRAINING_PATH, OUTPUT_DIR = sys.argv[1], sys.argv[2]

# Identical to rerank_qwen_8b.py. If either copy changes, both must.
INSTRUCTION = (
    "The Query asks about a Vietnamese listed company's financial statements. "
    "The Document is one table from a filing whose first line identifies its "
    "ticker, reporting period, and scope (hợp nhất or công ty mẹ). The candidate "
    "pool is already narrowed to compatible filings, but reject a Document when "
    "that filing line conflicts with the Query. Answer yes only when the table "
    "reports an exact requested accounting item with a value for the requested "
    "period. For a multi-step question, a table is yes when it supplies any one "
    "required input. Do not accept a merely related label, a reference to an "
    "item, a related-party or subsidiary listing, a movement or allocation "
    "schedule, or a column header. Prefer the statement or note that reports the "
    "figure; a note or segment breakdown that validly restates it is yes."
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_groups(path):
    """Rows regrouped by (question, query), which is the unit the loss ranks over."""
    grouped = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault((row["id"], row["query"]), []).append(row)
    groups = []
    for (identifier, query), rows in grouped.items():
        positives = [row for row in rows if row["label"] == 1]
        negatives = [row for row in rows if row["label"] == 0]
        if not positives or not negatives:
            # A group with only one class has no ordering to teach.
            continue
        groups.append({"id": identifier, "query": query, "positives": positives, "negatives": negatives})
    return groups


def split_by_question(groups, share, rng):
    """Hold out whole questions, so no document crosses the split."""
    identifiers = sorted({group["id"] for group in groups})
    rng.shuffle(identifiers)
    held = set(identifiers[: max(1, round(share * len(identifiers)))])
    return (
        [group for group in groups if group["id"] not in held],
        [group for group in groups if group["id"] in held],
    )


def encode(tokenizer, query, document):
    body = tokenizer(
        f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}",
        truncation=True, max_length=MAX_LENGTH, add_special_tokens=False,
    )["input_ids"]
    return tokenizer.encode(PREFIX, add_special_tokens=False) + body + tokenizer.encode(
        SUFFIX, add_special_tokens=False
    )


def group_scores(model, tokenizer, group, yes_id, no_id, rng, train=True):
    """Score one question's candidates, returning the log-odds of "yes" for each.

    Left padding, because the score is read at the final position and right
    padding would read the pad instead.
    """
    positives = group["positives"] if not train else [rng.choice(group["positives"])]
    negatives = group["negatives"][: MAX_GROUP - len(positives)]
    rows = positives + negatives
    encoded = [torch.tensor(encode(tokenizer, group["query"], row["document"])) for row in rows]
    flipped = pad_sequence(
        [ids.flip(0) for ids in encoded], batch_first=True, padding_value=tokenizer.pad_token_id
    ).flip(1)
    attention = pad_sequence(
        [torch.ones_like(ids).flip(0) for ids in encoded], batch_first=True, padding_value=0
    ).flip(1)
    # Call the PEFT-wrapped causal model directly. Some current Transformers
    # versions return a CausalLM output even through ``model.model``; ``logits``
    # is the stable public interface and gives the same yes/no margin.
    logits = model(
        input_ids=flipped.to(model.device), attention_mask=attention.to(model.device), use_cache=False,
    ).logits[:, -1, :].float()
    # The margin between "yes" and "no" is the quantity inference ranks on, so it
    # is the quantity the loss operates on. Training a separate head would leave
    # scoring and training optimising two different functions.
    return logits[:, yes_id] - logits[:, no_id], len(positives)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")
    global DTYPE
    # bfloat16 wherever the card has it. QLoRA on an 8B in fp16 is the setup
    # that diverges to NaN quietly, and the only thing that would catch it here
    # is the held-out evaluation, long after the session is over. Turing — the
    # Kaggle T4 — has no bf16 and has to take the risk; Ampere and later do not.
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    global QUANTIZATION
    if not os.environ.get("QUANTIZATION"):
        gigabytes = torch.cuda.get_device_properties(0).total_memory / 1e9
        QUANTIZATION = "fp16" if gigabytes >= NEEDS_GB else QUANTIZATION
    else:
        QUANTIZATION = os.environ["QUANTIZATION"]
    print(f"{torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB, "
          f"compute dtype {DTYPE}, weights {QUANTIZATION}", flush=True)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    groups = load_groups(TRAINING_PATH)
    if MAX_GROUPS:
        rng.shuffle(groups)
        groups = groups[:MAX_GROUPS]
    train_groups, validation_groups = split_by_question(groups, VALIDATION_SHARE, rng)
    # Trim to what the clock can actually reach. Without this the schedule is
    # laid out over every group in the file, the run stops a fifth of the way in,
    # and the learning rate never comes down off its peak — which is the one way
    # a time-boxed run quietly trains worse than a shorter planned one.
    affordable = max(1, int(TIME_LIMIT_HOURS * 3600 / ASSUMED_SECONDS_PER_STEP) // EPOCHS)
    dropped = max(0, len(train_groups) - affordable)
    train_groups = train_groups[:affordable]
    # A held-out pass costs a forward over every validation group, twice, and at
    # these step rates that is real money against the clock.
    validation_groups = validation_groups[:200]
    print(
        f"{len(groups)} groups over {len({g['id'] for g in groups})} questions; "
        f"{len(train_groups)} train / {len(validation_groups)} validation"
        + (f"; {dropped} groups past what {TIME_LIMIT_HOURS}h can reach" if dropped else ""),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    config = None
    if QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=DTYPE,
        device_map={"": 0}, attn_implementation="sdpa",
    )

    from peft import LoraConfig, get_peft_model
    # `prepare_model_for_kbit_training` upcasts every non-quantized parameter
    # to fp32. On the 24 GB contest card that transient allocation OOMs before
    # the first step; bf16 compute plus input gradients is sufficient here.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0
    )
    steps = EPOCHS * math.ceil(len(train_groups) / GROUPS_PER_STEP)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=steps, pct_start=0.1
    )

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for group in validation_groups:
                scores, positive_count = group_scores(
                    model, tokenizer, group, yes_id, no_id, rng, train=False
                )
                # The only thing that matters downstream is whether a gold table
                # outranks the hard negatives, so that is what is reported.
                correct += int(scores.argmax().item() < positive_count)
                total += 1
        model.train()
        return correct / max(1, total)

    print(f"validation top-1 before training: {evaluate():.4f}", flush=True)
    model.train()
    step, started, stopped = 0, time.time(), False
    for epoch in range(EPOCHS):
        rng.shuffle(train_groups)
        running = 0.0
        for index, group in enumerate(train_groups, 1):
            scores, positive_count = group_scores(model, tokenizer, group, yes_id, no_id, rng)
            # Softmax over one question's candidates: the positive has to beat the
            # negatives it is actually shown beside, which is the comparison the
            # submission budget makes.
            loss = torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=scores.device)
            )
            loss.backward()
            running += loss.item()
            if index % GROUPS_PER_STEP == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if index % 50 == 0:
                rate = (time.time() - started) / max(1, step)
                print(f"epoch {epoch + 1} {index}/{len(train_groups)} "
                      f"loss {running / 50:.4f} {rate:.1f}s/step "
                      f"{(time.time() - started) / 3600:.2f}h elapsed", flush=True)
                running = 0.0
            # The adapter is written as the run goes, not only at the end. A
            # Kaggle session can be reclaimed without warning, and an 8B QLoRA
            # step on a T4 is slow enough that the difference between nine hours
            # of training and nothing is one interrupted save.
            if index % SAVE_EVERY == 0:
                model.save_pretrained(OUTPUT_DIR)
            if time.time() - started > TIME_LIMIT_HOURS * 3600:
                print(f"stopping at the {TIME_LIMIT_HOURS}h limit, "
                      f"{step} steps done", flush=True)
                stopped = True
                break
        print(f"epoch {epoch + 1} validation top-1: {evaluate():.4f}", flush=True)
        if stopped:
            break

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"adapter written to {OUTPUT_DIR} after {step} steps, "
          f"{(time.time() - started) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
