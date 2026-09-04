#!/usr/bin/env python3
"""Show an open model some tables; ask it for a question those tables answer.

This is the only step that decides anything about the training data, and it is
deliberately the only one. The tables were drawn at random, so whatever question
comes back, the tables it was written from are its evidence — the label is the
input, not a judgement anyone made about the output.

The organizers permit open-weight models of 14B or smaller for data generation
and forbid closed models. Qwen2.5-14B-Instruct sits at that ceiling. Sampling is
seeded and the run resumes from a partial file, because the box is rented by the
hour.

    python kaggle/generate_questions.py data/tasks.jsonl questions.jsonl
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---- edit these, then run ----------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"   # the 14B ceiling the organizers allow
TASKS_PATH = "data/tasks.jsonl"
OUTPUT_PATH = "questions.jsonl"
# 14B: 9 GB in nf4, 28 GB in fp16. Add room for a batch of sixteen at 3,072
# tokens and fp16 wants a card of about 40 GB.
NEEDS_GB = 40
QUANTIZATION = "nf4"     # replaced at startup by fp16 where the card has room
MAX_BATCH = int(os.environ.get("MAX_BATCH", 16))
# Tasks are grouped by table count inside a block of this many, not across the
# file. Eight batches is enough that a batch is nearly uniform in width, and the
# block length is also the worst-case skew at whatever point the budget stops the
# run — the last partial block is sorted, so it is short by design.
BLOCK = MAX_BATCH * 8
TABLE_CHARS = 1100      # characters shown per table; p90 of a rendered table is
                        # 1,079, so this shows nine in ten of them whole
TABLE_BUDGET = 6000     # ...but a group of ten shown whole would not fit, so the
                        # per-table share shrinks with the group and the biggest
                        # groups see the head of each table only
MAX_INPUT_TOKENS = 3072  # a cross-company group carries up to ten tables
MAX_NEW_TOKENS = 160    # released questions reach 55 words, and Vietnamese runs
                        # well over one token per word
TEMPERATURE = 0.7   # Promptagator samples its queries at 0.7
SEED = 20260818
# ------------------------------------------------------------------------------

if len(sys.argv) > 2:
    TASKS_PATH, OUTPUT_PATH = sys.argv[1], sys.argv[2]

SYSTEM = (
    "Bạn là chuyên viên phân tích tài chính, viết câu hỏi khảo thí bằng tiếng Việt "
    "dựa trên báo cáo tài chính của các công ty niêm yết Việt Nam."
)

# The clauses below are switched on per task by `style`, which sample_groups.py
# drew from the released questions' measured distribution. Left to itself a
# generator states the scope every time and picks whatever unit it likes; the
# real questions state a scope 37% of the time, essentially never say "hợp nhất",
# and ask in tỷ đồng, phần trăm or triệu đồng in that order. Conditioning on the
# observed proportions matches the distribution without copying any question.
#
# Two clauses are fixed rather than sampled, because the released questions are
# near-unanimous on both: they run 19 to 55 words (p10 to p90, median 30), and
# only 0.5% of them refer to a table or report rather than asking straight for
# the figure. A generator shown "--- Bảng 1 ---" does the latter constantly
# unless told not to.

TEMPLATE = """{preamble}

{tables}

Viết MỘT câu hỏi tiếng Việt mà người đọc phải dùng {need} mới trả lời được.

Yêu cầu:
{rules}
- Không nhắc đến bảng, báo cáo hay nguồn dữ liệu; hỏi thẳng vào chỉ tiêu.
- Độ dài khoảng 20-55 từ.
- Chỉ trả về đúng câu hỏi, một dòng, không giải thích, không đánh số.

Câu hỏi:"""

SCOPE_CLAUSE = {
    "separate": "của công ty mẹ (báo cáo riêng)",
    "consolidated": "hợp nhất",
}

OPERATION_CLAUSE = {
    "tổng": "Yêu cầu tính tổng các chỉ tiêu được hỏi.",
    "chênh lệch": "Yêu cầu tính chênh lệch giữa các chỉ tiêu được hỏi.",
    "tỷ trọng": "Yêu cầu tính tỷ trọng của một chỉ tiêu so với chỉ tiêu còn lại.",
}

# The three shapes the organizers' own difficulty tiers turn on: their Hard tier
# is "multi-hop dependent — an intermediate result decides the next table,
# company or period", which is a condition followed by a lookup, and their
# Intermediate tier is repeated or grouped arithmetic such as a mean or a
# margin. Both are near-absent from single-filing questions and common in the
# cross-year and cross-company ones, so sample_groups.py draws them at the rate
# measured for each kind rather than at one rate overall.
SHAPE_CLAUSE = {
    # Terminal: the maximum is the answer. Dependent: the maximum only picks
    # which {axis} to read, and a second figure of that {axis} is the answer —
    # the organizers' Hard tier, and half of what a superlative actually is.
    "dependent": "- Xác định {axis} có một chỉ tiêu đạt giá trị cao nhất (hoặc thấp nhất), "
                 "rồi hỏi một chỉ tiêu KHÁC của đúng {axis} đó.",
    "superlative": "- Hỏi {axis} nào đạt giá trị cao nhất hoặc thấp nhất.",
    "conditional": "- Nêu một điều kiện lọc trước ({axis} thỏa điều kiện nào đó), "
                   "rồi hỏi số liệu trong nhóm đã lọc.",
    "aggregate": "- Yêu cầu tính giá trị trung bình hoặc trung vị của chỉ tiêu.",
}

AXIS = {"single": "chỉ tiêu", "years": "năm", "peers": "công ty"}


def preamble_for(task, count):
    """What the group is, in the words the question will have to use.

    A cross-year or cross-company group has to name which figure belongs to
    which filing, or the model cannot write a question that distinguishes them.
    """
    kind = task.get("kind", "single")
    if kind == "years":
        years = ", ".join(str(y) for y in sorted({t["year"] for t in task["tables"]}))
        return (f"Dưới đây là {count} bảng của cùng một công ty qua nhiều năm.\n\n"
                f"Công ty: {task['company']} ({task['ticker']})\n"
                f"Các năm: {years}")
    if kind == "peers":
        names = ", ".join(sorted({t["ticker"] for t in task["tables"]}))
        return (f"Dưới đây là {count} bảng của {count} công ty khác nhau trong cùng một năm.\n\n"
                f"Các công ty: {names}\n"
                f"Năm: {task['year']}")
    return (f"Dưới đây là {count} bảng trích từ "
            f"{'báo cáo tài chính' if count == 1 else 'cùng một báo cáo tài chính'}.\n\n"
            f"Công ty: {task['company']} ({task['ticker']})\n"
            f"Năm: {task['year']}")


def rules_for(task, count):
    style = task.get("style", {})
    kind = task.get("kind", "single")
    axis = AXIS[kind]
    if kind == "years":
        rules = ["- Nêu tên công ty và nêu rõ các năm được so sánh."]
    elif kind == "peers":
        rules = [f"- Nêu tên hoặc mã của các công ty được so sánh và năm {task['year']}."]
    else:
        rules = [f"- Nêu tên công ty và năm {task['year']}."]
    if style.get("name_scope"):
        clause = SCOPE_CLAUSE.get(task.get("scope"))
        if clause:
            rules.append(f"- Nêu rõ đây là số liệu {clause}.")
    if kind == "single":
        rules.append(
            "- Hỏi về một chỉ tiêu cụ thể có trong bảng, gọi đúng tên chỉ tiêu như bảng ghi."
            if count == 1 else
            f"- Hỏi về một chỉ tiêu cụ thể ở MỖI bảng ({count} chỉ tiêu), gọi đúng tên như bảng ghi."
        )
    elif style.get("dependent"):
        # A dependent question needs two items, not one: the first picks the
        # year or the company, the second is the answer. Both have to exist in
        # every table or the question cannot be answered from the group.
        rules.append("- Dùng HAI chỉ tiêu có mặt ở TẤT CẢ các bảng, gọi đúng tên như bảng ghi: "
                     "một chỉ tiêu để chọn ra đối tượng, một chỉ tiêu để trả lời.")
    else:
        rules.append("- Hỏi về CÙNG một chỉ tiêu ở tất cả các bảng, gọi đúng tên như bảng ghi.")
    rules.append(
        "- Hỏi số liệu tại thời điểm cuối năm." if style.get("period_end")
        else "- Hỏi số liệu trong năm."
    )
    unit = style.get("unit")
    if unit:
        rules.append(f"- Hỏi kết quả theo đơn vị {unit}.")
    operation = style.get("operation")
    if operation:
        rules.append(f"- {OPERATION_CLAUSE[operation]}")
    for name, clause in SHAPE_CLAUSE.items():
        # A dependent question already asks for the maximum, so the terminal
        # superlative clause would ask for it twice and contradict itself.
        if name == "superlative" and style.get("dependent"):
            continue
        if style.get(name):
            rules.append(clause.format(axis=axis))
    rules.append("- Câu hỏi phải trả lời được bằng số liệu trong bảng.")
    return "\n".join(rules)


def load_tasks(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_for(tokenizer, task, chars=TABLE_CHARS, budget=TABLE_BUDGET):
    tables = task["tables"]
    per_table = min(chars, max(350, budget // max(1, len(tables))))
    single = task.get("kind", "single") == "single"
    rendered = "\n\n".join(
        # Cross-filing tables carry their company and year in the heading, since
        # the whole point of the question is to tell them apart.
        f"--- Bảng {index + 1}"
        + ("" if single else f" ({table['ticker']} {table['year']})")
        + f" ---\n{table['text'][:per_table]}"
        for index, table in enumerate(tables)
    )
    count = len(tables)
    body = TEMPLATE.format(
        preamble=preamble_for(task, count),
        need="bảng trên" if count == 1 else f"cả {count} bảng",
        tables=rendered, rules=rules_for(task, count),
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True,
    )


def clean(text: str) -> str:
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    line = line.lstrip("-*0123456789. ").strip()
    if len(line) > 1 and line[0] in "\"'“”" and line[-1] in "\"'“”":
        line = line[1:-1].strip()
    return line


def pick_quantization(default="nf4"):
    """nf4 on a 24 GB card, fp16 where there is room for the weights whole.

    Qwen2.5-14B is 28 GB in fp16. On a 24 GB card it has to be quantized to
    NF4, which costs a dequantization step on every matmul; on a 48 GB card it
    does not, and the questions come out of the unquantized model. Detected
    rather than set, because getting it wrong either wastes the card or fails to
    allocate an hour into a paid run.
    """
    import os
    if os.environ.get("QUANTIZATION"):
        return os.environ["QUANTIZATION"]
    if not torch.cuda.is_available():
        return default
    gigabytes = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "fp16" if gigabytes >= NEEDS_GB else default

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this needs a GPU")
    global QUANTIZATION
    QUANTIZATION = pick_quantization()
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(SEED)

    tasks = load_tasks(TASKS_PATH)
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as handle:
            done = {json.loads(line)["task_id"] for line in handle if line.strip()}
        tasks = [task for task in tasks if task["task_id"] not in done]
        print(f"resuming, {len(done)} already written", flush=True)

    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    ) if QUANTIZATION == "nf4" else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=compute_dtype, device_map="cuda:0",
    ).eval()
    print(f"{MODEL_NAME} in {QUANTIZATION} ({compute_dtype}), batch {MAX_BATCH}, "
          f"{len(tasks)} groups to write", flush=True)

    # Batch by table count, so padding does not blow up on mixed input lengths —
    # but only inside a block, never across the whole file. The run is stopped by
    # the budget rather than by finishing, so what gets written is a prefix, and
    # a global sort would make that prefix every single-table group and no other
    # kind. Sorting within a block of BLOCK keeps batches near-uniform in width
    # while leaving the completed prefix a fair sample of the mix the sampler
    # drew. tasks.jsonl arrives shuffled, which is what makes a block fair.
    tasks = [task for start in range(0, len(tasks), BLOCK)
             for task in sorted(tasks[start:start + BLOCK],
                                key=lambda task: len(task["tables"]))]
    started, written = time.time(), 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for start in range(0, len(tasks), MAX_BATCH):
            batch = tasks[start:start + MAX_BATCH]
            encoded = tokenizer(
                [prompt_for(tokenizer, task) for task in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS,
            ).to(model.device)
            with torch.inference_mode():
                produced = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            for task, sequence in zip(batch, produced):
                question = clean(tokenizer.decode(
                    sequence[encoded["input_ids"].shape[1]:], skip_special_tokens=True
                ))
                if not question:
                    continue
                out.write(json.dumps(
                    {"task_id": task["task_id"], "question": question}, ensure_ascii=False
                ) + "\n")
                written += 1
            out.flush()
            elapsed = time.time() - started
            print(f"{written}/{len(tasks)} in {elapsed/60:.1f} min "
                  f"({written/max(1e-9, elapsed):.1f}/s)", flush=True)
    print(f"wrote {written} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
