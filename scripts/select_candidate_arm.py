#!/usr/bin/env python3
"""Write one retrieval arm from a combined candidate-pair file."""

import argparse
import json
from pathlib import Path

from vifinqa.fusion import first_stage
from vifinqa.jsonl import load_jsonl


def select_candidates(record: dict, arm: str) -> dict:
    return {
        **record,
        "candidates": [
            candidate for candidate in record["candidates"]
            if first_stage(candidate, arm) > 0
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("sparse", "dense", "hybrid"), required=True)
    args = parser.parse_args()

    records = [select_candidates(record, args.arm) for record in load_jsonl(args.pairs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "arm": args.arm,
        "questions": len(records),
        "pairs": sum(len(record["candidates"]) for record in records),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
