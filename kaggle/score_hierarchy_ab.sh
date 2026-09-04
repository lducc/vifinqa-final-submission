#!/usr/bin/env bash
# Score the flat control and hierarchy treatment in one GPU session.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCORER="${SCORER:-$ROOT/kaggle/rerank_qwen_8b.py}"
FLAT_PAIRS="${FLAT_PAIRS:-$ROOT/output/table_retrieval/pairs_flat_233.jsonl}"
HIERARCHY_PAIRS="${HIERARCHY_PAIRS:-$ROOT/output/table_retrieval/pairs_hierarchy_233.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/table_retrieval/gpu_ab}"

for path in "$SCORER" "$FLAT_PAIRS" "$HIERARCHY_PAIRS"; do
    test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
flat_scores="$OUTPUT_DIR/scores_flat_233.jsonl"
hierarchy_scores="$OUTPUT_DIR/scores_hierarchy_233.jsonl"
for path in "$flat_scores" "$hierarchy_scores"; do
    test ! -e "$path" || {
        echo "refusing to append to existing score file: $path" >&2
        echo "move it away or choose another OUTPUT_DIR" >&2
        exit 2
    }
done

python "$SCORER" "$FLAT_PAIRS" "$flat_scores"
python "$ROOT/scripts/validate_rerank_scores.py" --pairs "$FLAT_PAIRS" --scores "$flat_scores"
python "$SCORER" "$HIERARCHY_PAIRS" "$hierarchy_scores"
python "$ROOT/scripts/validate_rerank_scores.py" \
    --pairs "$HIERARCHY_PAIRS" --scores "$hierarchy_scores"

sha256sum "$FLAT_PAIRS" "$HIERARCHY_PAIRS" "$flat_scores" "$hierarchy_scores" \
    > "$OUTPUT_DIR/sha256sum.txt"
echo "paired scores written to $OUTPUT_DIR"
