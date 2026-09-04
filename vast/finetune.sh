#!/usr/bin/env bash
# Ship the training data to the same box the questions were written on, and
# start the fine-tune.
#
#   bash vast/finetune.sh root@ssh4.vast.ai 41521
#
# Use the box that already generated the questions. It has the weights cached
# and a fresh instance would re-download tens of gigabytes on the meter.
#
# On a 4090 this is bfloat16 and roughly three seconds a step, against ten to
# twenty-five on a Kaggle T4 in float16 — which is the setup that diverges to
# NaN quietly, with nothing to catch it until the held-out evaluation runs.
set -euo pipefail
HOST="${1:?usage: finetune.sh user@host port}"
PORT="${2:?usage: finetune.sh user@host port}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# vast.ai instances authorise one of your keys, and it is rarely the default id_*.
KEY="${SSH_KEY:-$HOME/.ssh/vastai}"
SSH=(ssh -o StrictHostKeyChecking=accept-new); SCP=(scp -o StrictHostKeyChecking=accept-new)
[ -f "$KEY" ] && { SSH+=(-i "$KEY"); SCP+=(-i "$KEY"); }

TRAINING="${TRAINING:-output/rerank/training_qwen.jsonl}"

test -s "$ROOT/$TRAINING" || {
    echo "no $TRAINING — run scripts/assemble_training.py first" >&2
    exit 1
}
echo "sending $(wc -l < "$ROOT/$TRAINING") rows, $(du -h "$ROOT/$TRAINING" | cut -f1)"

"${SSH[@]}" -p "$PORT" "$HOST" 'mkdir -p ~/tune'
"${SCP[@]}" -P "$PORT" "$ROOT/$TRAINING" "${HOST}:tune/training.jsonl"
"${SCP[@]}" -P "$PORT" "$ROOT/kaggle/train_reranker.py" "$ROOT/vast/train.sh" "${HOST}:tune/"
"${SSH[@]}" -p "$PORT" -t "$HOST" 'bash ~/tune/train.sh'
