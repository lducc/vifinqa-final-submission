#!/usr/bin/env bash
# Bring the questions home and audit them before anything is assembled.
#
#   bash vast/fetch.sh root@ssh4.vast.ai 41521
set -euo pipefail
HOST="${1:?usage: fetch.sh user@host port}"
PORT="${2:?usage: fetch.sh user@host port}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# vast.ai instances authorise one of your keys, and it is rarely the default id_*.
KEY="${SSH_KEY:-$HOME/.ssh/vastai}"
SSH=(ssh -o StrictHostKeyChecking=accept-new); SCP=(scp -o StrictHostKeyChecking=accept-new)
[ -f "$KEY" ] && { SSH+=(-i "$KEY"); SCP+=(-i "$KEY"); }

DIR="${DIR:-gen}"
OUT="${OUT:-output/rerank/questions.jsonl}"
"${SCP[@]}" -P "$PORT" "${HOST}:$DIR/questions.jsonl" "$ROOT/$OUT"
wc -l "$ROOT/$OUT"
# The audit compares surface features against the released questions. It is the
# training set that has to match them; the held-out evaluation set only has to
# be answerable, so read the drift there as description rather than as a gate.
"$ROOT/.venv/bin/python" "$ROOT/scripts/audit_questions.py" --questions "$ROOT/$OUT"
