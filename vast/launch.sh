#!/usr/bin/env bash
# Ship the two files a generation run needs to a vast.ai box and start it.
#
#   bash vast/launch.sh root@ssh4.vast.ai 41521          # every sampled group
#   GROUPS=4000 bash vast/launch.sh root@ssh4.vast.ai 41521
#
# The host and port come from the instance's "Connect" button. Nothing else is
# needed on the box: generate_questions.py imports only torch and transformers,
# so the repository is not cloned there.
#
# The default sends everything and lets the budget decide when to stop. The run
# writes each question as it finishes and resumes from what is already on disk,
# so destroying the instance costs only the batch in flight — which means a group
# count chosen in advance can only be wrong in one of two directions, either
# paying for groups past the point of usefulness or stopping while credit is
# left. Fetch whenever you like; the file is complete up to that moment.
set -euo pipefail
HOST="${1:?usage: launch.sh user@host port}"
PORT="${2:?usage: launch.sh user@host port}"
GROUPS="${GROUPS:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# vast.ai instances authorise one of your keys, and it is rarely the default id_*.
KEY="${SSH_KEY:-$HOME/.ssh/vastai}"
SSH=(ssh -o StrictHostKeyChecking=accept-new); SCP=(scp -o StrictHostKeyChecking=accept-new)
[ -f "$KEY" ] && { SSH+=(-i "$KEY"); SCP+=(-i "$KEY"); }

# The held-out evaluation questions are a second, much shorter pass over the
# same box: same script, different task file, its own directory so the resume
# file does not collide.
#
#   DIR=eval TASKS=output/rerank/tasks_holdout.jsonl bash vast/launch.sh HOST PORT
DIR="${DIR:-gen}"
TASKS="${TASKS:-output/rerank/tasks.jsonl}"
SEND="$ROOT/output/rerank/tasks_send.jsonl"

# Always shuffled, seeded, whatever the count. tasks.jsonl is ordered by report,
# so any prefix of it is one alphabetical slice of the corpus — and since the run
# is stopped by the budget rather than by finishing, a prefix is exactly what
# gets generated. Shuffling first is what makes that prefix a random sample of
# the corpus instead of the companies whose tickers sort early.
if [ "$GROUPS" = "all" ]; then
    shuf --random-source=<(yes 20260818) "$ROOT/$TASKS" > "$SEND"
else
    shuf --random-source=<(yes 20260818) "$ROOT/$TASKS" \
        | head -n "$GROUPS" > "$SEND"
fi
echo "sending $(wc -l < "$SEND") groups, $(du -h "$SEND" | cut -f1)"

"${SSH[@]}" -p "$PORT" "$HOST" "mkdir -p ~/$DIR"
"${SCP[@]}" -P "$PORT" "$SEND" "${HOST}:$DIR/tasks.jsonl"
"${SCP[@]}" -P "$PORT" "$ROOT/kaggle/generate_questions.py" "$ROOT/vast/generate.sh" "${HOST}:$DIR/"
"${SSH[@]}" -p "$PORT" -t "$HOST" "bash ~/$DIR/generate.sh"
