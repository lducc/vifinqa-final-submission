#!/usr/bin/env bash
# Bring the adapter home.
#
#   bash vast/collect.sh root@ssh4.vast.ai 41521
set -euo pipefail
HOST="${1:?usage: collect.sh user@host port}"
PORT="${2:?usage: collect.sh user@host port}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# vast.ai instances authorise one of your keys, and it is rarely the default id_*.
KEY="${SSH_KEY:-$HOME/.ssh/vastai}"
SSH=(ssh -o StrictHostKeyChecking=accept-new); SCP=(scp -o StrictHostKeyChecking=accept-new)
[ -f "$KEY" ] && { SSH+=(-i "$KEY"); SCP+=(-i "$KEY"); }

mkdir -p "$ROOT/output/adapter"
"${SCP[@]}" -rP "$PORT" "${HOST}:tune/adapter/." "$ROOT/output/adapter/"
ls -la "$ROOT/output/adapter"
echo
echo "next: score the candidate pool with ADAPTER_PATH set to output/adapter"
