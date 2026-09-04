#!/usr/bin/env bash
# Runs on the rented box. Fine-tunes the reranker on the questions the
# generation pass wrote.
#
#   bash train.sh
#
# Expects ~/tune/training.jsonl and ~/tune/train_reranker.py, which finetune.sh
# ships. Output goes to ~/tune/adapter and the script rewrites it every few
# hundred groups, so losing the box costs the last few minutes rather than the
# run.
set -euo pipefail
cd "$(dirname "$0")"

# The Vast base image keeps torch in a venv rather than the system python, and
# ships uv. A plain PyTorch template has neither, so both are conditional.
if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi
INSTALL="pip install -q --no-input"
command -v uv >/dev/null && INSTALL="uv pip install -q"

python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())"
$INSTALL transformers accelerate bitsandbytes peft

export HF_HOME="${HF_HOME:-$PWD/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
$INSTALL hf_transfer

# The defaults in train_reranker.py are Kaggle's: a twelve-hour session wall and
# a T4's step rate. This box is neither, and the step rate is what decides how
# much of the training data the run will reach.
export TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-4}"
export SECONDS_PER_STEP="${SECONDS_PER_STEP:-3}"
echo "budget ${TIME_LIMIT_HOURS}h at ${SECONDS_PER_STEP}s/step"

nohup python train_reranker.py training.jsonl adapter > train.log 2>&1 &
echo $! > train.pid
echo "started pid $(cat train.pid)"
# Only follow the log when a person is watching. Run from a script, tail -f
# never returns and the caller hangs on a run that is already detached.
if [ -t 1 ]; then
    echo "tailing, ctrl-c is safe"
    tail -f train.log
fi
