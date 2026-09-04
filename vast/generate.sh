#!/usr/bin/env bash
# Runs on the rented box. Installs what the PyTorch template lacks, then writes
# one Vietnamese question per sampled table group.
#
#   bash generate.sh
#
# Expects ~/gen/tasks.jsonl and ~/gen/generate_questions.py, which launch.sh
# ships. Output goes to ~/gen/questions.jsonl and the run resumes from it, so a
# dropped connection or a killed instance costs only the batch in flight.
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

# The template ships torch built against its own CUDA. Upgrading it is how a paid
# box breaks, so install only what is missing.
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), round(torch.cuda.get_device_properties(0).total_memory/1e9), 'GB')"
$INSTALL transformers accelerate bitsandbytes

# HF_HOME on the big volume: 14B in fp16 is a 28 GB download before it is
# quantized on load, and the template's root filesystem is often smaller.
export HF_HOME="${HF_HOME:-$PWD/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1
$INSTALL hf_transfer

nohup python generate_questions.py tasks.jsonl questions.jsonl > generate.log 2>&1 &
echo $! > generate.pid
echo "started pid $(cat generate.pid)"
# Only follow the log when a person is watching. Run from a script, tail -f
# never returns and the caller hangs on a run that is already detached.
if [ -t 1 ]; then
    echo "tailing, ctrl-c is safe"
    tail -f generate.log
fi
