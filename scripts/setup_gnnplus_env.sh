#!/usr/bin/env bash
# Build a separate Python 3.10 environment for the authors' GNNPlus code (their README's stack).
# Usage: scripts/setup_gnnplus_env.sh [VENV_DIR]   (default: third_party/.venv-gnnplus)
set -euo pipefail
cd "$(dirname "$0")/.."
V=${1:-third_party/.venv-gnnplus}
uv venv -q --python 3.10 "$V"
if [ "$(uname)" = "Linux" ]; then
  uv pip install -q --python "$V/bin/python" torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
  uv pip install -q --python "$V/bin/python" torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
else
  # No prebuilt torch_scatter wheel for macOS + torch 2.2: compile it against the installed torch (~1 min).
  uv pip install -q --python "$V/bin/python" torch==2.2.0 setuptools==69.5.1 wheel
  uv pip install -q --python "$V/bin/python" --no-build-isolation torch_scatter==2.1.2
fi
uv pip install -q --python "$V/bin/python" -r configs/gnnplus/requirements.txt
"$V/bin/python" -c "import torch, torch_geometric, torch_scatter, rdkit; print('gnnplus env:', torch.__version__, 'pyg', torch_geometric.__version__, 'cuda', torch.cuda.is_available())"
