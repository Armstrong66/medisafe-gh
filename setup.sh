#!/usr/bin/env bash
set -euo pipefail

echo "=== G-MASS Setup ==="

# 1. Python check
PYTHON_CMD=python3
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "FAIL Python 3.10+ required"
  exit 1
fi
"$PYTHON_CMD" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" \
  && echo "OK Python $($PYTHON_CMD --version)" \
  || { echo "FAIL Python 3.10+ required"; exit 1; }

# 2. .env
if [ -f .env ]; then
  echo "OK .env exists"
elif [ -f .env.example ]; then
  cp .env.example .env
  echo "OK .env created from .env.example"
  echo "Fill in your real API keys in .env before running the project."
else
  echo "FAIL .env missing and .env.example was not found"
  exit 1
fi

# Warn if the new .env still contains obvious placeholders
if [ -f .env ] && grep -qE 'your_[A-Za-z0-9_]+|YOUR_[A-Z0-9_]+' .env 2>/dev/null; then
  echo "WARNING: .env contains placeholder values. Update .env with real keys before use."
fi

# 3. Install dependencies
"$PYTHON_CMD" -m pip install -r requirements.txt --quiet
echo "OK dependencies installed"

# 4. fasttext LID model
mkdir -p scorer/models
if [ -f scorer/models/lid.176.ftz ]; then
  echo "OK fasttext LID model present"
else
  curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
       -o scorer/models/lid.176.ftz
  echo "OK fasttext LID downloaded"
fi

# 5. Verify current restored pipeline environment
"$PYTHON_CMD" scripts/check_environment.py && echo "=== Setup complete ==="

echo ""
echo "Next steps:"
echo "  1. Fill in .env with your API keys if needed"
echo "  2. $PYTHON_CMD scripts/check_environment.py"
echo "  3. python run_bilingual_eval.py phi3 --probe-file data/probes/simulation_set_6_probes.jsonl --full"
