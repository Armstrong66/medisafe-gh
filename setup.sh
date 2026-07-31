#!/usr/bin/env bash
set -e
echo "=== G-MASS Setup ==="

# 1. Python check
python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" \
  && echo "OK Python $(python3 --version)" \
  || { echo "FAIL Python 3.10+ required"; exit 1; }

# 2. .env
[ -f .env ] && echo "OK .env exists" || {
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "OK .env created - fill in your API keys"
  else
    echo "FAIL .env missing and .env.example was not found"
    exit 1
  fi
}

# 3. Install dependencies
pip install -r requirements.txt --quiet
echo "OK dependencies installed"

# 4. fasttext LID model
mkdir -p scorer/models
[ -f scorer/models/lid.176.ftz ] \
  && echo "OK fasttext LID model present" \
  || { curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
       -o scorer/models/lid.176.ftz && echo "OK fasttext LID downloaded"; }

# 5. Verify current restored pipeline environment
python scripts/check_environment.py && echo "=== Setup complete ==="

echo ""
echo "Next steps:"
echo "  1. Fill in .env with your API keys"
echo "  2. python scripts/check_environment.py"
echo "  3. python run_bilingual_eval.py phi3 --probe-file data/probes/simulation_set_6_probes.jsonl --full"


