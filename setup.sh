#!/usr/bin/env bash
# G-MASS Setup Script
# Run once after cloning: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   G-MASS · MediSafe-GH · Setup               ║"
echo "║   Africa AI Safety Prize 2026                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required="3.11"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "      ✓ Python $python_version"
else
    echo "      ✗ Python $python_version detected — Python 3.11+ required"
    echo "        Install from: https://www.python.org/downloads/"
    exit 1
fi

# ── 2. Create .env from template ─────────────────────────────
echo "[2/6] Setting up .env file..."
if [ -f ".env" ]; then
    echo "      ✓ .env already exists — skipping (edit manually if needed)"
else
    cp .env.example .env
    echo "      ✓ .env created from .env.example"
    echo "      ⚠ IMPORTANT: Open .env and fill in your API keys before running gmass"
fi

# ── 3. Install Python package ─────────────────────────────────
echo "[3/6] Installing medisafe-gh package..."
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"
echo "      ✓ Package installed (gmass CLI available)"

# ── 4. Download fasttext LID model ───────────────────────────
echo "[4/6] Downloading fasttext language identification model..."
mkdir -p models/fasttext
LID_PATH="models/fasttext/lid.176.bin"
if [ -f "$LID_PATH" ]; then
    echo "      ✓ Already downloaded: $LID_PATH"
else
    curl -sL --progress-bar \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
        -o "$LID_PATH"
    echo "      ✓ Downloaded: $LID_PATH (~1MB)"
fi

# ── 5. Ollama setup (local models) ────────────────────────────
echo "[5/6] Checking Ollama for local model inference..."
if command -v ollama &> /dev/null; then
    echo "      ✓ Ollama installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    echo ""
    echo "      To pull local models (do this once, then keep 'ollama serve' running):"
    echo "        ollama pull llama3.2:3b"
    echo "        ollama pull phi3:mini"
    echo ""
else
    echo "      ⚠ Ollama not found — needed for LLaMA-3.2 3B and Phi-3 Mini"
    echo "        Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "        Then:    ollama pull llama3.2:3b && ollama pull phi3:mini"
    echo ""
fi

# ── 6. Verify gmass CLI ───────────────────────────────────────
echo "[6/6] Verifying gmass CLI..."
if gmass --version &> /dev/null; then
    echo "      ✓ $(gmass --version)"
else
    echo "      ✗ gmass CLI not found — check pip install step above"
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Setup complete ✓                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Fill in your API keys in .env"
echo "  2. Run probe check:  gmass probe check"
echo "  3. Dry run:          gmass evaluate --model gpt-4o-mini --language english --dry-run"
echo "  4. Pilot batch:      gmass evaluate --model gpt-4o-mini --language english"
echo ""
echo "For nohup batch runs on RTX:"
echo "  nohup bash scripts/run_full_eval.sh > logs/nohup_out.txt 2>&1 &"
echo "  tail -f logs/gmass_eval.log"
echo ""