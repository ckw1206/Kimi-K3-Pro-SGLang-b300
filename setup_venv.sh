#!/bin/sh
# Create venv and install dependencies (avoids system Python / externally-managed-environment).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Activate the venv and run the test:"
echo "  source .venv/bin/activate"
echo "  python test_openai_api.py"
