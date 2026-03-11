#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
NOTEBOOK="$SCRIPT_DIR/main.ipynb"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install torch torchvision pandas scikit-learn matplotlib kaggle jupyter
fi

source "$VENV_DIR/bin/activate"

if [ "${1:-}" = "--headless" ]; then
    echo "Running notebook headless..."
    jupyter nbconvert --to notebook --execute "$NOTEBOOK" --output main.ipynb
else
    jupyter notebook "$NOTEBOOK"
fi
