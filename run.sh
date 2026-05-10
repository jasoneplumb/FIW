#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
NOTEBOOK="$SCRIPT_DIR/main.ipynb"
OS="$(uname -s)"
REQUIRED_PYTHON="3.11"

# --- Check for Python 3.11 -------------------------------------------------------
python_version() {
    "$1" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null
}

is_required_python() {
    local candidate="$1"
    [ "$(python_version "$candidate")" = "$REQUIRED_PYTHON" ]
}

python_candidates=("${PYTHON:-}" python3.11)
if [ -n "${PYTHON:-}" ]; then
    python_candidates=("$PYTHON")
fi

if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
    brew_prefix="$(brew --prefix python@3.11 2>/dev/null || true)"
    if [ -n "$brew_prefix" ]; then
        python_candidates+=("$brew_prefix/bin/python3")
    fi
fi

PYTHON=""
for candidate in "${python_candidates[@]}"; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ] || command -v "$candidate" &>/dev/null; then
        if is_required_python "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python $REQUIRED_PYTHON not found."
    echo "  This project uses Python $REQUIRED_PYTHON for its PyTorch and facenet-pytorch dependency stack."
    case "$OS" in
        Darwin) echo "  Install via Homebrew:  brew install python@3.11" ;;
        Linux)  echo "  Install via apt:       sudo apt install python3.11 python3.11-venv" ;;
        *)      echo "  Please install Python $REQUIRED_PYTHON for your platform." ;;
    esac
    exit 1
fi

# On macOS, the system Python ships with LibreSSL which breaks urllib3 v2.
# Reject it when it is the selected interpreter.
if [ "$OS" = "Darwin" ]; then
    SSL_LIB="$("$PYTHON" -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null || true)"
    if [[ "$SSL_LIB" == *LibreSSL* ]]; then
        echo "Warning: selected Python is linked against LibreSSL, which causes urllib3 errors."
        echo "  Fix:  brew install python@3.11"
        exit 1
    fi
fi

# --- Check for Kaggle API key ----------------------------------------------------
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
    echo "Error: Kaggle API key not found."
    echo "  Option 1: Place kaggle.json in ~/.kaggle/"
    echo "  Option 2: Export KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo "  See: https://www.kaggle.com/docs/api"
    exit 1
fi

# --- Create virtual environment ---------------------------------------------------
CREATE_VENV=0
if [ ! -d "$VENV_DIR" ]; then
    CREATE_VENV=1
elif [ ! -x "$VENV_DIR/bin/python" ]; then
    CREATE_VENV=1
elif ! is_required_python "$VENV_DIR/bin/python"; then
    venv_version="$(python_version "$VENV_DIR/bin/python" || echo unknown)"
    echo "Existing virtual environment uses Python $venv_version; recreating it with Python $REQUIRED_PYTHON."
    rm -rf "$VENV_DIR"
    CREATE_VENV=1
elif ! "$VENV_DIR/bin/python" -c "import torch, torchvision, pandas, sklearn, matplotlib, kaggle, notebook, facenet_pytorch" &>/dev/null; then
    CREATE_VENV=1
fi

if [ "$CREATE_VENV" -eq 1 ]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"

    echo "Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip

    # PyTorch: use CPU-only wheels on Linux to avoid downloading CUDA bundles;
    # on macOS pip defaults to the correct variant (MPS-capable on Apple Silicon).
    case "$OS" in
        Darwin)
            "$VENV_DIR/bin/pip" install torch torchvision
            ;;
        *)
            "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac

    "$VENV_DIR/bin/pip" install pandas scikit-learn matplotlib kaggle jupyter facenet-pytorch
fi

source "$VENV_DIR/bin/activate"

if [ "${1:-}" = "--headless" ]; then
    echo "Running notebook headless..."
    jupyter nbconvert --to notebook --execute "$NOTEBOOK" --output main.ipynb
else
    jupyter notebook "$NOTEBOOK"
fi
