#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
NOTEBOOK="$SCRIPT_DIR/main.ipynb"
OS="$(uname -s)"

# --- Check for python3 -----------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found."
    case "$OS" in
        Darwin) echo "  Install via Homebrew:  brew install python" ;;
        Linux)  echo "  Install via apt:       sudo apt install python3 python3-venv" ;;
        *)      echo "  Please install Python 3.8+ for your platform." ;;
    esac
    exit 1
fi

# On macOS, the system Python ships with LibreSSL which breaks urllib3 v2.
# Prefer Homebrew Python if available.
if [ "$OS" = "Darwin" ]; then
    BREW_PYTHON="$(brew --prefix 2>/dev/null)/bin/python3"
    if [ -x "$BREW_PYTHON" ]; then
        PYTHON="$BREW_PYTHON"
    else
        SSL_LIB="$(python3 -c "import ssl; print(ssl.OPENSSL_VERSION)" 2>/dev/null || true)"
        if [[ "$SSL_LIB" == *LibreSSL* ]]; then
            echo "Warning: system Python is linked against LibreSSL, which causes urllib3 errors."
            echo "  Fix:  brew install python"
            exit 1
        fi
        PYTHON="python3"
    fi
else
    PYTHON="python3"
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
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
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
