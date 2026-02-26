#!/bin/bash

# Configuration
VENV_DIR=".venv_voc"
REQUIRED_PYTHON="3.11"
KERNEL_NAME="voc_biometric"
DISPLAY_NAME="Python 3.11 (VOC Training)"

echo "--- Starting Robust Environment Setup for VOC Training ---"

# 1. Detect or Install Python 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN=$(command -v python3.11)
    echo "Found system Python 3.11 at $PYTHON_BIN"
elif [[ "$OSTYPE" == "linux-gnu"* ]] && command -v pyenv &> /dev/null; then
    echo "Python 3.11 not found in path. Trying to find/install via pyenv..."
    if ! pyenv versions | grep -q "3.11"; then
        echo "Installing Python 3.11.10 via pyenv (this may take a while)..."
        # Ensure dependencies for compilation are present if we have sudo
        if command -v sudo &> /dev/null; then
            echo "Attempting to install build dependencies..."
            sudo apt-get update -y && sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils \
            tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        fi
        pyenv install 3.11.10
    fi
    PYTHON_BIN="$(pyenv prefix 3.11.10 2>/dev/null || pyenv prefix 3.11)/bin/python"
else
    echo "Error: Python 3.11 is not installed."
    echo "Please install it using 'sudo apt install python3.11' or 'pyenv install 3.11'"
    exit 1
fi

# 2. Clean up existing venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# 3. Create Virtual Environment
echo "Creating virtual environment using $PYTHON_BIN..."
$PYTHON_BIN -m venv "$VENV_DIR" || { echo "Failed to create venv. Ensure python3-venv is installed."; exit 1; }

# 4. Upgrade pip and install requirements
echo "Installing/Updating dependencies..."
./"$VENV_DIR"/bin/pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    ./"$VENV_DIR"/bin/pip install -r requirements.txt
fi

# 5. Kernel Registration
echo "Registering Jupyter kernel..."
./"$VENV_DIR"/bin/pip install ipykernel
./"$VENV_DIR"/bin/python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$DISPLAY_NAME"

# 6. Final Verification
echo "Verifying environment..."
./"$VENV_DIR"/bin/python -c "import sqlite3; import matplotlib; print('SUCCESS: Environment ready with SQLite and Matplotlib')"

echo "--- Setup Complete! ---"
echo "Select kernel '$DISPLAY_NAME' in your notebook."
