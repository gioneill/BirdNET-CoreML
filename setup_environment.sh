#!/bin/bash

# BirdNET-CoreML Environment Setup Script
# This script creates a clean Python virtual environment and installs all dependencies

set -e  # Exit on any error

VENV_NAME="birdnet_clean_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🐦 Setting up BirdNET-CoreML environment..."

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📍 Using Python $PYTHON_VERSION"

# Create virtual environment
echo "🔧 Creating virtual environment: $VENV_NAME"
cd "$SCRIPT_DIR"
rm -rf "$VENV_NAME"
python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing TensorFlow for macOS..."
pip install tensorflow-macos==2.15.0

echo "📦 Installing CoreML Tools..."
pip install coremltools==8.3.0

# Install additional dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing additional dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "📦 Installing common dependencies..."
    pip install numpy librosa scikit-learn
fi

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  cd $SCRIPT_DIR"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To test the export script, run:"
echo "  python coreml_export/export_coreml_gpt03.py --help"
echo ""
echo "🎉 You're ready to convert BirdNET models to CoreML!"
