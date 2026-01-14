#!/bin/bash

# VAID JI Quick Setup Script
# This script automates the setup process for VAID JI

set -e  # Exit on error

echo "================================================"
echo "   VAID JI - Medical Research Assistant Setup   "
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
ENV_NAME="medical_assistant_env"
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source $ENV_NAME/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your API keys:"
    echo "   - OPENROUTER_API_KEY (get from https://openrouter.ai/)"
    echo "   - GROQ_API_KEY (get from https://console.groq.com/)"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p chroma_db
mkdir -p .sentence_transformers_cache
echo "✅ Directories created"
echo ""

# Run configuration validation
echo "Validating configuration..."
python config.py || echo "⚠️  Configuration validation failed. Please check your .env file."
echo ""

echo "================================================"
echo "              Setup Complete! 🎉                "
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "2. Make sure you've added API keys to .env file"
echo ""
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "For more information, see README.md and QUICKSTART.md"
echo ""
echo "Happy researching! 📚🔬"
