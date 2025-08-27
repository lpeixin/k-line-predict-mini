#!/bin/bash

# Exit on any error
set -e

echo "Setting up k-line-predict-mini environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Installation complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the stock predictor, use:"
echo "  python main.py <SYMBOL> <MARKET>"
echo "  or"
echo "  stock-predict <SYMBOL> <MARKET>"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"