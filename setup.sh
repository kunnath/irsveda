#!/bin/bash
# Setup script for irisayush project
echo "Setting up irisayush project..."

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for specific packages that might be missing
if ! pip list | grep -q fpdf; then
    echo "Installing fpdf package specifically..."
    pip install fpdf>=1.7.2
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p fonts

# Download fonts if they don't exist
if [ ! -f "fonts/DejaVuSans.ttf" ]; then
    echo "Downloading DejaVu Sans fonts..."
    curl -L -o fonts/DejaVuSans.ttf https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf
    curl -L -o fonts/DejaVuSans-Bold.ttf https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf
fi

echo "Setup complete. You can now run the application with: streamlit run advanced_app.py"
