#!/bin/bash
# IridoVeda Advanced Setup Script

echo "🔧 Setting up IridoVeda with Advanced Iris Analysis..."

# Create a Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating a new Python virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Update pip first
echo "🔄 Updating pip to the latest version..."
pip install --upgrade pip

# Install build dependencies first (required for spacy)
echo "📥 Installing build dependencies..."
pip install wheel setuptools cython

# Install spacy separately with --no-build-isolation flag
echo "📥 Installing spacy..."
pip install spacy --no-build-isolation

# Download spacy language models
echo "📥 Downloading spacy language models..."
python -m spacy download en_core_web_sm

# Install the rest of the dependencies
echo "📥 Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p static
mkdir -p fonts

# Check for Qdrant
echo "🔍 Checking if Qdrant is running..."
curl -s http://localhost:6333/healthz > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Qdrant is not running. Starting with Docker..."
    docker run -d -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_data:/qdrant/storage \
        qdrant/qdrant
fi

echo "✅ Setup completed successfully!"
echo "🚀 To start the application, run: ./run_enhanced.sh"
