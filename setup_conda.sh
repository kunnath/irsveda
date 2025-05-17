#!/bin/bash
# IridoVeda Advanced Setup Script using Conda
# Use this script if you encounter issues with pip installation

echo "🔧 Setting up IridoVeda with Advanced Iris Analysis using Conda..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create a conda environment
echo "📦 Creating conda environment 'iridoveda'..."
conda create -n iridoveda python=3.10 -y

# Activate the environment
echo "🔌 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iridoveda

# Install core dependencies with conda
echo "📥 Installing core dependencies with conda..."
conda install -c conda-forge spacy scikit-learn scikit-image numpy pandas matplotlib pillow -y
conda install -c pytorch pytorch -y

# Install spacy language model
echo "📥 Installing spacy language model..."
python -m spacy download en_core_web_sm

# Install other dependencies with pip
echo "📥 Installing remaining dependencies with pip..."
pip install streamlit>=1.22.0 pymupdf>=1.21.1 sentence-transformers>=2.2.2 qdrant-client>=1.1.1 pytesseract>=0.3.10 nltk>=3.8.1 fpdf>=1.7.2

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
echo "🚀 To start the application, run: ./run_conda.sh"
