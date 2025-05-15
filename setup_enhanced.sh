#!/bin/bash
# Setup script for enhanced PDF processing system

echo "Setting up enhanced PDF processing system..."

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Installing additional dependencies..."
pip install scikit-learn

echo "Setup complete!"
echo "To start the enhanced application, run: streamlit run advanced_app.py"
