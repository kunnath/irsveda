#!/bin/bash

# run_dosha_app.sh
# Script to run the IridoVeda app with dosha analysis enabled

echo "Starting IridoVeda with Dosha Analysis..."

# Check if required libraries are installed
pip install -q streamlit pandas matplotlib numpy opencv-python scikit-learn nltk

# Download NLTK resources
echo "Checking and downloading required NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run the app
streamlit run app.py
