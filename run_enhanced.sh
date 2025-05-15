#!/bin/bash
# Run the enhanced application

# Ensure models are downloaded
python -m spacy download en_core_web_sm 2>/dev/null || echo "spaCy model already downloaded"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null || echo "NLTK data already downloaded"

# Start the application
streamlit run advanced_app.py
