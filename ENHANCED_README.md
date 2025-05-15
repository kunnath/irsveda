# Enhanced Iridology Knowledge Base - Advanced PDF Analysis

This project provides advanced PDF analysis for iridology and Ayurvedic texts using advanced NLP techniques and vector search.

## Features

### Advanced PDF Processing
- Context-aware text extraction with semantic understanding
- Advanced OCR with higher quality processing
- Keyword and entity recognition
- Relevance scoring for chunks
- Duplicate detection with semantic similarity

### Enhanced Vector Search
- Hybrid search combining vector similarity with keyword matching
- Multi-query expansion for improved recall
- Highlighting of relevant text passages
- Contextual ranking algorithm
- Integration with a more robust embedding model

### Context-Aware Answer Generation
- Synthesizes answers from multiple sources
- Highlights key insights from search results
- Extracts relevant entities and keywords
- Provides confidence scoring
- Lists sources with relevance metrics

## Installation

### Using Docker
```bash
docker-compose build
docker-compose up -d
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install models and additional data
./setup_enhanced.sh

# Run the application
streamlit run advanced_app.py
```

## Architecture

The system consists of several interconnected components:

1. **Enhanced PDF Extractor** - Processes PDFs using NLP techniques to extract structured chunks
2. **Enhanced Qdrant Client** - Provides improved vector search capabilities
3. **Context-Aware Answer Generator** - Generates coherent answers from multiple sources
4. **Advanced UI** - Rich Streamlit interface with visualization and control

## Usage

1. Start the application
2. Upload PDF files in the "PDF Upload & Processing" tab
3. Choose between standard and enhanced processing
4. Store the extracted chunks in the knowledge base
5. Use the "Knowledge Query" tab to search the knowledge base
6. View statistics and insights in the "Statistics" tab

## Configuration

The application provides a configuration tab where you can customize:
- Vector search parameters
- NLP processing settings
- Extraction parameters
- Collection names and models

## Comparison with Standard Mode

The enhanced mode provides several advantages:

| Feature | Standard Mode | Enhanced Mode |
|---------|--------------|--------------|
| Text Extraction | Basic paragraph splitting | NLP-aware text extraction |
| Relevance Scoring | Basic vector similarity | Hybrid scoring with keyword boosting |
| OCR Processing | Basic | Higher resolution with custom configuration |
| Answer Generation | None | Context-aware answer synthesis |
| Content Filtering | Keyword-based | Semantic and keyword-based |
| Visualization | Basic | Advanced metrics and visualizations |

## Requirements

- Python 3.10+
- Required Python packages (see requirements.txt)
- spaCy and NLTK models
- Tesseract OCR
- Qdrant vector database
