# AyushIris: Ayurvedic Iridology Knowledge Base

<p align="center">
  <img src="irs.png" alt="AyushIris Logo" width="300">
</p>

## Overview

AyushIris is a Streamlit application that extracts iris-related information from Ayurvedic and Iridology books, stores it as vector embeddings in Qdrant, and provides a query interface to access this knowledge. The application can also analyze iris images to generate relevant health queries.

## Features

- PDF upload and extraction of iris-related information
- Focus on "how", "why", and "when" questions about iridology
- Vector database storage using Qdrant
- Natural language querying of the knowledge base
- (Optional) Iris image analysis and health prediction

## Getting Started

### Option 1: Using Docker Compose (Recommended)

1. Make sure Docker and Docker Compose are installed on your system
2. Clone this repository
3. Navigate to the project directory
4. Run the application:

```bash
docker-compose up
```

5. Access the application at: http://localhost:8501

### Option 2: Local Setup

1. Install Python 3.10 or higher
2. Install Tesseract OCR (for PDF OCR capabilities)
3. Run Qdrant vector database with Docker:

```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
```

4. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. Run the application:

```bash
./run.sh
```

## Project Structure

```
irisayush/
├── app.py                 # Streamlit UI
├── pdf_extractor.py       # Extract & filter iris text
├── iris_qdrant.py         # Qdrant vector DB integration
├── iris_predictor.py      # ML model for iris image analysis
├── ocr_pdf.py             # OCR processing for scanned PDFs
├── docker-compose.yml     # Multi-container Docker setup
├── Dockerfile             # Container configuration
├── run.sh                 # Local startup script
├── static/                # Static assets (logos, etc.)
└── uploads/               # Directory for uploaded PDFs
├── qdrant_client.py       # Embedding & vector DB interface
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build instructions
└── docker-compose.yml     # Docker Compose configuration
```

## Setup and Installation

### Local Development

1. Set up a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Qdrant (either locally with Docker or connect to a cloud instance):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

### Using Docker Compose

1. Build and start the containers:

```bash
docker-compose up --build
```

2. Access the application at `http://localhost:8501`

## Usage

1. Upload Ayurvedic/Iridology books in PDF format
2. Process the PDFs to extract iris-related information
3. Store the extracted information in the knowledge base
4. Query the knowledge base with natural language questions
5. (Optional) Upload iris images for analysis and recommendations

## Future Enhancements

- OCR support for image-based PDFs
- Multi-language support (Sanskrit, Malayalam, etc.)
- Advanced iris image analysis with deep learning
- Integration with electronic health records
- Mobile-friendly UI for field use by practitioners

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# irsveda
