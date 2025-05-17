# IridoVeda: Ayurvedic Iridology Knowledge Base

<p align="center">
  <img src="irs.png" alt="IridoVeda Logo" width="300">
</p>

## Overview

IridoVeda is a Streamlit application that extracts iris-related information from Ayurvedic and Iridology books, stores it as vector embeddings in Qdrant, and provides a query interface to access this knowledge. The application can also analyze iris images to generate relevant health queries and performs advanced iris analysis using computer vision and machine learning techniques.

Powered by [Dinexora](https://www.dinexora.de)

## Features

- PDF upload and extraction of iris-related information
- Focus on "how", "why", and "when" questions about iridology
- Vector database storage using Qdrant
- Natural language querying of the knowledge base
- Comprehensive iris image analysis with zone mapping
- Advanced iris analysis with color, texture, and spot detection
- Pattern matching with similar iris patterns
- Health insights and Ayurvedic recommendations
- PDF and HTML report generation

## Getting Started

### Option 1: Using Docker Compose (Recommended)

1. Make sure Docker and Docker Compose are installed on your system
2. Clone this repository
3. Navigate to the project directory
4. Run the Docker setup script (handles common build issues):

```bash
./setup_docker.sh
```

Or manually run the application:

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
iridoveda/
├── app.py                          # Basic Streamlit UI
├── advanced_app.py                 # Advanced Streamlit UI with enhanced features
├── pdf_extractor.py                # Extract & filter iris text
├── enhanced_pdf_extractor.py       # Advanced text extraction with NLP
├── iris_qdrant.py                  # Basic Qdrant vector DB integration
├── enhanced_iris_qdrant.py         # Advanced vector search capabilities
├── iris_predictor.py               # ML model for iris image analysis
├── iris_zone_analyzer.py           # Advanced zone-based iris analysis
├── iris_report_generator.py        # PDF report generation
├── context_aware_answers.py        # Generate answers from search results
├── ocr_pdf.py                      # OCR processing for scanned PDFs
├── advanced_iris_analyzer.py       # Integration of advanced iris analysis components
├── iris_advanced_segmentation.py   # Enhanced iris boundary detection algorithms
├── iris_feature_extractor.py       # Color, texture, and spot feature extraction
├── iris_pattern_matcher.py         # Pattern storage and similarity search using Qdrant
├── docker-compose.yml              # Multi-container Docker setup
├── Dockerfile                      # Container configuration
├── run.sh                          # Basic mode startup script
├── run_enhanced.sh                 # Enhanced mode startup script
├── setup.sh                        # Basic setup script
├── setup_advanced.sh               # Advanced setup script with all features
├── static/                         # Static assets (logos, etc.)
├── uploads/                        # Directory for uploaded PDFs
├── fonts/                          # Fonts for PDF report generation
├── requirements.txt                # Python dependencies
└── docker-compose.yml              # Docker Compose configuration
```

## Setup and Installation

### Quick Start with Advanced Features

1. Run the setup script to create a virtual environment and install dependencies:

```bash
./setup_advanced.sh
```

2. Start the application with all advanced features:

```bash
./run_advanced.sh
```

3. Access the application at: http://localhost:8501

### Alternative Setup with Conda (Recommended for spacy installation issues)

If you encounter issues installing dependencies with pip, especially with spacy, use the conda setup:

1. Install Miniconda or Anaconda if not already installed
2. Run the conda setup script:

```bash
./setup_conda.sh
```

3. Start the application using the conda environment:

```bash
./run_conda.sh
```

4. Access the application at: http://localhost:8501

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
# For basic version:
streamlit run app.py

# For enhanced version (recommended):
streamlit run advanced_app.py
```

### Using Docker Compose

1. Build and start the containers:

```bash
# Build the containers
docker compose build

# Start in detached mode
docker compose up -d

# View logs if needed
docker compose logs -f
```

2. Access the application at `http://localhost:8501`

3. Stop the containers when done:

```bash
docker compose down
```

## Docker Hub Integration

IridoVeda can be easily deployed using the official Docker image from Docker Hub. This is the recommended approach for production deployments or if you're experiencing build issues.

### Using the Pre-built Image

```bash
docker pull dinexora/iridoveda:latest
docker run -p 8501:8501 dinexora/iridoveda:latest
```

For a complete setup with Qdrant, use the provided docker-compose:

```bash
# Create a docker-compose.yml file with pre-built image configuration
cat > docker-compose.prebuilt.yml << EOL
services:
  app:
    image: dinexora/iridoveda:latest
    ports:
      - "8501:8501"
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
    driver: local
EOL

# Run with the pre-built image configuration
docker compose -f docker-compose.prebuilt.yml up -d
```

### Publishing to Docker Hub

If you've made custom modifications and want to publish your own version to Docker Hub:

1. Run the Docker setup script:

```bash
./setup_docker.sh
```

2. Select option 4: "Build and push to Docker Hub"
3. Follow the prompts to provide:
   - Your Docker Hub username
   - Repository name (default: iridoveda)
   - Image tag (default: latest)
4. The script will build the image, log in to Docker Hub, and push your image

You can then use your custom image by replacing `dinexora/iridoveda:latest` with `yourusername/yourrepo:tag` in the examples above.

## Usage

1. Upload Ayurvedic/Iridology books in PDF format
2. Process the PDFs to extract iris-related information (choose between standard and enhanced processing)
3. Store the extracted information in the knowledge base
4. Query the knowledge base with natural language questions
5. Upload iris images for analysis and recommendations
6. Generate detailed iris zone analysis reports
7. Explore statistics and insights about your knowledge base

## Advanced Features

- **Enhanced NLP Processing**: More accurate text extraction with semantic understanding
- **Multi-Query Search**: Improved search accuracy through query expansion
- **Iris Zone Analysis**: Detailed mapping of iris zones to body systems
- **PDF Report Generation**: Comprehensive reports with Ayurvedic insights
- **Context-Aware Answers**: AI-generated responses synthesized from multiple sources
- **Statistical Insights**: Analyze the content of your knowledge base

## Advanced Iris Analysis

The Advanced Analysis tab provides comprehensive iris analysis using computer vision and machine learning techniques:

### Features

1. **Color Analysis**
   - Identifies dominant colors in the iris
   - Quantifies color distribution percentages
   - Provides Ayurvedic interpretation of colors

2. **Spot Detection**
   - Identifies and counts spots/freckles in the iris
   - Maps spots to organ systems based on iridology principles
   - Offers detoxification suggestions based on spot patterns

3. **Texture Analysis**
   - Analyzes iris fiber patterns and structures
   - Calculates quantitative metrics: contrast, uniformity, energy, entropy
   - Links texture patterns to dosha constitutions

4. **Pattern Matching**
   - Stores iris patterns in a vector database
   - Compares new iris patterns with historical data
   - Provides similarity scores and feature comparisons

5. **Health Insights**
   - Generates comprehensive health assessment
   - Visualizes dosha distribution
   - Provides tailored Ayurvedic recommendations

## Accessing Advanced Analysis

1. Upload an iris image in the "🧬 Advanced Analysis" tab
2. Wait for the system to process the image (typically 5-10 seconds)
3. Explore the different analysis views through the subtabs
4. Generate a comprehensive report with the button at the bottom of the page

## Contact & Support

For questions, support, or custom implementations:
- 📧 Email: [contact@dinexora.de](mailto:contact@dinexora.de)
- 🌐 Website: [www.dinexora.de](https://www.dinexora.de)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Installation Issues

#### Spacy Installation Errors

If you see errors like:
```
Failed to build spacy
ERROR: Could not build wheels for spacy, which is required to install pyproject.toml-based projects
```

Try these solutions:

1. **Update pip first:**
   ```bash
   pip install --upgrade pip
   ```

2. **Install build dependencies separately:**
   ```bash
   pip install wheel setuptools cython
   pip install spacy --no-build-isolation
   ```

3. **Use the Conda installation method:**
   Use the `./setup_conda.sh` script which handles these dependencies better.

#### Docker Build Issues

If you encounter errors when building the Docker image like:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

This is typically caused by a version mismatch between numpy and packages that depend on it. Try these solutions:

1. **Use the pre-built image** (recommended):
   ```bash
   docker pull dinexora/iridoveda:latest
   docker run -p 8501:8501 dinexora/iridoveda:latest
   ```

2. **Build with the fixed Dockerfile**:
   The repository includes an updated Dockerfile that fixes these compatibility issues by:
   - Installing numpy with a fixed version first
   - Installing spacy with the `--no-build-isolation` flag
   - Directly downloading the spacy language model from GitHub

3. **Clean Docker cache and rebuild**:
   ```bash
   docker system prune -a
   docker-compose build --no-cache
   docker-compose up
   ```

#### Missing Language Model Errors

If you encounter errors related to missing spacy language models:

```bash
python -m spacy download en_core_web_sm
```

### Runtime Issues

#### Qdrant Connection Problems

If the application can't connect to Qdrant:

1. Check if the Qdrant container is running:
   ```bash
   docker ps | grep qdrant
   ```

2. Start Qdrant if it's not running:
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

3. Ensure the environment variables are set correctly:
   ```bash
   export QDRANT_HOST=localhost
   export QDRANT_PORT=6333
   ```

#### Image Processing Errors

If you encounter errors when processing iris images:

1. Check if OpenCV is installed correctly
2. Verify that the image is a valid iris image
3. Try processing a different image to see if the issue is with a specific file

# IridoVeda - Powered by Dinexora
