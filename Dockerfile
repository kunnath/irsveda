FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR and OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with specific versions to avoid compatibility issues
COPY requirements.txt .

# Install numpy first with a specific version
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fpdf>=1.7.2

# Download NLTK resources - ensuring punkt_tab is available
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
# Download all NLTK corpora to ensure punkt_tab is available
RUN python -c "import nltk; nltk.download('all')"
# Verify punkt_tab is available
RUN python -c "import nltk; import os; print('NLTK Data Path:', nltk.data.path); print('punkt_tab exists:', os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers/punkt_tab')))"

# Copy application code
COPY . .

# Create uploads and fonts directories
RUN mkdir -p uploads && chmod 777 uploads
RUN mkdir -p fonts && chmod 777 fonts

# Ensure fonts are available for PDF generation
RUN apt-get update && apt-get install -y curl \
    && curl -L -o fonts/DejaVuSans.ttf https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf \
    && curl -L -o fonts/DejaVuSans-Bold.ttf https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies and models with proper versioning
RUN pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir spacy==3.5.3 --no-build-isolation && \
    python -m pip install --no-cache-dir --no-deps https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl

# Expose port for Streamlit
EXPOSE 8501

# Run the application - using the advanced_app.py with optimized features
CMD ["streamlit", "run", "advanced_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
