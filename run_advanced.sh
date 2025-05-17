#!/bin/bash
# IridoVeda Application Launcher with Advanced Features

echo "🚀 Starting IridoVeda with Advanced Iris Analysis..."

# Set environment variables if needed
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Activate the virtual environment
source venv/bin/activate

# Check if Qdrant is running
echo "🔍 Checking if Qdrant is running..."
curl -s http://localhost:6333/healthz > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Qdrant is not running. Starting with Docker..."
    docker run -d -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_data:/qdrant/storage \
        qdrant/qdrant
    echo "⏳ Waiting for Qdrant to start..."
    sleep 5  # Give Qdrant time to start
fi

# Run the application
echo "👁️ Starting IridoVeda application..."
streamlit run advanced_app.py -- --server.port=8501 --server.address=0.0.0.0
