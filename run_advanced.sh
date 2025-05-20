#!/bin/bash
# IridoVeda Application Launcher with Advanced Features

echo "🚀 Starting IridoVeda with Advanced Iris Analysis..."

# Set environment variables if needed
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Activate the virtual environment
source venv/bin/activate

# Start services using Docker Compose
echo "🔍 Starting services with Docker Compose..."
docker-compose up -d

# Check if services are running
echo "⏳ Waiting for services to be ready..."
sleep 5  # Give services time to start

# Run the application
if [ "$1" == "local" ]; then
    # Run application locally (outside Docker)
    echo "👁️ Starting IridoVeda application locally..."
    streamlit run advanced_app.py -- --server.port=8501 --server.address=0.0.0.0
else
    # Application is already running in Docker
    echo "👁️ IridoVeda application is running in Docker container"
    echo "📊 Access the application at http://localhost:8501"
    echo "💡 Use Ctrl+C to stop watching logs"
    docker-compose logs -f app
fi
