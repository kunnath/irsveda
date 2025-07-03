#!/bin/bash

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the Qdrant server with Docker if not already running
if ! docker ps | grep -q qdrant; then
    echo "Starting Qdrant container..."
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
    echo "Waiting for Qdrant to initialize..."
    sleep 5
else
    echo "Qdrant container is already running."
fi

# Run the Streamlit application
echo "Starting AyushIris application..."
streamlit run app.py
