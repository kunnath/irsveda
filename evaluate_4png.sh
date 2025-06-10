#!/bin/bash
# Script to evaluate 4.png iris image

echo "==========================================="
echo "Iris Recognition System - Evaluating 4.png"
echo "==========================================="

# Make sure we have a datasets directory
mkdir -p datasets/casia_thousand

# Check if Qdrant is running using our new connection checker
echo "Checking if Qdrant is available..."
python check_qdrant_connection.py
QDRANT_AVAILABLE=$?

if [ $QDRANT_AVAILABLE -ne 0 ]; then
    echo "WARNING: Qdrant is not available. Attempting to start with Docker..."
    
    # Check if Docker is running
    docker ps > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: Docker doesn't seem to be running. Please start Docker first."
        echo "If you want to continue without Qdrant (limited functionality),"
        echo "run with: python evaluate_iris_image.py --force-offline 4.png"
        exit 1
    fi
    
    # Start Qdrant with Docker Compose
    echo "Starting Qdrant container..."
    docker-compose up -d qdrant
    
    # Wait for Qdrant to start (up to 30 seconds)
    echo "Waiting for Qdrant to start (max 30 seconds)..."
    for i in {1..6}; do
        sleep 5
        echo "Checking Qdrant connection (attempt $i)..."
        python check_qdrant_connection.py
        if [ $? -eq 0 ]; then
            echo "Qdrant is now available!"
            QDRANT_AVAILABLE=0
            break
        fi
    done
    
    if [ $QDRANT_AVAILABLE -ne 0 ]; then
        echo "WARNING: Qdrant still not available after waiting. Will continue with limited functionality."
        read -p "Press Enter to continue or Ctrl+C to cancel..."
    fi
fi

# Run the evaluation script with setup
if [ $QDRANT_AVAILABLE -eq 0 ]; then
    echo -e "\nSetting up dataset and evaluating 4.png with full functionality..."
    python iris_recognition_model.py --build --force --dataset casia_thousand --evaluate 4.png
    
    # For comparison, run the other evaluation tool as well
    echo -e "\nRunning alternative evaluation on 4.png..."
    python evaluate_iris_image.py --setup 4.png
else
    # Run in offline mode
    echo -e "\nRunning in offline mode with limited functionality..."
    python evaluate_iris_image.py --force-offline 4.png
fi

echo -e "\nEvaluation complete. Results are in the evaluation_results directory."
