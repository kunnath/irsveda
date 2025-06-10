#!/bin/bash

# fix_nltk_docker.sh
# Script to fix NLTK resource issues in Docker container

echo "Fixing NLTK resources in Docker container..."

# Get container ID
CONTAINER_ID=$(docker ps | grep irisayush_app | awk '{print $1}')

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No running container found for irisayush_app"
    echo "Make sure the Docker container is running before executing this script."
    exit 1
fi

echo "Found container: $CONTAINER_ID"

# Execute commands inside the container
docker exec -it $CONTAINER_ID bash -c '
    echo "Inside container, fixing NLTK resources..."
    
    # Install NLTK if not already installed
    pip install -q nltk
    
    # Download all NLTK resources (includes punkt_tab)
    echo "Downloading all NLTK resources..."
    python -c "import nltk; nltk.download(\"all\")"
    
    # Verify punkt_tab is available
    echo "Verifying punkt_tab resource..."
    python -c "import nltk; import os; path = nltk.data.path[0]; punkt_tab_path = os.path.join(path, \"tokenizers/punkt_tab\"); print(\"Checking\", punkt_tab_path); print(\"punkt_tab exists:\", os.path.exists(punkt_tab_path))"
    
    # Test tokenization
    echo "Testing tokenization..."
    python -c "import nltk; from nltk.tokenize import sent_tokenize; print(\"Test sentence tokenization:\", sent_tokenize(\"This is a test. This is another test.\")); print(\"NLTK resources verified successfully!\")"
'

echo "Fix completed. If you still encounter issues, please rebuild the Docker container:"
echo "docker-compose down && docker-compose up --build"
