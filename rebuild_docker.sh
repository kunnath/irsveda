#!/bin/bash

# rebuild_docker.sh
# Script to rebuild Docker containers with NLTK fixes

echo "Rebuilding Docker containers with NLTK fixes..."

# Create the NLTK data directory if it doesn't exist
mkdir -p nltk_data

# Stop existing containers
echo "Stopping existing containers..."
docker-compose down

# Clean dangling images
echo "Cleaning dangling images..."
docker image prune -f

# Rebuild without using cache
echo "Rebuilding containers..."
docker-compose up --build --no-cache -d

# Wait for containers to start
echo "Waiting for containers to start..."
sleep 10

# Verify NLTK resources in the container
echo "Verifying NLTK resources in the container..."
CONTAINER_ID=$(docker ps | grep irisayush_app | awk '{print $1}')

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No running container found for irisayush_app"
    echo "Container may have failed to start. Check docker-compose logs with: docker-compose logs"
    exit 1
fi

echo "Running verification in container $CONTAINER_ID..."
docker exec -it $CONTAINER_ID bash -c '
    echo "Checking NLTK resources inside container..."
    python -c "import nltk; import os; print(\"NLTK data paths:\", nltk.data.path)"
    
    # Check for punkt_tab
    python -c "import nltk; import os; punkt_tab_exists=False; 
    for path in nltk.data.path:
        if os.path.exists(os.path.join(path, \"tokenizers/punkt_tab\")):
            punkt_tab_exists=True
            print(\"Found punkt_tab at\", os.path.join(path, \"tokenizers/punkt_tab\"))
            break
    if not punkt_tab_exists:
        print(\"punkt_tab not found, downloading all NLTK resources...\")
        nltk.download(\"all\")
    else:
        print(\"punkt_tab resource is available\")"
    
    # Test tokenization
    python -c "import nltk; from nltk.tokenize import sent_tokenize; print(\"Test tokenization:\", sent_tokenize(\"This is a test. This is another test.\"))"
'

echo "Containers rebuilt. The application should be running at http://localhost:8501"
echo "If you still encounter NLTK issues, please run ./fix_nltk_docker.sh"
