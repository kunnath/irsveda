#!/bin/bash
# IridoVeda Docker Setup Script
# This script handles common Docker build issues

# Function to push to Docker Hub
push_to_dockerhub() {
    echo "🚀 Preparing to push to Docker Hub..."
    
    # Ask for Docker Hub credentials
    read -p "Enter Docker Hub username: " docker_username
    if [ -z "$docker_username" ]; then
        echo "❌ Username cannot be empty. Aborting push."
        return 1
    fi
    
    # Ask for repository name or use default
    read -p "Enter repository name (default: iridoveda): " docker_repo
    docker_repo=${docker_repo:-iridoveda}
    
    # Ask for tag or use default
    read -p "Enter tag (default: latest): " docker_tag
    docker_tag=${docker_tag:-latest}
    
    # Ask for image description
    read -p "Enter image description (optional): " docker_description
    
    full_image_name="$docker_username/$docker_repo:$docker_tag"
    
    echo "📋 Image will be tagged as: $full_image_name"
    if [ ! -z "$docker_description" ]; then
        echo "📝 Description: $docker_description"
    fi
    read -p "Continue? (y/n): " confirm
    
    if [ "$confirm" != "y" ]; then
        echo "❌ Push aborted."
        return 1
    fi
    
    # Tag the image
    echo "🔖 Tagging image..."
    docker tag irisayush_app:latest $full_image_name
    
    # Login to Docker Hub
    echo "🔑 Logging in to Docker Hub..."
    docker login -u $docker_username
    
    if [ $? -ne 0 ]; then
        echo "❌ Login failed. Aborting push."
        return 1
    fi
    
    # Push the image
    echo "⬆️ Pushing image to Docker Hub..."
    docker push $full_image_name
    
    if [ $? -eq 0 ]; then
        echo "✅ Image successfully pushed to Docker Hub as $full_image_name"
    else
        echo "❌ Failed to push image to Docker Hub."
        return 1
    fi
    
    return 0
}

echo "🐳 Setting up IridoVeda with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Ask the user which setup method they want to use
echo "Please select a setup method:"
echo "1. Build from source (may take longer but provides latest code)"
echo "2. Use pre-built image (faster, recommended if having build issues)"
echo "3. Clean everything and rebuild from scratch"
echo "4. Build and push to Docker Hub"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔨 Building Docker images from source..."
        docker compose build
        ;;
    2)
        echo "🔽 Using pre-built Docker image..."
        # Create a temporary docker-compose file that uses the pre-built image
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
        echo "🚀 Starting containers with pre-built image..."
        docker compose -f docker-compose.prebuilt.yml pull
        docker compose -f docker-compose.prebuilt.yml up -d
        exit 0
        ;;
    3)
        echo "🧹 Cleaning Docker cache and rebuilding from scratch..."
        docker compose down
        docker system prune -a --volumes --force
        docker compose build --no-cache
        ;;
    4)
        echo "🏗️ Building and pushing Docker image to Docker Hub..."
        
        # Build the image first
        echo "🔨 Building Docker image..."
        docker compose build
        
        if [ $? -ne 0 ]; then
            echo "❌ Build failed. Cannot push to Docker Hub."
            exit 1
        fi
        
        # Push to Docker Hub
        push_to_dockerhub
        
        # Ask if user wants to run the containers locally after pushing
        read -p "Do you want to run the containers locally? (y/n): " run_local
        
        if [ "$run_local" == "y" ]; then
            echo "🚀 Starting containers..."
            docker compose up -d
        else
            echo "✅ Image built and pushed successfully. Exiting without running containers."
            exit 0
        fi
        ;;
    *)
        echo "❌ Invalid choice. Exiting..."
        exit 1
        ;;
esac

# Start the containers
echo "🚀 Starting containers..."
docker compose up -d

# Check if containers are running
if [ $(docker compose ps -q | wc -l) -eq 2 ]; then
    echo "✅ IridoVeda is now running!"
    echo "🌐 Access the application at: http://localhost:8501"
else
    echo "❌ Something went wrong. Check Docker logs:"
    echo "   docker compose logs"
fi
