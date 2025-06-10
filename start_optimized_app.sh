#!/bin/bash

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down

# Build and start the services with optimization
echo "Building and starting optimized services..."
docker-compose build --no-cache
docker-compose up -d

# Wait for Qdrant to be fully healthy
echo "Waiting for Qdrant to be healthy..."
while ! docker-compose exec -T qdrant bash -c "exec 3<>/dev/tcp/localhost/6333; echo -e 'GET /livez HTTP/1.1\r\nHost: localhost\r\n\r\n' >&3; cat <&3 | grep -q '200 OK'" 2>/dev/null; do
  echo "Qdrant is starting up..."
  sleep 2
done

# Show running containers
echo "Services are running:"
docker-compose ps

echo "Access the application at http://localhost:8501"
