#!/bin/bash

echo "=== IrisAyush Application Status ==="
docker-compose ps

echo ""
echo "=== Application Logs ==="
docker-compose logs --tail=50 app

echo ""
echo "=== Qdrant Vector Database Logs ==="
docker-compose logs --tail=20 qdrant

echo ""
echo "To view real-time logs, use: docker-compose logs -f"
