#!/bin/bash
# Script to stop all Docker Compose services

echo "🛑 Stopping all IridoVeda services..."
docker-compose down

echo "✅ All services stopped"
