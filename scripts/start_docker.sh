#!/bin/bash
# Script to start Adam in Docker environment

# Ensure .env exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "Error: .env.example not found."
        exit 1
    fi
fi

echo "Starting Adam via Docker Compose..."
docker-compose up --build
