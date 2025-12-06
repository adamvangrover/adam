#!/bin/bash

# Adam Project - One-Click Launcher

echo "Starting Adam Project..."

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env from example..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "Warning: No .env or .env.example found. Configuration might be missing."
    fi
fi

# Check if Docker is installed and running
if command -v docker-compose &> /dev/null; then
    echo "Docker Compose found. Starting services..."
    docker-compose up -d
    echo "Services started."
    echo "UI available at http://localhost:80" # Assuming port 80 from docker-compose
    echo "API available at http://localhost:5001"
elif command -v docker &> /dev/null; then
    echo "Docker found (Compose V2). Starting services..."
    docker compose up -d
    echo "Services started."
    echo "UI available at http://localhost:80"
    echo "API available at http://localhost:5001"
else
    echo "Docker not found. Attempting local startup..."

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 not found."
        exit 1
    fi

    echo "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e .

    # Export localhost connection strings for local execution
    export REDIS_URL="redis://localhost:6379/0"
    export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/adam"
    export NEO4J_URI="bolt://localhost:7687"

    echo "Starting Backend..."
    python scripts/run_adam.py &
    BACKEND_PID=$!

    echo "Starting UI (requires npm)..."
    if command -v npm &> /dev/null; then
        cd services/webapp/client
        npm install --legacy-peer-deps
        npm start &
        UI_PID=$!
        cd ../../..
    else
        echo "Warning: npm not found. UI will not start."
    fi

    echo "Adam running locally."
    echo "Press Ctrl+C to stop."

    wait $BACKEND_PID $UI_PID
fi
