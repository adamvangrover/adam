#!/bin/bash

# This script automates the setup and deployment of the ADAM v21.0 platform.

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for prerequisites
echo "Checking for prerequisites..."
if ! command_exists git; then
  echo "Error: Git is not installed." >&2
  exit 1
fi

if ! command_exists docker; then
  echo "Error: Docker is not installed." >&2
  exit 1
fi

if ! command_exists docker-compose; then
  echo "Error: Docker Compose is not installed." >&2
  exit 1
fi
echo "All prerequisites are installed."

# Prompt for configuration values
echo "Please enter your configuration values:"
read -p "OpenAI API Key: " OPENAI_API_KEY
read -p "SharePoint Site URL: " SHAREPOINT_SITE_URL
read -p "SharePoint Username: " SHAREPOINT_USERNAME
read -s -p "SharePoint Password: " SHAREPOINT_PASSWORD
echo
read -p "Data Warehouse Host: " DATA_WAREHOUSE_HOST
read -p "Data Warehouse Port: " DATA_WAREHOUSE_PORT
read -p "Data Warehouse Username: " DATA_WAREHOUSE_USERNAME
read -s -p "Data Warehouse Password: " DATA_WAREHOUSE_PASSWORD
echo
read -p "Data Warehouse Database: " DATA_WAREHOUSE_DATABASE

# Create .env file
echo "Creating .env file..."
cat > .env << EOL
OPENAI_API_KEY=${OPENAI_API_KEY}
SHAREPOINT_SITE_URL=${SHAREPOINT_SITE_URL}
SHAREPOINT_USERNAME=${SHAREPOINT_USERNAME}
SHAREPOINT_PASSWORD=${SHAREPOINT_PASSWORD}
DATA_WAREHOUSE_HOST=${DATA_WAREHOUSE_HOST}
DATA_WAREHOUSE_PORT=${DATA_WAREHOUSE_PORT}
DATA_WAREHOUSE_USERNAME=${DATA_WAREHOUSE_USERNAME}
DATA_WAREHOUSE_PASSWORD=${DATA_WAREHOUSE_PASSWORD}
DATA_WAREHOUSE_DATABASE=${DATA_WAREHOUSE_DATABASE}
EOL
echo ".env file created successfully."

# Build and run Docker containers
echo "Building Docker containers..."
docker-compose build
echo "Starting the system..."
docker-compose up -d

echo "ADAM v21.0 platform has been deployed successfully!"
echo "Web application is available at http://localhost:8080"
echo "API is available at http://localhost:8000"
