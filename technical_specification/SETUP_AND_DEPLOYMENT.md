# Setup and Deployment Guide

## 1. Introduction

This guide provides step-by-step instructions for setting up and deploying the ADAM v21.0 platform. The system is designed to be deployed using Docker, which ensures a consistent and portable environment.

## 2. Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   **Git:** For cloning the repository.
*   **Docker:** For running the application in containers.
*   **Docker Compose:** For orchestrating the multi-container application.

## 3. Setup

### 3.1. Clone the Repository

First, clone the project repository from GitHub:

```bash
git clone https://github.com/adamvangrover/adam.git
cd adam
```

### 3.2. Configuration

The system is configured using environment variables. You can create a `.env` file in the root of the project to store your configuration. A sample configuration file is provided at `config.sample.json`.

Create a `.env` file and add the following variables:

```
# API Keys
# Your OpenAI API key for accessing the LLM.
OPENAI_API_KEY=<your_openai_api_key>

# SharePoint Configuration
# The URL of the SharePoint site to connect to.
SHAREPOINT_SITE_URL=<your_sharepoint_site_url>
# The username for authenticating with SharePoint.
SHAREPOINT_USERNAME=<your_sharepoint_username>
# The password for authenticating with SharePoint.
SHAREPOINT_PASSWORD=<your_sharepoint_password>

# Data Warehouse Configuration
# The hostname or IP address of the data warehouse.
DATA_WAREHOUSE_HOST=<your_data_warehouse_host>
# The port number of the data warehouse.
DATA_WAREHOUSE_PORT=<your_data_warehouse_port>
# The username for authenticating with the data warehouse.
DATA_WAREHOUSE_USERNAME=<your_data_warehouse_username>
# The password for authenticating with the data warehouse.
DATA_WAREHOUSE_PASSWORD=<your_data_warehouse_password>
# The name of the database to connect to.
DATA_WAREHOUSE_DATABASE=<your_data_warehouse_database>
```

## 4. Automated Deployment

A `deploy.sh` script is provided to automate the setup and deployment process. The script will prompt you for the necessary configuration values and then build and run the system.

To use the script, run the following command from the `technical_specification` directory:

```bash
bash deploy.sh
```

## 5. Manual Deployment

If you prefer to deploy the system manually, follow these steps:

### 5.1. Build the Docker Containers

The `docker-compose.yml` file in the root of the project defines the services that make up the application. To build the Docker containers, run the following command from the root directory of the project:

```bash
docker-compose build
```

### 5.2. Run the System

To run the system, use the following command:

```bash
docker-compose up -d
```

This will start all the services in detached mode.

### 5.3. Verifying the Deployment

To verify that the system is running correctly, you can check the logs for each container:

```bash
docker-compose logs -f <service_name>
```

You can also access the web application by navigating to `http://localhost:8080` in your web browser. The API will be available at `http://localhost:8000`.

## 6. Troubleshooting

**Issue:** The web application is not accessible.

**Solution:**
*   Ensure that the `webapp` container is running (`docker-compose ps`).
*   Check the logs for the `webapp` container for any errors (`docker-compose logs -f webapp`).

**Issue:** The API is returning errors.

**Solution:**
*   Ensure that the `api` and `core` containers are running.
*   Check the logs for the `api` and `core` containers for any errors.
*   Verify that your API keys and other configuration values in the `.env` file are correct.
